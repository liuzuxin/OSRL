import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


# "normal" or "uniform" or None
INIT_METHOD = "normal"


def mlp(sizes, activation, output_activation=nn.Identity):
    if INIT_METHOD == "normal":
        initializer = nn.init.xavier_normal_
    elif INIT_METHOD == "uniform":
        initializer = nn.init.xavier_uniform_
    else:
        initializer = None
    bias_init = 0.0
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        if initializer is not None:
            # init layer weight
            initializer(layer.weight)
            nn.init.constant_(layer.bias, bias_init)
        layers += [layer, act()]
    return nn.Sequential(*layers)


class MLPGaussianPerturbationActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, phi=0.05, act_limit=1):
        super().__init__()
        pi_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit
        self.phi = phi

    def forward(self, obs, act):
        # Return output from network scaled to action space limits.
        a = self.phi * self.act_limit * self.pi(torch.cat([obs, act], 1))
        return (a + act).clamp(-self.act_limit, self.act_limit)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit=1):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_sizes,
                 activation, device="cpu"):
        super().__init__()
        self.device = device
        self.action_low = torch.nn.Parameter(
                            torch.tensor(action_low, device=device)[None, ...], 
                            requires_grad=False)  # (1, act_dim)
        self.action_high = torch.nn.Parameter(
                            torch.tensor(action_high, device=device)[None, ...],
                            requires_grad=False)  # (1, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = torch.sigmoid(self.mu_net(obs))
        mu = self.action_low + (self.action_high - self.action_low) * mu
        std = torch.exp(self.log_std)
        return mu, Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, deterministic=False):
        '''
        Produce action distributions for given observations, and
        optionally compute the log likelihood of given actions under
        those distributions.
        If act is None, sample an action
        '''
        mu, pi = self._distribution(obs)
        if act is None:
            act = pi.sample()
        if deterministic:
            act = mu
        logp_a = self._log_prob_from_distribution(pi, act)
        return pi, act, logp_a


LOG_STD_MAX = 2
LOG_STD_MIN = -20
class SquashedGaussianMLPActor(nn.Module):
    '''
    Probablistic actor, can also be used as a deterministic actor
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self,
                obs,
                deterministic=False,
                with_logprob=True,
                with_distribution=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # print("actor: ", torch.sum(mu), torch.sum(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        return [torch.squeeze(q(data), -1) for q in self.q_nets]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)


class EnsembleDoubleQCritic(nn.Module):
    '''
    An ensemble of double Q network to address the overestimation issue.
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q1_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])
        self.q2_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        q1 = [torch.squeeze(q(data), -1) for q in self.q1_nets]
        q2 = [torch.squeeze(q(data), -1) for q in self.q2_nets]
        return q1, q2

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        # qs = torch.vstack(q_list)  # [num_q, batch_size]
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values
        return qs1_min, qs2_min, q1_list, q2_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)


class VAE(nn.Module):
    """
    Variational Auto-Encoder
    """
    def __init__(self, obs_dim, act_dim, hidden_size, latent_dim, act_lim, device="cpu"):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(obs_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, act_dim)

        self.act_lim = act_lim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, obs, act):
        z = F.relu(self.e1(torch.cat([obs, act], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs, z=None):
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).clamp(-0.5, 0.5).to(self.device)
            
        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.act_lim * torch.tanh(self.d3(a))
