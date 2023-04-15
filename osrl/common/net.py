from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch import distributions as pyd

import numpy as np
import math


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)


class MLPGaussianPerturbationActor(nn.Module):
    """
    A MLP actor that adds Gaussian noise to the output.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        phi (float): The standard deviation of the Gaussian noise to add to the output.
        act_limit (float): The absolute value of the limits of the action space.
    """

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
    """
    A MLP actor
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        act_limit (float, optional): The upper limit of the action space.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit=1):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPGaussianActor(nn.Module):
    """
    A MLP Gaussian actor
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        action_low (np.ndarray): A 1D numpy array of lower bounds for each action dimension.
        action_high (np.ndarray): A 1D numpy array of upper bounds for each action dimension.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        device (str): The device to use for computation (cpu or cuda).
    """
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
    A MLP Gaussian actor, can also be used as a deterministic actor
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
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
                with_distribution=False,
                return_pretanh_value=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # for BEARL only
        if return_pretanh_value:
            return torch.tanh(pi_action), pi_action
        
        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act=None):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = obs if act is None else torch.cat([obs, act], dim=-1)
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
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
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
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_size (int): The number of hidden units in the encoder and decoder networks.
        latent_dim (int): The dimensionality of the latent space.
        act_lim (float): The upper limit of the action space.
        device (str): The device to use for computation (cpu or cuda).
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

    # for BEARL only
    def decode_multiple(self, obs, z=None, num_decode=10):
        if z is None:
            z = torch.randn((obs.shape[0], num_decode, self.latent_dim)).clamp(-0.5, 0.5).to(self.device)
            
        a = F.relu(self.d1(torch.cat([obs.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)


class LagrangianPIDController:
    '''
    Lagrangian multiplier controller
    
    Args:
        KP (float): The proportional gain.
        KI (float): The integral gain.
        KD (float): The derivative gain.
        thres (float): The setpoint for the controller.
    '''

    def __init__(self, KP, KI, KD, thres) -> None:
        super().__init__()
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.thres = thres
        self.error_old = 0
        self.error_integral = 0

    def control(self, qc):
        '''
        @param qc [batch,]
        '''
        error_new = torch.mean(qc - self.thres)  # [batch]
        error_diff = F.relu(error_new - self.error_old)
        self.error_integral = torch.mean(F.relu(self.error_integral + error_new))
        self.error_old = error_new

        multiplier = F.relu(self.KP * F.relu(error_new) + self.KI * self.error_integral +
                          self.KD * error_diff)
        return torch.mean(multiplier)


# Decision Transformer implementation
class TransformerBlock(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x
    
    
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)
    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute the empirical entropy:
        x = self.rsample((N, ))
        log_p = self.log_prob(x)

        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # sum up along the action dimensions
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """
    torch.distributions implementation of an diagonal Gaussian policy.
    """

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        std = log_std.exp()
        return Normal(mu, std)
