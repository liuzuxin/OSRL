import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch import distributions as pyd
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa

from osrl.common.net import EnsembleQCritic, SquashedGaussianMLPActor


class PDCA(nn.Module):
    """
    Offline Constrained Policy Optimization
    via stationary DIstribution Correction Estimation (COptiDICE)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        observations_std (np.ndarray): The standard deviation of the observation space.
        actions_std (np.ndarray): The standard deviation of the action space.
        a_hidden_sizes (list): List of integers specifying the sizes
                               of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes
                               of the layers in the critic network (nu and chi networks).
        gamma (float): Discount factor for the reward.
        alpha (float): The coefficient for the cost term in the loss function.
        cost_ub_epsilon (float): A small value added to the upper bound on the cost term.
        num_nu (int): The number of critics to use for the nu-network.
        num_chi (int): The number of critics to use for the chi-network.
        cost_threshold (float): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 gamma: float = 0.99,
                 B: float = 5,
                 cost_threshold: float = 10,
                 device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.gamma = gamma
        self.B = B
        self.cost_threshold = cost_threshold
        self.device = device

        # actor
        self.actor = SquashedGaussianMLPActor(
                self.state_dim,
                self.action_dim,
                self.a_hidden_sizes,
                nn.ReLU).to(self.device)

        # critics
        self.reward_network = EnsembleQCritic(
                self.state_dim,
                self.action_dim,
                self.c_hidden_sizes,
                nn.ReLU,
                num_q=1).to(self.device)
        self.cost_network = EnsembleQCritic(
                self.state_dim,
                self.action_dim,
                self.c_hidden_sizes,
                nn.ReLU,
                num_q=1).to(self.device)

        # lambda
        self.lambdas = 0

    def func_E(self, f, R, policy, states, actions, next_states, dones):
        next_actions, *_ = policy.forward(next_states)
        X = f(states, actions)[0] - R - self.gamma * f(next_states, next_actions)[0] * dones
        sum_positive = torch.mean(torch.clamp(X, min=0))
        sum_negative = torch.mean(torch.clamp(X, max=0))
        return torch.max(sum_positive, -sum_negative)

    def func_A(self, f, policy, states, actions):
        policy_actions, *_ = policy.forward(states)
        return f(states, policy_actions)[0] - f(states, actions)[0]

    def combined_reward(self, states, actions):
        return self.reward_network(states, actions)[0] + (
            self.cost_threshold - self.cost_network(states, actions)[0]
        ) * self.lambdas

    def update(self, batch):
        states, next_states, actions, rewards, costs, dones, is_init = batch

        # reward critic
        self.reward_optimizer.zero_grad()
        loss_reward = (
            2 * self.func_E(self.reward_network, rewards, self.actor, states, actions, next_states, dones) +
            self.func_A(self.reward_network, self.actor, states, actions)
        )
        loss_reward.sum().backward()
        nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
        self.reward_optimizer.step()

        # cost critic
        loss_cost = (
            2 * self.func_E(self.cost_network, costs, self.actor, states, actions, next_states, dones) -
            self.func_A(self.cost_network, self.actor, states, actions)
        )
        loss_cost.sum().backward()
        nn.utils.clip_grad_norm_(self.cost_network.parameters(), max_norm=1.0)
        self.cost_optimizer.step()

        # actor
        self.actor_optimizer.zero_grad()
        loss_actor = -self.func_A(self.combined_reward, self.actor, states, actions)
        loss_actor.sum().backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        stats_loss = {
            "loss/cost_critic_loss": loss_cost.mean().item(),
            "loss/reward_critic_loss": loss_reward.mean().item(),
            "loss/actor_loss": loss_actor.mean().item(),
        }
        return stats_loss

    def update_lambdas(self, init_batch):
        init_states, = init_batch
        with torch.no_grad():
            init_actions, *_ = self.actor(init_states)
            w = torch.mean(self.cost_threshold - self.cost_network(init_states, init_actions)[0], axis=0)
            self.lambdas = self.B if w < 0 else 0
        return {
            "params/lambda": self.lambdas,
        }

    def setup_optimizers(self, actor_lr, critic_lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.reward_optimizer = torch.optim.Adam(
            self.reward_network.parameters(),
            lr=critic_lr
        )
        self.cost_optimizer = torch.optim.Adam(
            self.cost_network.parameters(),
            lr=critic_lr
        )

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self.actor.forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()
        return np.squeeze(a, axis=0), np.squeeze(logp_a)


class PDCATrainer:
    """
    PDCA trainer

    Args:
        model (COptiDICE): The COptiDICE model to train.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic (nu and chi networks)
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(self,
                 model: PDCA,
                 env: gym.Env,
                 logger: WandbLogger = DummyLogger(),
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 device="cpu"):
        self.model = model
        self.logger = logger
        self.env = env
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr)

    def train_one_step(self, batch, init_batch):
        stats_loss = self.model.update(batch)
        stats_lambdas = self.model.update_lambdas(init_batch)
        self.logger.store(**stats_loss, **stats_lambdas)

    def evaluate(self, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(self):
        """
        Evaluates the performance of the model on a single episode.
        """
        obs, info = self.env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.model.episode_len):
            act, _ = self.model.act(obs, True, True)
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost

