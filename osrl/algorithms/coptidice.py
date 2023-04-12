from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass

import gym
import numpy as np
from tqdm.auto import tqdm, trange  # noqa

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch import distributions as pyd
from torch.distributions.beta import Beta

from saferl.utils import WandbLogger, DummyLogger
from osrl.common.net import SquashedGaussianMLPActor, EnsembleQCritic


class COptiDICE(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128], 
                 gamma: float = 0.99,
                 alpha: float = 0.5,
                 num_q: int = 1,
                 num_qc: int = 1,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 device: str = "cpu"
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.gamma = gamma
        self.alpha = alpha
        self.num_q = num_q
        self.num_qc = num_qc
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        self.tau = torch.tensor(0, requires_grad=True).to(self.device)
        self.lmbda = torch.tensor(0, requires_grad=True).to(self.device)
        self.actor = SquashedGaussianMLPActor(
            self.state_dim, self.action_dim, self.a_hidden_sizes, nn.ReLU).to(self.device)
        self.critic = EnsembleQCritic(self.state_dim, self.action_dim, self.c_hidden_sizes, 
                                      nn.ReLU, num_q=self.num_q).to(self.device)
        
        
class COptiDICETrainer:
    def __init__(self,
                 model: COptiDICE,
                 env: gym.Env,
                 logger: WandbLogger = DummyLogger(),
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 device="cpu"):
        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimiers()
        
    def train_one_step(self, observations, next_observations,
                       actions, rewards, costs, done):
        pass
    
    def evaluate(self, eval_episodes):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(episode_costs) / self.cost_scale, np.mean(
            episode_lens)
        
    @torch.no_grad()
    def rollout(self):
        obs, info = self.env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.model.episode_len):
            act, _ = self.model.act(obs)
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost
