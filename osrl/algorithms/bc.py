from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import gym
import dsrl
import numpy as np
from tqdm.auto import tqdm, trange  # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from saferl.utils import WandbLogger, DummyLogger
from osrl.common.net import MLPActor


class BC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 episode_len: int = 300,
                 device: str = "cpu"):
        """
        Behavior Cloning
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.episode_len = episode_len
        self.device = device
        
        self.actor = MLPActor(self.state_dim, self.action_dim, self.a_hidden_sizes, 
                              nn.ReLU, self.max_action).to(self.device)
        
    def actor_loss(self, observations, actions):
        pred_actions = self.actor(observations)
        loss_actor = F.mse_loss(pred_actions, actions)
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}
        return loss_actor, stats_actor
        
    def setup_optimiers(self, actor_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
    def act(self, obs):
        '''
        Given a single obs, return the action.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        act = self.actor(obs)
        act = act.data.numpy() if self.device == "cpu" else act.data.cpu().numpy()
        return np.squeeze(act, axis=0)


class BCTrainer:
    """
    Constraints Penalized Q-learning Trainer
    """
    def __init__(
            self,
            model: BC,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            bc_mode: str = "all",
            cost_limit: int = 10,
            device="cpu"):
        
        self.model = model
        self.logger = logger
        self.env = env
        self.device = device
        self.bc_mode = bc_mode
        self.cost_limit = cost_limit
        self.model.setup_optimiers(actor_lr)
        
    def set_target_cost(self, target_cost):
        self.cost_limit = target_cost
        
    def train_one_step(self, observations, actions):
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
        self.logger.store(**stats_actor)
        
    def evaluate(self, eval_episodes):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets), np.mean(episode_costs), np.mean(episode_lens)
        
    @torch.no_grad()
    def rollout(self):
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        obs, info = self.env.reset()
        if self.bc_mode == "multi-task":
            obs = np.append(obs, self.cost_limit)
        for _ in range(self.model.episode_len):
            act = self.model.act(obs)
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            if self.bc_mode == "multi-task":
                obs_next = np.append(obs_next, self.cost_limit)
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost
