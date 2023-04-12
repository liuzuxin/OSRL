from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import os
import uuid

import gym
import dsrl
import pyrallis
import numpy as np
from tqdm.auto import tqdm, trange  # noqa

import torch
import torch.nn as nn
from saferl.utils import WandbLogger, DummyLogger
from osrl.net import MLPGaussianPerturbationActor, \
    EnsembleDoubleQCritic, VAE, LagrangianPIDController


class BCQL(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 sample_action_num: int = 10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 phi: float = 0.05,
                 lmbda: float = 0.75,
                 beta: float = 0.5,
                 PID: list = [0.1, 0.003, 0.001],
                 num_q: int = 1,
                 num_qc: int = 1,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 device: str = "cpu"):
        """
        Batch-Constrained deep Q-learning with PID Lagrangian (BCQL)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.latent_dim = self.action_dim * 2
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.sample_action_num = sample_action_num
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.lmbda = lmbda
        self.beta = beta
        self.KP, self.KI, self.KD = PID
        self.num_q = num_q
        self.num_qc = num_qc
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        ################ create actor critic model ###############
        self.actor = MLPGaussianPerturbationActor(self.state_dim, self.action_dim, 
                                             self.a_hidden_sizes, nn.Tanh, 
                                             self.phi, self.max_action).to(self.device)
        self.critic = EnsembleDoubleQCritic(self.state_dim, self.action_dim, 
                                       self.c_hidden_sizes, nn.ReLU, 
                                       num_q=self.num_q).to(self.device)
        self.cost_critic = EnsembleDoubleQCritic(self.state_dim, self.action_dim, 
                                            self.c_hidden_sizes, nn.ReLU, 
                                            num_q=self.num_qc).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.vae_hidden_sizes, 
                  self.latent_dim, self.max_action, self.device).to(self.device)
        
        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()

        self.qc_thres = cost_limit * (1 - self.gamma**self.episode_len) / (
                1 - self.gamma) / self.episode_len
        self.controller = LagrangianPIDController(self.KP, self.KI, self.KD, self.qc_thres)

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def vae_loss(self, observations, actions):
        recon, mean, std = self.vae(observations, actions)
        recon_loss = nn.functional.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss
        
        self.vae_optim.zero_grad()
        loss_vae.backward()
        self.vae_optim.step()
        stats_vae = {"loss/loss_vae": loss_vae.item()}
        return loss_vae, stats_vae
    
    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num, 0).to(self.device)
            
            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.critic_old.predict(obs_next, act_targ_next)
            
            q_targ = self.lmbda * torch.min(q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]

            backup = rewards + self.gamma * (1 - done) * q_targ
        loss_critic = self.critic.loss(backup, q1_list) + self.critic.loss(backup, q2_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        _, _, q1_list, q2_list = self.cost_critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num, 0).to(self.device)
            
            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.cost_critic_old.predict(obs_next, act_targ_next)
            
            q_targ = self.lmbda * torch.min(q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]

            backup = costs + self.gamma * q_targ
        loss_cost_critic = self.cost_critic.loss(backup, q1_list) + self.cost_critic.loss(backup, q2_list)
        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()
        stats_cost_critic = {"loss/cost_critic_loss": loss_cost_critic.item()}
        return loss_cost_critic, stats_cost_critic 

    def actor_loss(self, observations):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False
            
        actions = self.actor(observations, self.vae.decode(observations))
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)  # [batch_size]
        qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, actions)
        qc_pi = torch.min(qc1_pi, qc2_pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        
        with torch.no_grad():
            multiplier = self.controller.control(qc_pi).detach()
        qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()
        loss_actor = -q_pi.mean() + qc_penalty
        
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        
        stats_actor = {"loss/actor_loss": loss_actor.item(),
                       "loss/qc_penalty": qc_penalty.item(),
                       "loss/lagrangian": multiplier.item()}

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        for p in self.vae.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor

    def setup_optimiers(self, actor_lr, critic_lr, vae_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def sync_weight(self):
        """Soft-update the weight for the target network."""
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, value, logp.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        act = self.actor(obs, self.vae.decode(obs))
        if self.device == "cpu":
            act = act.data.numpy()
        else:
            act = act.data.cpu().numpy()
        return np.squeeze(act, axis=0), None


class BCQLTrainer:
    """
    Constraints Penalized Q-learning Trainer
    """
    def __init__(
            self,
            model: BCQL,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            vae_lr: float = 1e-4,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device="cpu"):
        
        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimiers(actor_lr, critic_lr, vae_lr)
        
    def train_one_step(self, observations, next_observations,
                       actions, rewards, costs, done):
        # update VAE
        loss_vae, stats_vae = self.model.vae_loss(observations, actions)
        # update critic
        loss_critic, stats_critic = self.model.critic_loss(observations, next_observations, 
                                                           actions, rewards, done)
        # update cost critic
        loss_cost_critic, stats_cost_critic = self.model.cost_critic_loss(observations, next_observations, 
                                                                          actions, costs, done)
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations)

        self.model.sync_weight()

        self.logger.store(**stats_vae)
        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
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
