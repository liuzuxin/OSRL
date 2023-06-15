import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import MLPActor


class BC(nn.Module):
    """
    Behavior Cloning (BC)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list, optional): List of integers specifying the sizes 
            of the layers in the actor network.
        episode_len (int, optional): Maximum length of an episode.
        device (str, optional): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 episode_len: int = 300,
                 device: str = "cpu"):

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

    def setup_optimizers(self, actor_lr):
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
    Behavior Cloning Trainer
    
    Args:
        model (BC): The BC model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        bc_mode (str): specify bc mode
        cost_limit (int): Upper limit on the cost per episode.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
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
        self.model.setup_optimizers(actor_lr)

    def set_target_cost(self, target_cost):
        self.cost_limit = target_cost

    def train_one_step(self, observations, actions):
        """
        Trains the model by updating the actor.
        """
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
        self.logger.store(**stats_actor)

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
        return np.mean(episode_rets), np.mean(episode_costs), np.mean(episode_lens)

    @torch.no_grad()
    def rollout(self):
        """
        Evaluates the performance of the model on a single episode.
        """
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
