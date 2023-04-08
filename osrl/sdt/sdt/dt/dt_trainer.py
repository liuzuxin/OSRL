from typing import Tuple

import gym  # noqa
import numpy as np
import torch
from torch.nn import functional as F  # noqa
from tqdm.auto import tqdm, trange  # noqa

from saferl.utils import WandbLogger
from sdt.dt.dt_model import DecisionTransformer


class Trainer:

    def __init__(
            self,
            model: DecisionTransformer,
            logger: WandbLogger,
            env: gym.Env,
            # training params
            learning_rate=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            clip_grad=0.25,
            lr_warmup_steps=10000,
            reward_scale=1,
            device="cpu") -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.device = device

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )

    def train_one_step(self, states, actions, returns, costs, time_steps, mask):
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        predicted_actions = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()
        self.scheduler.step()

        self.logger.store(tab="train", train_loss=loss.item(), learning_rate=self.scheduler.get_last_lr()[0])

    def train_dt(self, data_loader, target_returns, training_steps=100000, eval_every=1000, eval_num_rollouts=4):
        self.model.train()

        for i, batch in tqdm(enumerate(data_loader), desc="Training", total=training_steps, leave=False):
            states, actions, returns, time_steps, mask = [b.to(self.device) for b in batch]
            # True value indicates that the corresponding key value will be ignored
            padding_mask = ~mask.to(torch.bool)

            predicted_actions = self.model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
            )
            loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
            # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
            loss = (loss * mask.unsqueeze(-1)).mean()

            self.optim.zero_grad()
            loss.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optim.step()
            self.scheduler.step()

            self.logger.store(tab="train", train_loss=loss.item(), learning_rate=self.scheduler.get_last_lr()[0])

            if i % eval_every == 0 or i == training_steps:
                self.evaluate(eval_num_rollouts, target_returns)

            if i >= training_steps:
                break

    def evaluate(self, num_rollouts, target_return):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env, target_return)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(episode_costs), np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        model: DecisionTransformer,
        env: gym.Env,
        target_return: float,
    ) -> Tuple[float, float]:

        states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=self.device)
        actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=self.device)
        returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=self.device)
        time_steps = torch.arange(model.episode_len, dtype=torch.long, device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important)
            acts = model(  # fix this noqa!!!
                states[:, :step + 1][:, -model.seq_len:],  # noqa
                actions[:, :step + 1][:, -model.seq_len:],  # noqa
                returns[:, :step + 1][:, -model.seq_len:],  # noqa
                time_steps[:, :step + 1][:, -model.seq_len:],  # noqa
            )
            act = acts[0, -1].cpu().numpy()
            obs_next, reward, terminated, truncated, info = env.step(act)
            cost = info["cost"]
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += cost

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
