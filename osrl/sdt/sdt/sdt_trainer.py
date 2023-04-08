from typing import Tuple

import gym  # noqa
import numpy as np
import torch
from torch.nn import functional as F  # noqa
from torch.distributions.beta import Beta
from tqdm.auto import tqdm, trange  # noqa

from saferl.utils import WandbLogger, DummyLogger
from sdt.sdt_model import DecisionTransformer


class Trainer:

    def __init__(
            self,
            model: DecisionTransformer,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            cost_reverse: bool = False,
            no_entropy: bool = False,
            device="cpu") -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy

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
        self.stochastic = self.model.stochastic
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        self.max_action = self.model.max_action

        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask, episode_cost, costs):
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        action_preds, cost_preds, state_preds = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
            episode_cost=episode_cost,
        )

        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            entropy_reg_item = entropy_reg.item()
            if self.no_entropy:
                entropy_reg = 0.0
                entropy_reg_item = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            self.logger.store(tab="train", nll=-log_likelihood.item(), ent=entropy.item(), ent_reg=entropy_reg_item)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        # cost_preds: [batch_size * seq_len, 2], costs: [batch_size * seq_len]
        cost_preds = cost_preds.reshape(-1, 2)
        costs = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
        # cost_loss = F.mse_loss(cost_preds, costs.detach(), reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()
        # compute the accuracy, 0 value, 1 indice, [batch_size, seq_len]
        pred = cost_preds.data.max(dim=1)[1]
        correct = pred.eq(costs.data.view_as(pred)) * mask.flatten()
        correct = correct.sum()
        total_num = mask.sum()
        acc = correct / total_num

        # [batch_size, seq_len, state_dim]
        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:].detach(), reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() * (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        self.scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            act_loss=act_loss.item(),
            cost_loss=cost_loss.item(),
            cost_acc=acc.item(),
            state_loss=state_loss.item(),
            train_lr=self.scheduler.get_last_lr()[0],
        )

    def evaluate(self, num_rollouts, target_return, target_cost, eval_data: dict = None):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env, target_return, target_cost)
            if self.env.noise_scale is None:
                log_info = {
                    "TR" + str(target_return / self.reward_scale) + "TC" + str(target_cost / self.cost_scale) + "_EpRet":
                    epi_ret / self.reward_scale,
                    "TR" + str(target_return / self.reward_scale) + "TC" + str(target_cost / self.cost_scale) + "_EpCost":
                    epi_cost / self.cost_scale,
                    "TR" + str(target_return / self.reward_scale) + "TC" + str(target_cost / self.cost_scale) + "_EpLen":
                    epi_len,
                }
                # log_info = {
                #     "TC" + str(target_cost / self.cost_scale) + "_EpRet":
                #     epi_ret / self.reward_scale,
                #     "TC" + str(target_cost / self.cost_scale) + "_EpCost":
                #     epi_cost / self.cost_scale,
                #     "TC" + str(target_cost / self.cost_scale) + "_EpLen":
                #     epi_len,
                # }
            else:
                log_info = {
                    "Noise_"+str(self.env.noise_scale)+"_EpRet":  epi_ret / self.reward_scale,
                    "Noise_"+str(self.env.noise_scale)+"_EpCost": epi_cost / self.cost_scale,
                    "Noise_"+str(self.env.noise_scale)+"_EpLen":  epi_len,
                }
            if eval_data is not None:
                for k, v in log_info.items():
                    if k not in eval_data.keys():
                        eval_data[k] = [v]
                    else:
                        eval_data[k].append(v)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(episode_costs) / self.cost_scale, np.mean(
            episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        model: DecisionTransformer,
        env: gym.Env,
        target_return: float,
        target_cost: float,
    ) -> Tuple[float, float]:

        states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=self.device)
        actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=self.device)
        returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=self.device)
        costs = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=self.device)
        time_steps = torch.arange(model.episode_len, dtype=torch.long, device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        epi_cost = torch.tensor(np.array([target_cost]), dtype=torch.float, device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            acts, _, _ = model(s, a, r, c, t, None, epi_cost)
            if self.stochastic:
                acts = acts.mean
            acts = acts.clamp(-self.max_action, self.max_action)
            act = acts[0, -1].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            # the costs could not be negative
            # if not self.cost_reverse:
            #     costs = F.relu(costs)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)

        # cost_noise = -self.beta_dist.sample([size, 1]) * 10
        # cost_noise = torch.randn([size, 1], device=self.device) * 10 - 5
        # cost_noise = torch.repeat_interleave(cost_noise, c.shape[1], dim=1)
        # # cost_noise = torch.randn_like(c[:, -1], device=self.device) * 0.0
        # state_noise = torch.randn_like(s, device=self.device) * 0.0
        # return_noise = torch.randn_like(r, device=self.device) * 0.0

        # c = c + cost_noise
        # s = s + state_noise
        # r = r + return_noise

        acts, _, _ = model(s, a, r, c, t, None, epi_cost)
        if self.stochastic:
            acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        acts = acts.clamp(-self.max_action, self.max_action)
        act = acts[0, -1].cpu().numpy()
        return act

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
