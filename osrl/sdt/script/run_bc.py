from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import bullet_safety_gym  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange  # noqa
from dsrl.offline_env import OfflineEnvWrapper  # noqa
from saferl.utils import WandbLogger, DummyLogger
from sdt.utils import wrap_env, set_seed, auto_name


TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # wandb params
    project: str = "offline-srl-baselines_benchmark"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = ""
    suffix: Optional[str] = ""
    # training params
    env_name: str = "SafetyCarCircle-v0"
    dataset: str = "dataset/data/cc_cost_20_70_245_trajs.hdf5"
    learning_rate: float = 1e-4
    eval_rollouts: int = 4
    episode_len: int = 300
    eval_every: int = 2500  # How often (time steps) we evaluate
    update_steps: int = 100_000  # Max time steps to run environment
    lr_warmup_steps: int = 500
    reward_scale: float = 1
    cost_scale: float = 1
    batch_size: int = 512  # Batch size for all networks
    gamma: float = 1.0  # Discount factor
    buffer_size: int = 2_000_000  # Replay buffer size
    cost_limit: int = 10  # Best data fraction to use
    max_action: float = 1.0
    aug_cost: bool = False
    # general params
    log_path: Optional[str] = "log"  # Save path
    deterministic_torch: bool = False
    seed: int = 0
    device: str = "cpu"
    thread: int = 4


@dataclass
class CarRunConfig(TrainConfig):
    group: str = "CarRun"
    episode_len: int = 200
    # training params
    env_name: str = "SafetyCarRun-v0"
    dataset: str = "./dataset/data/dr_cost_0_50_1243.hdf5"


@dataclass
class AntRunConfig(TrainConfig):
    group: str = "AntRun"
    episode_len: int = 200
    # training params
    env_name: str = "SafetyAntRun-v0"
    dataset: str = "./dataset/data/ar_cost_0_30_663.hdf5"


@dataclass
class DroneCircleConfig(TrainConfig):
    group: str = "DroneCircle"
    episode_len: int = 300
    # training params
    env_name: str = "SafetyDroneCircle-v0"
    dataset: str = "./dataset/data/dr_cost_0_50_1243.hdf5"


@dataclass
class DroneRunConfig(TrainConfig):
    group: str = "DroneRun"
    episode_len: int = 100
    # training params
    env_name: str = "SafetyDroneRun-v0"
    dataset: str = "./dataset/data/dr_cost_20_70_452_trajs.hdf5"


@dataclass
class CarReachConfig(TrainConfig):
    group: str = "CarReach"
    episode_len: int = 500
    # training params
    env_name: str = "SafetyCarReach-v0"
    dataset: str = "./dataset/data/creach_cost_0_100_2121.hdf5"


DEFAULT_CONFIG = {
    "SafetyAntRun-v0": AntRunConfig,
    "SafetyCarCircle-v0": TrainConfig,
    "SafetyCarRun-v0": CarRunConfig,
    "SafetyDroneCircle-v0": DroneCircleConfig,
    "SafetyDroneRun-v0": DroneRunConfig,
    "SafetyCarReach-v0": CarReachConfig
}


class ReplayBuffer:
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        self._device = device
        self._states = self._to_tensor(data["observations"])
        self._actions = self._to_tensor(data["actions"])
        self._rewards = self._to_tensor(data["rewards"][..., None])
        self._next_states = self._to_tensor(data["next_observations"])
        self._dones = self._to_tensor(data["terminals"][..., None])
        self._size = data["observations"].shape[0]

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def process_dataset(
    dataset: Dict[str, np.ndarray],
    cost_limit: float,
    gamma: float,
    aug_cost: bool):

    dataset["actions"] = np.clip(dataset["actions"], -1, 1)
    done_mask = (dataset["terminals"] == 1) + (dataset["timeouts"] == 1)
    dataset["terminals"][done_mask] = 1

    n_transitions = dataset["rewards"].shape[0]
    idx = np.arange(n_transitions)
    done_idx = idx[done_mask]
    valid_transition = np.zeros((n_transitions,), dtype=int)
    if aug_cost:
        valid_transition = np.ones((n_transitions,), dtype=int)

    dataset["cost_returns"] = np.zeros_like(dataset["costs"])
    for i in range(done_idx.shape[0]-1):
        if i == 0:
            start = 0
            end = done_idx[i] + 1
        else:
            start = done_idx[i] + 1
            end = done_idx[i+1] + 1
        cost_returns = discounted_cumsum(dataset["costs"][start:end], gamma=gamma)
        dataset["cost_returns"][start:end] = cost_returns[0]
        if cost_returns[0] <= cost_limit:
            valid_transition[start:end] = 1

    mask = valid_transition == 1
    dataset["observations"] = dataset["observations"][mask]
    dataset["actions"] = dataset["actions"][mask]
    dataset["next_observations"] = dataset["next_observations"][mask]
    dataset["rewards"] = dataset["rewards"][mask]
    dataset["costs"] = dataset["costs"][mask]
    dataset["terminals"] = dataset["terminals"][mask]
    if aug_cost:
        dataset["observations"] = np.hstack((dataset["observations"], dataset["cost_returns"].reshape(-1, 1)))
    print(f"original size = {n_transitions}, cost limit = {cost_limit}, filtered size = {np.sum(mask)}")


class BehaviorCloning(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 max_action: float = 1.0,
                 episode_len: int = 1000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action
        self.episode_len = episode_len

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Trainer:  # noqa
    def __init__(
        self,
        model: BehaviorCloning,
        env: gym.Env,
        logger: WandbLogger = DummyLogger(),
        learning_rate: float = 1e-4,
        gamma: float = 1.0,
        reward_scale: float = 1.0,
        cost_scale: float = 1.0,
        cost_limit: float = 10,
        device: str = "cpu",
        aug_cost: bool = False
    ):
        self.model = model
        self.logger = logger
        self.env = env
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate)
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.cost_limit = cost_limit
        self.device = device
        self.gamma = gamma
        self.aug_cost = aug_cost

    def train_one_step(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, _, _, _ = batch
        # Compute actor loss
        # print(state.type(), state.shape)
        pred_action = self.model(state)
        actor_loss = F.mse_loss(pred_action, action)
        # Optimize the actor
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()
        self.logger.store(
            tab="train",
            act_loss=actor_loss.item())

    def state_dict(self) -> Dict[str, Any]:
        return {"actor": self.model.state_dict(),
                "actor_optimizer": self.optim.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.model.load_state_dict(state_dict["actor"])
        self.optim.load_state_dict(state_dict["actor_optimizer"])

    def evaluate(self, num_rollouts):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.array(episode_rets), np.array(episode_costs), np.array(episode_lens)

    @torch.no_grad()
    def rollout(self, 
                model: BehaviorCloning,
                env: gym.Env):
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        obs, info = env.reset()
        if self.aug_cost:
            obs = np.append(obs, self.cost_limit)
        obs = torch.tensor(obs.reshape(1, -1), 
                           dtype=torch.float32,
                           device=self.device)
        for _ in range(model.episode_len):
            # print(obs.type(), obs.shape)
            act = torch.squeeze(model(obs)).cpu().numpy()
            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.aug_cost:
                obs_next = np.append(obs_next, self.cost_limit)
            obs_next = torch.tensor(obs_next.reshape(1, -1),
                                    dtype=torch.float32,
                                    device=self.device)
            obs = obs_next
            
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost


def gen_name(config: TrainConfig):
    assert config.env_name in DEFAULT_CONFIG, f"{config.env_name} doesn't have a default config!"
    default_train_config = DEFAULT_CONFIG[config.env_name]
    name = config.name
    if name is None:
        skip_keys = [
            "project", "group", "name", "env_name", "log_path", "device", "thread", "target_returns", "eval_every",
            "prefix", "suffix", "update_steps", "dataset"
        ]
        # generate the name by comparing the difference with the default config
        name = auto_name(asdict(default_train_config()),
                         asdict(config),
                         skip_keys=skip_keys,
                         prefix=config.prefix,
                         suffix=config.suffix)
    name = "default" if not len(name) else name
    # name = f"{name}-{config.env_name}-{str(uuid.uuid4())[:4]}"
    name = f"{name}-{str(uuid.uuid4())[:4]}"
    config.name = name
    if config.log_path is not None:
        config.log_path = os.path.join(config.log_path, config.name)


@pyrallis.wrap()
def train(config: TrainConfig):
    if config.device == "cpu":
        torch.set_num_threads(config.thread)
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    gen_name(config)
    # init wandb session for logging
    cfg = asdict(config)
    logger = WandbLogger(cfg, config.project, config.group, config.name, config.log_path)
    logger.save_config(cfg, verbose=False)

    # the cost scale is down in trainer rollout
    env = wrap_env(
        env=gym.make(config.env_name),
        reward_scale=config.reward_scale,
    )
    env = OfflineEnvWrapper(env)
    data = env.get_dataset(config.dataset)
    process_dataset(data, config.cost_limit, config.gamma, config.aug_cost)

    state_dim = env.observation_space.shape[0]
    if config.aug_cost:
        state_dim += 1
    action_dim = env.action_space.shape[0]

    # model & optimizer setup
    model = BehaviorCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=config.max_action,
        episode_len=config.episode_len,
    ).to(config.device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    trainer = Trainer(
        model=model,
        env=env,
        logger=logger,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        reward_scale=config.reward_scale,
        cost_scale=config.cost_scale,
        cost_limit=config.cost_limit,
        device=config.device,
        aug_cost=config.aug_cost
    )

    def checkpoint_fn():
        return trainer.state_dict()
    logger.setup_checkpoint_fn(checkpoint_fn)

    replay_buffer = ReplayBuffer(
        data=data,
        device=config.device,
    )

    # for saving the best
    best_reward = 0
    best_violation = 10000
    best_idx = 0

    for step in trange(config.update_steps, desc="Training"):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        trainer.train_one_step(batch)

        # evaluation
        if (step + 1) % config.eval_every == 0 or step == config.update_steps - 1:
            ret, cost, length = trainer.evaluate(config.eval_rollouts)
            
            mask = cost > config.cost_limit
            violation = sum(mask) / config.eval_rollouts * 100
            logger.store(tab="eval", EpCost=np.mean(cost))
            logger.store(tab="eval", EpRet=np.mean(ret))
            logger.store(tab="eval", EpLen=np.mean(length))
            logger.store(tab="eval", Violation=violation)

            # save the current weight
            logger.save_checkpoint()

            # save the best weight
            cond1 = violation < best_violation
            cond2 = (violation == best_violation) and (np.mean(ret) > best_reward)
            if sum(~mask) != 0:
                cond2 = (violation == best_violation) and (np.mean(ret[~mask]) > best_reward)
            if cond1 or cond2:
                best_violation = violation
                best_reward = np.mean(ret)
                if sum(~mask) != 0:
                    best_reward = np.mean(ret[~mask])
                best_idx = step
                logger.save_checkpoint(suffix="optimal")
            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
