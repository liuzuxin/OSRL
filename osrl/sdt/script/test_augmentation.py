import os
import uuid
from dataclasses import asdict, dataclass
from turtle import color
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import gym  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pyrallis

from d4srl.offline_env import OfflineEnvWrapper  # noqa
from sdt.dataset import SequenceDataset
from sdt.utils import set_seed, wrap_env


@dataclass
class TestConfig:
    env_name: str = "SafetyCarCircle-v0"
    dataset: str = "./dataset/data/cc_cost_20_70_245_trajs.hdf5"
    num_workers: int = 8
    deterministic_torch: bool = False
    seed: int = 10
    # augmentation param
    deg: int = 4
    beta: float = 0.25
    augment_percent: float = 0.8
    # maximum absolute value of reward for the augmented trajs
    max_reward: float = 550.0
    # minimum reward above the PF curve
    min_reward: float = 5.0
    # the max drecrease of ret between the associated traj w.r.t the nearest pf traj
    max_rew_decrease: float = 50.0


# @dataclass
# class TestConfig:
#     env_name: str = "SafetyAntRun-v0"
#     dataset: str = "./dataset/data/ar_cost_20_80_416_trajs.hdf5"
#     num_workers: int = 8
#     deterministic_torch: bool = False
#     seed: int = 10
#     # augmentation param
#     deg: int = 4
#     beta: float = 0.25
#     augment_percent: float = 0.8
#     # maximum absolute value of reward for the augmented trajs
#     max_reward: float = 750.0
#     # minimum reward above the PF curve
#     min_reward: float = 5.0
#     # the max drecrease of ret between the associated traj w.r.t the nearest pf traj
#     max_rew_decrease: float = 50.0

# @dataclass
# class TestConfig:
#     env_name: str = "SafetyDroneRun-v0"
#     dataset: str = "./dataset/data/dr_cost_20_70_452_trajs.hdf5"
#     num_workers: int = 8
#     deterministic_torch: bool = False
#     seed: int = 10
#     # augmentation param
#     deg: int = 4
#     beta: float = 0.25
#     augment_percent: float = 0.8
#     # maximum absolute value of reward for the augmented trajs
#     max_reward: float = 500.0
#     # minimum reward above the PF curve
#     min_reward: float = 5.0
#     # the max drecrease of ret between the associated traj w.r.t the nearest pf traj
#     max_rew_decrease: float = 50.0


@pyrallis.wrap()
def test(config: TestConfig):
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)

    # the cost scale is down in tester rollout
    env = wrap_env(
        env=gym.make(config.env_name),
        reward_scale=1,
    )
    env = OfflineEnvWrapper(env)
    data = env.get_dataset(config.dataset)

    print("Data & dataloader setup...")
    dataset = SequenceDataset(data,
                              seq_len=20,
                              reward_scale=1,
                              cost_scale=1,
                              deg=config.deg,
                              max_rew_decrease=config.max_rew_decrease,
                              beta=config.beta,
                              augment_percent=config.augment_percent,
                              max_reward=config.max_reward,
                              min_reward=config.min_reward)

    idx, aug_data = dataset.idx, dataset.aug_data
    data = dataset.original_data

    print(f"original trajectories num: {len(data)}")
    print(f"augmented trajectories num: {len(aug_data)}")

    def plot_scatter(plot_step=0):
        original_ret = []
        original_cost = []
        for d in data:
            original_ret.append(d["returns"][plot_step])
            original_cost.append(d["cost_returns"][plot_step])

        aug_ret = []
        aug_cost = []
        for d in aug_data:
            aug_ret.append(d["returns"][plot_step])
            aug_cost.append(d["cost_returns"][plot_step])

        ass_ret, ass_cost = [], []
        for i in idx:
            ass_ret.append(data[i]["returns"][plot_step])
            ass_cost.append(data[i]["cost_returns"][plot_step])

        plt.scatter(original_cost, original_ret, c="blue", label="original")
        plt.scatter(aug_cost, aug_ret, c="red", label="augmented")
        plt.scatter(ass_cost, ass_ret, c="green", label="associated")
        plt.legend()
        plt.savefig("test_" + config.env_name + "_" + str(plot_step) + ".png")
        plt.clf()

    plot_scatter(0)
    # plot_scatter(90)


if __name__ == "__main__":
    test()
