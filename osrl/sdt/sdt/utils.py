import os
import os.path as osp
import random
from dataclasses import asdict, dataclass
from typing import (Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union)

import gym  # noqa
import numpy as np
import torch
import yaml


# general utils
def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:

    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def to_string(values):
    name = ""
    if isinstance(values, Sequence) and not isinstance(values, str):
        for i, v in enumerate(values):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(v)
        return name
    elif isinstance(values, Dict):
        for i, k in enumerate(sorted(values.keys())):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(values[k])
        return name
    else:
        return str(values)


key_abbre = {
    "embedding_dim": "embed",
    "num_layers": "layer",
    "num_heads": "head",
    "time_emb": "time",
    "seq_len": "seq",
    "attention_dropout": "ad",
    "residual_dropout": "rd",
    "embedding_dropout": "ed",
    "learning_rate": "lr",
    "batch_size": "bs",
    "augment_percent": "ap",
    "max_reward": "max_r",
    "min_reward": "min_r",
    "max_rew_decrease": "max_rd",
    "dataset": "data",
    "reward_scale": "rs",
    "cost_scale": "cs",
    "action_head_layers": "act_layer",
    "cost_transform": "ct",
    "cost_only": "conly",
    "add_cost_feat": "cfa",
    "mul_cost_feat": "cfm",
    "cat_cost_feat": "cfc",
    "loss_cost_weight": "cw",
    "loss_state_weight": "sw",
}


def auto_name(default_cfg: dict, current_cfg: dict, prefix="", suffix="", skip_keys=[]):
    '''
    Automatic generate the experiment name by comparing the current config with the default one
    '''
    name = prefix
    for i, k in enumerate(sorted(default_cfg.keys())):
        if default_cfg[k] == current_cfg[k] or k in skip_keys:
            continue
        prefix = "_" if len(name) else ""
        if k == "dataset":
            value = current_cfg[k].split("/")[-1].split(".")[0]
        else:
            value = to_string(current_cfg[k])
        # replace the name with abbreviation
        if k in key_abbre:
            k = key_abbre[k]
        name += prefix + k + value
    if len(suffix):
        name = name + "_" + suffix if len(name) else suffix
    return name


def load_config(path: str):
    if osp.exists(path):
        with open(path) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config
    else:
        raise ValueError(f"{path} doesn't exist!")
