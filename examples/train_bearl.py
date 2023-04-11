from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os
import uuid

import gym  # noqa
import bullet_safety_gym  # noqa
import dsrl
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from saferl.utils import WandbLogger

from osrl.dataset import TransitionDataset
from osrl.bearl import BEARL, BEARLTrainer
from saferl.utils.exp_util import auto_name, seed_all
from .configs.bearl_configs import BEARLTrainConfig, BEARL_DEFAULT_CONFIG


@pyrallis.wrap()
def train(args: BEARLTrainConfig):
    seed_all(args.seed)

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(BEARL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # the cost scale is down in trainer rollout
    env = gym.make(args.task)
    data = env.get_dataset()
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)
