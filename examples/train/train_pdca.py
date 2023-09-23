import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.pdca_configs import (PDCA_DEFAULT_CONFIG,
                                                PDCATrainConfig)
from osrl.algorithms import PDCA, PDCATrainer
from osrl.common import TransitionDataset, InitialStateDataset
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: PDCATrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(PDCATrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(PDCA_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(PDCA_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_threshold))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym  # noqa
    env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_threshold)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # setup dataset
    dataset = TransitionDataset(
        data,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        state_init=True)
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    initial_state_dataset = InitialStateDataset(data)
    initial_state_loader = DataLoader(
        initial_state_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1,
    )
    initial_state_loader_iter = iter(initial_state_loader)

    # setup model
    model = PDCA(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        gamma=args.gamma,
        B=args.B,
        cost_threshold=args.cost_threshold,
        device=args.device,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = PDCATrainer(
            model,
            env,
            logger=logger,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            device=args.device)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        batch = [b.to(args.device) for b in batch]
        init_batch = next(initial_state_loader_iter)
        trainer.train_one_step(batch, init_batch)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            ret, cost, length = trainer.evaluate(args.eval_episodes)
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
