from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os
import uuid

import gymnasium as gym  # noqa
import bullet_safety_gym  # noqa
import dsrl
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger

from osrl.common import TransitionDataset
from osrl.common.dataset import process_bc_dataset
from osrl.algorithms import BC, BCTrainer
from osrl.common.exp_util import auto_name, seed_all
from examples.configs.bc_configs import BCTrainConfig, BC_DEFAULT_CONFIG


@pyrallis.wrap()
def train(args: BCTrainConfig):
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.prefix += "-" + args.bc_mode
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # the cost scale is down in trainer rollout
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task+"_density"+str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data, args.outliers_percent, args.noise_scale,
                                args.inpaint_ranges, args.epsilon, args.density,
                                cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb)

    # function w.r.t episode cost
    frontier_fn = {}
    frontier_fn["OfflineAntCircle-v0"] = lambda x: 600 + 4 * x
    frontier_fn["OfflineAntRun-v0"] = lambda x: 600 + 10 / 3 * x
    frontier_fn["OfflineCarCircle-v0"] = lambda x: 450 + 5 / 3 * x
    frontier_fn["OfflineCarRun-v0"] = lambda x: 600
    frontier_fn["OfflineDroneRun-v0"] = lambda x: 325 + 125 / 70 * x
    frontier_fn["OfflineDroneCircle-v0"] = lambda x: 600 + 4 * x
    frontier_range = 50

    process_bc_dataset(data, args.cost_limit, args.gamma, args.bc_mode,
                    #    frontier_fn[args.task], 
                       None,
                       frontier_range)

    # model & optimizer & scheduler setup
    state_dim = env.observation_space.shape[0]
    if args.bc_mode == "multi-task":
        state_dim += 1
    model = BC(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = BCTrainer(model,
                        env,
                        logger=logger,
                        actor_lr=args.actor_lr,
                        bc_mode=args.bc_mode,
                        cost_limit=args.cost_limit,
                        device=args.device)

    trainloader = DataLoader(
        TransitionDataset(data),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, _, actions, _, _, _ = [b.to(args.device) for b in batch]
        trainer.train_one_step(observations, actions)

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
