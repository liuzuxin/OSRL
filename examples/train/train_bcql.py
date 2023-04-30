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
from dsrl.infos import DEFAULT_MAX_EPISODE_STEPS
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from saferl.utils import WandbLogger

from osrl.common import TransitionDataset
from osrl.algorithms import BCQL, BCQLTrainer
from saferl.utils.exp_util import auto_name, seed_all
from examples.configs.bcql_configs import BCQLTrainConfig, BCQL_DEFAULT_CONFIG


@pyrallis.wrap()
def train(args: BCQLTrainConfig):
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # setup logger
    args.episode_len = DEFAULT_MAX_EPISODE_STEPS[args.task.split("-")[1]]
    cfg = asdict(args)
    default_cfg = asdict(BCQL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # initialize environment
    env = gym.make(args.task)
    
    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    data = env.pre_process_data(data, 
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # model & optimizer setup
    model = BCQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        vae_hidden_sizes=args.vae_hidden_sizes,
        sample_action_num=args.sample_action_num,
        PID=args.PID,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda,
        beta=args.beta,
        phi=args.phi,
        num_q=args.num_q,
        num_qc=args.num_qc,
        cost_limit=args.cost_limit,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = BCQLTrainer(model,
                          env,
                          logger=logger,
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          vae_lr=args.vae_lr,
                          reward_scale=args.reward_scale,
                          cost_scale=args.cost_scale,
                          device=args.device)

    # initialize pytorch dataloader
    dataset = TransitionDataset(
        data,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale)
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    # training
    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, next_observations, actions, rewards, costs, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, costs, done)

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
