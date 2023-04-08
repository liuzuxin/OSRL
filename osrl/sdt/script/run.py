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
from dsrl.offline_env import OfflineEnvWrapper  # noqa
from saferl.utils import WandbLogger

from sdt.sdt_model import DecisionTransformer
from sdt.dataset import SequenceDataset
from sdt.utils import wrap_env, set_seed, auto_name
from sdt.sdt_trainer import Trainer
from configs import TrainConfig, CarCircleConfig, AntRunConfig, DroneRunConfig, CarRunConfig, DroneCircleConfig, CarReachConfig, AntCircleConfig

DEFAULT_CONFIG = {
    "offline-CarCircle-v0": CarCircleConfig,
    "SafetyAntRun-v0": AntRunConfig,
    "SafetyDroneRun-v0": DroneRunConfig,
    "SafetyDroneCircle-v0": DroneCircleConfig,
    "SafetyCarRun-v0": CarRunConfig,
    "SafetyAntCircle-v0": AntCircleConfig,
    "SafetyCarReach-v0": CarReachConfig,
}


def gen_name(config: TrainConfig):
    assert config.env_name in DEFAULT_CONFIG, f"{config.env_name} doesn't have a default config!"
    default_train_config = DEFAULT_CONFIG[config.env_name]
    name = config.name
    if name is None:
        skip_keys = [
            "project", "group", "name", "env_name", "log_path", "device", "thread", "target_returns", "eval_every",
            "prefix", "suffix", "update_steps", "dataset", "eval_rollouts"
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
        config.log_path = os.path.join(config.log_path, config.group, config.name)


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
    env = gym.make(config.env_name)
    data = env.get_dataset()
    env = wrap_env(
        env=env,
        reward_scale=config.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    target_entropy = -env.action_space.shape[0]

    # model & optimizer & scheduler setup
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
        time_emb=config.time_emb,
        use_rew=config.use_rew,
        use_cost=config.use_cost,
        cost_transform=config.cost_transform,
        add_cost_feat=config.add_cost_feat,
        mul_cost_feat=config.mul_cost_feat,
        cat_cost_feat=config.cat_cost_feat,
        action_head_layers=config.action_head_layers,
        cost_prefix=config.cost_prefix,
        stochastic=config.stochastic,
        init_temperature=config.init_temperature,
        target_entropy=target_entropy,
    ).to(config.device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = Trainer(model,
                      env,
                      logger=logger,
                      learning_rate=config.learning_rate,
                      weight_decay=config.weight_decay,
                      betas=config.betas,
                      clip_grad=config.clip_grad,
                      lr_warmup_steps=config.lr_warmup_steps,
                      reward_scale=config.reward_scale,
                      cost_scale=config.cost_scale,
                      loss_cost_weight=config.loss_cost_weight,
                      loss_state_weight=config.loss_state_weight,
                      cost_reverse=config.cost_reverse,
                      no_entropy=config.no_entropy,
                      device=config.device)

    ct = lambda x: 70 - x if config.linear else 1 / (x + 10)

    dataset = SequenceDataset(
        data,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        cost_scale=config.cost_scale,
        deg=config.deg,
        pf_sample=config.pf_sample,
        max_rew_decrease=config.max_rew_decrease,
        beta=config.beta,
        augment_percent=config.augment_percent,
        cost_reverse=config.cost_reverse,
        max_reward=config.max_reward,
        min_reward=config.min_reward,
        pf_only=config.pf_only,
        rmin=config.rmin,
        cost_bins=config.cost_bins,
        npb=config.npb,
        cost_sample=config.cost_sample,
        cost_transform=ct,
        start_sampling=config.start_sampling,
        prob=config.prob,
        random_aug=config.random_aug,
        aug_rmin=config.aug_rmin,
        aug_rmax=config.aug_rmax,
        aug_cmin=config.aug_cmin,
        aug_cmax=config.aug_cmax,
        cgap=config.cgap,
        rstd=config.rstd,
        cstd=config.cstd,
        # cost_transform=lambda x: 1 / x + 1
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = 0
    best_violation = 10000
    best_idx = 0

    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            b.to(config.device) for b in batch
        ]
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask, episode_cost, costs)

        # evaluation
        if (step + 1) % config.eval_every == 0 or step == config.update_steps - 1:
            average_reward, average_violation = [], []
            log_cost, log_reward, log_violation, log_len = {}, {}, {}, {}
            for target_return in config.target_returns:
                reward_return, cost_return = target_return
                if config.cost_reverse:
                    # critical step, rescale the return!
                    ret, cost, length = trainer.evaluate(config.eval_rollouts, reward_return * config.reward_scale,
                                                         (config.episode_len - cost_return) * config.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(config.eval_rollouts, reward_return * config.reward_scale,
                                                         cost_return * config.cost_scale)
                violation = max(cost - cost_return, 0)
                average_violation.append(violation)
                average_reward.append(ret)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_return))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_violation.update({name: violation})
                log_len.update({name: length})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="violation", **log_violation)
            logger.store(tab="length", **log_len)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            mean_ret = np.mean(average_reward)
            mean_violation = np.mean(average_violation)
            if mean_violation < best_violation or (mean_violation == best_violation and mean_ret > best_reward):
                best_violation = mean_violation
                best_reward = mean_ret
                best_idx = step
                logger.save_checkpoint(suffix="optimal")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
