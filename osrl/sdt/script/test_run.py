from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os
import uuid

import bullet_safety_gym  # noqa
import gym  # noqa
import numpy as np
import pyrallis
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from dsrl.offline_env import OfflineEnvWrapper  # noqa
from saferl.utils import WandbLogger

from sdt.sdt_model import DecisionTransformer
from sdt.dataset import SequenceDataset
from sdt.utils import wrap_env, set_seed, auto_name
from sdt.sdt_trainer import Trainer
from configs import TrainConfig, CarCircleConfig, AntRunConfig, DroneRunConfig

DEFAULT_CONFIG = {
    "SafetyCarCircle-v0": CarCircleConfig,
    "SafetyAntRun-v0": AntRunConfig,
    "SafetyDroneRun-v0": DroneRunConfig
}


def gen_name(config: TrainConfig):
    assert config.env_name in DEFAULT_CONFIG, f"{config.env_name} doesn't have a default config!"
    default_train_config = DEFAULT_CONFIG[config.env_name]
    name = config.name
    if name is None:
        skip_keys = [
            "project", "group", "name", "dataset", "env_name", "log_path", "device", "target_returns", "eval_every"
        ]
        # generate the name by comparing the difference with the default config
        name = auto_name(asdict(default_train_config()), asdict(config), skip_keys=skip_keys)
    name = f"{name}-{config.env_name}-{str(uuid.uuid4())[:4]}"
    config.name = name
    if config.log_path is not None:
        config.log_path = os.path.join(config.log_path, config.name)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    gen_name(config)
    # init wandb session for logging
    cfg = asdict(config)
    logger = WandbLogger(cfg, config.project, config.group, config.name, config.log_path)
    logger.save_config(cfg, verbose=True)

    # the cost scale is down in trainer rollout
    env = wrap_env(
        env=gym.make(config.env_name),
        reward_scale=config.reward_scale,
    )
    env = OfflineEnvWrapper(env)
    data = env.get_dataset(config.dataset)

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
                      device=config.device)

    print("Data & dataloader setup...")
    dataset = SequenceDataset(data,
                              seq_len=config.seq_len,
                              reward_scale=config.reward_scale,
                              cost_scale=config.cost_scale,
                              deg=config.deg,
                              max_rew_decrease=config.max_rew_decrease,
                              beta=config.beta,
                              augment_percent=config.augment_percent,
                              max_reward=config.max_reward,
                              min_reward=config.min_reward)
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    trainloader_iter = iter(trainloader)

    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, costs, time_steps, mask = [b.to(config.device) for b in batch]
        trainer.train_one_step(states, actions, returns, costs, time_steps, mask)

        if step % config.eval_every == 0 or step == config.update_steps - 1:
            for target_return in config.target_returns:
                reward_return, cost_return = target_return
                # critical step, rescale the return!
                ret, cost, length = trainer.evaluate(config.eval_rollouts, reward_return * config.reward_scale,
                                                     cost_return * config.cost_scale)
                logger.store(tab="eval_rew_" + str(reward_return) + "_cost_" + str(cost_return), reward=ret, cost=cost)
            logger.write(step, True)
            # TODO: save the best
            logger.save_checkpoint()

        else:
            logger.write_without_reset(step)

    logger.save_checkpoint(suffix="final")


if __name__ == "__main__":
    train()
