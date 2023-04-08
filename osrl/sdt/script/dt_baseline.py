# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os
import uuid

import bullet_safety_gym  # noqa
import gym  # noqa
from dsrl.offline_env import OfflineEnvWrapper  # noqa
import numpy as np
import pyrallis
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from saferl.utils import WandbLogger

from sdt.dt.dt_model import DecisionTransformer
from sdt.dataset import SequenceDataset
from sdt.utils import wrap_env, set_seed
from sdt.dt.dt_trainer import Trainer


@dataclass
class TrainConfig:
    # wandb params
    project: str = "offline-srl"
    group: str = "SDT"
    name: str = "SDT"
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "SafetyCarCircle-v0"
    dataset: str = "/home/zuxin/.dsrl/cc-cost1-replay.hdf5"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 2048
    update_steps: int = 100_000
    lr_warmup_steps: int = 20_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: Tuple[float, ...] = (400.0, 1.0)  # reward, cost
    eval_num_rollouts: int = 4
    eval_every: int = 1000
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    device: str = "cuda:1"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    cfg = asdict(config)
    logger = WandbLogger(cfg, cfg["project"], cfg["group"], cfg["name"], cfg["checkpoints_path"])
    logger.save_config(cfg, verbose=True)

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
                      logger,
                      env,
                      learning_rate=config.learning_rate,
                      weight_decay=config.weight_decay,
                      betas=config.betas,
                      clip_grad=config.clip_grad,
                      lr_warmup_steps=config.lr_warmup_steps,
                      reward_scale=config.reward_scale,
                      device=config.device)

    print("Data & dataloader setup...")
    dataset = SequenceDataset(data, seq_len=config.seq_len, reward_scale=config.reward_scale)
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
                ret, cost, length = trainer.evaluate(config.eval_num_rollouts, target_return)
                logger.store(tab="eval" + str(target_return), reward=ret, cost=cost, length=length)
            logger.write(step, True)

        else:
            logger.write_without_reset(step)

    logger.save_checkpoint(suffix="final")


if __name__ == "__main__":
    train()
