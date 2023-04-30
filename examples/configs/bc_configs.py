from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BCTrainConfig:
    # wandb params
    project: str = "OSRL-baselines-new"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BC"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    threads: int = 4
    actor_lr: float = 0.001
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    bc_mode: str = "all"  # "all", "safe", "risky", "frontier", "boundary", "multi-task"
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    gamma: float = 1.0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BCCarCircleConfig(BCTrainConfig):
    pass


@dataclass
class BCAntRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCDroneRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BCDroneCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCCarRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BCAntCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


BC_DEFAULT_CONFIG = {
    "OfflineCarCircle-v0": BCCarCircleConfig,
    "OfflineAntRun-v0": BCAntRunConfig,
    "OfflineDroneRun-v0": BCDroneRunConfig,
    "OfflineDroneCircle-v0": BCDroneCircleConfig,
    "OfflineCarRun-v0": BCCarRunConfig,
    "OfflineAntCircle-v0": BCAntCircleConfig,
}