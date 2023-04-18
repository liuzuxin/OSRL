from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BCTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = "BC"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "log"
    verbose: bool = True
    # training params
    task: str = "offline-CarCircle-v0"
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
    # wandb params
    group: str = "AntRun"
    # training params
    task: str = "offline-AntRun-v0"
    episode_len: int = 200
    frontier_fn: callable = lambda x: 600 + 10/3*x
    frontier_range=50.0


@dataclass
class BCDroneRunConfig(BCTrainConfig):
    # wandb params
    group: str = "DroneRun"
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 100
    frontier_fn: callable = lambda x: 325 + 125/70*x
    frontier_range=50.0


@dataclass
class BCDroneCircleConfig(BCTrainConfig):
    # wandb params
    group: str = "DroneCircle"
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300
    frontier_fn: callable = lambda x: 600 + 4*x
    frontier_range=50.0


@dataclass
class BCCarRunConfig(BCTrainConfig):
    # wandb params
    group: str = "CarRun"
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200
    frontier_fn: callable = lambda x: 600
    frontier_range=50.0


@dataclass
class BCAntCircleConfig(BCTrainConfig):
    # wandb params
    group: str = "AntCircle"
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 200


BC_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": BCCarCircleConfig,
    "offline-AntRun-v0": BCAntRunConfig,
    "offline-DroneRun-v0": BCDroneRunConfig,
    "offline-DroneCircle-v0": BCDroneCircleConfig,
    "offline-CarRun-v0": BCCarRunConfig,
    "offline-AntCircle-v0": BCAntCircleConfig,
}