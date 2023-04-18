from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BCQLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # training params
    task: str = "offline-CarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BCQLCarCircleConfig(BCQLTrainConfig):
    pass


@dataclass
class BCQLAntRunConfig(BCQLTrainConfig):
    # wandb params
    group: str = "AntRun"
    prefix: str = "BCQL"
    # training params
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class BCQLDroneRunConfig(BCQLTrainConfig):
    # wandb params
    group: str = "DroneRun"
    prefix: str = "BCQL"
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 100


@dataclass
class BCQLDroneCircleConfig(BCQLTrainConfig):
    # wandb params
    group: str = "DroneCircle"
    prefix: str = "BCQL"
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCQLCarRunConfig(BCQLTrainConfig):
    # wandb params
    group: str = "CarRun"
    prefix: str = "BCQL"
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class BCQLAntCircleConfig(BCQLTrainConfig):
    # wandb params
    group: str = "AntCircle"
    prefix: str = "BCQL"
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 200


BCQL_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": BCQLCarCircleConfig,
    "offline-AntRun-v0": BCQLAntRunConfig,
    "offline-DroneRun-v0": BCQLDroneRunConfig,
    "offline-DroneCircle-v0": BCQLDroneCircleConfig,
    "offline-CarRun-v0": BCQLCarRunConfig,
    "offline-AntCircle-v0": BCQLAntCircleConfig,
}