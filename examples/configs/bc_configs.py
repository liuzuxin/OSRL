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
    density: float = 1.0
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


@dataclass
class BCCarButton1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarButton2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarCircle1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCCarCircle2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCCarGoal1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarGoal2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarPush1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarPush2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointButton1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointButton2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointCircle1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPointCircle2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPointGoal1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointGoal2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointPush1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointPush2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


BC_DEFAULT_CONFIG = {
    "OfflineCarCircle-v0": BCCarCircleConfig,
    "OfflineAntRun-v0": BCAntRunConfig,
    "OfflineDroneRun-v0": BCDroneRunConfig,
    "OfflineDroneCircle-v0": BCDroneCircleConfig,
    "OfflineCarRun-v0": BCCarRunConfig,
    "OfflineAntCircle-v0": BCAntCircleConfig,
    "OfflineCarButton1Gymnasium-v0": BCCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCCarPush2Config,

    "OfflinePointButton1Gymnasium-v0": BCPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCPointPush2Config,
}