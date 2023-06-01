from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BCQLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines-new"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL"
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
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCQLDroneRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BCQLDroneCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCQLCarRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BCQLAntCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BCQLCarButton1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarButton2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarCircle1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLCarCircle2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLCarGoal1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarGoal2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarPush1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarPush2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointButton1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointButton2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointCircle1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLPointCircle2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLPointGoal1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointGoal2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointPush1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointPush2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


BCQL_DEFAULT_CONFIG = {
    "OfflineCarCircle-v0": BCQLCarCircleConfig,
    "OfflineAntRun-v0": BCQLAntRunConfig,
    "OfflineDroneRun-v0": BCQLDroneRunConfig,
    "OfflineDroneCircle-v0": BCQLDroneCircleConfig,
    "OfflineCarRun-v0": BCQLCarRunConfig,
    "OfflineAntCircle-v0": BCQLAntCircleConfig,

    "OfflineCarButton1Gymnasium-v0": BCQLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCQLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCQLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCQLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCQLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCQLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCQLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCQLCarPush2Config,

    "OfflinePointButton1Gymnasium-v0": BCQLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCQLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCQLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCQLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCQLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCQLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCQLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCQLPointPush2Config,
}