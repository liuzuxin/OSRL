from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCQLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
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
class BCQLBallRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BCQLBallCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


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


@dataclass
class BCQLAntVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLHalfCheetahVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLHopperVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLSwimmerVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLWalker2dVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLEasySparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLEasyMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLEasyDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumSparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardSparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


BCQL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCQLCarCircleConfig,
    "OfflineAntRun-v0": BCQLAntRunConfig,
    "OfflineDroneRun-v0": BCQLDroneRunConfig,
    "OfflineDroneCircle-v0": BCQLDroneCircleConfig,
    "OfflineCarRun-v0": BCQLCarRunConfig,
    "OfflineAntCircle-v0": BCQLAntCircleConfig,
    "OfflineBallCircle-v0": BCQLBallCircleConfig,
    "OfflineBallRun-v0": BCQLBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": BCQLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCQLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCQLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCQLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCQLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCQLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCQLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCQLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BCQLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCQLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCQLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCQLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCQLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCQLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCQLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCQLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BCQLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BCQLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BCQLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BCQLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BCQLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BCQLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BCQLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BCQLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BCQLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BCQLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BCQLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BCQLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BCQLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BCQLHardDenseConfig
}