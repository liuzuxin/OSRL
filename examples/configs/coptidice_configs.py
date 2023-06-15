from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class COptiDICETrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "COptiDICE"
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
    actor_lr: float = 0.0001
    critic_lr: float = 0.0001
    scalar_lr: float = 0.0001
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    alpha: float = 0.5
    gamma: float = 0.99
    cost_ub_epsilon: float = 0.01
    f_type: str = "softchi"
    num_nu: int = 2
    num_chi: int = 2
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class COptiDICECarCircleConfig(COptiDICETrainConfig):
    pass


@dataclass
class COptiDICEAntRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEDroneRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEDroneCircleConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class COptiDICECarRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEAntCircleConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class COptiDICEBallRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class COptiDICEBallCircleConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class COptiDICECarButton1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICECarButton2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICECarCircle1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class COptiDICECarCircle2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class COptiDICECarGoal1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICECarGoal2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICECarPush1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICECarPush2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointButton1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointButton2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointCircle1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class COptiDICEPointCircle2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class COptiDICEPointGoal1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointGoal2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointPush1Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEPointPush2Config(COptiDICETrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class COptiDICEAntVelocityConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class COptiDICEHalfCheetahVelocityConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class COptiDICEHopperVelocityConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class COptiDICESwimmerVelocityConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class COptiDICEWalker2dVelocityConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class COptiDICEEasySparseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEEasyMeanConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEEasyDenseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEMediumSparseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEMediumMeanConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEMediumDenseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEHardSparseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEHardMeanConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class COptiDICEHardDenseConfig(COptiDICETrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


COptiDICE_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": COptiDICECarCircleConfig,
    "OfflineAntRun-v0": COptiDICEAntRunConfig,
    "OfflineDroneRun-v0": COptiDICEDroneRunConfig,
    "OfflineDroneCircle-v0": COptiDICEDroneCircleConfig,
    "OfflineCarRun-v0": COptiDICECarRunConfig,
    "OfflineAntCircle-v0": COptiDICEAntCircleConfig,
    "OfflineBallCircle-v0": COptiDICEBallCircleConfig,
    "OfflineBallRun-v0": COptiDICEBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": COptiDICECarButton1Config,
    "OfflineCarButton2Gymnasium-v0": COptiDICECarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": COptiDICECarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": COptiDICECarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": COptiDICECarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": COptiDICECarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": COptiDICECarPush1Config,
    "OfflineCarPush2Gymnasium-v0": COptiDICECarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": COptiDICEPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": COptiDICEPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": COptiDICEPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": COptiDICEPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": COptiDICEPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": COptiDICEPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": COptiDICEPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": COptiDICEPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": COptiDICEAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": COptiDICEHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": COptiDICEHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": COptiDICESwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": COptiDICEWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": COptiDICEEasySparseConfig,
    "OfflineMetadrive-easymean-v0": COptiDICEEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": COptiDICEEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": COptiDICEMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": COptiDICEMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": COptiDICEMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": COptiDICEHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": COptiDICEHardMeanConfig,
    "OfflineMetadrive-harddense-v0": COptiDICEHardDenseConfig
}
