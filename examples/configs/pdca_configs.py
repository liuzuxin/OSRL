from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class PDCATrainConfig:
    # wandb params
    project: str = "PDCA"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "PDCA"
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
    B: float = 5
    cost_threshold: float = 10
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
class PDCACarCircleConfig(PDCATrainConfig):
    pass


@dataclass
class PDCAAntRunConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class PDCADroneRunConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class PDCADroneCircleConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class PDCACarRunConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class PDCAAntCircleConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class PDCABallRunConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class PDCABallCircleConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class PDCACarButton1Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCACarButton2Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCACarCircle1Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDCACarCircle2Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDCACarGoal1Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCACarGoal2Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCACarPush1Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCACarPush2Config(PDCATrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointButton1Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointButton2Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointCircle1Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDCAPointCircle2Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDCAPointGoal1Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointGoal2Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointPush1Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAPointPush2Config(PDCATrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDCAAntVelocityConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDCAHalfCheetahVelocityConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDCAHopperVelocityConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDCASwimmerVelocityConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDCAWalker2dVelocityConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDCAEasySparseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAEasyMeanConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAEasyDenseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAMediumSparseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAMediumMeanConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAMediumDenseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAHardSparseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAHardMeanConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDCAHardDenseConfig(PDCATrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


PDCA_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": PDCACarCircleConfig,
    "OfflineAntRun-v0": PDCAAntRunConfig,
    "OfflineDroneRun-v0": PDCADroneRunConfig,
    "OfflineDroneCircle-v0": PDCADroneCircleConfig,
    "OfflineCarRun-v0": PDCACarRunConfig,
    "OfflineAntCircle-v0": PDCAAntCircleConfig,
    "OfflineBallCircle-v0": PDCABallCircleConfig,
    "OfflineBallRun-v0": PDCABallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": PDCACarButton1Config,
    "OfflineCarButton2Gymnasium-v0": PDCACarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": PDCACarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": PDCACarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": PDCACarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": PDCACarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": PDCACarPush1Config,
    "OfflineCarPush2Gymnasium-v0": PDCACarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": PDCAPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": PDCAPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": PDCAPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": PDCAPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": PDCAPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": PDCAPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": PDCAPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": PDCAPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": PDCAAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": PDCAHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": PDCAHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": PDCASwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": PDCAWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": PDCAEasySparseConfig,
    "OfflineMetadrive-easymean-v0": PDCAEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": PDCAEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": PDCAMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": PDCAMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": PDCAMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": PDCAHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": PDCAHardMeanConfig,
    "OfflineMetadrive-harddense-v0": PDCAHardDenseConfig
}
