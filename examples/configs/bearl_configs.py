from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BEARLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BEARL"
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
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 300_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    lmbda: float = 0.75
    mmd_sigma: float = 50
    target_mmd_thresh: float = 0.05
    num_samples_mmd_match: int = 10
    start_update_policy_step: int = 0
    kernel: str = "gaussian"  # or "laplacian"
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BEARLCarCircleConfig(BEARLTrainConfig):
    pass


@dataclass
class BEARLAntRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BEARLCarRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BEARLAntCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BEARLBallRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BEARLBallCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class BEARLCarButton1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLCarButton2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLCarCircle1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BEARLCarCircle2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BEARLCarGoal1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLCarGoal2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLCarPush1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLCarPush2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointButton1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointButton2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointCircle1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BEARLPointCircle2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BEARLPointGoal1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointGoal2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointPush1Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLPointPush2Config(BEARLTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BEARLAntVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLHalfCheetahVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLHopperVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLSwimmerVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLWalker2dVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLEasySparseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLEasyMeanConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLEasyDenseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLMediumSparseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLMediumMeanConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLMediumDenseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLHardSparseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLHardMeanConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BEARLHardDenseConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


BEARL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BEARLCarCircleConfig,
    "OfflineAntRun-v0": BEARLAntRunConfig,
    "OfflineDroneRun-v0": BEARLDroneRunConfig,
    "OfflineDroneCircle-v0": BEARLDroneCircleConfig,
    "OfflineCarRun-v0": BEARLCarRunConfig,
    "OfflineAntCircle-v0": BEARLAntCircleConfig,
    "OfflineBallCircle-v0": BEARLBallCircleConfig,
    "OfflineBallRun-v0": BEARLBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": BEARLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BEARLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BEARLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BEARLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BEARLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BEARLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BEARLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BEARLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BEARLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BEARLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BEARLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BEARLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BEARLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BEARLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BEARLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BEARLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BEARLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BEARLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BEARLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BEARLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BEARLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BEARLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BEARLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BEARLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BEARLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BEARLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BEARLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BEARLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BEARLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BEARLHardDenseConfig
}
