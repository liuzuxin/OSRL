from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class CPQTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CPQ"
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
    critic_lr: float = 0.001
    alpha_lr: float = 0.0001
    vae_lr: float = 0.001
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
    beta: float = 0.5
    num_q: int = 2
    num_qc: int = 2
    qc_scalar: float = 1.5
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class CPQCarCircleConfig(CPQTrainConfig):
    pass


@dataclass
class CPQAntRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class CPQCarRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class CPQAntCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class CPQBallRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class CPQBallCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class CPQCarButton1Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQCarButton2Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQCarCircle1Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CPQCarCircle2Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CPQCarGoal1Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQCarGoal2Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQCarPush1Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQCarPush2Config(CPQTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointButton1Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointButton2Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointCircle1Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CPQPointCircle2Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CPQPointGoal1Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointGoal2Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointPush1Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQPointPush2Config(CPQTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CPQAntVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQHalfCheetahVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQHopperVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQSwimmerVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQWalker2dVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQEasySparseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQEasyMeanConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQEasyDenseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQMediumSparseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQMediumMeanConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQMediumDenseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQHardSparseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQHardMeanConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CPQHardDenseConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


CPQ_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CPQCarCircleConfig,
    "OfflineAntRun-v0": CPQAntRunConfig,
    "OfflineDroneRun-v0": CPQDroneRunConfig,
    "OfflineDroneCircle-v0": CPQDroneCircleConfig,
    "OfflineCarRun-v0": CPQCarRunConfig,
    "OfflineAntCircle-v0": CPQAntCircleConfig,
    "OfflineBallCircle-v0": CPQBallCircleConfig,
    "OfflineBallRun-v0": CPQBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": CPQCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": CPQCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": CPQCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": CPQCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": CPQCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": CPQCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": CPQCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": CPQCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": CPQPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": CPQPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": CPQPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": CPQPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": CPQPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": CPQPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": CPQPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": CPQPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": CPQAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": CPQHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CPQHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CPQSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": CPQWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": CPQEasySparseConfig,
    "OfflineMetadrive-easymean-v0": CPQEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": CPQEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": CPQMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": CPQMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": CPQMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": CPQHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": CPQHardMeanConfig,
    "OfflineMetadrive-harddense-v0": CPQHardDenseConfig
}