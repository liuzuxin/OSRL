from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


@dataclass
class CDTTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CDT"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    action_head_layers: int = 1
    seq_len: int = 10
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    time_emb: bool = True
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 2048
    update_steps: int = 100_000
    lr_warmup_steps: int = 500
    reward_scale: float = 0.1
    cost_scale: float = 1
    num_workers: int = 8
    # evaluation params
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((450.0, 10), (500.0, 20), (550.0, 50))  # reward, cost
    cost_limit: int = 10
    eval_episodes: int = 10
    eval_every: int = 2500
    # general params
    seed: int = 0
    device: str = "cuda:2"
    threads: int = 6
    # augmentation param
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2
    # maximum absolute value of reward for the augmented trajs
    max_reward: float = 600.0
    # minimum reward above the PF curve
    min_reward: float = 1.0
    # the max drecrease of ret between the associated traj
    # w.r.t the nearest pf traj
    max_rew_decrease: float = 100.0
    # model mode params
    use_rew: bool = True
    use_cost: bool = True
    cost_transform: bool = True
    cost_prefix: bool = False
    add_cost_feat: bool = False
    mul_cost_feat: bool = False
    cat_cost_feat: bool = False
    loss_cost_weight: float = 0.02
    loss_state_weight: float = 0
    cost_reverse: bool = False
    # pf only mode param
    pf_only: bool = False
    rmin: float = 300
    cost_bins: int = 60
    npb: int = 5
    cost_sample: bool = True
    linear: bool = True  # linear or inverse
    start_sampling: bool = False
    prob: float = 0.2
    stochastic: bool = True
    init_temperature: float = 0.1
    no_entropy: bool = False
    # random augmentation
    random_aug: float = 0
    aug_rmin: float = 400
    aug_rmax: float = 500
    aug_cmin: float = -2
    aug_cmax: float = 25
    cgap: float = 5
    rstd: float = 1
    cstd: float = 0.2


@dataclass
class CDTCarCircleConfig(CDTTrainConfig):
    pass


@dataclass
class CDTAntRunConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineAntRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((700.0, 10), (750.0, 20), (800.0, 40))
    # augmentation param
    deg: int = 3
    max_reward: float = 1000.0
    max_rew_decrease: float = 150
    device: str = "cuda:2"


@dataclass
class CDTDroneRunConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineDroneRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((400.0, 10), (500.0, 20), (600.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 700.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTDroneCircleConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 300
    # training params
    task: str = "OfflineDroneCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((700.0, 10), (750.0, 20), (800.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 1000.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTCarRunConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineCarRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((575.0, 10), (575.0, 20), (575.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 600.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTAntCircleConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineAntCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTBallRunConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 100
    # training params
    task: str = "OfflineBallRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((500.0, 10), (500.0, 20), (700.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 1400.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTBallCircleConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineBallCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((700.0, 10), (750.0, 20), (800.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 1000.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTCarButton1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((35.0, 20), (35.0, 40), (35.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class CDTCarButton2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class CDTCarCircle1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((20.0, 20), (22.5, 40), (25.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class CDTCarCircle2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((20.0, 20), (21.0, 40), (22.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class CDTCarGoal1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 50.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTCarGoal2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTCarPush1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((15.0, 20), (15.0, 40), (15.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTCarPush2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((12.0, 20), (12.0, 40), (12.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTPointButton1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTPointButton2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTPointCircle1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((50.0, 20), (52.5, 40), (55.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 65.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTPointCircle2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((45.0, 20), (47.5, 40), (50.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 55.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTPointGoal1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTPointGoal2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTPointPush1Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((15.0, 20), (15.0, 40), (15.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTPointPush2Config(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((12.0, 20), (12.0, 40), (12.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTAntVelocityConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((2800.0, 20), (2800.0, 40), (2800.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class CDTHalfCheetahVelocityConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((3000.0, 20), (3000.0, 40), (3000.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTHopperVelocityConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((1750.0, 20), (1750.0, 40), (1750.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 2000.0
    max_rew_decrease: float = 300
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTSwimmerVelocityConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((160.0, 20), (160.0, 40), (160.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 250.0
    max_rew_decrease: float = 50
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTWalker2dVelocityConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((2800.0, 20), (2800.0, 40), (2800.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3600.0
    max_rew_decrease: float = 800
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTEasySparseConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTEasyMeanConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTEasyDenseConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTMediumSparseConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (300.0, 20), (300.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class CDTMediumMeanConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (300.0, 20), (300.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTMediumDenseConfig(CDTTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CDTHardSparseConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTHardMeanConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class CDTHardDenseConfig(CDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


CDT_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CDTCarCircleConfig,
    "OfflineAntRun-v0": CDTAntRunConfig,
    "OfflineDroneRun-v0": CDTDroneRunConfig,
    "OfflineDroneCircle-v0": CDTDroneCircleConfig,
    "OfflineCarRun-v0": CDTCarRunConfig,
    "OfflineAntCircle-v0": CDTAntCircleConfig,
    "OfflineBallCircle-v0": CDTBallCircleConfig,
    "OfflineBallRun-v0": CDTBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": CDTCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": CDTCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": CDTCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": CDTCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": CDTCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": CDTCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": CDTCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": CDTCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": CDTPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": CDTPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": CDTPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": CDTPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": CDTPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": CDTPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": CDTPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": CDTPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": CDTAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": CDTHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CDTHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CDTSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": CDTWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": CDTEasySparseConfig,
    "OfflineMetadrive-easymean-v0": CDTEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": CDTEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": CDTMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": CDTMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": CDTMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": CDTHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": CDTHardMeanConfig,
    "OfflineMetadrive-harddense-v0": CDTHardDenseConfig
}
