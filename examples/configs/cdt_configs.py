from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass


@dataclass
class CDTTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = "CDT"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
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
    task: str = "offline-CarCircle-v0"
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
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 20), 
                                                     (300, 0), 
                                                     (400, 0), 
                                                     (200, 0))  # reward, cost
    eval_episodes: int = 10
    eval_every: int = 2500
    # general params
    seed: int = 11
    device: str = "cuda:0"
    threads: int = 6
    # augmentation param
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2
    # maximum absolute value of reward for the augmented trajs
    max_reward: float = 500.0
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
    # wandb params
    group: str = "AntRun"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "offline-AntRun-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((500, 0), (600.0, 0), 
                                                     (600, 10), (600, 20), 
                                                     (700, 20), (600, 40),
                                                     (700, 40), (700, 60))
    # augmentation param
    max_reward: float = 700.0
    max_rew_decrease: float = 250
    # random augmentation
    aug_rmin: float = 500
    aug_rmax: float = 700
    aug_cmin: float = -2
    aug_cmax: float = 25


@dataclass
class CDTDroneRunConfig(CDTTrainConfig):
    # wandb params
    group: str = "DroneRun"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 100
    # training params
    task: str = "offline-DroneRun-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 0), (300, 10), 
                                                     (300, 20), (400, 20), 
                                                     (500, 20), (300, 40),
                                                     (400, 40), (500, 40), 
                                                     (500, 60))
    # augmentation param
    max_reward: float = 450.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    # random augmentation
    aug_rmin: float = 300
    aug_rmax: float = 450
    aug_cmin: float = -2
    aug_cmax: float = 40


@dataclass
class CDTDroneCircleConfig(CDTTrainConfig):
    # wandb params
    group: str = "DroneCircle"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 300
    # training params
    task: str = "offline-DroneCircle-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 0), (300, 10), 
                                                     (300, 20), (400, 20), 
                                                     (500, 20), (300, 40),
                                                     (400, 40), (500, 40), 
                                                     (500, 60))
    # augmentation param
    max_reward: float = 800.0
    max_rew_decrease: float = 300
    min_reward: float = 1


@dataclass
class CDTCarRunConfig(CDTTrainConfig):
    # wandb params
    group: str = "CarRun"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "offline-CarRun-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 0), (300, 10), 
                                                     (300, 20), (400, 20), 
                                                     (500, 20), (300, 40),
                                                     (400, 40), (500, 40), 
                                                     (500, 60))
    # augmentation param
    max_reward: float = 600.0
    max_rew_decrease: float = 100
    min_reward: float = 1


@dataclass
class CDTAntCircleConfig(CDTTrainConfig):
    # wandb params
    group: str = "AntCircle"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "offline-AntCircle-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 0), (300, 10), 
                                                     (300, 20), (400, 20), 
                                                     (500, 20), (300, 40),
                                                     (400, 40), (500, 40), 
                                                     (500, 60))
    # augmentation param
    max_reward: float = 250.0
    max_rew_decrease: float = 120
    min_reward: float = 1


@dataclass
class CDTCarReachConfig(CDTTrainConfig):
    # wandb params
    group: str = "CarReach"
    prefix: str = "CDT"
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "offline-CarReach-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((300.0, 0), (300, 10), 
                                                     (300, 20), (400, 20), 
                                                     (500, 20), (300, 40),
                                                     (400, 40), (500, 40), 
                                                     (500, 60))
    # augmentation param
    max_reward: float = 300.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    
    
CDT_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": CDTCarCircleConfig,
    "offline-AntRun-v0": CDTAntRunConfig,
    "offline-DroneRun-v0": CDTDroneRunConfig,
    "offline-DroneCircle-v0": CDTDroneCircleConfig,
    "offline-CarRun-v0": CDTCarRunConfig,
    "offline-AntCircle-v0": CDTAntCircleConfig,
}