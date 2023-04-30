from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class COptiDICETrainConfig:
    # wandb params
    project: str = "OSRL-baselines-new"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "COptiDICE"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    # training params
    task: str = "offline-CarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
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
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEDroneRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEDroneCircleConfig(COptiDICETrainConfig):
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class COptiDICECarRunConfig(COptiDICETrainConfig):
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class COptiDICEAntCircleConfig(COptiDICETrainConfig):
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 500


COptiDICE_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": COptiDICECarCircleConfig,
    "offline-AntRun-v0": COptiDICEAntRunConfig,
    "offline-DroneRun-v0": COptiDICEDroneRunConfig,
    "offline-DroneCircle-v0": COptiDICEDroneCircleConfig,
    "offline-CarRun-v0": COptiDICECarRunConfig,
    "offline-AntCircle-v0": COptiDICEAntCircleConfig,
}