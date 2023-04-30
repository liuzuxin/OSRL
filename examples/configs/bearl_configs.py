from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BEARLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines-new"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BEARL"
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
    actor_lr: float = 0.001
    critic_lr: float = 0.001
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
    lmbda: float = 0.75
    mmd_sigma: float = 50
    target_mmd_thresh: float = 0.05
    num_samples_mmd_match: int = 10
    start_update_policy_step: int = 0
    kernel: str = "gaussian" # or "laplacian"
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
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneRunConfig(BEARLTrainConfig):
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class BEARLCarRunConfig(BEARLTrainConfig):
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class BEARLAntCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 500


BEARL_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": BEARLCarCircleConfig,
    "offline-AntRun-v0": BEARLAntRunConfig,
    "offline-DroneRun-v0": BEARLDroneRunConfig,
    "offline-DroneCircle-v0": BEARLDroneCircleConfig,
    "offline-CarRun-v0": BEARLCarRunConfig,
    "offline-AntCircle-v0": BEARLAntCircleConfig,
}
