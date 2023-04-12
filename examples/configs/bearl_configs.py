from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BEARLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = "BEARL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "log"
    verbose: bool = True
    # training params
    task: str = "offline-CarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    thread: int = 4
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
    num_q: int = 1
    num_qc: int = 1
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BEARLCarCircleConfig(BEARLTrainConfig):
    pass


@dataclass
class BEARLAntRunConfig(BEARLTrainConfig):
    # wandb params
    group: str = "AntRun"
    prefix: str = "BEARL"
    # training params
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneRunConfig(BEARLTrainConfig):
    # wandb params
    group: str = "DroneRun"
    prefix: str = "BEARL"
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 100


@dataclass
class BEARLDroneCircleConfig(BEARLTrainConfig):
    # wandb params
    group: str = "DroneCircle"
    prefix: str = "BEARL"
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class BEARLCarRunConfig(BEARLTrainConfig):
    # wandb params
    group: str = "CarRun"
    prefix: str = "BEARL"
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class BEARLAntCircleConfig(BEARLTrainConfig):
    # wandb params
    group: str = "AntCircle"
    prefix: str = "BEARL"
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 200


BEARL_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": BEARLCarCircleConfig,
    "offline-AntRun-v0": BEARLAntRunConfig,
    "offline-DroneRun-v0": BEARLDroneRunConfig,
    "offline-DroneCircle-v0": BEARLDroneCircleConfig,
    "offline-CarRun-v0": BEARLCarRunConfig,
    "offline-AntCircle-v0": BEARLAntCircleConfig,
}
