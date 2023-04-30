from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class CPQTrainConfig:
    # wandb params
    project: str = "OSRL-baselines-new"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CPQ"
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
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneRunConfig(CPQTrainConfig):
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneCircleConfig(CPQTrainConfig):
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class CPQCarRunConfig(CPQTrainConfig):
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class CPQAntCircleConfig(CPQTrainConfig):
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 500


CPQ_DEFAULT_CONFIG = {
    "offline-CarCircle-v0": CPQCarCircleConfig,
    "offline-AntRun-v0": CPQAntRunConfig,
    "offline-DroneRun-v0": CPQDroneRunConfig,
    "offline-DroneCircle-v0": CPQDroneCircleConfig,
    "offline-CarRun-v0": CPQCarRunConfig,
    "offline-AntCircle-v0": CPQAntCircleConfig,
}