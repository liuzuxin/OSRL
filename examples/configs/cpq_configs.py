from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class CPQTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = "CPQ"
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
    actor_lr: float = 0.0001
    critic_lr: float = 0.0001
    alpha_lr: float = 0.0001
    vae_lr: float = 0.0001
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    max_action: float = 1.0
    a_hidden_sizes: List[float] = field(default=[300, 300], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[400, 400], is_mutable=True)
    vae_hidden_sizes: int = 400
    alpha_max: float = 0.2
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    lmbda: float = 0.75
    beta: float = 1.5
    num_q: int = 1
    num_qc: int = 1
    qc_scalar: float = 1.5
    # evaluation params
    eval_episodes: int = 4
    eval_every: int = 2500


@dataclass
class CPQCarCircleConfig(CPQTrainConfig):
    pass


@dataclass
class CPQAntRunConfig(CPQTrainConfig):
    # wandb params
    group: str = "AntRun"
    prefix: str = "CPQ"
    # training params
    task: str = "offline-AntRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneRunConfig(CPQTrainConfig):
    # wandb params
    group: str = "DroneRun"
    prefix: str = "CPQ"
    # training params
    task: str = "offline-DroneRun-v0"
    episode_len: int = 100


@dataclass
class CPQDroneCircleConfig(CPQTrainConfig):
    # wandb params
    group: str = "DroneCircle"
    prefix: str = "CPQ"
    # training params
    task: str = "offline-DroneCircle-v0"
    episode_len: int = 300


@dataclass
class CPQCarRunConfig(CPQTrainConfig):
    # wandb params
    group: str = "CarRun"
    prefix: str = "CPQ"
    # training params
    task: str = "offline-CarRun-v0"
    episode_len: int = 200


@dataclass
class CPQAntCircleConfig(CPQTrainConfig):
    # wandb params
    group: str = "AntCircle"
    prefix: str = "CPQ"
    # training params
    task: str = "offline-AntCircle-v0"
    episode_len: int = 200
