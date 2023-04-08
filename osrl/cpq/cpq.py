import pyrallis
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


@dataclass
class TrainConfig:
    # wandb params
    project: str = "DSRL"
    group: str = "CarCircle"
    name: Optional[str] = None
    prefix: Optional[str] = ""
    suffix: Optional[str] = ""
    log_path: Optional[str] = "log"
    # general params
    env_name: str = "SafetyCarCircle-v0"
    deterministic_torch: bool = False
    seed: int = 0
    device: str = "cuda:0"
    thread: int = 4
    # model params
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    a_hidden_sizes: list = [128, 128]
    c_hidden_sizes: list = [128, 128]
    vae_hidden_sizes: int = 64
    alpha_max: float = 0.2
    sample_action_num: int = 10
    gamma: float = 0.99
    polyak: float = 0.995
    lmbda: float = 0.75
    beta: float = 1.5
    num_q: int = 1
    num_qc: int = 1
    qc_scalar: float = 1.5
    cost_limit: int = 20
    timeout_steps: int = 200
    # evaluation params
    eval_rollouts: int = 4
    eval_every: int = 2500
    

@pyrallis.wrap()
def train(config: TrainConfig):
    pass