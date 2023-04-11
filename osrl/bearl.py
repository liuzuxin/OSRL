from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass

import gym
import numpy as np
from tqdm.auto import tqdm, trange  # noqa

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch import distributions as pyd
from torch.distributions.beta import Beta

from saferl.utils import WandbLogger, DummyLogger


class BEARL(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        
        
class BEARLTrainer:
    def __init__(self,
                 ):
        pass