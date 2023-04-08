from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os.path as osp
import uuid

import bullet_safety_gym  # noqa
import gym  # noqa
import numpy as np
import pyrallis
from pyrallis import field
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from d4srl.offline_env import OfflineEnvWrapper  # noqa
from saferl.utils import WandbLogger

from sdt.sdt_model import DecisionTransformer
from sdt.dataset import SequenceDataset
from sdt.utils import wrap_env, set_seed, load_config
from sdt.sdt_trainer import Trainer

import pandas as pd
import time, os

@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    returns: List[float] = field(default=[300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000], is_mutable=True)
    costs: List[float] = field(default=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], is_mutable=True)

    # returns: List[float] = field(default=[300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000], is_mutable=True)
    # costs: List[float] = field(default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], is_mutable=True)
    
    # returns: List[float] = field(default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], is_mutable=True)
    # costs: List[float] = field(default=[40, 40, 40, 40, 40, 40, 40, 40, 40, 40], is_mutable=True)
    noise_scale: List[float] = None # field(default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], is_mutable=True)
    eval_rollouts: int = 20
    optimal: bool = False
    deterministic_torch: bool = False
    seed: int = 0
    device: str = "cpu"
    threads: int = 4


def parse_model_config(path: str, optimal: bool):
    # exp_dir = osp.join(*path.split("/")[:-2])
    # config_file = osp.join(exp_dir, "config.yaml")
    config_file = osp.join(path, "config.yaml")
    model_name = "model.pt"
    if optimal:
        model_name = "model_optimal.pt"
    model_path = osp.join(path, osp.join("checkpoint", model_name))
    print(f"Load model from {model_path}")
    return load_config(config_file), model_path


def gen_output_dir(cfg: EvalConfig, config: dict):
    ymd_time = time.strftime("%Y-%m-%d")
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    dir = "/home/zijian/code/offline-safe-rl/log/"+config["env_name"]+"_optimal_eval"
    # mode = ""
    # if config["augment_percent"] == 0.:
    #     mode += "_no_aug"
    # if not config["pf_sample"]:
    #     mode += "_no_sample"
    mode = "_no_sample" # "_no_entropy", "_no_stochastic", "_no_cost", "_no_aug", "_no_aug_no_sample", ""
    if not cfg.optimal:
        dir = "/home/zijian/code/offline-safe-rl/log/"+config["env_name"]+"_last_eval"
    output_dir = osp.join(
        osp.join(dir, ymd_time+"_CDT"+mode), hms_time+"-CDT"+mode+"_s"+str(config["seed"])
    )
    # return 
    if osp.exists(output_dir):
        print("Warning: Log dir %s already exists! Storing info there anyway." % output_dir)
    else:
        os.makedirs(output_dir)
    print(f"Saving to {output_dir}")
    return output_dir

@pyrallis.wrap()
def train(cfg: EvalConfig):
    print(f"Reading model from {cfg.path}")
    assert cfg.path is not None, f"The path should not be None"
    config, model_path = parse_model_config(cfg.path, cfg.optimal)
    output_dir = gen_output_dir(cfg, config)

    torch.set_num_threads(cfg.threads)
    set_seed(config["seed"], deterministic_torch=config["deterministic_torch"])
    
    env = wrap_env(
        env=gym.make(config["env_name"]),
        reward_scale=config["reward_scale"],
    )
    env = OfflineEnvWrapper(env)

    target_entropy = -env.action_space.shape[0]

    # model & optimizer & scheduler setup
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        embedding_dim=config["embedding_dim"],
        seq_len=config["seq_len"],
        episode_len=config["episode_len"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        attention_dropout=config["attention_dropout"],
        residual_dropout=config["residual_dropout"],
        embedding_dropout=config["embedding_dropout"],
        max_action=config["max_action"],
        time_emb=config["time_emb"],
        use_rew=config["use_rew"],
        use_cost=config["use_cost"],
        cost_transform=config["cost_transform"],
        add_cost_feat=config["add_cost_feat"],
        mul_cost_feat=config["mul_cost_feat"],
        cat_cost_feat=config["cat_cost_feat"],
        action_head_layers=config["action_head_layers"],
        cost_prefix=config["cost_prefix"],
        stochastic=config["stochastic"],
        init_temperature=config["init_temperature"],
        target_entropy=target_entropy,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    model.load_state_dict(torch.load(model_path)["model_state"])
    model.to(cfg.device)
    
    eval_data = {}
    eval_data["Epoch"] = np.arange(cfg.eval_rollouts)

    trainer = Trainer(model,
                      env,
                      reward_scale=config["reward_scale"],
                      cost_scale=config["cost_scale"],
                      cost_reverse=config["cost_reverse"],
                      device=cfg.device)

    '''
    CDT_optimal = { "Ant-Run": "TR800.0TC10.0",
                    "Car-Circle": "TR550.0TC10.0",
                    "Car-Run": "TR850.0TC10.0",
                    "Drone-Circle": "TR850.0TC10.0",
                    "Drone-Run": "TR550.0TC10.0",}
                    # "Car-Reach": "TR350.0TC40.0"}
    '''
    trtc = {"SafetyAntRun-v0": {"returns": [400, 450, 800, 525, 700, 650, 600, 600, 600, 550, 550, 550],
                                "costs":   [0,   5,   10,  15,  20,  25,  30,  35,  40,  50,  60,  70]},
            "SafetyCarCircle-v0": {"returns": [200, 550, 550, 550, 550, 550, 550, 450, 450, 450, 450, 450],
                                   "costs":   [0,   5,   10,  15,  20,  25,  30,  35,  40,  50,  60,  70]},
            "SafetyCarRun-v0": {"returns":  [450, 500, 850, 550, 575, 575, 575, 575, 575, 575, 575, 575],
                                   "costs": [0,   5,   10,  15,  20,  25,  30,  35,  40,  50,  60,  70]},
            "SafetyDroneCircle-v0": {"returns": [250, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 750, 750, 750],
                                     "costs":   [0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,  60,  70,  80]},
            "SafetyDroneRun-v0": {"returns":  [350, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 600, 450, 500, 450],
                                     "costs": [0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  80,  90,  100]},
        }

    if cfg.noise_scale is None:
        rets = cfg.returns
        costs = cfg.costs
        
        # rets = np.arange(400, 850, 50).tolist()
        # costs = [100] * len(rets)
        
        # rets = [700, 750, 800, 850, 900]
        # costs = [60] * len(rets)
        
        # rets = [550+i*50 for i in range(len(costs))]
        
        # rets = trtc[config["env_name"]]["returns"]
        # costs = trtc[config["env_name"]]["costs"]
        assert len(rets) == len(costs), f"The length of returns {len(rets)} should be equal to costs {len(costs)}!"
        for target_ret, target_cost in zip(rets, costs):
            # critical step, rescale the return!
            set_seed(config["seed"], deterministic_torch=config["deterministic_torch"])
            ret, cost, length = trainer.evaluate(cfg.eval_rollouts, target_ret * config["reward_scale"],
                                                target_cost * config["cost_scale"], eval_data)
            print(f"Target reward {target_ret}, real reward {ret}; target cost {target_cost}, real cost {cost}")
    else:
        target_ret, target_cost = cfg.returns[0], cfg.costs[0]
        print(f"Target reward {target_ret}, target cost {target_cost}")
        for noise_scale in cfg.noise_scale:
            set_seed(config["seed"], deterministic_torch=config["deterministic_torch"])
            trainer.env.set_noise_scale(noise_scale)
            ret, cost, length = trainer.evaluate(cfg.eval_rollouts, target_ret * config["reward_scale"],
                                                 target_cost * config["cost_scale"], eval_data)
            print(f"Noise scale {noise_scale}, real reward {ret}; real cost {cost}")

    df = pd.DataFrame(eval_data)
    df.to_csv(osp.join(output_dir, "progress.txt"))

if __name__ == "__main__":
    train()
