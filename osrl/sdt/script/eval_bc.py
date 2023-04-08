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
from tqdm.auto import trange  # noqa
from dsrl.offline_env import OfflineEnvWrapper  # noqa

from run_bc import BehaviorCloning, Trainer
from sdt.utils import wrap_env, set_seed, load_config

import pandas as pd
import time, os

@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    eval_rollouts: int = 20
    optimal: bool = False
    costs: List[float] = field(default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], is_mutable=True)
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


def gen_output_dir(cfg, config):
    ymd_time = time.strftime("%Y-%m-%d")
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    dir = "/home/zijian/code/offline-safe-rl/log/"+config["env_name"]+"_optimal_eval"
    if not cfg.optimal:
        dir = "/home/zijian/code/offline-safe-rl/log/"+config["env_name"]+"_last_eval"
    
    policy_name = "BC-Safe"
    if config["aug_cost"]:
        policy_name = "MTBC"
    
    output_dir = osp.join(osp.join(dir, ymd_time+"_"+policy_name), hms_time+"-"+policy_name+"_s"+str(config["seed"]))
    
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

    state_dim = env.observation_space.shape[0]
    if config["aug_cost"]:
        state_dim += 1
    # model & optimizer setup
    model = BehaviorCloning(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=config["max_action"],
        episode_len=config["episode_len"],
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    model.load_state_dict(torch.load(model_path)["actor"])
    model.to(cfg.device)

    trainer = Trainer(model,
                      env,
                      device=cfg.device,
                      cost_limit=config["cost_limit"],
                      aug_cost=config["aug_cost"])

    costs = cfg.costs
    eval_data = {"Epoch": np.arange(cfg.eval_rollouts)}
    if config["aug_cost"]:
        for target_cost in costs:
            set_seed(config["seed"], deterministic_torch=config["deterministic_torch"])
            trainer.cost_limit = target_cost
            ret, cost, length = trainer.evaluate(cfg.eval_rollouts)
            print(f"Real reward {np.mean(ret)}; target cost {target_cost}, real cost {np.mean(cost)}")
            
            eval_data["TC"+str(target_cost)+"_EpRet"] = ret
            eval_data["TC"+str(target_cost)+"_EpCost"] = cost
            eval_data["TC"+str(target_cost)+"_EpLen"] = length
    else:
        ret, cost, length = trainer.evaluate(cfg.eval_rollouts)
        print(f"EpRet = {np.mean(ret)}, EpCost = {np.mean(cost)}, EpLen =  {np.mean(length)}")
        eval_data["EpRet"] = ret
        eval_data["EpCost"] = cost
        eval_data["EpLen"] = length

    df = pd.DataFrame(eval_data)
    df.to_csv(osp.join(output_dir, "progress.txt"))

if __name__ == "__main__":
    train()
