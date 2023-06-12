import datetime
import itertools
import json
import os
import os.path as osp
import random
import subprocess
import time
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from fsrl.utils.logger.logger_util import colorize


def seed_all(seed=1029, others: Optional[list] = None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def load_config_and_model(path: str, best: bool = False):
    '''
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one. Defaults to False.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    '''
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")


###################### naming utils ######################


def to_string(values):
    '''
    Recursively convert a sequence or dictionary of values to a string representation.
    :param values: the sequence or dictionary of values to be converted to a string.
    :return: a string representation of the input values.
    '''
    name = ""
    if isinstance(values, Sequence) and not isinstance(values, str):
        for i, v in enumerate(values):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(v)
        return name
    elif isinstance(values, Dict):
        for i, k in enumerate(sorted(values.keys())):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(values[k])
        return name
    else:
        return str(values)


DEFAULT_SKIP_KEY = [
    "task", "reward_threshold", "logdir", "worker", "project", "group", "name", "prefix",
    "suffix", "save_interval", "render", "verbose", "save_ckpt", "training_num",
    "testing_num", "epoch", "device", "thread"
]

DEFAULT_KEY_ABBRE = {
    "cost_limit": "cost",
    "mstep_iter_num": "mnum",
    "estep_iter_num": "enum",
    "estep_kl": "ekl",
    "mstep_kl_mu": "kl_mu",
    "mstep_kl_std": "kl_std",
    "mstep_dual_lr": "mlr",
    "estep_dual_lr": "elr",
    "update_per_step": "update"
}


def auto_name(default_cfg: dict,
              current_cfg: dict,
              prefix: str = "",
              suffix: str = "",
              skip_keys: list = DEFAULT_SKIP_KEY,
              key_abbre: dict = DEFAULT_KEY_ABBRE) -> str:
    '''
    Automatic generate the experiment name by comparing the current config with the default one.

    :param dict default_cfg: a dictionary containing the default configuration values.
    :param dict current_cfg: a dictionary containing the current configuration values.
    :param str prefix: (optional) a string to be added at the beginning of the generated name.
    :param str suffix: (optional) a string to be added at the end of the generated name.
    :param list skip_keys: (optional) a list of keys to be skipped when generating the name.
    :param dict key_abbre: (optional) a dictionary containing abbreviations for keys in the generated name.

    :return str: a string representing the generated experiment name.
    '''
    name = prefix
    for i, k in enumerate(sorted(default_cfg.keys())):
        if default_cfg[k] == current_cfg[k] or k in skip_keys:
            continue
        prefix = "_" if len(name) else ""
        value = to_string(current_cfg[k])
        # replace the name with abbreviation if key has abbreviation in key_abbre
        if k in key_abbre:
            k = key_abbre[k]
        # Add the key-value pair to the name variable with the prefix
        name += prefix + k + value
    if len(suffix):
        name = name + "_" + suffix if len(name) else suffix

    name = "default" if not len(name) else name
    name = f"{name}-{str(uuid.uuid4())[:4]}"
    return name


class ExperimentGrid():

    def __init__(self, log_name=None) -> None:
        """
        log_name : str, optional
            The name of the grid experiment.
        """
        self.log_root = log_name or datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        os.makedirs(os.path.join(os.getcwd(), 'logs', self.log_root), exist_ok=True)
        self.log_root = os.path.join(os.getcwd(), 'logs', self.log_root)

    def run(self,
            instructions: List[str],
            exp_names: Optional[List[str]] = None,
            gpus: List[int] = [0],
            max_parallel: int = 1,
            **kwargs: dict) -> None:
        """
        Execute experiments in parallel and save logs.

        Parameters:
        ----------
        instructions : List[str]
            The command line instruction for each experiment.
        exp_names : List[str], optional
            The name for each experiment.
        gpus : List[int], default [0]
            The GPU ID for each experiment.
        max_parallel : int, default 1
            The maximum number of experiments to run in parallel.
        """
        exp_names = exp_names or [f"exp_{i}" for i in range(len(instructions))]
        num_gpu = len(gpus)
        running_experiments = []

        for i, ins in enumerate(instructions):
            while len(running_experiments) >= max_parallel:
                running_experiments = [
                    p for p in running_experiments if p.poll() is None
                ]
                time.sleep(1)

            print(colorize(f"Running experiment {i}", "green", True) + f": {ins}")
            redirect = f"> {self.log_root}/{exp_names[i]}_gpu{gpus[i % num_gpu]}.out"
            command = f"CUDA_VISIBLE_DEVICES={gpus[i % num_gpu]} {instructions[i]} {redirect}"
            p = subprocess.Popen(command, shell=True)
            running_experiments.append(p)

        while running_experiments:
            running_experiments = [p for p in running_experiments if p.poll() is None]
            time.sleep(1)

    def compose(self,
                template: str,
                args: List[List],
                dump_param: Optional[bool] = True,
                suffix: Optional[str] = None) -> List[str]:
        """
        Generate a list of instructions from a template and arguments.

        Parameters:
        ----------
        template : str
            The template of the instruction in the form of a format string.
        args : List[List]
            The list of arguments for each instruction.
        dump_param : bool, optional, default True
            If True, dump the args and template to a file.
        suffix : str, optional
            The suffix for the dumped file.

        Returns:
        ----------
        List[str]
            The list of instructions ready to run.
        """
        if dump_param:
            os.makedirs(os.path.join(self.log_root, 'param'), exist_ok=True)
            with open(os.path.join(self.log_root, f"param/args_{suffix}.json"),
                      "w") as f:
                json.dump(args, f)
            with open(os.path.join(self.log_root, f"param/template_{suffix}.json"),
                      "w") as f:
                json.dump(template, f)

        all_combinations = list(itertools.product(*args))
        return [template.format(*combination) for combination in all_combinations]
