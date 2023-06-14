<div align="center">
  <a href="http://fsrl.readthedocs.io"><img width="300px" height="auto" src="docs/_static/images/osrl-logo.png"></a>
</div>

<br/>

<div align="center">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
  <!-- [![Documentation Status](https://img.shields.io/readthedocs/fsrl?logo=readthedocs)](https://fsrl.readthedocs.io) -->
  <!-- [![CodeCov](https://codecov.io/github/liuzuxin/fsrl/branch/main/graph/badge.svg?token=BU27LTW9F3)](https://codecov.io/github/liuzuxin/fsrl)
  [![Tests](https://github.com/liuzuxin/fsrl/actions/workflows/test.yml/badge.svg)](https://github.com/liuzuxin/fsrl/actions/workflows/test.yml) -->
  <!-- [![CodeCov](https://img.shields.io/codecov/c/github/liuzuxin/fsrl/main?logo=codecov)](https://app.codecov.io/gh/liuzuxin/fsrl) -->
  <!-- [![tests](https://img.shields.io/github/actions/workflow/status/liuzuxin/fsrl/test.yml?label=tests&logo=github)](https://github.com/liuzuxin/fsrl/tree/HEAD/tests) -->
  <!-- [![PyPI](https://img.shields.io/pypi/v/fsrl?logo=pypi)](https://pypi.org/project/fsrl) -->
  <!-- [![GitHub Repo Stars](https://img.shields.io/github/stars/liuzuxin/fsrl?color=brightgreen&logo=github)](https://github.com/liuzuxin/fsrl/stargazers)
  [![Downloads](https://static.pepy.tech/personalized-badge/fsrl?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/fsrl) -->
   <!-- [![License](https://img.shields.io/github/license/liuzuxin/fsrl?label=license)](#license) -->

</div>

---

**OSRL (Offline Safe Reinforcement Learning)** offers a collection of elegant and extensible implementations of state-of-the-art offline safe reinforcement learning (RL) algorithms. Aimed at propelling research in offline safe RL, OSRL serves as a solid foundation to implement, benchmark, and iterate on safe RL solutions.

The OSRL package is a crucial component of our larger benchmarking suite for offline safe learning, which also includes [FSRL](https://github.com/liuzuxin/FSRL) and [DSRL](https://github.com/liuzuxin/DSRL), and is built to facilitate the development of robust and reliable offline safe RL solutions.

To learn more, please visit our [project website](http://www.offline-saferl.org).

## Structure
The structure of this repo is as follows:
```
├── examples
│   ├── configs  # the training configs of each algorithm
│   ├── eval     # the evaluation escipts
│   ├── train    # the training scipts
├── osrl
│   ├── algorithms  # offline safe RL algorithms
│   ├── common      # base networks and utils
```
The implemented offline safe RL and imitation learning algorithms include:

| Algorithm           | Type           | Description           |
|:-------------------:|:-----------------:|:------------------------:|
| BCQ-Lag             | Q-learning           | [BCQ](https://arxiv.org/pdf/1812.02900.pdf) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| BEAR-Lag            | Q-learning           | [BEARL](https://arxiv.org/abs/1906.00949) with [PID Lagrangian](https://arxiv.org/abs/2007.03964)   |
| CPQ                 | Q-learning           | [Constraints Penalized Q-learning (CPQ))](https://arxiv.org/abs/2107.09003) |
| COptiDICE           | Distribution Correction Estimation           | [Offline Constrained Policy Optimization via stationary DIstribution Correction Estimation](https://arxiv.org/abs/2204.08957) |
| CDT                 | Sequential Modeling | [Constrained Decision Transformer](https://arxiv.org/abs/2302.07351) |
| BC-All                 | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) with all datasets |
| BC-Safe                 | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) with safe trajectories |


## Installation
Pull the repo and install:
```
git clone https://github.com/liuzuxin/OSRL.git
cd osrl
pip install -e .
pip install OApackage==2.7.6
```

## How to use OSRL

The example usage are in the `examples` folder, where you can find the training and evaluation scripts for all the algorithms. 
All the parameters and their default configs for each algorithm are available in the `examples/configs` folder. 
OSRL uses the `WandbLogger` in [FSRL](https://github.com/liuzuxin/FSRL). The offline dataset and offline environments are provided in [DSRL](https://github.com/liuzuxin/DSRL), so make sure you install both of them first.

### Training
For example, to train the `bcql` method, simply run by overriding the default parameters:

```shell
python examples/train/train_bcql.py --task OfflineCarCirvle-v0 --param1 args1 ...
```
By default, the config file and the logs during training will be written to `logs\` folder and the training plots can be viewed online using Wandb.

### Evaluation
To evaluate a trained agent, for example, a BCQ agent, simply run
```
python example/eval/eval_bcql.py --path path_to_model --eval_episodes 20
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run 20 episodes, and print the average normalized reward and cost.
