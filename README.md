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

OSRL (Offline Safe Reinforcement Learning) is an open-source implementation for offline safe reinforcement learning algorithms.

## Structure
The structure of this repo is as follows:
```
├── osrl  # offline safe RL algorithms
│   ├── common_net.py
│   ├── common_util.py
│   ├── xx_algorithm.py
│   ├── xx_algorithm_util.py
│   ├── ...
```

## Installation
Pull the repo and install:
```
git clone https://github.com/liuzuxin/offline-safe-rl-baselines.git
cd offline-safe-rl-baselines
pip install -e .
```

## How to use DSRL
DSRL uses the [OpenAI Gym](https://github.com/openai/gym) API. Tasks are created via the `gym.make` function. Each task is associated with a fixed offline dataset, which can be obtained with the `env.get_dataset()` method. This method returns a dictionary with:
- `observations`: An N × obs_dim array of observations.
- `next_observations`: An N × obs_dim of next observations.
- `actions`: An N × act_dim array of actions.
- `rewards`: An N dimensional array of rewards.
- `costs`: An N dimensional array of costs.
- `terminals`: An N dimensional array of episode termination flags. This is true when episodes end due to termination conditions such as falling over.
- `timeouts`: An N dimensional array of termination flags. This is true when episodes end due to reaching the maximum episode length.

```python
import gym
import dsrl

# set seed
seed = 0

# Create the environment
env = gym.make('offline-CarCircle-v0')

# dsrl abides by the OpenAI gym interface
obs, info = env.reset(seed=seed)
obs, reward, terminal, timeout, info = env.step(env.action_space.sample())
cost = info["cost"]

# Each task is associated with a dataset
# dataset contains observations, next_observatiosn, actions, rewards, costs, terminals, timeouts
dataset = env.get_dataset()
print(dataset['observations']) # An N x obs_dim Numpy array of observations
```

Datasets are automatically downloaded to the `~/.dsrl/datasets` directory when `get_dataset()` is called. If you would like to change the location of this directory, you can set the `$DSRL_DATASET_DIR` environment variable to the directory of your choosing, or pass in the dataset filepath directly into the `get_dataset` method.

### Normalizing Scores
- Set target cost by using `env.set_target_cost(target_cost)` function, where `target_cost` is the undiscounted sum of costs of an episode
- You can use the `env.get_normalized_score(return, cost_return)` function to compute a normalized reward and cost for an episode, where `returns` and `cost_returns` are the undiscounted sum of rewards and costs respectively of an episode. 
- The individual min and max reference returns are stored in `dsrl/infos.py` for reference.



