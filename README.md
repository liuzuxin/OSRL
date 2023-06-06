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

The OSRL package is a crucial component of our larger benchmarking suite for offline safe learning, which also includes [FSRL](https://github.com/liuzuxin/fsrl) and [DSRL](https://github.com/liuzuxin/dsrl), and is built to facilitate the development of robust and reliable offline safe RL solutions.

To learn more, please visit our [project website](http://www.offline-saferl.org).

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
git clone https://github.com/liuzuxin/osrl.git
cd osrl
pip install -e .
```

## How to use OSRL

The example usage are in the `examples` folder, where you can find the training and evaluation scripts for all the algorithms.

For example, to train the `bcql` method, simply run by overriding the default parameters:

```shell
python examples/train/train_bcql.py --param1 args1
```
All the parameters and their default configs for each algorithm are available in the `examples/configs` folder.