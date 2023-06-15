#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("osrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        "dsrl",
        "fast-safe-rl",
        "pyrallis==0.3.1",
        "pyyaml~=6.0",
        "scipy~=1.10.1",
        "tqdm",
        "numpy>1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch~=1.13.0",
        "numba>=0.51.0",
        "wandb~=0.14.0",
        "h5py>=2.10.0",
        "protobuf~=3.19.0",  # breaking change, sphinx fail
        "python-dateutil==2.8.2",
        "easy_runner",
        "swig==4.1.1",
    ]


def get_extras_require() -> str:
    req = {
        "dev": [
            "sphinx==6.2.1",
            "sphinx_rtd_theme==1.2.0",
            "jinja2==3.0.3",  # temporary fix
            "sphinxcontrib-bibtex==2.5.0",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest~=7.3.1",
            "pytest-cov~=4.0.0",
            "networkx",
            "mypy",
            "pydocstyle",
            "doc8==0.11.2",
            "scipy",
            "pre-commit",
        ]
    }
    return req


setup(
    name="osrl-lib",
    version=get_version(),
    description=
    "Elegant Implementations of Offline Safe Reinforcement Learning Algorithms",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liuzuxin/offline-safe-rl-baselines.git",
    author="Zijian Guo; Zuxin Liu",
    author_email="zuxin1997@gmail.com",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="offline safe reinforcement learning algorithms pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
