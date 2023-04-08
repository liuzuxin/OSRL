#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("sdt", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        'gym>=0.26.0',
        'numpy',
        'pybullet>=3.0.6',
        'wandb~=0.13.6',
        'SWIG~=4.1.1',
        'OApackage~=2.7.6',
        'scipy~=1.9.3',
    ]


setup(
    name="sdt",
    version=get_version(),
    description="Decision Transformer for Offline Safe Reinforcement Learning.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    python_requires=">=3.6",
    keywords="safe reinforcement learning with decision transfomer",
    packages=find_packages(exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]),
    install_requires=get_install_requires(),
)
