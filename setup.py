#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

requirements = [
    "torch",
    "gpytorch",
    "botorch>=0.6",
    "scipy",
    "jupyter",
    "matplotlib",
    "sklearn",
    "joblib",
]

dev_requires = [
    "black",
    "flake8",
    "pytest",
    "coverage",
]

setup(
    name="robust_mobo",
    version="0.1",
    description="Robust Multi-Objective Bayesian Optimization Under Input Noise",
    author="Anonymous Authors",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requires},
)
