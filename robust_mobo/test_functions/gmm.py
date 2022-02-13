#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
GMM test problem.

This is a multi-objective extension that is adapted from
the single objective version available at:
https://github.com/boschresearch/NoisyInputEntropySearch/blob/master/core/util/objectives.py
"""

import numpy as np
from typing import Optional
import torch
from botorch.exceptions import UnsupportedError
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)
from scipy import stats
from torch import Tensor


def gmm_2d(x, num_objectives: int = 2):
    if num_objectives not in (2, 3, 4):
        raise UnsupportedError("GMM only currently supports 2 to 4 objectives.")
    x = np.atleast_2d(x)
    gmm_pos = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]])
    gmm_var = np.array([0.20, 0.10, 0.10]) ** 2
    gmm_norm = 2 * np.pi * gmm_var * np.array([0.5, 0.7, 0.7])
    gaussians = [
        stats.multivariate_normal(mean=gmm_pos[i], cov=gmm_var[i])
        for i in range(gmm_var.shape[0])
    ]
    f1 = [gmm_norm[i] * g.pdf(x) for i, g in enumerate(gaussians)]
    f1 = np.atleast_1d(np.sum(np.asarray(f1), axis=0))[:, None]

    gmm_pos2 = np.array([[0.07, 0.2], [0.4, 0.8], [0.85, 0.1]])
    gmm_var2 = np.array([0.2, 0.1, 0.05]) ** 2
    gmm_norm2 = 2 * np.pi * gmm_var2 * np.array([0.5, 0.7, 0.7])
    gaussians = [
        stats.multivariate_normal(mean=gmm_pos2[i], cov=gmm_var2[i])
        for i in range(gmm_var2.shape[0])
    ]
    f2 = [gmm_norm2[i] * g.pdf(x) for i, g in enumerate(gaussians)]
    f2 = np.atleast_1d(np.sum(np.asarray(f2), axis=0))[:, None]
    objs = [f1, f2]
    if num_objectives > 2:
        gmm_pos3 = np.array([[0.08, 0.21], [0.45, 0.75], [0.86, 0.11]])
        gmm_var3 = np.array([0.2, 0.1, 0.07]) ** 2
        gmm_norm3 = 2 * np.pi * gmm_var3 * np.array([0.5, 0.7, 0.9])
        gaussians = [
            stats.multivariate_normal(mean=gmm_pos3[i], cov=gmm_var3[i])
            for i in range(gmm_var3.shape[0])
        ]
        f3 = [gmm_norm3[i] * g.pdf(x) for i, g in enumerate(gaussians)]
        f3 = np.atleast_1d(np.sum(np.asarray(f3), axis=0))[:, None]
        objs.append(f3)
    if num_objectives > 3:
        gmm_pos4 = np.array([[0.09, 0.19], [0.44, 0.72], [0.89, 0.13]])
        gmm_var4 = np.array([0.15, 0.07, 0.09]) ** 2
        gmm_norm4 = 2 * np.pi * gmm_var4 * np.array([0.5, 0.7, 0.9])
        gaussians = [
            stats.multivariate_normal(mean=gmm_pos4[i], cov=gmm_var4[i])
            for i in range(gmm_var4.shape[0])
        ]
        f4 = [gmm_norm4[i] * g.pdf(x) for i, g in enumerate(gaussians)]
        f4 = np.atleast_1d(np.sum(np.asarray(f4), axis=0))[:, None]
        objs.append(f4)

    return np.hstack(objs)


class GMM(MultiObjectiveTestProblem):
    r"""A two-output Gaussian mixture model.
    This has different optimal regions for robust and non-robust solutions.
    """
    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        num_objectives: int = 2,
    ) -> None:
        r"""Base constructor for multi-objective test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
            num_objectives: The number of objectives.
        """
        if num_objectives not in (2, 3, 4):
            raise UnsupportedError("GMM only currently supports 2 to 4 objectives.")
        self._ref_point = [0.2338, 0.2211]
        if num_objectives > 2:
            self._ref_point.append(0.5180)
        if num_objectives > 3:
            self._ref_point.append(0.1866)
        self.num_objectives = num_objectives
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (
            torch.from_numpy(
                gmm_2d(X.view(-1, 2).cpu().numpy(), num_objectives=self.num_objectives)
            )
            .to(X)
            .view(*X.shape[:-1], self.num_objectives)
        )
