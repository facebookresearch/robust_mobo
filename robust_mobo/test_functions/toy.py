#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.test_functions import Levy
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor


class Toy(MultiObjectiveTestProblem):
    r"""A 1D toy problem constructed by combining the Levy function
    with a simple sinusoidal function.

    This function is intended for maximization.
    A reasonable MVaR ref point is [-14.1951,  -3.1887].
    """
    dim = 1
    _bounds = [(0.0, 0.7)]
    _ref_point = [6.1397, 8.1942]
    num_objectives = 2
    levy = Levy()

    def f_1(self, X: Tensor) -> Tensor:
        p1 = 2.4 - 10 * X - 0.1 * X.pow(2)
        p2 = 2 * X - 0.1 * X.pow(2)
        smoother = (X - 0.5).pow(2) + torch.sin(30 * X) * 0.1
        x_mask = torch.sigmoid((0.2 - X) / 0.005)
        return -(p1 * x_mask + p2 * (1 - x_mask) + smoother).view(-1) * 30 + 30

    def f_2(self, X: Tensor) -> Tensor:
        X = torch.cat(
            [X, torch.zeros_like(X)],
            dim=-1,
        )
        # Cut out the first part of the function.
        X = X * 0.95 + 0.03
        X = unnormalize(X, self.levy.bounds.to(X))
        Y = self.levy(X)
        Y -= X[..., 0].pow(2) * 0.75
        return Y.view(-1)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack([self.f_1(X), self.f_2(X)], dim=-1)
