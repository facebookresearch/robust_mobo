#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Disc Brake Problem.

TODO: Move this upstream.


References

.. [Tanabe2020]
    Ryoji Tanabe, Hisao Ishibuchi, An easy-to-use real-world multi-objective
    optimization problem suite, Applied Soft Computing,Volume 89, 2020.
"""

import torch
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from torch import Tensor


class DiscBrake(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The Disc Brake problem.

    There are 2 objectives and 4 constraints.

    Both objectives should be minimized.

    See [Tanabe2020]_ for details.

    A reasonable reference point for the MVaR frontier is
    [5.9988, 5.8384]
    """

    dim = 4
    num_objectives = 2
    num_constraints = 4
    _bounds = [(55.0, 80.0), (75.0, 110.0), (1000.0, 3000.0), (11.0, 20.0)]
    _ref_point = [5.7771, 3.9651]  # Taken from the notebook.

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.zeros(
            *X.shape[:-1], self.num_objectives, dtype=X.dtype, device=X.device
        )

        X1, X2, X3, X4 = torch.split(X, 1, -1)

        f[..., :1] = 4.9 * 1e-5 * (X2 * X2 - X1 * X1) * (X4 - 1.0)
        f[..., 1:] = ((9.82 * 1e6) * (X2 * X2 - X1 * X1)) / (
            X3 * X4 * (X2 * X2 * X2 - X1 * X1 * X1)
        )

        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        g = torch.zeros(
            *X.shape[:-1], self.num_constraints, dtype=X.dtype, device=X.device
        )

        X1, X2, X3, X4 = torch.split(X, 1, -1)
        g[..., :1] = (X2 - X1) - 20.0
        g[..., 1:2] = 0.4 - (X3 / (3.14 * (X2 * X2 - X1 * X1)))
        g[..., 2:3] = 1.0 - (
            2.22 * 1e-3 * X3 * (X2 * X2 * X2 - X1 * X1 * X1)
        ) / torch.pow((X2 * X2 - X1 * X1), 2)
        g[..., 3:] = (2.66 * 1e-2 * X3 * X4 * (X2 * X2 * X2 - X1 * X1 * X1)) / (
            X2 * X2 - X1 * X1
        ) - 900.0
        return g
