#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch
from botorch.acquisition.multi_objective.analytic import (
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.acquisition.risk_measures import VaR
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils import t_batch_mode_transform
from botorch.utils.multi_objective import get_chebyshev_scalarization
from torch import Tensor


class ChVUCB(MultiObjectiveAnalyticAcquisitionFunction):
    r"""The upper confidence bound acquisition function for
    VaR of Chebyshev scalarizations.
    """

    def __init__(
        self,
        model: Union[SingleTaskGP, ModelListGP],
        weights: Tensor,
        var: VaR,
        t: int,
    ) -> None:
        r"""VaR UCB acquisition function supporting `m >= 2` objectives
        via Chebyshev scalarizations.

        Args:
            model: A SingleTaskGP model, modeling `m` outcomes independently.
                `model` must have an `InputPerturbation`.
            weights: An `m`-dim tensor denoting the weights to use for
                Chebyshev scalarization.
            var: A VaR module used to calculate the VaR of UCB of scalarizations.
                Must be compatible with `model`'s input perturbation.
            t: An integer denoting the iteration count.
        """
        super().__init__(model=model)
        if isinstance(model, ModelListGP):
            train_targets = torch.stack(model.train_targets, dim=-1)
        else:
            train_targets = model.train_targets.t()
        self.m = self.model.num_outputs
        self.betas = self.get_betas(torch.as_tensor(t).to(device=train_targets.device))
        self.var = var
        self.scalarization = get_chebyshev_scalarization(
            weights=weights, Y=train_targets, alpha=0
        )
        self.weights_sign = torch.sign(weights)

    def get_betas(self, t: Tensor) -> Tensor:
        r"""Get the betas corresponding to each of `m` outcomes.

        NOTE: Following Nguyen et al., 2021, we use beta = 2 log(t^2 \pi^2 / 0.6),
        which has nothing to do with the theoretical values.

        Args:
            t: A scalar tensor denoting the iteration number.

        Returns:
            An `m`-dim tensor of betas.
        """
        return (2 * torch.log(t.pow(2) * math.pi ** 2 / 0.6)).sqrt().expand(self.m)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Calculate VaR(u(s[f(X + \xi), w])).

        Args:
            X: A `batch_shape x 1 x d`-dim tensor of candidate solutions.

        Returns:
            A `batch_shape`-dim tensor of UCB values.
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean
        sigma = posterior.variance.sqrt()
        # If weights are negative, we want the LCB.
        individual_ucb = mean + self.weights_sign * self.betas * sigma
        scalarized_ucb = self.scalarization(individual_ucb, X)
        return self.var(scalarized_ucb.unsqueeze(-1)).squeeze(-1)
