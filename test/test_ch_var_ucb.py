#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from unittest import TestCase

import torch
from botorch.acquisition.risk_measures import VaR
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.transforms import normalize
from robust_mobo.ch_var_ucb import ChVUCB
from robust_mobo.input_transform import InputPerturbation


class TestChVUCB(TestCase):
    def test_chvucb(self):
        tkwargs = {"device": torch.device("cpu"), "dtype": torch.float}
        train_X = torch.rand(10, 2, **tkwargs)
        train_Y = torch.randn(10, 2, **tkwargs)
        weights = torch.tensor([0.5, -0.5], **tkwargs)
        perturbation = InputPerturbation(torch.zeros(1, 2, **tkwargs))
        var = VaR(n_w=1, alpha=0.5)
        model = SingleTaskGP(train_X, train_Y, input_transform=perturbation).eval()

        acqf = ChVUCB(model=model, weights=weights, var=var, t=1)
        expected_betas = torch.full(
            (2,), 2 * torch.log(torch.tensor(math.pi ** 2 / 0.6, **tkwargs))
        ).sqrt()
        self.assertTrue(torch.allclose(acqf.betas, expected_betas))
        ch_bounds = torch.stack(
            [train_Y.min(dim=-2).values, train_Y.max(dim=-2).values]
        )
        test_X = torch.rand(3, 1, 2, **tkwargs)
        posterior = model.posterior(test_X)
        pm, pstd = posterior.mean.squeeze(-2), posterior.variance.squeeze(-2).sqrt()
        ucb = pm + expected_betas * torch.tensor([1, -1], **tkwargs) * pstd
        normalized_ucb = normalize(ucb, ch_bounds)
        normalized_ucb[..., -1] -= 1
        scalarized_ucb = torch.min(normalized_ucb * weights, dim=-1).values

        v_ucb = acqf(test_X)
        self.assertTrue(torch.allclose(v_ucb, scalarized_ucb))

        # Test with actual perturbations
        perturbation = InputPerturbation(torch.rand(4, 2, **tkwargs) * 0.1)
        var = VaR(n_w=4, alpha=0.5)
        model = SingleTaskGP(train_X, train_Y, input_transform=perturbation).eval()
        acqf = ChVUCB(model=model, weights=weights, var=var, t=1)
        self.assertTrue(acqf(test_X).shape == torch.Size([3]))

        # Test with ModelListGP
        models = [
            SingleTaskGP(train_X, train_Y[..., i : i + 1], input_transform=perturbation)
            for i in range(train_Y.shape[-1])
        ]
        model = ModelListGP(*models).eval()
        acqf = ChVUCB(model=model, weights=weights, var=var, t=1)
        self.assertTrue(acqf(test_X).shape == torch.Size([3]))
