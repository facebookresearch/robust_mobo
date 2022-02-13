#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import torch
from robust_mobo.experiment_utils import get_perturbations
from robust_mobo.input_transform import InputPerturbation


class TestInputPerturbation(TestCase):
    def test_input_perturbation(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype}
            perturbations = get_perturbations(
                n_w=5,
                dim=2,
                tkwargs=tkwargs,
                bounds=torch.tensor([[0, 0], [1, 1.0]]),  # dummy
                method="heteroscedastic_linear_normal",
                scale=1.0,
            )
            inpt = InputPerturbation(perturbation_set=perturbations).eval()
            # Get the base samples.
            base_samples = perturbations(torch.ones(1, 2, **tkwargs)).squeeze()
            # Check that it is applied properly.
            # Adding 0.1 to avoid numerical issues in the check below.
            X = torch.rand(3, 10, 2, **tkwargs) + 0.1
            perturbed_X = inpt(X)
            repeated_X = X.repeat_interleave(5, dim=-2)
            scaled_diff = (perturbed_X - repeated_X) / repeated_X
            self.assertTrue(
                torch.allclose(base_samples.repeat(3, 10, 1), scaled_diff, atol=1e-3)
            )
