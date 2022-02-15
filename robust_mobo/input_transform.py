#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Input Transformations.

Modified from original BoTorch implementation.
"""

from typing import Optional, Union, Callable

import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor
from torch.nn import Module


class InputPerturbation(InputTransform, Module):
    r"""A transform that adds the set of perturbations to the given input.

    Similar to `AppendFeatures`, this can be used with `RiskMeasureMCObjective`
    to optimize risk measures. See `AppendFeatures` for additional discussion
    on optimizing risk measures.

    A tutorial notebook using this with `qNoisyExpectedImprovement` can be found at
    https://botorch.org/tutorials/risk_averse_bo_with_input_perturbations.
    """

    def __init__(
        self,
        perturbation_set: Union[Tensor, Callable[[Tensor], Tensor]],
        bounds: Optional[Tensor] = None,
        multiplicative: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Add `perturbation_set` to each input.

        Args:
            perturbation_set: An `n_p x d`-dim tensor denoting the perturbations
                to be added to the inputs. Alternatively, this can be a callable that
                returns `batch x n_p x d`-dim tensor of perturbations for input of
                shape `batch x d`. This is useful for heteroscedastic perturbations.
            bounds: A `2 x d`-dim tensor of lower and upper bounds for each
                column of the input. If given, the perturbed inputs will be
                clamped to these bounds.
            multiplicative: A boolean indicating whether the input perturbations
                are additive or multiplicative. If True, inputs will be multiplied
                with the perturbations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if isinstance(perturbation_set, Tensor):
            if perturbation_set.dim() != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self.register_buffer("perturbation_set", perturbation_set)
        else:
            self.perturbation_set = perturbation_set
        if bounds is not None:
            if (
                isinstance(perturbation_set, Tensor)
                and bounds.shape[-1] != perturbation_set.shape[-1]
            ):
                raise ValueError(
                    "`bounds` must have the same number of columns (last dimension) as "
                    f"the `perturbation_set`! Got {bounds.shape[-1]} and "
                    f"{perturbation_set.shape[-1]}."
                )
            self.register_buffer("bounds", bounds)
        else:
            self.bounds = None
        self.multiplicative = multiplicative
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by adding `perturbation_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_p x d`-dim tensor with the `perturbation_set` added to the input.
        For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_p) x d`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_w: (i + 1) * n_w, :]`.

        Note: Adding the `perturbation_set` on the `q-batch` dimension is necessary
        to avoid introducing additional bias by evaluating the inputs on independent
        GP sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_p) x d`-dim tensor of perturbed inputs.
        """
        if isinstance(self.perturbation_set, Tensor):
            perturbations = self.perturbation_set
        else:
            perturbations = self.perturbation_set(X)
        expanded_X = X.unsqueeze(dim=-2).expand(
            *X.shape[:-1], perturbations.shape[-2], -1
        )
        expanded_perturbations = perturbations.expand(*expanded_X.shape[:-1], -1)
        if self.multiplicative:
            perturbed_inputs = expanded_X * expanded_perturbations
        else:
            perturbed_inputs = expanded_X + expanded_perturbations
        perturbed_inputs = perturbed_inputs.reshape(*X.shape[:-2], -1, X.shape[-1])
        if self.bounds is not None:
            perturbed_inputs = torch.maximum(
                torch.minimum(perturbed_inputs, self.bounds[1]), self.bounds[0]
            )
        return perturbed_inputs
