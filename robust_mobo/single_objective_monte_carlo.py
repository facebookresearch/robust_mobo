#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling. See [Rezende2014reparam]_, [Wilson2017reparam]_ and
[Balandat2020botorch]_.

Modified from BoTorch implementations.

TODO: use upstream version when
https://github.com/pytorch/botorch/pull/1056 is merged.

References
.. [Rezende2014reparam]
    D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and
    approximate inference in deep generative models. ICML 2014.

.. [Wilson2017reparam]
    J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
    The reparameterization trick for acquisition functions. ArXiv 2017.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import MCSampler
from botorch.utils.low_rank import (
    extract_batch_covar,
    sample_cached_cholesky,
)
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.utils.errors import NotPSDError, NanError
import gpytorch.settings as gpt_settings
from torch import Tensor


class qNoisyExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Improvement using SAA with low-rank cholesky updates.

        TODO: similar to NEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental NEI from the new point. This would greatly increase
        efficiency for large batches.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        self.base_sampler = deepcopy(self.sampler)
        if not self.sampler.collapse_batch_dims:
            raise UnsupportedError(
                "qNoisyExpectedImprovement currently requires sampler "
                "to use `collapse_batch_dims=True`."
            )
        elif self.sampler.base_samples is not None:
            warnings.warn(
                message=(
                    "sampler.base_samples is not None. qNoisyExpectedImprovement "
                    "requires that the base_samples be initialized to None. "
                    "Resetting sampler.base_samples to None."
                ),
                category=BotorchWarning,
            )
            self.sampler.base_samples = None
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("X_baseline", X_baseline)
        models = model.models if isinstance(model, ModelListGP) else [model]
        self._is_mt = any(isinstance(m, MultiTaskGP) for m in models)
        self.q = -1
        # set baseline samples
        with torch.no_grad():
            posterior = self.model.posterior(X_baseline)
            baseline_samples = self.base_sampler(posterior)
        baseline_obj = self.objective(baseline_samples, X=X_baseline)
        self.register_buffer("baseline_samples", baseline_samples)
        self.register_buffer("baseline_obj_max_values", baseline_obj.max(dim=-1).values)
        self._cache_root_decomposition(posterior=posterior)

    def _cache_root_decomposition(
        self,
        posterior: Posterior,
    ) -> None:
        if isinstance(posterior.mvn, MultitaskMultivariateNormal):
            lazy_covar = extract_batch_covar(posterior.mvn)
        else:
            lazy_covar = posterior.mvn.lazy_covariance_matrix
        with gpt_settings.fast_computations.covar_root_decomposition(False):
            baseline_L = lazy_covar.root_decomposition().root.evaluate()
        self.register_buffer("_baseline_L", baseline_L)

    def _set_sampler(
        self,
        q: int,
        posterior: Posterior,
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        Args:
            q: the batch size
            posterior: the posterior

        TODO: refactor some/all of this into the MCSampler.
        """
        if self.q != q:
            # create new base_samples
            base_sample_shape = self.sampler._get_base_sample_shape(posterior=posterior)
            self.sampler._construct_base_samples(
                posterior=posterior, shape=base_sample_shape
            )
            if (
                self.X_baseline.shape[0] > 0
                and self.base_sampler.base_samples is not None
            ):
                current_base_samples = self.base_sampler.base_samples.detach().clone()
                view_shape = (
                    base_sample_shape[0:1]
                    + torch.Size([1] * (len(base_sample_shape) - 3))
                    + current_base_samples.shape[-2:]
                )
                expanded_shape = (
                    base_sample_shape[:-2] + current_base_samples.shape[-2:]
                )
                # Use stored base samples:
                # Use all base_samples from the current sampler
                # this includes the base_samples from the base_sampler
                # and any base_samples for the new points in the sampler.
                # For example, when using sequential greedy candidate generation
                # then generate the new candidate point using last (-1) base_sample
                # in sampler. This copies that base sample.
                end_idx = current_base_samples.shape[-2]
                self.sampler.base_samples[..., :end_idx, :] = current_base_samples.view(
                    view_shape
                ).expand(expanded_shape)
            self.q = q

    def _get_f_X_samples(self, posterior: GPyTorchPosterior, q: int) -> Tensor:
        r"""Get posterior samples at the `q` new points from the joint posterior.

        Args:
            posterior: The joint posterior is over (X_baseline, X).
            q: The number of new points in X.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of posterior
                samples at the new points.
        """
        # technically we should make sure that we add a consistent nugget to the cached covariance (and box decompositions)
        # and the new block.
        if not self._is_mt and hasattr(self, "_baseline_L"):
            try:
                return sample_cached_cholesky(
                    posterior=posterior,
                    baseline_L=self._baseline_L,
                    q=q,
                    base_samples=self.sampler.base_samples,
                    sample_shape=self.sampler.sample_shape,
                )
            except (NanError, NotPSDError):
                warnings.warn(
                    "Low-rank cholesky updates failed due NaNs or due to an ill-conditioned covariance matrix. "
                    "Falling back to standard sampling.",
                    BotorchWarning,
                )

        # TODO: improve efficiency for multi-task models
        samples = self.sampler(posterior)
        return samples[..., -q:, :]

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(X_full)
        self._set_sampler(q=q, posterior=posterior)
        n_w = posterior.event_shape[-2] // X_full.shape[-2]
        new_samples = self._get_f_X_samples(posterior=posterior, q=n_w * q)
        new_obj = self.objective(new_samples, X=X_full[..., -q:, :])
        new_obj_max_values = new_obj.max(dim=-1).values
        view_shape = torch.Size(
            self.baseline_obj_max_values.shape
            + torch.Size([1] * (new_obj_max_values.ndim - 1))
        )
        diffs = new_obj_max_values - self.baseline_obj_max_values.view(view_shape)
        return diffs.clamp_min(0).mean(dim=0)
