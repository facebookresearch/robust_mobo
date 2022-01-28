from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional

import torch
from botorch import settings
from botorch.acquisition import monte_carlo  # noqa F401
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import MCAcquisitionObjective, GenericMCObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.models.model import Model
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils import apply_constraints
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize_indices, normalize
from torch import Tensor
from torch.quasirandom import SobolEngine

MAX_BYTES = 5e6  # 5 MB


class GenericMCMultiOutputObjective(GenericMCObjective, MCMultiOutputObjective):
    pass


def prune_inferior_points_multi_objective(
    model: Model,
    X: Tensor,
    ref_point: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    min_prob: float = 0.0,
    marginalize_dim: Optional[int] = None,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be pareto optimal.

    Given a model, an objective, and an input tensor `X`, this function returns
    the subset of points in `X` that have some probability of being pareto
    optimal, better than the reference point, and feasible. This function uses
    sampling to estimate the probabilities, the higher the number of points `n`
    in `X` the higher the number of samples `num_samples` should be to obtain
    accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape `n x d`. Batched inputs are currently not
            supported.
        ref_point: The reference point.
        objective: The objective under which to evaluate the posterior.
        constraints: A list of callables, each mapping a Tensor of dimension
            `sample_shape x batch-shape x q x m` to a Tensor of dimension
            `sample_shape x batch-shape x q`, where negative values imply
            feasibility.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being pareto optimal.
    """
    if constraints is not None:
        raise NotImplementedError
    if X.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by "
            "prune_inferior_points_multi_objective"
        )
    max_points = math.ceil(max_frac * X.size(-2))
    if max_points < 1 or max_points > X.size(-2):
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")

    with torch.no_grad():
        posterior = model.posterior(X=X)
    test_sample = posterior.rsample()
    obj_vals = objective(test_sample, X=X)
    n_w = obj_vals.shape[-2] // X.shape[-2]
    if obj_vals.ndim > 3:
        if obj_vals.ndim == 4 and marginalize_dim is not None:
            obj_vals = obj_vals.mean(dim=marginalize_dim)
        else:
            # TODO: support batched inputs (req. dealing with ragged tensors)
            raise UnsupportedError(
                "Models with multiple batch dims are currently unsupported by"
                " prune_inferior_points_multi_objective."
            )
    el_size = 64 if X.dtype == torch.double else 32
    # set mini-batch size to generate tensors no larger than 5MB
    mini_batch_size = min(
        num_samples, math.ceil(MAX_BYTES / (obj_vals.numel() * el_size / 8))
    )
    if posterior.event_shape.numel() > SobolEngine.MAXDIM:
        if settings.debug.on():
            warnings.warn(
                f"Sample dimension q*m={posterior.event_shape.numel()} exceeding Sobol "
                f"max dimension ({SobolEngine.MAXDIM}). Using iid samples instead.",
                SamplingWarning,
            )
        sampler = IIDNormalSampler(num_samples=mini_batch_size, resample=True)
    else:
        sampler = SobolQMCNormalSampler(num_samples=mini_batch_size, resample=True)
    if objective is None:
        objective = IdentityMCMultiOutputObjective()
    all_counts = torch.zeros(X.shape[0], dtype=torch.int, device=X.device)
    total_samples = 0
    while total_samples < num_samples:
        if mini_batch_size > num_samples - total_samples:
            mini_batch_size = num_samples - total_samples
            sampler = sampler.__class__(num_samples=mini_batch_size, resample=True)
        samples = sampler(posterior)
        obj_vals = objective(samples, X=X)
        pareto_mask = is_non_dominated(obj_vals, deduplicate=False) & (
            obj_vals > ref_point
        ).all(dim=-1)
        pareto_mask = pareto_mask.view(*pareto_mask.shape[:-1], -1, n_w).any(dim=-1)
        all_counts += pareto_mask.to(dtype=all_counts.dtype).sum(dim=0)
        total_samples += mini_batch_size
        # I observe reserved mem going up while allocated is roughly constant.
        # This helps.
        torch.cuda.empty_cache()
    if min_prob > 0:
        probs = all_counts / num_samples
        idcs = (probs > min_prob).nonzero().view(-1)
    else:
        idcs = all_counts.nonzero().view(-1)
    if idcs.shape[0] > max_points:
        idcs = torch.topk(all_counts, k=max_points, largest=True).indices.view(-1)
    return X[idcs]


def get_Y_normalization_bounds(
    Y: Tensor,
    ref_point: Optional[Tensor] = None,
) -> Tensor:
    r"""Get normalization bounds for scalarizations.

    Args:
        Y: A `n x m`-dim tensor of outcomes
        ref_point: The reference point.

    Returns:
        A `2 x m`-dim tensor containing the normalization bounds.
    """
    if Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    if Y.shape[-2] == 0:
        # If there are no observations, we do not need to normalize the objectives
        Y_bounds = torch.zeros(2, Y.shape[-1], dtype=Y.dtype, device=Y.device)
        Y_bounds[1] = 1
    pareto_Y = Y[is_non_dominated(Y)]
    if pareto_Y.shape[-2] == 1:
        if ref_point is not None and (pareto_Y > ref_point).all():
            Y_bounds = torch.cat([ref_point.unsqueeze(0), pareto_Y], dim=0)
        else:
            # If there is only one observation, set the bounds to be
            # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
            # divide by zero
            Y_bounds = torch.cat([pareto_Y, pareto_Y + 1], dim=0)
    else:
        if ref_point is None:
            better_than_ref = torch.ones(
                pareto_Y.shape[0], device=pareto_Y.device, dtype=torch.long
            )
        else:
            better_than_ref = (pareto_Y > ref_point).all(dim=-1)
        if ref_point is not None and better_than_ref.any():
            nadir = ref_point
            pareto_Y = pareto_Y[better_than_ref]
        else:
            nadir = pareto_Y.min(dim=-2).values
        ideal = pareto_Y.max(dim=-2).values
        Y_bounds = torch.stack([nadir, ideal])
    return Y_bounds


def get_chebyshev_scalarization(
    weights: Tensor,
    Y: Tensor,
    alpha: float = 0.05,
    ref_point: Optional[Tensor] = None,
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""Construct an augmented Chebyshev scalarization.

    Outcomes are first normalized to [0,1] and then an augmented
    Chebyshev scalarization is applied.

    Augmented Chebyshev scalarization:
        objective(y) = min(w * y) + alpha * sum(w * y)

    Note: this assumes maximization.

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim tensor of weights.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.
        ref_point: A `m`-dim tensor containing the reference value for each
            objective.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, 0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    Y_bounds = get_Y_normalization_bounds(Y=Y, ref_point=ref_point)

    if weights.shape[0] == 1:
        einsum_str = "...m,m->...m"
        weights = weights.squeeze(0)
    else:
        einsum_str = "...m,nm->n...m"
    if ref_point is not None:

        ref_point = normalize(ref_point.unsqueeze(0), bounds=Y_bounds).squeeze(0)

        def chebyshev_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
            Y_obj = normalize(Y, bounds=Y_bounds)
            product = torch.einsum(einsum_str, Y_obj - ref_point, weights)
            return product.min(dim=-1).values + alpha * product.sum(dim=-1)

    else:

        def chebyshev_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
            Y_obj = normalize(Y, bounds=Y_bounds)
            product = torch.einsum(einsum_str, Y_obj, weights)
            return product.min(dim=-1).values + alpha * product.sum(dim=-1)

    return chebyshev_obj


def get_constraint_indexer(i: int) -> Callable[[Tensor], Tensor]:
    def idxr(Y):
        # negative constraint slack implies feasibility in botorch's
        # sigmoid approximation of the feasibility indicator
        return -Y[..., i]

    return idxr


class FeasibilityWeightedMCMultiOutputObjective(MCMultiOutputObjective):
    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        num_constraints: int,
        objective: Optional[MCMultiOutputObjective] = None,
    ) -> None:
        super().__init__()
        if num_constraints > 0:
            num_objectives = model.num_outputs - num_constraints
            inf_cost = get_infeasible_cost(
                X=X_baseline, model=model, objective=lambda y: y
            )[:num_objectives]

            def apply_feasibility_weights(
                Y: Tensor, X: Optional[Tensor] = None
            ) -> Tensor:
                return apply_constraints(
                    obj=Y[..., :num_objectives],
                    constraints=[
                        get_constraint_indexer(i=i)
                        for i in range(num_objectives, model.num_outputs)
                    ],
                    samples=Y,
                    # this ensures that the dtype/device is set properly
                    infeasible_cost=inf_cost.to(Y),
                )

            self.apply_feasibility_weights = apply_feasibility_weights
        else:
            self.apply_feasibility_weights = lambda Y: Y
        if objective is None:
            objective = lambda Y, X: Y
        self.objective = objective

    def forward(self, Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return self.objective(self.apply_feasibility_weights(Y), X=X)


def get_objective_after_feasibility_weighting(
    model: Model,
    X_baseline: Tensor,
    num_constraints: int,
    objective: Optional[MCMultiOutputObjective] = None,
) -> Optional[MCMultiOutputObjective]:
    r"""Returns feasibility weighted objective.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        num_constraints: The number of constraints. Constraints are handled by
            weighting the samples according to a sigmoid approximation of feasibility.
            If objective is given, it is applied after weighting.
        objective: This is the optional objective for optimizing independent
            risk measures, such as expectation.

    Returns:
        A multi output objective that applies feasibility weighting before calculating
        the objective value.
    """
    return FeasibilityWeightedMCMultiOutputObjective(
        model=model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        objective=objective,
    )


def get_infeasible_cost(
    X: Tensor,
    model: Model,
    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
) -> Tensor:
    r"""Get infeasible cost for a model and objective.

    Computes an infeasible cost `M` such that `-M < min_x f(x)` almost always,
        so that feasible points are preferred.

    Args:
        X: A `n x d` Tensor of `n` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted botorch model.
        objective: The objective with which to evaluate the model output.

    Returns:
        A `m`-dim tensor containing the infeasible cost `M` value.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> objective = lambda Y: Y[..., -1] ** 2
        >>> M = get_infeasible_cost(train_X, model, obj)
    """
    if objective is None:

        def objective(Y: Tensor, X: Optional[Tensor] = None):
            return Y.squeeze(-1)

    with torch.no_grad():
        posterior = model.posterior(X)
    mean = posterior.mean
    lb = objective(mean - 6 * posterior.variance.clamp_min(0).sqrt())
    if lb.ndim < mean.ndim:
        lb = lb.unsqueeze(-1)
    lb = lb.min(dim=-2).values  # take component-wise min
    return -(lb.clamp_max(0.0))
