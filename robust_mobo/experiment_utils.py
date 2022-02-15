#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for experiments.
"""

from typing import Callable, Dict, Optional, Tuple, Union, List

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
)
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    IndependentVaR,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    MCAcquisitionObjective,
)
from botorch.acquisition.risk_measures import VaR
from botorch.exceptions.errors import UnsupportedError
from botorch.models import FixedNoiseGP, SingleTaskGP, ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import ConstrainedBraninCurrin, WeldedBeam
from botorch.test_functions.multi_objective import DiscBrake, GMM, Penicillin, ToyRobust
from botorch.utils import apply_constraints
from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions import (
    DominatedPartitioning,
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import (
    draw_sobol_normal_samples,
    draw_sobol_samples,
    sample_simplex,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from robust_mobo.ch_var_ucb import ChVUCB
from robust_mobo.constraint_active_search import ExpectedCoverageImprovement
from robust_mobo.input_transform import InputPerturbation
from robust_mobo.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
)
from robust_mobo.multi_objective_risk_measures import MultiOutputExpectation, MVaR
from robust_mobo.rffs import get_gp_sample_w_transforms
from torch import Tensor
from torch.nn import Module
from robust_mobo.single_objective_monte_carlo import qNoisyExpectedImprovement
from robust_mobo.utils import (
    get_chebyshev_scalarization,
    get_objective_after_feasibility_weighting,
    get_infeasible_cost,
    get_constraint_indexer,
)


def generate_initial_data(
    n: int,
    eval_problem: Callable[[Tensor], Tensor],
    bounds: Tensor,
    tkwargs: dict,
) -> Tuple[Tensor, Tensor]:
    r"""
    Generates the initial data for the experiments.

    Args:
        n: Number of training points.
        eval_problem: The callable used to evaluate the objective function.
        bounds: The bounds to generate the training points from. `2 x d`-dim tensor.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The train_X and train_Y. `n x d` and `n x m`.
    """
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    train_obj = eval_problem(train_x)
    return train_x, train_obj


def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
    use_model_list: bool = True,
    use_fixed_noise: bool = True,
) -> Tuple[
    Union[ExactMarginalLogLikelihood, SumMarginalLogLikelihood],
    Union[FixedNoiseGP, SingleTaskGP, ModelListGP],
]:
    r"""Constructs the model and its MLL.

    Args:
        train_x: An `n x d`-dim tensor of training inputs.
        train_y: An `n x m`-dim tensor of training outcomes.
        use_model_list: If True, returns a ModelListGP with models for each outcome.
        use_fixed_noise: If True, assumes noise-free outcomes and uses FixedNoiseGP.

    Returns:
        The MLL and the model. Note: the model is not trained!
    """
    base_model_class = FixedNoiseGP if use_fixed_noise else SingleTaskGP
    # define models for objective and constraint
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_y, 1e-7) * train_y.std(dim=0).pow(2)
    if use_model_list:
        model_kwargs = []
        for i in range(train_y.shape[-1]):
            model_kwargs.append(
                {
                    "train_X": train_x,
                    "train_Y": train_y[..., i : i + 1],
                    "outcome_transform": Standardize(m=1),
                }
            )
            if use_fixed_noise:
                model_kwargs[i]["train_Yvar"] = train_Yvar[..., i : i + 1]
        models = [base_model_class(**model_kwargs[i]) for i in range(train_y.shape[-1])]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    else:
        model_kwargs = {
            "train_X": train_x,
            "train_Y": train_y,
            "outcome_transform": Standardize(m=train_y.shape[-1]),
        }
        if use_fixed_noise:
            model_kwargs["train_Yvar"] = train_Yvar
        model = base_model_class(**model_kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model


def get_ch_var_UCB(
    model: GPyTorchModel,
    var: VaR,
    iteration: int,
    tkwargs,
) -> ChVUCB:
    r"""Construct the UCB acquisition function with VaR of Chebyshev scalarizations.

    Args:
        model: A fitted multi-output GPyTorchModel.
        var: The VaR object to use.
        iteration: The iteration number, used to get the beta for UCB.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The ChVUCB acquisition function.
    """
    weights = sample_simplex(d=model.num_outputs, n=1, **tkwargs).view(-1)
    acq_func = ChVUCB(
        model=model,
        weights=weights,
        var=var,
        t=max(iteration, 1),
    )
    return acq_func


def get_chebyshev_objective(
    model: Model,
    X_baseline: Tensor,
    num_constraints: int,
    alpha: float,
    pre_output_transform: Optional[MCAcquisitionObjective] = None,
    n_scalarizations: int = 1,
    ref_point: Optional[Tensor] = None,
    mvar: Optional[MVaR] = None,
) -> Union[Callable[[Tensor, Optional[Tensor]], Tensor], ConstrainedMCObjective]:
    r"""Create the (constrained) MC objective.

    Args:
        model: The model.
        X_baseline: A `n x d`-dim tensor of baseline points.
        num_constraints: The number of constraints.
        alpha: The mixing parameter for the augmented chebyshev scalarization.
        pre_output_transform: An output transformation to apply before applying
            the scalarization. For example, using independent VaR, we need to apply
            VaR to each output before scalarizing.
        n_scalarizations: The number of scalarizations to sample.
        ref_point: The reference point.
        mvar: An optional MVaR objective used for determining normalization bounds.
    """
    num_objectives = model.num_outputs - num_constraints
    weights = sample_simplex(
        d=num_objectives,
        n=n_scalarizations,
        dtype=X_baseline.dtype,
        device=X_baseline.device,
    )
    # Use posterior mean over perturbed X_baseline to estimate
    # the range of each objective.
    with torch.no_grad():
        Y = model.posterior(X_baseline).mean
    if pre_output_transform is not None or mvar is not None:
        assert not (pre_output_transform is not None and mvar is not None)
        if num_constraints > 0:
            # Calculate the scalarization bounds over feasibility weighted objectives.
            inf_cost = get_infeasible_cost(
                X=X_baseline, model=model, objective=lambda y: y
            )[:num_objectives]
            Y = apply_constraints(
                obj=Y[..., :num_objectives],
                constraints=[
                    get_constraint_indexer(i=i)
                    for i in range(num_objectives, model.num_outputs)
                ],
                samples=Y,
                infeasible_cost=inf_cost,
            )
        if pre_output_transform is not None:
            Y = pre_output_transform(Y)
        elif mvar is not None:
            Y = mvar(Y)
    # If model has an InputPerturbation input transform then the posterior
    # is a batched posterior.
    Y_obj = Y[..., :num_objectives].view(-1, num_objectives)
    # Handle the case where all values of an objective have the same value.
    # This can happen if feasiblity weighting is applied and there are no
    # feasible objectives.
    Y_range = Y_obj.max(dim=0).values - Y_obj.min(dim=0).values
    mask = Y_range <= 0
    Y_obj[-1, mask] = Y_obj[-1, mask] + 1
    scalarization = get_chebyshev_scalarization(
        weights=weights, Y=Y_obj, alpha=alpha, ref_point=ref_point
    )
    if pre_output_transform is None:
        if num_constraints == 0:
            return scalarization
        else:
            scalar_obj = lambda Y: scalarization(Y[..., :num_objectives])
            inf_cost = get_infeasible_cost(
                X=X_baseline, model=model, objective=scalar_obj
            )
            return ConstrainedMCObjective(
                objective=scalar_obj,
                constraints=[
                    get_constraint_indexer(i=i)
                    for i in range(num_objectives, model.num_outputs)
                ],
                infeasible_cost=inf_cost,
            )
    else:
        if num_constraints == 0:
            return lambda Y: scalarization(pre_output_transform(Y))
        else:
            inf_cost = get_infeasible_cost(
                X=X_baseline, model=model, objective=lambda y: y
            )[:num_objectives]

            def constrain_transform_scalarize(Y: Tensor) -> Tensor:
                constrained_Y = apply_constraints(
                    obj=Y[..., :num_objectives],
                    constraints=[
                        get_constraint_indexer(i=i)
                        for i in range(num_objectives, model.num_outputs)
                    ],
                    samples=Y,
                    infeasible_cost=inf_cost,
                )
                return scalarization(pre_output_transform(constrained_Y))

            return constrain_transform_scalarize


def get_ch_var_NEI(
    model: GPyTorchModel,
    var: VaR,
    X_baseline: Tensor,
    sampler: MCSampler,
    num_constraints: int,
    mvar_ref_point: Tensor,
    augmented: bool = False,
    ref_aware: bool = False,
) -> qNoisyExpectedImprovement:
    r"""Construct the NEI acquisition function with VaR of Chebyshev scalarizations.

    Args:
        model: A fitted multi-output GPyTorchModel.
        var: The VaR object to use.
        X_baseline: An `r x d`-dim tensor of points already observed.
        sampler: The sampler used to draw the base samples.
        num_constraints: The number of constraints.
        mvar_ref_point: The mvar reference point.
        augmented: If True, use augmented Chebyshev scalarization.
        ref_aware: If True, use reference point aware bounds for normalization.

    Returns:
        The NEI acquisition function.
    """
    if ref_aware:
        mvar = MVaR(n_w=var.n_w, alpha=var.alpha)
    else:
        mvar = None
        mvar_ref_point = None

    mc_obj = get_chebyshev_objective(
        model=model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        alpha=0.05 if augmented else 0.0,
        ref_point=mvar_ref_point,
        mvar=mvar,
    )

    def f_objective(Y, X):
        return var(mc_obj(Y).unsqueeze(-1))

    objective = GenericMCObjective(f_objective)
    acq_func = qNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        objective=objective,
        prune_baseline=True,
        sampler=sampler,
    )
    return acq_func


def get_NParEGO(
    model: GPyTorchModel,
    X_baseline: Tensor,
    sampler: MCSampler,
    num_constraints: int,
    pre_output_transform: Optional[MCAcquisitionObjective] = None,
) -> qNoisyExpectedImprovement:
    r"""Construct the NEI acquisition function with Chebyshev scalarizations.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        sampler: The sampler used to draw the base samples.
        num_constraints: The number of constraints.
        pre_output_transform: An output transformation to apply before applying
            the scalarization or constraints. For example, using indepedent VaR,
            we need to apply VaR to each output before scalarizing or evaluating
            constraints.

    Returns:
        The NEI acquisition function.
    """
    mc_obj = get_chebyshev_objective(
        model=model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        alpha=0.05,
        pre_output_transform=pre_output_transform,
    )
    objective = GenericMCObjective(mc_obj)
    acq_func = qNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        objective=objective,
        prune_baseline=True,
        sampler=sampler,
    )
    return acq_func


def get_nehvi_ref_point(
    model: GPyTorchModel,
    X_baseline: Tensor,
    objective: MCMultiOutputObjective,
) -> Tensor:
    r"""Estimate the reference point for NEHVI using the model posterior on
    `X_baseline` and the `infer_reference_point` objective. This applies the
    feasibility weighted objective to the posterior mean, then uses the
    heuristic.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        objective: The feasibility weighted MC objective.

    Returns:
        A `num_objectives`-dim tensor representing the reference point.
    """
    with torch.no_grad():
        post_mean = model.posterior(X_baseline).mean
    if objective is not None:
        obj = objective(post_mean)
    else:
        obj = post_mean
    return infer_reference_point(obj)


def get_nehvi(
    model: GPyTorchModel,
    X_baseline: Tensor,
    sampler: MCSampler,
    num_constraints: int,
    use_rff: bool,
    ref_point: Tensor,
    ref_aware: bool = False,
    num_rff_features: int = 512,
    objective: Optional[MCMultiOutputObjective] = None,
) -> Union[qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement]:
    r"""Construct the NEHVI acquisition function.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        sampler: The sampler used to draw the base samples.
        num_constraints: The number of constraints. Constraints are handled by
            weighting the samples according to a sigmoid approximation of feasibility.
            If objective is given, it is applied after weighting.
        use_rff: If True, replace the model with a single RFF sample. NEHVI-1.
        ref_point: The reference point.
        ref_aware: If True, use the given ref point. Otherwise, approximate it.
        num_rff_features: The number of RFF features.
        objective: This is the optional objective for optimizing independent
            risk measures, such as expectation.

    Returns:
        The qNEHVI acquisition function.
    """
    if use_rff:
        assert not isinstance(model, ModelListGP)
        model = get_gp_sample_w_transforms(
            model=model,
            num_outputs=model.num_outputs,
            n_samples=1,
            num_rff_features=num_rff_features,
        )
    objective = get_objective_after_feasibility_weighting(
        model=model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        objective=objective,
    )
    if ref_point is None or not ref_aware:
        ref_point = get_nehvi_ref_point(
            model=model, X_baseline=X_baseline, objective=objective
        )
    if use_rff:
        y_baseline = model.posterior(X_baseline).mean
        if objective is not None:
            y_baseline = objective(y_baseline)
        partitioning = FastNondominatedPartitioning(
            Y=y_baseline,
            ref_point=ref_point,
        )
        return qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
        )
    else:
        return qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )


def get_perturbations(
    n_w: int, dim: int, tkwargs: dict, bounds: Tensor, method: str, **kwargs
) -> Union[Tensor, Callable[[Tensor], Tensor]]:
    r"""Generate an `n_w x dim`-dim tensor of input perturbations.

    Args:
        n_w: Number of perturbations to generate.
        dim: The dimension of the perturbations.
        tkwargs: Arguments for tensors, dtype and device.
        bounds: A `2 x d`-dim tensor containing the bounds of the
            search space. The standard deviation or range of the
            perturbations will be specified in the raw search space,
            but internally we operate [0,1]^d and only unnormalize
            when we evaluate the objective. Hence, we need to scale
            perturbations from the original domain to the unit cube.
        method: Specifies the method used to generate the perturbation.
        **kwargs: Additional arguments as needed depending on the type.

    Returns:
        An `n_w x dim`-dim tensor of perturbations.
        Alternatively, this may return a Callable that takes an input X and
        produces perturbations based the given X using fixed base samples.
        Base samples should be produced here and kept fixed for SAA type use.
        Given a `batch x dim`-dim `X`, the Callable should have a return
        shape of `batch x n_w x dim`.
    """
    search_space_range = bounds[1] - bounds[0]
    if "mean" in kwargs:
        mean = kwargs["mean"]
        scaled_mean = mean / search_space_range
    elif "scaled_mean" in kwargs:
        scaled_mean = kwargs["scaled_mean"]
    else:
        scaled_mean = torch.zeros_like(search_space_range)
    if method == "sobol-normal":
        std_dev = kwargs.get("std_dev")
        if std_dev is None:
            raise ValueError(f"std_dev is required for {method} perturbations.")
        scaled_std_dev = (
            torch.tensor(std_dev, dtype=bounds.dtype, device=bounds.device)
            / search_space_range
        )
        perturbations = (
            scaled_mean
            + (draw_sobol_normal_samples(d=dim, n=n_w, **tkwargs)) * scaled_std_dev
        )

    elif method == "sobol-mvn":
        covar = kwargs.get("covariance_matrix")
        if covar is None:
            raise ValueError(
                f"covariance_matrix is required for {method} perturbations."
            )
        scaled_covar = torch.tensor(covar, dtype=bounds.dtype, device=bounds.device) / (
            search_space_range.unsqueeze(0) * search_space_range.unsqueeze(1)
        )
        base_samples = draw_sobol_normal_samples(d=dim, n=n_w, **tkwargs)
        mvn = MultivariateNormal(mean=scaled_mean, covariance_matrix=scaled_covar)
        perturbations = mvn.sample(
            sample_shape=torch.Size([n_w]),
            base_samples=base_samples,
        )

    elif method == "uniform":
        delta = kwargs.get("delta")
        if delta is None:
            raise ValueError(f"delta is required for {method} perturbations.")
        std_bounds = torch.zeros_like(bounds)
        std_bounds[1] = 1
        scaled_delta = (
            torch.tensor(delta, dtype=bounds.dtype, device=bounds.device)
            / search_space_range
        )
        perturbations = scaled_mean + (
            draw_sobol_samples(bounds=std_bounds, n=n_w, q=1).squeeze(1) - 0.5
        ) * (2 * scaled_delta)
    elif method == "heteroscedastic_linear_normal":
        # This produces sobol-normal perturbations where variance scales with X.
        base_samples = draw_sobol_normal_samples(d=dim, n=n_w, **tkwargs) * kwargs.get(
            "scale"
        )

        def perturbations(X: Tensor) -> Tensor:
            return scaled_mean + (
                kwargs.get("constant_offset", 0) + X.unsqueeze(-2)
            ) * base_samples.to(X)

    elif method == "heteroscedastic_sigmoid_1d":
        # This produces sobol-normal perturbations where variance increases as
        # X[..., 0] decreases according to a sigmoid function
        base_samples = draw_sobol_normal_samples(d=dim, n=n_w, **tkwargs) * kwargs.get(
            "scale"
        )

        def perturbations(X: Tensor) -> Tensor:
            return scaled_mean + (
                1 + torch.sigmoid((0.5 - X[..., :1].unsqueeze(-2)) / 0.5)
            ) * base_samples.to(X)

    else:
        raise ValueError(f"Unknown method: {method}!")
    return perturbations


def get_TS(
    model: GPyTorchModel,
    X_baseline: Tensor,
    num_constraints: int,
    num_rff_features: int,
    pre_output_transform: Optional[MCAcquisitionObjective] = None,
) -> qSimpleRegret:
    r"""Construct the TS acquisition function with Chebyshev scalarizations.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        num_constraints: The number of constraints.
        num_rff_features: The number of RFF features.
        pre_output_transform: An output transformation to apply before applying
            the scalarization or constraints. For example, using indepedent VaR,
            we need to apply VaR to each output before scalarizing or evaluating
            constraints.

    Returns:
        The TS acquisition function.
    """
    sampled_model = get_gp_sample_w_transforms(
        model=model,
        num_outputs=model.num_outputs,
        n_samples=1,
        num_rff_features=num_rff_features,
    )
    mc_obj = get_chebyshev_objective(
        model=sampled_model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        alpha=0.05,
        pre_output_transform=pre_output_transform,
    )
    objective = GenericMCObjective(mc_obj)
    acq_func = qSimpleRegret(
        model=sampled_model,
        sampler=SobolQMCNormalSampler(1),  # dummy sampler
        objective=objective,
    )
    # register X_baseline for sample around best initialization heuristic
    acq_func.register_buffer("X_baseline", X_baseline)
    return acq_func


def get_ch_var_TS(
    model: GPyTorchModel,
    var: VaR,
    X_baseline: Tensor,
    num_constraints: int,
    num_rff_features: int,
    mvar_ref_point: Tensor,
    ref_aware: bool = False,
) -> qSimpleRegret:
    r"""Construct the TS acquisition function with VaR of Chebyshev scalarizations.

    Args:
        model: A fitted multi-output GPyTorchModel.
        var: The VaR object to use.
        X_baseline: An `r x d`-dim tensor of points already observed.
        num_constraints: The number of constraints.
        num_rff_features: The number of RFF features.
        mvar_ref_point: The mvar reference point.
        ref_aware: If True, use reference point aware bounds for normalization.

    Returns:
        The TS acquisition function.
    """
    # get a deterministic model using RFF
    sampled_model = get_gp_sample_w_transforms(
        model=model,
        num_outputs=model.num_outputs,
        n_samples=1,
        num_rff_features=num_rff_features,
    )

    if ref_aware:
        mvar = MVaR(n_w=var.n_w, alpha=var.alpha)
    else:
        mvar = None
        mvar_ref_point = None

    mc_obj = get_chebyshev_objective(
        model=sampled_model,
        X_baseline=X_baseline,
        num_constraints=num_constraints,
        alpha=0.0,
        ref_point=mvar_ref_point,
        mvar=mvar,
    )

    def f_objective(Y, X):
        return var(mc_obj(Y).unsqueeze(-1))

    objective = GenericMCObjective(f_objective)
    acq_func = qSimpleRegret(
        model=sampled_model,
        sampler=SobolQMCNormalSampler(1),  # dummy sampler
        objective=objective,
    )
    # register X_baseline for sample around best initialization heuristic
    acq_func.register_buffer("X_baseline", X_baseline)
    return acq_func


def get_cas(
    model: ModelListGP,
    mvar_ref_point: Tensor,
    punchout_radius: float,
    tkwargs: dict,
    num_samples: int = 512,
) -> ExpectedCoverageImprovement:
    r"""Construct the expected improvement acquisition function.
    The goal is to cover the space dominating the MVaR ref point
    and satisfying the outcome constraints.

    Args:
        model: A ModelListGP modeling the outcomes and the constraints.
        mvar_ref_point: The MVaR ref point.
        punchout_radius: Positive value defining the desired minimum
            distance between points.
        tkwargs: Tensor arguments.
        num_samples: Number of samples for MC integration

    Returns:
        An ExpectedCoverageImprovement acquisition function.
    """
    assert isinstance(model, ModelListGP)
    input_dim = model.models[0].train_inputs[0].shape[-1]
    standard_bounds = torch.zeros(2, input_dim, **tkwargs)
    standard_bounds[1] = 1.0
    constraints = []
    for i, ref in enumerate(mvar_ref_point):
        constraints.append(("gt", ref))
    for i in range(len(mvar_ref_point), model.num_outputs):
        constraints.append(("gt", 0.0))
    return ExpectedCoverageImprovement(
        model=model,
        constraints=constraints,
        punchout_radius=punchout_radius,
        bounds=standard_bounds,
        tkwargs=tkwargs,
        num_samples=num_samples,
    )


def get_acqf(
    label: str,
    mc_samples: int,
    model: GPyTorchModel,
    perturbation_set: Union[Tensor, Callable[[Tensor], Tensor]],
    var: VaR,
    X_baseline: Tensor,
    iteration: int,
    tkwargs: dict,
    num_constraints: int,
    mvar_ref_point: Tensor,
    batch_size: int = 1,
    input_tf_kwargs: Optional[Dict[str, bool]] = None,
    **kwargs,
) -> Union[AcquisitionFunction, List[AcquisitionFunction]]:
    r"""Combines a few of the above utils to construct the acqf."""
    input_tf_kwargs = input_tf_kwargs or {}
    if batch_size > 1:
        if "nehvi" not in label:
            # Return a list of acqfs, each with a different scalarization.
            # The ones with "nehvi" naturally support batch evaluation.
            return [
                get_acqf(
                    label=label,
                    mc_samples=mc_samples,
                    model=model,
                    perturbation_set=perturbation_set,
                    var=var,
                    X_baseline=X_baseline,
                    iteration=iteration,
                    tkwargs=tkwargs,
                    num_constraints=num_constraints,
                    mvar_ref_point=mvar_ref_point,
                    batch_size=1,
                    input_tf_kwargs=input_tf_kwargs,
                    **kwargs,
                )
                for _ in range(batch_size)
            ]
    ref_aware = "ref" in label
    if "rff" in label:
        mc_samples = 1
    sampler = SobolQMCNormalSampler(num_samples=mc_samples)
    if label not in ("nparego", "nehvi", "ts", "nehvi_rff", "cas"):
        intf = InputPerturbation(
            perturbation_set=perturbation_set, **input_tf_kwargs
        ).eval()
        models = model.models if isinstance(model, ModelListGP) else [model]
        for m in models:
            m.input_transform = intf

    if "ch-var-nei" in label:
        acq_func = get_ch_var_NEI(
            model=model,
            var=var,
            X_baseline=X_baseline,
            sampler=sampler,
            num_constraints=num_constraints,
            augmented="aug_" in label,
            ref_aware=ref_aware,
            mvar_ref_point=mvar_ref_point,
        )
    elif label == "ch-var-ucb":
        if num_constraints > 0:
            raise UnsupportedError(f"{label} does not support outcome constraints.")
        acq_func = get_ch_var_UCB(
            model=model,
            var=var,
            iteration=iteration,
            tkwargs=tkwargs,
        )
    elif "ch-var-ts" in label:
        acq_func = get_ch_var_TS(
            model=model,
            var=var,
            X_baseline=X_baseline,
            num_constraints=num_constraints,
            num_rff_features=kwargs.get("num_rff_features", 512),
            ref_aware=ref_aware,
            mvar_ref_point=mvar_ref_point,
        )
    elif "nparego" in label:
        if label == "independent_var_nparego":
            pre_output_transform = IndependentVaR(alpha=var.alpha, n_w=var.n_w)
        elif label == "expectation_nparego":
            pre_output_transform = MultiOutputExpectation(n_w=var.n_w)
        else:
            pre_output_transform = None
        acq_func = get_NParEGO(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            num_constraints=num_constraints,
            pre_output_transform=pre_output_transform,
        )
    elif "ts" in label:
        if label == "independent_var_ts":
            pre_output_transform = IndependentVaR(alpha=var.alpha, n_w=var.n_w)
        elif label == "expectation_ts":
            pre_output_transform = MultiOutputExpectation(n_w=var.n_w)
        else:
            pre_output_transform = None
        acq_func = get_TS(
            model=model,
            X_baseline=X_baseline,
            num_constraints=num_constraints,
            num_rff_features=kwargs.get("num_rff_features", 512),
            pre_output_transform=pre_output_transform,
        )
    elif "nehvi" in label:
        if "expectation" in label:
            objective = MultiOutputExpectation(n_w=var.n_w)
        elif "independent_var" in label:
            objective = IndependentVaR(alpha=var.alpha, n_w=var.n_w)
        elif "mvar" in label:
            objective = MVaR(n_w=var.n_w, alpha=var.alpha)
        else:
            objective = None
        acq_func = get_nehvi(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            num_constraints=num_constraints,
            use_rff="rff" in label,
            num_rff_features=kwargs.get("num_rff_features", 512),
            objective=objective,
            ref_point=mvar_ref_point if "mvar" in label else None,
            ref_aware=ref_aware,
        )
    elif "cas" in label:
        acq_func = get_cas(
            model=model,
            mvar_ref_point=mvar_ref_point,
            punchout_radius=kwargs.get("punchout_radius"),
            tkwargs=tkwargs,
            num_samples=kwargs.get("cas_num_samples", 512),
        )
    else:
        raise NotImplementedError
    return acq_func


class MVaRHV(Module):
    r"""A helper class that calculates the HV of the MVaR set."""

    def __init__(
        self,
        alpha: float,
        eval_problem: Callable,
        ref_point: Tensor,
        n_w: int,
        perturbation_set: Union[Tensor, Callable[[Tensor], Tensor]],
        num_constraints: int,
        calculate_mvar_on_cpu: bool = False,
        input_tf_kwargs: Optional[Dict[str, bool]] = None,
    ) -> None:
        input_tf_kwargs = input_tf_kwargs or {}
        super().__init__()
        self.hv = DominatedPartitioning(ref_point=ref_point)
        self.mvar = MVaR(n_w=n_w, alpha=alpha)
        self.perturbation = InputPerturbation(
            perturbation_set=perturbation_set, **input_tf_kwargs
        ).eval()
        self.eval_problem = eval_problem
        self.num_constraints = num_constraints
        self.calculate_mvar_on_cpu = calculate_mvar_on_cpu

    def forward(self, new_X: Tensor) -> float:
        r"""Calculate the resulting HV by adding the MVaR corresponding to the new_X
        to the Pareto set.

        Args:
            new_X: `q x dim`-dim tensor of candidate points.

        Returns:
            The cumulative MVaR HV of all points evaluated so far.
        """
        # Get the corresponding MVaR set.
        perturbed_X = self.perturbation(new_X)
        perturbed_Y = self.eval_problem(perturbed_X)

        if self.num_constraints > 0:
            infeas = (perturbed_Y[..., -self.num_constraints :] < 0).any(dim=-1)
            perturbed_Y = perturbed_Y[..., : -self.num_constraints]
            perturbed_Y[infeas] = self.hv.ref_point
        if self.calculate_mvar_on_cpu:
            new_mvar = (
                self.mvar(
                    perturbed_Y.cpu(),
                    use_cpu=True,
                )
                .view(-1, perturbed_Y.shape[-1])
                .to(new_X)
            )
        else:
            new_mvar = self.mvar(perturbed_Y).view(-1, perturbed_Y.shape[-1])
        # Update and return the new MVaR HV.
        self.hv.update(new_mvar)
        return self.hv.compute_hypervolume().item()


def get_problem(name: str) -> MultiObjectiveTestProblem:
    r"""Initialize the test function."""
    if name == "gmm2":
        return GMM(negate=True, num_objectives=2)
    elif name == "gmm3":
        return GMM(negate=True, num_objectives=3)
    elif name == "gmm4":
        return GMM(negate=True, num_objectives=4)
    elif name == "welded_beam":
        return WeldedBeam(negate=True)
    elif name == "constrained_branin_currin":
        return ConstrainedBraninCurrin(negate=True)
    elif name == "disc_brake":
        return DiscBrake(negate=True)
    elif name == "penicillin":
        return Penicillin(negate=True)
    elif name == "toy":
        return ToyRobust(negate=True)
    else:
        raise ValueError(f"Unknown function name: {name}!")
