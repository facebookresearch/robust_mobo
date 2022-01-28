import json
import os
import sys

import torch
import numpy as np
from botorch.utils.transforms import unnormalize
from botorch.test_functions.base import ConstrainedBaseTestProblem
from torch import Tensor

from robust_mobo.experiment_utils import get_problem
from robust_mobo.experiment_utils import MVaRHV, get_perturbations

exp_dir = "../experiments/experiment_v1/"
max_hv_dir = "./max_hvs/"


def construct_mvar_hv(
    function_name: str,
    alpha: float,
    perturbation_kwargs: dict,
    hv_n_w: int,
    mvar_ref_point: tuple,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device("cpu"),
    use_cpu: bool = True,
    **kwargs,
):
    r"""Get the MVaRHV corresponding to the given experiment."""
    tkwargs = {"dtype": dtype, "device": device}
    base_function = get_problem(name=function_name)
    base_function.to(**tkwargs)

    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    num_constraints = base_function.num_constraints if is_constrained else 0

    def eval_problem(X: Tensor) -> Tensor:
        X = unnormalize(X, base_function.bounds)
        Y = base_function(X)
        if is_constrained:
            # here, non-negative Y_con implies feasibility
            Y_con = base_function.evaluate_slack(X)
            Y = torch.cat([Y, Y_con], dim=-1)
        return Y

    old_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    perturbations = get_perturbations(
        n_w=hv_n_w,
        dim=base_function.dim,
        tkwargs={"dtype": dtype, "device": "cpu"},
        bounds=base_function.bounds.cpu(),
        **perturbation_kwargs,
    )
    if not callable(perturbations):
        perturbations = perturbations.to(device=device)
    mvar_hv = MVaRHV(
        alpha=alpha,
        eval_problem=eval_problem,
        ref_point=torch.tensor(mvar_ref_point, **tkwargs),
        n_w=hv_n_w,
        perturbation_set=perturbations,
        num_constraints=num_constraints,
        calculate_mvar_on_cpu=True,
        input_tf_kwargs=kwargs.get("input_tf_kwargs"),
    )
    torch.random.set_rng_state(old_state)
    return mvar_hv


def get_max_hv(problem, output_name, use_cpu=True):
    all_Xs = []
    problem_path = os.path.join(exp_dir, problem)
    for p in os.listdir(problem_path):
        p_path = os.path.join(problem_path, p)
        if os.path.isdir(p_path):
            for fp in os.listdir(p_path):
                fp_path = os.path.join(p_path, fp)
                try:
                    output = torch.load(fp_path)
                    all_Xs.append(output["X"])
                except:
                    continue
    all_Xs = torch.cat(all_Xs).unique(dim=0)

    config_path = os.path.join(problem_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config.pop("device")
    mvar_hv = construct_mvar_hv(**config, use_cpu=use_cpu)
    # Do not re-calculate if the output exists.
    output_path = os.path.join(max_hv_dir, f"{output_name}.pt")
    if os.path.exists(output_path):
        raise FileExistsError
    if not use_cpu:
        all_Xs = all_Xs.to(device="cuda")
    max_hv = mvar_hv(all_Xs)
    neg_Y = mvar_hv.hv._neg_Y
    output = {
        "max_hv": max_hv,
        "neg_Y": neg_Y.cpu(),
        "all_Xs": all_Xs.cpu(),
    }
    torch.save(output, output_path)


if __name__ == "__main__":
    problem = sys.argv[1]
    if len(sys.argv) > 2:
        output_name = sys.argv[2]
    else:
        output_name = problem
    use_cpu = len(sys.argv) > 3
    get_max_hv(problem, output_name)
