#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The main script to run the experiments.
"""

import json
import os
import sys
from time import time
from typing import Optional, Dict
import errno
import gc

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.risk_measures import VaR
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from botorch.test_functions.base import ConstrainedBaseTestProblem
from torch import Tensor

from robust_mobo.experiment_utils import (
    generate_initial_data,
    get_perturbations,
    get_problem,
    initialize_model,
    get_acqf,
    MVaRHV,
)
import gpytorch.settings as gpt_settings


supported_labels = [
    "ch-var-ucb",
    "nparego",
    "expectation_nparego",
    "nehvi",
    "expectation_nehvi",
    "nehvi_rff",
    "expectation_nehvi_rff",
    "ref_ch-var-nei",
    "ref_ch-var-ts",
    "ref_mvar_nehvi_rff",
    "ref_mvar_nehvi",
    "sobol",
    "cas",
]


def main(
    seed: int,
    label: str,
    input_dict: Optional[dict],
    mode: Optional[str],
    output_path: str,
    iterations: int,
    n_initial_points: int,
    function_name: str,
    batch_size: int,
    n_w: int,
    alpha: float,
    perturbation_kwargs: dict,
    mc_samples: int,
    optimization_kwargs: dict,
    hv_n_w: int,
    mvar_ref_point: tuple,
    input_tf_kwargs: Optional[Dict[str, bool]] = None,
    model_kwargs: Optional[dict] = None,
    save_frequency: int = 5,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device("cpu"),
) -> None:
    r"""Run the BO loop for given number of iterations. Supports restarting of
    prematurely killed experiments.

    Args:
        seed: The experiment seed.
        label: The label / algorithm to use.
        input_dict: If continuing an existing experiment, this is the output saved by
            the incomplete run.
        mode: Should be `-a` if appending outputs to an existing experiment.
        output_path: The path for the output file.
        iterations: Number of iterations of the BO loop to perform.
        n_initial_points: Number of initial evaluations to use.
        function_name: The name of the test function to use.
        batch_size: The q-batch size, i.e., number of parallel function evaluations.
        n_w: The size of the perturbation set to use.
        alpha: The risk level alpha for VaR / MVaR.
        perturbation_kwargs: Arguments to pass into the `get_perturbations` function.
        mc_samples: Number of MC samples used for MC acquisition functions (e.g., NEI).
        optimization_kwargs: Arguments passed to `optimize_acqf`. Includes `num_restarts`
            and `raw_samples` and other optional arguments.
        hv_n_w: Number of perturbations used to calculate the MVaR HV for reporting.
        mvar_ref_point: The reference point for MVaR HV calculations.
        input_tf_kwargs: Kwargs for input transform, such as `multiplicative`.
        model_kwargs: Arguments for `initialize_model`. The default behavior is to use
            a ModelListGP consisting of noise-free FixedNoiseGP models.
        save_frequency: How often to save the output.
        dtype: The tensor dtype to use.
        device: The device to use.
    """
    assert label in supported_labels, "Label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)

    tkwargs = {"dtype": dtype, "device": device}
    model_kwargs = model_kwargs or {}
    input_tf_kwargs = input_tf_kwargs or {}
    model_kwargs.setdefault(
        "use_model_list", False if "ts" in label or "rff" in label else True
    )
    base_function = get_problem(name=function_name)
    base_function.to(**tkwargs)

    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    num_constraints = base_function.num_constraints if is_constrained else 0

    # set default optimization parameters
    optimization_kwargs.setdefault("num_restarts", 20)
    optimization_kwargs.setdefault("raw_samples", 1024)
    options = optimization_kwargs.get("options")
    if options is None:
        options = {}
        optimization_kwargs["options"] = options
    options.setdefault("batch_limit", 5)
    options.setdefault("maxiter", 200)

    def eval_problem(X: Tensor) -> Tensor:
        X = unnormalize(X, base_function.bounds)
        Y = base_function(X)
        if is_constrained:
            # here, non-negative Y_con implies feasibility
            Y_con = base_function.evaluate_slack(X)
            Y = torch.cat([Y, Y_con], dim=-1)
        return Y

    standard_bounds = torch.ones(2, base_function.dim, **tkwargs)
    standard_bounds[0] = 0
    # Get the initial data.
    X, Y = generate_initial_data(
        n=n_initial_points,
        eval_problem=eval_problem,
        bounds=standard_bounds,
        tkwargs=tkwargs,
    )

    # Initialize other commonly used things.
    var = VaR(n_w=n_w, alpha=alpha)
    # Ensure consistency of MVaRHV across seeds by using same perturbations.
    # This sets the random seed and generates the perturbations on CPU.
    # MVaR calculations are also moved to CPU.
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
        input_tf_kwargs=input_tf_kwargs,
    )
    torch.random.set_rng_state(old_state)

    # Set some counters to keep track of things.
    start = time()
    existing_iterations = 0
    wall_time = torch.zeros(iterations, dtype=dtype)
    # If in the "append" mode, load the existing outputs.
    if input_dict is not None:
        assert torch.allclose(X, input_dict["X"][: X.shape[0]].to(**tkwargs))
        if mode == "-a":
            # Adding iterations to existing output.
            assert input_dict["label"] == label
            # Ensure that correct perturbation set is used for MVaR HV
            assert input_dict.get("uses_fixed_perturbations", False)
            existing_iterations = torch.div(
                input_dict["X"].shape[0] - X.shape[0], batch_size, rounding_mode="floor"
            )
            if existing_iterations >= iterations:
                raise ValueError("Existing output has as many or more iterations!")
            wall_time[:existing_iterations] = (
                input_dict["wall_time"].cpu().to(dtype=dtype)
            )
            start -= float(input_dict["wall_time"][-1])
            X = input_dict["X"].to(**tkwargs)
            Y = input_dict["Y"].to(**tkwargs)
            all_mvar_hvs = input_dict["all_mvar_hvs"]
            # Update the internal state of mvar hv.
            mvar_hv(X)
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")
    else:
        try:
            all_mvar_hvs = torch.tensor([mvar_hv(X)], dtype=dtype)
        except RuntimeError:
            # Try to feed them one by one. This helps with memory.
            initial_mvar_hv = 0.0
            for j in range(X.shape[0]):
                initial_mvar_hv = mvar_hv(X[j : j + 1])
            all_mvar_hvs = torch.tensor([initial_mvar_hv], dtype=dtype)

    # BO loop for as many iterations as needed.
    for i in range(existing_iterations, iterations):
        print(
            f"Starting label {label}, seed {seed}, iteration {i}, "
            f"time: {time()-start}, current MVaR HV: {all_mvar_hvs[-1]}."
        )
        # Fit the model.
        mll, model = initialize_model(train_x=X, train_y=Y, **model_kwargs)
        fit_gpytorch_model(mll)

        if label == "sobol":
            candidates = (
                draw_sobol_samples(
                    bounds=standard_bounds,
                    n=1,
                    q=batch_size,
                )
                .squeeze(0)
                .to(**tkwargs)
            )
        else:
            # Generate the perturbations.
            perturbation_set = get_perturbations(
                n_w=n_w,
                dim=base_function.dim,
                tkwargs=tkwargs,
                bounds=base_function.bounds,
                **perturbation_kwargs,
            )

            with gpt_settings.cholesky_max_tries(6):
                # Construct the acqf.
                pr = perturbation_kwargs.get(
                    "std_dev", None
                ) or perturbation_kwargs.get("delta", None)
                if pr is None and label == "cas":
                    raise NotImplementedError
                pr = torch.tensor(pr, **tkwargs)
                bounds_range = base_function.bounds[1] - base_function.bounds[0]
                pr = pr / bounds_range
                if pr.numel() > 1:
                    if pr.max() > 1.5 * pr.min():
                        raise NotImplementedError
                    else:
                        pr = pr.min()

                acq_func = get_acqf(
                    label=label,
                    mc_samples=mc_samples,
                    model=model,
                    perturbation_set=perturbation_set,
                    var=var,
                    X_baseline=X,
                    iteration=i,
                    tkwargs=tkwargs,
                    num_constraints=num_constraints,
                    mvar_ref_point=torch.tensor(mvar_ref_point, **tkwargs),
                    batch_size=batch_size,
                    input_tf_kwargs=input_tf_kwargs,
                    punchout_radius=pr,
                )

                # Optimize the acqf.
                while options["batch_limit"] >= 1:
                    # Try to get around OOM by reducing batch_limit.
                    try:
                        torch.cuda.empty_cache()
                        if isinstance(acq_func, list):
                            candidates, _ = optimize_acqf_list(
                                acq_function_list=acq_func,
                                bounds=standard_bounds,
                                **optimization_kwargs,
                            )
                        else:
                            candidates, _ = optimize_acqf(
                                acq_function=acq_func,
                                bounds=standard_bounds,
                                q=batch_size,
                                **optimization_kwargs,
                            )
                        torch.cuda.empty_cache()
                        break
                    except RuntimeError as e:
                        if options["batch_limit"] > 1:
                            print(
                                "Got a RuntimeError in `optimize_acqf`. "
                                "Trying with reduced `batch_limit`."
                            )
                            options["batch_limit"] //= 2
                            continue
                        else:
                            raise e
            # free memory
            del acq_func, mll, model
            gc.collect()
            torch.cuda.empty_cache()

        # Get the new observations and update the data.
        new_y = eval_problem(candidates)
        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        wall_time[i] = time() - start
        new_mvar_hv = mvar_hv(candidates)
        all_mvar_hvs = torch.cat(
            [all_mvar_hvs, torch.tensor([new_mvar_hv], dtype=dtype)], dim=0
        )

        # Periodically save the output.
        if iterations % save_frequency == 0:
            output_dict = {
                "label": label,
                "X": X.cpu(),
                "Y": Y.cpu(),
                "wall_time": wall_time[: i + 1],
                "all_mvar_hvs": all_mvar_hvs,
                "uses_fixed_perturbations": True,
            }
            torch.save(output_dict, output_path)

    # Save the final output
    output_dict = {
        "label": label,
        "X": X.cpu(),
        "Y": Y.cpu(),
        "wall_time": wall_time,
        "all_mvar_hvs": all_mvar_hvs,
        "uses_fixed_perturbations": True,
    }
    torch.save(output_dict, output_path)
    return


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, sys.argv[1])
    config_path = os.path.join(exp_dir, "config.json")
    label = sys.argv[2]
    seed = int(float(sys.argv[3]))
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
    input_dict = None
    mode = None
    if os.path.exists(output_path):
        if last_arg and last_arg in ["-a", "-f"]:
            mode = last_arg
            if last_arg == "-f":
                print("Overwriting the existing output!")
            elif last_arg == "-a":
                print(
                    "Appending iterations to existing output!"
                    "Warning: If parameters other than `iterations` have "
                    "been changed, this will corrupt the output!"
                )
                input_dict = torch.load(output_path)
            else:
                raise RuntimeError
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 4th argument to overwrite!"
                "Pass -a as the 4th argument to add more iterations!"
            )
            quit()
    elif not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    main(
        seed=seed,
        label=label,
        input_dict=input_dict,
        mode=mode,
        output_path=output_path,
        **kwargs,
    )
