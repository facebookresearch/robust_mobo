import json
import os
import sys
import torch
from botorch.utils.multi_objective import is_non_dominated

from approximate_max_hv_run import construct_mvar_hv, max_hv_dir, exp_dir


def batch_evaluate_mvar(problem, batch_i):
    x_path = os.path.join(max_hv_dir, f"{problem}_Xs.pt")
    # Using nd since we remove the dominated ones
    out_path = os.path.join(max_hv_dir, f"{problem}_mvar_{batch_i}_nd.pt")
    if os.path.exists(out_path):
        raise FileExistsError
    all_Xs = torch.load(x_path)["all_Xs"]
    active_X = torch.split(all_Xs, 1000)[batch_i]

    config_path = os.path.join(exp_dir, problem, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config.pop("device")
    mvar_hv = construct_mvar_hv(**config)

    # Calculate the MVaR
    perturbed_X = mvar_hv.perturbation(active_X)
    perturbed_Y = mvar_hv.eval_problem(perturbed_X)
    if mvar_hv.num_constraints > 0:
        infeas = (perturbed_Y[..., -mvar_hv.num_constraints :] < 0).any(dim=-1)
        perturbed_Y = perturbed_Y[..., : -mvar_hv.num_constraints]
        perturbed_Y[infeas] = mvar_hv.hv.ref_point
    new_mvar = (
        mvar_hv.mvar(
            perturbed_Y.cpu(),
            use_cpu=True,
        )
        .view(-1, perturbed_Y.shape[-1])
        .to(active_X)
    )
    # Remove dominated ones
    mask = is_non_dominated(new_mvar)
    new_mvar = new_mvar[mask]
    torch.save(new_mvar, out_path)


if __name__ == "__main__":
    batch_evaluate_mvar(sys.argv[1], int(sys.argv[2]))
