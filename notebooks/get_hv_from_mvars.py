import os
import sys
import json
import torch
from time import time
from botorch.utils.multi_objective.pareto import is_non_dominated
from approximate_max_hv_run import construct_mvar_hv, exp_dir, max_hv_dir


def get_hv_from_mvars(problem, output_name=None, device="cpu"):
    start = time()
    if output_name is None:
        output_name = problem
    output_path = os.path.join(max_hv_dir, f"{output_name}.pt")
    if os.path.exists(output_path):
        raise FileExistsError
    all_mvar_path = os.path.join(max_hv_dir, f"{output_name}_all_mvars.pt")
    if os.path.exists(all_mvar_path):
        all_mvars = torch.load(all_mvar_path)
    else:
        all_mvars = []
        for p in os.listdir(max_hv_dir):
            if f"{problem}_mvar" not in p:
                continue
            p_path = os.path.join(max_hv_dir, p)
            if "_nd.pt" in p_path:
                # These are non-dominated
                mvars = torch.load(p_path)
                all_mvars.append(mvars)
                print(f"Reading {p}, found {mvars.shape}. Time {time()-start}.")
            else:
                # remove the non-dominated entries
                mvars = torch.load(p_path)
                print(f"Reading {p}, found {mvars.shape}. Time {time()-start}.")
                mask = is_non_dominated(mvars.to(device=device)).cpu()
                print(f"Number of dominated points: {(~mask).sum()}")
                all_mvars.append(mvars[mask])
                # replace the file
                torch.save(mvars[mask], f"{p_path[:-3]}_nd.pt")
                os.remove(p_path)

        all_mvars = torch.cat(all_mvars, dim=0)
        mask = is_non_dominated(all_mvars)
        all_mvars = all_mvars[mask]
    print(f"MVaRs found: {all_mvars.shape}.")
    all_Xs = torch.load(os.path.join(max_hv_dir, f"{problem}_Xs.pt"))["all_Xs"]

    config_path = os.path.join(exp_dir, problem, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config.pop("device")
    mvar_hv = construct_mvar_hv(**config)

    mvar_hv.hv.to(device=device)
    mvar_hv.hv.update(all_mvars.to(device=device))
    max_hv = mvar_hv.hv.compute_hypervolume().item()
    neg_Y = mvar_hv.hv._neg_Y
    output = {
        "max_hv": max_hv,
        "neg_Y": neg_Y.cpu(),
        "all_Xs": all_Xs.cpu(),
    }
    torch.save(output, output_path)


if __name__ == "__main__":
    get_hv_from_mvars(sys.argv[1], device=sys.argv[2])
