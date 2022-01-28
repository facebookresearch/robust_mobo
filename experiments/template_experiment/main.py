r"""
This would have the code necessary to run the BO loop.
"""
import sys
import os

import torch


def main(
    seed=seed,
    label=label,
    input_dict=input_dict,
    mode=mode,
    output_path=output_path,
):
    r"""
    The main BO loop goes here.
    Args:
        seed:
        label:
        input_dict:
        mode:
        output_path:

    Returns:

    """
    # Setup the label / algorithm specific things here

    # If in the "append" mode, load the existing outputs here

    # BO loop for as many iterations as needed.
    # Make sure to save the output periodically in case of errors, premature death.
    for i in range(existing_iterations, itearations):
        # Do Stuff.
        pass
    # Save the final output


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, sys.argv[1])
    config_path = os.path.join(exp_dir, "config.json")
    label, seed, last_arg = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
    input_dict = None
    mode = None
    if path.exists(output_path):
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
