## This directory contains the experiments (obviously).

### It is structured as follows:
- Each experiment type has its own sub-directory.
- Within this sub-directory, the main Python script for this experiment is found. 
- The script is runnable with `python main.py <dirname> <label> <seed>`.
- `<dirname>` specifies the location for both the configuration and output files.
- The experiment script will read its configuration from `<dirname>/config.json`.
- `config.json` includes a single dictionary (call it `kwargs`) with simple Python objects.
- The main function of script should be called as `main(seed=seed, label=label, **kwargs)`.
- `<label>` is a string used to specify the algorithm to use. 
- The script will write its output file in `<dirname>/<label>/<label>_<seed>.pt` where `<seed>` is 
  written with 4 digits with as many zeros filling in as needed.

- See below for the script for reading the config, running `main(...)` and saving the 
  output. It also checks whether the output exists. Pass `-f` as a 3rd argument to 
  overwrite existing output, or `-a` to add more iterations to the existing output. 
```python
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, sys.argv[1])
    config_path = os.path.join(exp_dir, "config.json")
    label, seed, last_arg = sys.argv[2], int(sys.argv[3]), sys.argv[4]
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
```

- Bash script for running seeds 0 to 9 in a for loop:
```bash
for i in {0..9}; do python main.py <dirname> $i; done
```

- Bash script for running seeds 20 to 29 with batches of 4 in parallel:
```bash
N=4; for i in {20..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py <dirname> $i & done
```
