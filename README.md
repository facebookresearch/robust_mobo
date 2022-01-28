# Robust Multi-Objective Bayesian Optimization Under Input Noise

## Getting started

Parts of this code relies on latest additions to BoTorch, which are not available
on the current (at the time of writing) `0.6.0` release. Unless a later version has been
released, first use the following to install from the main branch:
```bash
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
pip install --upgrade git+https://github.com/pytorch/botorch.git
```

From the base `robust_mobo` directory run:

`pip install -e .`

## Structure

The code is structured in three parts.
- The utilities for constructing the acquisition functions and other helper methods are defined in `robust_mobo/`.
- The notebooks for analyzing the experiment output and constructing the plots presented in the paper are found under `notebooks/`.
- The experiments are found in and ran from within `experiments/experiment_v1/`. The `main.py` is used to run the experiments, and the experiment configurations are found in the `config.json` file of each sub-directory. 

The individual experiment outputs were left out to avoid inflating the file size. An aggregate output of all experiments is presented in `notebooks/final_outputs.csv`.

## Running Experiments

To run a basic benchmark based on the `config.json` file in `experiments/experiments_v1/<experiment_name>` using `<algorithm>`:

```
cd experiments/experiments_v1
python main.py <experiment_name> <algorithm> <seed>
```

The code refers to the algorithms using the following labels:
```
algorithms = [
    ("sobol", "Sobol"),
    ("nparego", "qNParEGO"),
    ("nehvi", "qNEHVI"),
    ("nehvi_rff", "qNEHVI-RFF"),
    ("expectation_nparego", "Exp-qNParEGO"),
    ("expectation_nehvi_rff", "Exp-NEHVI-RFF"),
    ("ch-var-ucb", "MARS-UCB"),
    ("ref_ch-var-ts", "MARS-TS"),
    ("ref_ch-var-nei", "MARS-NEI"),
    ("ref_mvar_nehvi", "MVaR-NEHVI"),
    ("ref_mvar_nehvi_rff", "MVaR-NEHVI-RFF"),
    ("cas", "Constraint Active Search"),
]
```

Each folder under `experiments/experiment_v1/` corresponds to the experiments in the paper according to the following mapping:
```
experiments = {
    "1d_toy": "Toy Problem",
    "1d_toy_q2": "Toy Problem, q=2",
    "1d_toy_q4": "Toy Problem, q=4",
    "1d_toy_q8": "Toy Problem, q=8",
    "bc_v2": "Constrained Branin Currin",
    "bc_q2": "Constrained Branin Currin, q=2",
    "bc_q4": "Constrained Branin Currin, q=4",
    "bc_q8": "Constrained Branin Currin, q=8",
    "bc_8": "Constrained Branin Currin, n_xi=8",
    "bc_16": "Constrained Branin Currin, n_xi=16",
    "bc_64": "Constrained Branin Currin, n_xi=64",
    "bc_96": "Constrained Branin Currin, n_xi=96",
    "bc_128": "Constrained Branin Currin, n_xi=128",
    "bc_heteroskedastic_v2": "Constrained Branin Currin, Heteroscedastic",
    "disc_brake": "Disc Brake",
    "disc_brake_8": "Disc Brake, n_xi=8",
    "disc_brake_16": "Disc Brake, n_xi=16",
    "disc_brake_64": "Disc Brake, n_xi=64",
    "disc_brake_96": "Disc Brake, n_xi=64",
    "disc_brake_128": "Disc Brake, n_xi=128",
    "penicillin_v2": "Penicillin",
    "gmm_demo": "GMM, Std=0.05",
    "gmm_noise2x": "GMM, Std=0.10",
    "gmm_noise4x": "GMM, Std=0.20",
    "gmm_hetero": "GMM, Std=0.2X",
    "gmm_corr": "GMM, Correlated",
    "gmm_mul": "GMM, Multiplicative",
    "gmm3": "GMM, M=3, alpha=0.9",
    "gmm3_08": "GMM, M=3, alpha=0.8",
    "gmm3_07": "GMM, M=3, alpha=0.7",
    "gmm_4obj": "GMM, M=4, alpha=0.9",
    "gmm_4obj_08": "GMM, M=4, alpha=0.8",
}
```