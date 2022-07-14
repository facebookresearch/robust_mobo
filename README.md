[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Robust Multi-Objective Bayesian Optimization Under Input Noise

This is the code associated with the paper "[Robust Multi-Objective Bayesian Optimization Under Input Noise](https://proceedings.mlr.press/v162/daulton22a.html)."

For a simple demo, check out our [tutorial in BoTorch](https://github.com/pytorch/botorch/blob/main/tutorials/robust_multi_objective_bo.ipynb).

Please cite our work if you find it useful.

    
    @InProceedings{pmlr-v162-daulton22a,
      title = 	 {Robust Multi-Objective {B}ayesian Optimization Under Input Noise},
      author =       {Daulton, Samuel and Cakmak, Sait and Balandat, Maximilian and Osborne, Michael A. and Zhou, Enlu and Bakshy, Eytan},
      booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
      pages = 	 {4831--4866},
      year = 	 {2022},
      editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
      volume = 	 {162},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {17--23 Jul},
      publisher =    {PMLR},
      pdf = 	 {https://proceedings.mlr.press/v162/daulton22a/daulton22a.pdf},
      url = 	 {https://proceedings.mlr.press/v162/daulton22a.html},
      abstract = 	 {Bayesian optimization (BO) is a sample-efficient approach for tuning design parameters to optimize expensive-to-evaluate, black-box performance metrics. In many manufacturing processes, the design parameters are subject to random input noise, resulting in a product that is often less performant than expected. Although BO methods have been proposed for optimizing a single objective under input noise, no existing method addresses the practical scenario where there are multiple objectives that are sensitive to input perturbations. In this work, we propose the first multi-objective BO method that is robust to input noise. We formalize our goal as optimizing the multivariate value-at-risk (MVaR), a risk measure of the uncertain objectives. Since directly optimizing MVaR is computationally infeasible in many settings, we propose a scalable, theoretically-grounded approach for optimizing MVaR using random scalarizations. Empirically, we find that our approach significantly outperforms alternative methods and efficiently identifies optimal robust designs that will satisfy specifications across multiple metrics with high probability.}
    }


## Getting started

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

## License
This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
