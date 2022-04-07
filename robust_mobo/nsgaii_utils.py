#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
NSGA-II utilities
"""
from typing import Dict
from torch import Tensor
import torch
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from pymoo.util.termination.no_termination import NoTermination


def get_nsgaii(X_init: Tensor, Y_init: Tensor, base_function: BaseTestProblem) -> NSGA2:
    pymoo_problem = Problem(
        n_var=_problem.dim,
        n_obj=_problem.num_objectives,
        n_constr=_problem.num_constraints,
        xl=np.zeros(_problem.dim),
        xu=np.ones(_problem.dim),
    )
    pop = Population.new("X", X_init.numpy())
    # objectives
    pop.set("F", Y_init[:, : base_function.num_objectives].numpy())
    if isinstance(base_function, ConstrainedBaseTestProblem):
        # constraints
        pop.set("G", Y_init[:, base_function.num_objectives :].numpy())
    algorithm = NSGA2(pop_size=10, sampling=pop)
    # let the algorithm object never terminate and let the loop control it
    termination = NoTermination()

    # create an algorithm object that never terminates
    algorithm.setup(pymoo_problem, termination=termination, seed=0)

    return algorithm
