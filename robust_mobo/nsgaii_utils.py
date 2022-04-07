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
    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    pymoo_problem = Problem(
        n_var=base_function.dim,
        n_obj=base_function.num_objectives,
        n_constr=base_function.num_constraints if is_constrained else 0,
        xl=np.zeros(base_function.dim),
        xu=np.ones(base_function.dim),
    )
    pop = Population.new("X", X_init.cpu().numpy())
    # pymoo minimizes objectives
    pop.set("F", -Y_init[:, : base_function.num_objectives].cpu().numpy())
    if is_constrained:
        # negative constraint slack implies feasibility
        pop.set("G", -Y_init[:, base_function.num_objectives :].cpu().numpy())
    algorithm = NSGA2(pop_size=10, sampling=pop)
    # let the algorithm object never terminate and let the loop control it
    termination = NoTermination()

    # create an algorithm object that never terminates
    algorithm.setup(pymoo_problem, termination=termination, seed=0)

    return algorithm
