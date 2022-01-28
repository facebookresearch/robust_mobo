r"""
References

.. [Liang]
    Q. Liang and L. Lai, Scalable Bayesian Optimization Accelerates Process
    Optimization of Penicillin Production.
    NeurIPS 2021 AI for Science Workshop, 2021.

"""
import torch
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)
from torch import Tensor


Y_xs = 0.45
Y_ps = 0.90
K_1 = 10 ** (-10)
K_2 = 7 * 10 ** (-5)
m_X = 0.014
alpha_1 = 0.143
alpha_2 = 4 * 10 ** (-7)
alpha_3 = 10 ** (-4)
mu_X = 0.092
K_X = 0.15
mu_p = 0.005
K_p = 0.0002
K_I = 0.10
K = 0.04
k_g = 7.0 * 10 ** 3
E_g = 5100.0
k_d = 10.0 ** 33
E_d = 50000.0
lambd = 2.5 * 10 ** (-4)
T_v = 273.0  # Kelvin
T_o = 373.0
R = 1.9872  # CAL/(MOL K)
V_max = 180.0


def penicillin_vectorized(X_input: Tensor) -> Tensor:
    r"""Penicillin simulator, simplified and vectorized.

    Args:
        X_input: A `n x 7`-dim tensor of inputs.

    Returns:
        An `n x 3`-dim tensor of (negative) penicillin, CO2 and time.
    """
    V, X, T, S, F, s_f, H_ = torch.split(X_input, 1, -1)
    P, CO2 = torch.zeros_like(V), torch.zeros_like(V)
    H = torch.full_like(H_, 10.0).pow(-H_)

    active = torch.ones_like(V).bool()
    t_tensor = torch.full_like(V, 2500)

    for t in range(1, 2501):
        if active.sum() == 0:
            break
        F_loss = (
            V[active] * lambd * (torch.exp(5 * ((T[active] - T_o) / (T_v - T_o))) - 1)
        )
        dV_dt = F[active] - F_loss
        mu = (
            (mu_X / (1 + K_1 / H[active] + H[active] / K_2))
            * (S[active] / (K_X * X[active] + S[active]))
            * (
                (k_g * torch.exp(-E_g / (R * T[active])))
                - (k_d * torch.exp(-E_d / (R * T[active])))
            )
        )
        dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt
        mu_pp = mu_p * (S[active] / (K_p + S[active] + S[active].pow(2) / K_I))
        dS_dt = (
            -(mu / Y_xs) * X[active]
            - (mu_pp / Y_ps) * X[active]
            - m_X * X[active]
            + F[active] * s_f[active] / V[active]
            - (S[active] / V[active]) * dV_dt
        )
        dP_dt = (mu_pp * X[active]) - K * P[active] - (P[active] / V[active]) * dV_dt
        dCO2_dt = alpha_1 * dX_dt + alpha_2 * X[active] + alpha_3

        # UPDATE
        P[active] = P[active] + dP_dt  # Penicillin concentration
        V[active] = V[active] + dV_dt  # Culture medium volume
        X[active] = X[active] + dX_dt  # Biomass concentration
        S[active] = S[active] + dS_dt  # Glucose concentration
        CO2[active] = CO2[active] + dCO2_dt  # CO2 concentration

        # Update active indices
        full_dpdt = torch.ones_like(P)
        full_dpdt[active] = dP_dt
        inactive = (V > V_max) + (S < 0) + (full_dpdt < 10e-12)
        t_tensor[inactive] = torch.minimum(
            t_tensor[inactive], torch.full_like(t_tensor[inactive], t)
        )
        active[inactive] = 0

    return torch.stack([-P, CO2, t_tensor], dim=-1)


class Penicillin(MultiObjectiveTestProblem):
    r"""A penicillin production simulator from [Liang]_. The goal is to
    maximize the penicillin yield while minimizing time to ferment and the
    CO2 byproduct.

    The function is defined for minimization of all objectives.

    A reasonable MVaR reference point is [2.37, 86.86, 442.10].
    Both reference points are obtained using `infer_reference_point` with
    random search.
    """
    dim = 7
    num_objectives = 3
    _bounds = [
        (60.0, 120.0),
        (0.05, 18.0),
        (293.0, 303.0),
        (0.05, 18.0),
        (0.01, 0.5),
        (500.0, 700.0),
        (5.0, 6.5),
    ]
    _ref_point = [1.85, 86.93, 514.70]

    def evaluate_true(self, X: Tensor) -> Tensor:
        # This uses in-place ops. Using clone to avoid modifying X.
        return penicillin_vectorized(X.view(-1, self.dim).clone()).view(
            *X.shape[:-1], self.num_objectives
        )
