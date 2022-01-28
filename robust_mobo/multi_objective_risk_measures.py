"""
This is used for customizing the MO risk measures and adding new ones that
are not available on the BoTorch version, such as expectation.
"""
from math import ceil
from typing import Optional, List

import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputRiskMeasureMCObjective,
)
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MVaR as BoTorchMVaR,
)
from botorch.acquisition.objective import AcquisitionObjective
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.multi_objective import is_non_dominated
from gpytorch import lazify
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor


class MultiOutputExpectation(MultiOutputRiskMeasureMCObjective):
    r"""A multi-output MC expectation risk measure."""

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the expectation of the given samples. Expectation is
        calculated over each `n_w` samples in the q-batch dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of expectation samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.mean(dim=-2)


class ExpectationObjective(AcquisitionObjective):
    r"""Transform the `batch x (q * n_w) x m` posterior into a
    `batch x q x m` posterior of the expectation.
    This is intended for use with `InputPerturbation` or
    `AppendFeatures` for optimizing the expectation over
    `n_w` points. TODO: add reference
    Note: This is different than `ScalarizedObjective` in that
    this operates over the q-batch dimension.
    """

    def __init__(self, n_w: int) -> None:
        super().__init__()
        self.n_w = n_w

    def evaluate(self, Y: Tensor) -> Tensor:
        batch_shape = Y.shape[:-2]
        m = Y.shape[-1]
        return Y.view(*batch_shape, -1, self.n_w, m).mean(dim=-2)

    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        org_mvn = posterior.mvn
        if org_mvn._interleaved:
            raise NotImplementedError
        org_event_shape = org_mvn.event_shape
        batch_shape = org_mvn.batch_shape
        q = org_event_shape[-2] // self.n_w
        m = org_event_shape[-1]
        weights = torch.zeros(q * m, q * self.n_w * m).to(org_mvn.loc)
        for i in range(q * m):
            weights[i, self.n_w * i : self.n_w * (i + 1)] = 1.0 / self.n_w
        new_mean = (
            (weights @ org_mvn.loc.unsqueeze(-1))
            .view(*batch_shape, m, q)
            .transpose(-1, -2)
        )
        org_cov = (
            org_mvn.lazy_covariance_matrix
            if org_mvn.islazy
            else org_mvn.covariance_matrix
        )
        new_cov = weights @ (org_cov @ weights.t())
        new_mvn = MultitaskMultivariateNormal(
            new_mean, lazify(new_cov), interleaved=False
        )
        return GPyTorchPosterior(mvn=new_mvn)


class MVaR(BoTorchMVaR):
    r"""An approximately differentiable version of MVaR.
    The gradients produced here agree with the finite difference gradients.
    """

    def get_mvar_set_cpu(self, Y: Tensor) -> Tensor:
        r"""Find MVaR set based on the definition in [Prekopa2012MVaR]_.

        NOTE: This is much faster on CPU for large `n_w` than the alternative but it
        is significantly slower on GPU. Based on empirical evidence, this is recommended
        when running on CPU with `n_w > 64`.

        This first calculates the CDF for each point on the extended domain of the
        random variable (the grid defined by the given samples), then takes the
        values with CDF equal to (rounded if necessary) `alpha`. The non-dominated
        subset of these form the MVaR set.

        Args:
            Y: A `batch x n_w x m`-dim tensor of outcomes.

        Returns:
            A `batch` length list of `k x m`-dim tensor of MVaR values, where `k`
            depends on the corresponding batch inputs. Note that MVaR values in general
            are not in-sample points.
        """
        if Y.dim() == 3:
            return [self.get_mvar_set_cpu(y_) for y_ in Y]
        m = Y.shape[-1]
        # Generate sets of all unique values in each output dimension.
        # Note that points in MVaR are bounded from above by the
        # independent VaR of each objective. Hence, we only need to
        # consider the unique outcomes that are less than or equal to
        # the VaR of the independent objectives
        var_alpha_idx = ceil(self.alpha * self.n_w) - 1
        Y_sorted = Y.topk(Y.shape[0] - var_alpha_idx, dim=0, largest=False).values
        unique_outcomes_list = []
        for i in range(m):
            sorted_i = Y_sorted[:, i].cpu().clone(memory_format=torch.contiguous_format)
            unique_outcomes_list.append(sorted_i.unique().tolist()[::-1])
        # Convert this into a list of m dictionaries mapping values to indices.
        unique_outcomes = [
            dict(zip(outcomes, range(len(outcomes))))
            for outcomes in unique_outcomes_list
        ]
        # Initialize a tensor counting the number of points in Y that a given grid point
        # is dominated by. This will essentially be a non-normalized CDF.
        counter_tensor = torch.zeros(
            [len(outcomes) for outcomes in unique_outcomes],
            dtype=torch.long,
            device=Y.device,
        )
        # populate the tensor, counting the dominated points.
        # we only need to consider points in Y where at least one
        # objective is less than the max objective value in
        # unique_outcomes_list
        max_vals = torch.tensor(
            [o[0] for o in unique_outcomes_list], dtype=Y.dtype, device=Y.device
        )
        mask = (Y < max_vals).any(dim=-1)
        counter_tensor += self.n_w - mask.sum()
        Y_pruned = Y[mask]
        for y_ in Y_pruned:
            starting_idcs = [unique_outcomes[i].get(y_[i].item(), 0) for i in range(m)]
            slices = [slice(s_idx, None) for s_idx in starting_idcs]
            counter_tensor[slices] += 1

        # Get the count alpha-level points should have.
        alpha_count = ceil(self.alpha * self.n_w)
        # Get the alpha level indices.
        alpha_level_indices = (counter_tensor == alpha_count).nonzero(as_tuple=False)
        # If there are no exact alpha level points, get the smallest alpha' > alpha
        # and find the corresponding alpha level indices.
        if alpha_level_indices.numel() == 0:
            min_greater_than_alpha = counter_tensor[counter_tensor > alpha_count].min()
            alpha_level_indices = (counter_tensor == min_greater_than_alpha).nonzero(
                as_tuple=False
            )
        unique_outcomes = [
            torch.as_tensor(list(outcomes.keys()), device=Y.device, dtype=Y.dtype)
            for outcomes in unique_outcomes
        ]
        alpha_level_points = torch.stack(
            [
                unique_outcomes[i][alpha_level_indices[:, i]]
                for i in range(len(unique_outcomes))
            ],
            dim=-1,
        )
        # MVaR is simply the non-dominated subset of alpha level points.
        if self.filter_dominated:
            mask = is_non_dominated(alpha_level_points)
            mvar = alpha_level_points[mask]
        else:
            mvar = alpha_level_points
        return mvar

    def make_diffable(self, prepared_samples: Tensor, mvars: Tensor) -> List[Tensor]:
        r"""An experimental approach for obtaining the gradient of the MVaR via
        component-wise mapping to original samples.

        Args:
            prepared_samples: A `(sample_shape * batch_shape * q) x n_w x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            mvars: A `(sample_shape * batch_shape * q) x k x m`-dim tensor
                of padded MVaR values.

        Returns:
            The same `mvars` with entries mapped to inputs to produce gradients.
        """
        samples = prepared_samples.unsqueeze(-2).repeat(1, 1, mvars.shape[-2], 1)
        mask = samples == mvars.unsqueeze(-3)
        samples[~mask] = 0
        return samples.sum(dim=-3) / mask.sum(dim=-3)

    def forward(
        self,
        samples: Tensor,
        X: Optional[Tensor] = None,
        use_cpu: bool = False,
    ) -> Tensor:
        r"""Calculate the MVaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.
            use_cpu: If True, uses `get_mvar_set_cpu`. This is beneficial when
                `n_w` is quite large.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of MVaR values,
            if `self.expectation=True`.
            Otherwise, this returns a `sample_shape x batch_shape x (q * k') x m`-dim
            tensor, where `k'` is the maximum `k` across all batches that is returned
            by `get_mvar_set_...`. Each `(q * k') x m` corresponds to the `k` MVaR
            values for each `q` batch of `n_w` inputs, padded up to `k'` by repeating
            the last element. If `self.pad_to_n_w`, we set `k' = self.n_w`, producing
            a deterministic return shape.
        """
        batch_shape, m = samples.shape[:-2], samples.shape[-1]
        prepared_samples = self._prepare_samples(samples)
        # This is -1 x n_w x m.
        prepared_samples = prepared_samples.reshape(-1, *prepared_samples.shape[-2:])
        with torch.no_grad():
            if use_cpu:
                mvar_set = self.get_mvar_set_cpu(prepared_samples)
            else:
                mvar_set = self.get_mvar_set_gpu(prepared_samples)
        # Set the `pad_size` to either `self.n_w` or the size of the largest MVaR set.
        pad_size = self.n_w if self.pad_to_n_w else max([_.shape[0] for _ in mvar_set])
        padded_mvar_list = []
        for mvar_ in mvar_set:
            if self.expectation:
                padded_mvar_list.append(mvar_.mean(dim=0))
            else:
                # Repeat the last entry to make `mvar_set` `pad_size x m`.
                repeats_needed = pad_size - mvar_.shape[0]
                padded_mvar_list.append(
                    torch.cat([mvar_, mvar_[-1].expand(repeats_needed, m)], dim=0)
                )
        mvars = torch.stack(padded_mvar_list, dim=0)
        if samples.requires_grad:
            mvars = self.make_diffable(prepared_samples, mvars)
        return mvars.view(*batch_shape, -1, m)
