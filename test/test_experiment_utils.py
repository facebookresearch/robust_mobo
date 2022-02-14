from copy import deepcopy
from unittest import mock, TestCase

import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    IndependentVaR,
)
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.risk_measures import VaR
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.deterministic import DeterministicModel
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import apply_constraints
from botorch.utils.testing import MockModel, MockPosterior
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood

from robust_mobo.ch_var_ucb import ChVUCB
from robust_mobo.input_transform import InputPerturbation
from robust_mobo.experiment_utils import (
    generate_initial_data,
    initialize_model,
    get_chebyshev_objective,
    get_ch_var_UCB,
    get_ch_var_NEI,
    get_NParEGO,
    get_nehvi,
    get_perturbations,
    get_acqf,
    MVaRHV,
    get_constraint_indexer,
    get_infeasible_cost,
)
from robust_mobo.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
)
from robust_mobo.multi_objective_risk_measures import MultiOutputExpectation
from robust_mobo.single_objective_monte_carlo import qNoisyExpectedImprovement
from robust_mobo.utils import (
    get_chebyshev_scalarization,
    FeasibilityWeightedMCMultiOutputObjective,
)


class TestExperimentUtils(TestCase):
    def test_generate_initial_data(self):
        n = 5
        eval_problem = lambda X: X.sum(dim=-1, keepdim=True)
        bounds = torch.ones(2, 3)
        bounds[0] = 0
        tkwargs = {"dtype": torch.double}
        x, y = generate_initial_data(
            n=n,
            eval_problem=eval_problem,
            bounds=bounds,
            tkwargs=tkwargs,
        )
        self.assertEqual(x.shape, torch.Size([n, 3]))
        self.assertTrue(torch.equal(y, x.sum(dim=-1, keepdim=True)))
        self.assertEqual(x.dtype, tkwargs["dtype"])
        self.assertTrue(torch.all(0 <= x) and torch.all(x <= 1))

    def test_initialize_model(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        mll, model = initialize_model(x, y, False, False)
        self.assertTrue(isinstance(model, SingleTaskGP))
        self.assertTrue(torch.equal(model.train_inputs[0], x.repeat(2, 1, 1)))
        self.assertTrue(isinstance(mll, ExactMarginalLogLikelihood))
        mll, model = initialize_model(x, y, False, True)
        self.assertTrue(isinstance(model, FixedNoiseGP))
        mll, model = initialize_model(x, y, True, True)
        self.assertTrue(isinstance(model, ModelListGP))
        self.assertTrue(isinstance(model.models[0], FixedNoiseGP))
        self.assertEqual(len(model.models), 2)
        self.assertEqual(model.models[0].train_targets.shape, torch.Size([5]))
        self.assertTrue(isinstance(mll, SumMarginalLogLikelihood))

    def test_get_ch_var_UCB(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        model = SingleTaskGP(x, y).eval()
        perturbation_set = torch.randn(7, 3) * 0.1
        input_tf = InputPerturbation(perturbation_set)
        model.input_transform = input_tf
        var = VaR(n_w=7, alpha=0.5)
        acqf = get_ch_var_UCB(
            model=model,
            var=var,
            iteration=-1,
            tkwargs={},
        )
        self.assertTrue(isinstance(acqf, ChVUCB))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        model = initialize_model(x, y, True, True)[1].eval()
        for m in model.models:
            m.input_transform = input_tf
        acqf = get_ch_var_UCB(
            model=model,
            var=var,
            iteration=-1,
            tkwargs={},
        )
        self.assertTrue(isinstance(acqf, ChVUCB))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

    def test_get_ch_var_NEI(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        model = SingleTaskGP(x, y).eval()
        perturbation_set = torch.randn(7, 3) * 0.1
        input_tf = InputPerturbation(perturbation_set)
        model.input_transform = input_tf
        var = VaR(n_w=7, alpha=0.5)
        acqf = get_ch_var_NEI(
            model=model,
            var=var,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
            mvar_ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertTrue(isinstance(acqf, qNoisyExpectedImprovement))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        model = initialize_model(x, y, True, True)[1].eval()
        for m in model.models:
            m.input_transform = input_tf
        acqf = get_ch_var_NEI(
            model=model,
            var=var,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
            mvar_ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertTrue(isinstance(acqf, qNoisyExpectedImprovement))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))
        with mock.patch(
            "robust_mobo.experiment_utils.get_chebyshev_objective",
            wraps=get_chebyshev_objective,
        ) as mock_get_chebyshev_objective:
            y = torch.randn(5, 3)
            model = initialize_model(x, y, True, True)[1]
            for m in model.models:
                m.input_transform = input_tf
            get_ch_var_NEI(
                model=model,
                var=var,
                X_baseline=x,
                sampler=SobolQMCNormalSampler(5),
                num_constraints=1,
                mvar_ref_point=torch.tensor([0.0, 0.0]),
            )
            mock_get_chebyshev_objective.assert_called_once()

    def test_get_nparego(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        model = SingleTaskGP(x, y).eval()
        acqf = get_NParEGO(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
        )
        self.assertTrue(isinstance(acqf, qNoisyExpectedImprovement))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        model = initialize_model(x, y, True, True)[1]
        acqf = get_NParEGO(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
        )
        self.assertTrue(isinstance(acqf, qNoisyExpectedImprovement))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))
        # test constraints
        y = torch.randn(5, 3)
        model = initialize_model(x, y, True, True)[1]
        with mock.patch(
            "robust_mobo.experiment_utils.get_chebyshev_objective",
            wraps=get_chebyshev_objective,
        ) as mock_get_chebyshev_objective:
            acqf = get_NParEGO(
                model=model,
                X_baseline=x,
                sampler=SobolQMCNormalSampler(5),
                num_constraints=1,
            )
            mock_get_chebyshev_objective.assert_called_once_with(
                model=model,
                X_baseline=x,
                num_constraints=1,
                alpha=0.05,
                pre_output_transform=None,
            )
        # test pre_output_transform
        model = initialize_model(x, y, True, True)[1]
        perturbation_set = torch.randn(7, 3) * 0.1
        input_tf = InputPerturbation(perturbation_set)
        for m in model.models:
            m.input_transform = input_tf
        ivar = IndependentVaR(alpha=0.5, n_w=7)
        with mock.patch(
            "robust_mobo.experiment_utils.get_chebyshev_objective",
            wraps=get_chebyshev_objective,
        ) as mock_get_chebyshev_objective:
            acqf = get_NParEGO(
                model=model,
                X_baseline=x,
                sampler=SobolQMCNormalSampler(5),
                num_constraints=0,
                pre_output_transform=ivar,
            )
            mock_get_chebyshev_objective.assert_called_once_with(
                model=model,
                X_baseline=x,
                num_constraints=0,
                alpha=0.05,
                pre_output_transform=ivar,
            )

    def test_get_nehvi(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        model = SingleTaskGP(x, y).eval()
        acqf = get_nehvi(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
            use_rff=False,
            objective=None,
            ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertTrue(isinstance(acqf, qNoisyExpectedHypervolumeImprovement))
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        perturbation_set = torch.randn(7, 3) * 0.1
        input_tf = InputPerturbation(perturbation_set)
        model.input_transform = input_tf.eval()
        acqf = get_nehvi(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=0,
            use_rff=False,
            objective=MultiOutputExpectation(n_w=7),
            ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertIsInstance(acqf, qNoisyExpectedHypervolumeImprovement)
        self.assertIsInstance(acqf.objective, FeasibilityWeightedMCMultiOutputObjective)
        self.assertIsInstance(acqf.objective.objective, MultiOutputExpectation)
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        y = torch.randn(5, 3)
        model = SingleTaskGP(x, y).eval()
        model.input_transform = input_tf.eval()
        acqf = get_nehvi(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=1,
            use_rff=False,
            objective=MultiOutputExpectation(n_w=7),
            ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertIsInstance(acqf, qNoisyExpectedHypervolumeImprovement)
        self.assertIsInstance(acqf.objective, FeasibilityWeightedMCMultiOutputObjective)
        self.assertIsInstance(acqf.objective.objective, MultiOutputExpectation)
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

        model = FixedNoiseGP(x, y, torch.full_like(y, 1e-6)).eval()
        model.input_transform = input_tf.eval()
        acqf = get_nehvi(
            model=model,
            X_baseline=x,
            sampler=SobolQMCNormalSampler(5),
            num_constraints=1,
            use_rff=True,
            objective=MultiOutputExpectation(n_w=7),
            ref_point=torch.tensor([0.0, 0.0]),
        )
        self.assertIsInstance(acqf.model, DeterministicModel)
        self.assertIsInstance(acqf, qExpectedHypervolumeImprovement)
        self.assertIsInstance(acqf.objective, FeasibilityWeightedMCMultiOutputObjective)
        self.assertIsInstance(acqf.objective.objective, MultiOutputExpectation)
        self.assertEqual(acqf(torch.rand(2, 1, 3)).shape, torch.Size([2]))

    def test_get_perturbations(self):
        n_w = 128
        dim = 2
        bounds = torch.tensor([[-100.0, 0.0], [0.0, 100.0]])
        perturbations = get_perturbations(
            n_w=n_w,
            dim=dim,
            tkwargs={},
            bounds=bounds,
            method="sobol-normal",
            std_dev=10.0,
        )
        self.assertEqual(perturbations.shape, torch.Size([n_w, dim]))
        self.assertTrue(torch.any(perturbations[:, 0] < -0.1))
        self.assertTrue(torch.any(perturbations[:, 1] > 0.1))
        self.assertFalse(torch.any(perturbations[:, 0] < -1))

        perturbations = get_perturbations(
            n_w=n_w,
            dim=dim,
            tkwargs={},
            bounds=bounds,
            method="uniform",
            delta=10.0,
        )
        self.assertEqual(perturbations.shape, torch.Size([n_w, dim]))
        self.assertTrue(torch.any(perturbations[:, 0] < -0.05))
        self.assertTrue(torch.any(perturbations[:, 1] > 0.05))
        self.assertFalse(torch.any(perturbations[:, 0] < -0.1))

    def test_get_acqf(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 2)
        model = SingleTaskGP(x, y).eval()
        y2 = torch.randn(5, 3)
        constrained_model = SingleTaskGP(x, y2).eval()
        perturbation_set = torch.randn(7, 3) * 0.1
        var = VaR(n_w=7, alpha=0.5)
        bounds = torch.zeros(2, 3)
        bounds[1] = 1.0
        labels = [
            "ch-var-ucb",
            "aug_ch-var-nei",
            "ch-var-nei",
            "ch-var-ts",
            "nparego",
            "independent_var_nparego",
            "expectation_nparego",
            "expectation_ts",
            "nehvi",
            "expectation_nehvi",
            "independent_var_nehvi",
            "ts",
            "nehvi_rff",
            "expectation_nehvi_rff",
            "independent_var_nehvi_rff",
            "mvar_nehvi_rff",
            "mvar_nehvi",
            "ref_aug_ch-var-nei",
            "ref_ch-var-nei",
            "ref_ch-var-ts",
            "ref_mvar_nehvi_rff",
            "ref_mvar_nehvi",
        ]
        for label in labels:
            if label == "ch-var-ucb":
                acqf_class = ChVUCB
            elif "nparego" in label or "nei" in label:
                acqf_class = qNoisyExpectedImprovement
            elif "ts" in label:
                acqf_class = qSimpleRegret
            elif "nehvi_rff" in label:
                acqf_class = qExpectedHypervolumeImprovement
            else:
                acqf_class = qNoisyExpectedHypervolumeImprovement
            acqf = get_acqf(
                label=label,
                mc_samples=5,
                model=deepcopy(model),
                perturbation_set=perturbation_set,
                var=var,
                X_baseline=x,
                iteration=-1,
                tkwargs={},
                num_constraints=0,
                mvar_ref_point=torch.tensor([0.0, 0.0]),
            )
            self.assertTrue(isinstance(acqf, acqf_class))
            self.assertEqual(acqf(x[:2].unsqueeze(-2)).shape, torch.Size([2]))
            # make sure optimization runs smoothly
            optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=4,
            )

            # test constraints for scalarized acqfs
            if label == "ch-var-nei" or "nparego" in label:
                with mock.patch(
                    "robust_mobo.experiment_utils.get_chebyshev_objective",
                    wraps=get_chebyshev_objective,
                ) as mock_get_chebyshev_objective:
                    m2 = deepcopy(constrained_model)
                    acqf = get_acqf(
                        label=label,
                        mc_samples=5,
                        model=m2,
                        perturbation_set=perturbation_set,
                        var=var,
                        X_baseline=x,
                        iteration=-1,
                        tkwargs={},
                        num_constraints=1,
                        mvar_ref_point=torch.tensor([0.0, 0.0]),
                    )
                    mock_get_chebyshev_objective.assert_called_once()
                    ckwargs = mock_get_chebyshev_objective.call_args[1]
                    self.assertIs(ckwargs["model"], m2)
                    self.assertIs(ckwargs["X_baseline"], x)
                    self.assertEqual(acqf(x[:2].unsqueeze(-2)).shape, torch.Size([2]))

            # test constraints for nehvi
            if "nehvi" in label:
                with mock.patch(
                    "robust_mobo.utils.apply_constraints",
                    wraps=apply_constraints,
                ) as mock_apply_constraints:
                    m2 = deepcopy(constrained_model)
                    acqf = get_acqf(
                        label=label,
                        mc_samples=5,
                        model=m2,
                        perturbation_set=perturbation_set,
                        var=var,
                        X_baseline=x,
                        iteration=-1,
                        tkwargs={},
                        num_constraints=1,
                        mvar_ref_point=torch.tensor([0.0, 0.0]),
                    )
                    expected = 4
                    if "rff" in label:
                        expected = 2
                    if "ref" in label:
                        expected -= 1
                    self.assertEqual(mock_apply_constraints.call_count, expected)
                    self.assertEqual(acqf(x[:2].unsqueeze(-2)).shape, torch.Size([2]))

    def test_mvarhv(self):
        eval_problem = lambda X: X
        mvar_hv = MVaRHV(
            alpha=0.5,
            eval_problem=eval_problem,
            ref_point=torch.tensor([0.0, 0.0]),
            n_w=3,
            perturbation_set=torch.zeros(3, 2),
            num_constraints=0,
        )
        new_X = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        hv_val = mvar_hv(new_X)
        self.assertEqual(hv_val, 0.75)
        # test constraints
        mvar_hv = MVaRHV(
            alpha=0.5,
            eval_problem=eval_problem,
            ref_point=torch.tensor([0.0, 0.0]),
            n_w=3,
            perturbation_set=torch.zeros(3, 3),
            num_constraints=1,
        )
        new_X = torch.tensor([[1.0, 0.5, -1.0], [0.5, 1.0, 1.0]])
        hv_val = mvar_hv(new_X)
        self.assertEqual(hv_val, 0.5)

    def test_get_chebyshev_objective(self):
        x = torch.rand(5, 3)
        y = torch.randn(5, 3)
        for use_input_tf in (False, True):
            if use_input_tf:
                perturbation_set = torch.randn(7, 3) * 0.1
                input_tf = InputPerturbation(perturbation_set)
            else:
                input_tf = None
            model = SingleTaskGP(x, y, input_transform=input_tf).eval()
            with torch.no_grad():
                y_pred = model.posterior(x).mean.view(-1, 3)
            with mock.patch(
                "robust_mobo.experiment_utils.get_chebyshev_scalarization",
                wraps=get_chebyshev_scalarization,
            ) as mock_get_chebyshev_scalar:
                obj = get_chebyshev_objective(
                    model=model, X_baseline=x, num_constraints=0, alpha=0.01
                )
                mock_get_chebyshev_scalar.assert_called_once()
                ckwargs = mock_get_chebyshev_scalar.call_args[1]
                self.assertTrue(torch.equal(ckwargs["Y"], y_pred))
                self.assertEqual(ckwargs["alpha"], 0.01)
                self.assertEqual(ckwargs["weights"].shape, torch.Size([1, 3]))

            with mock.patch(
                "robust_mobo.experiment_utils.get_chebyshev_scalarization",
                wraps=get_chebyshev_scalarization,
            ) as mock_get_chebyshev_scalar:
                with mock.patch(
                    "robust_mobo.experiment_utils.get_infeasible_cost",
                    wraps=get_infeasible_cost,
                ) as mock_get_infeasible_cost:
                    obj = get_chebyshev_objective(
                        model=model, X_baseline=x, num_constraints=1, alpha=0.01
                    )
                    mock_get_chebyshev_scalar.assert_called_once()
                    ckwargs = mock_get_chebyshev_scalar.call_args[1]
                    self.assertTrue(torch.equal(ckwargs["Y"], y_pred[:, :2]))
                    self.assertEqual(ckwargs["alpha"], 0.01)
                    self.assertEqual(ckwargs["weights"].shape, torch.Size([1, 2]))
                    self.assertTrue(mock_get_infeasible_cost.call_count, 2)
                    ckwargs = mock_get_infeasible_cost.call_args_list[-1][1]
                    self.assertIs(ckwargs["model"], model)
                    self.assertIs(ckwargs["X"], x)
                    self.assertIsNotNone(ckwargs["objective"])

                # test pre_output_transform
                if use_input_tf:
                    model = SingleTaskGP(x, y, input_transform=input_tf).eval()
                    with torch.no_grad():
                        y_pred = model.posterior(x).mean
                    ivar = IndependentVaR(alpha=0.5, n_w=7)
                    with mock.patch(
                        "robust_mobo.experiment_utils.get_chebyshev_scalarization",
                        wraps=get_chebyshev_scalarization,
                    ) as mock_get_chebyshev_scalar:
                        obj = get_chebyshev_objective(
                            model=model,
                            X_baseline=x,
                            num_constraints=0,
                            alpha=0.01,
                            pre_output_transform=ivar,
                        )
                        mock_get_chebyshev_scalar.assert_called_once()
                        ckwargs = mock_get_chebyshev_scalar.call_args[1]
                        self.assertTrue(torch.equal(ckwargs["Y"], ivar(y_pred)))
                        self.assertEqual(ckwargs["alpha"], 0.01)
                        self.assertEqual(ckwargs["weights"].shape, torch.Size([1, 3]))
                    with mock.patch(
                        "robust_mobo.experiment_utils.get_chebyshev_scalarization",
                        wraps=get_chebyshev_scalarization,
                    ) as mock_get_chebyshev_scalar:
                        obj = get_chebyshev_objective(
                            model=model,
                            X_baseline=x,
                            num_constraints=1,
                            alpha=0.01,
                            pre_output_transform=ivar,
                        )
                        inf_cost = get_infeasible_cost(
                            X=x, model=model, objective=lambda y: y
                        )[:-1]
                        Y_obj = apply_constraints(
                            obj=y_pred[..., :2],
                            constraints=[
                                get_constraint_indexer(i=i)
                                for i in range(2, model.num_outputs)
                            ],
                            samples=y_pred,
                            infeasible_cost=inf_cost,
                        ).view(-1, 2)
                        mock_get_chebyshev_scalar.assert_called_once()
                        ckwargs = mock_get_chebyshev_scalar.call_args[1]
                        expected_Y = ivar(Y_obj)
                        Y_range = (
                            expected_Y.max(dim=0).values - expected_Y.min(dim=0).values
                        )
                        mask = Y_range <= 0
                        expected_Y[-1, mask] += 1
                        self.assertTrue(torch.equal(ckwargs["Y"], expected_Y))
                        self.assertEqual(ckwargs["alpha"], 0.01)
                        self.assertEqual(ckwargs["weights"].shape, torch.Size([1, 2]))

                    with mock.patch(
                        "robust_mobo.experiment_utils.apply_constraints",
                        wraps=apply_constraints,
                    ) as mock_apply_constraints:
                        with mock.patch(
                            "robust_mobo.experiment_utils.get_chebyshev_scalarization",
                            wraps=get_chebyshev_scalarization,
                        ) as mock_get_chebyshev_scalar:
                            exp = MultiOutputExpectation(n_w=7)
                            mean = torch.tensor(
                                [[5.0, 5.0, -1.0], [5.0, 5.0, -1.0], [1.0, 1.0, 1.0]]
                            ).repeat(7, 1)
                            model = MockModel(
                                MockPosterior(
                                    mean=mean, variance=torch.zeros_like(mean)
                                )
                            )
                            obj = get_chebyshev_objective(
                                model=model,
                                X_baseline=x,
                                num_constraints=1,
                                alpha=0.0,
                                pre_output_transform=exp,
                            )
                            Y_arg = mock_get_chebyshev_scalar.call_args[1]["Y"]
                            self.assertTrue(
                                (
                                    Y_arg.max(dim=0).values - Y_arg.min(dim=0).values
                                    > 0
                                ).all()
                            )
                            Y = torch.tensor(
                                [[5.0, 5.0, -1.0], [1.0, 1.0, 1.0]]
                            ).repeat(7, 1)
                            self.assertEqual(obj(Y).shape, torch.Size([2]))
                            self.assertEqual(mock_apply_constraints.call_count, 2)
