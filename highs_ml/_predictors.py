"""MILP embeddings of scikit-learn predictors for HiGHS.

Each ``*Constr`` class mirrors its ``gurobi_ml`` counterpart: it links a
set of model variables (the predictor inputs) to an output variable (the
prediction) through constraints that HiGHS can solve exactly:

* ``LinearRegressionConstr``   -- exact linear equality.
* ``LogisticRegressionConstr`` -- sigmoid via certified piecewise-linear
  approximation (binary variables), since HiGHS has no general constraints.
* ``MLPRegressorConstr``       -- exact big-M formulation for ReLU layers,
  certified PWL for tanh/logistic activations.
* ``PipelineConstr``           -- chains supported sklearn steps.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from highspy import HighsVarType

from ._affine import Affine
from ._pwl import PWLStats, add_pwl_constr


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _coerce_inputs(predictor, input_vars) -> list[Affine]:
    """Turn a mapping/sequence of input variables into ordered Affine forms."""
    n_features = predictor.n_features_in_
    if isinstance(input_vars, Mapping):
        names = getattr(predictor, "feature_names_in_", None)
        if names is None:
            raise ValueError(
                "Predictor has no feature_names_in_; pass inputs as a sequence instead."
            )
        missing = [name for name in names if name not in input_vars]
        if missing:
            raise ValueError(f"Missing input features: {missing}")
        return [Affine.coerce(input_vars[name]) for name in names]
    inputs = [Affine.coerce(v) for v in input_vars]
    if len(inputs) != n_features:
        raise ValueError(f"Expected {n_features} input features, got {len(inputs)}.")
    return inputs


def _linear_form(coefs, intercept, inputs: list[Affine]) -> Affine:
    z = Affine(const=float(intercept))
    for c, x in zip(coefs, inputs):
        z = z + x * float(c)
    return z


class AbstractPredictorConstr:
    """Base class: bookkeeping, stats and solution-time error checks."""

    def __init__(self, h, predictor, inputs: list[Affine], stats: Optional[PWLStats]):
        self.model = h
        self.predictor = predictor
        self.inputs = inputs
        self.stats = stats if stats is not None else PWLStats()
        self.output_var = None
        self.refinable_embeddings: list = []

    # -- reporting ------------------------------------------------------
    def print_stats(self) -> None:
        s = self.stats.as_dict()
        name = type(self).__name__
        print(f"Model for {name}:")
        print(
            f"  {s['variables']} added variables ({s['binaries']} binary), "
            f"{s['constraints']} constraints"
        )
        if s["pwl_relations"]:
            print(f"  {s['pwl_relations']} piecewise-linear relations")
        if s["relu_relations"]:
            print(f"  {s['relu_relations']} exact ReLU relations")

    def _solution_values(self):
        sol = self.model.getSolution()
        if not sol.value_valid:
            raise RuntimeError("No valid primal solution in the HiGHS model; solve first.")
        return sol.col_value

    def _input_values(self) -> np.ndarray:
        values = self._solution_values()
        return np.array([x.evaluate(values) for x in self.inputs])

    # -- error check ----------------------------------------------------
    def get_error(self) -> float:
        """Max |exact prediction - modelled output| at the current solution."""
        import warnings

        x = self._input_values()
        with warnings.catch_warnings():
            # sklearn warns about missing feature names on numpy input;
            # irrelevant for a numerical error check.
            warnings.simplefilter("ignore", UserWarning)
            exact = float(self._exact_prediction(x))
        values = self._solution_values()
        modelled = float(values[self.output_var.index])
        return abs(exact - modelled)

    def _exact_prediction(self, x: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


class LinearRegressionConstr(AbstractPredictorConstr):
    """Exact embedding of ``sklearn.linear_model.LinearRegression``."""

    def __init__(self, h, predictor, input_vars, output_var=None,
                 stats: Optional[PWLStats] = None, name: str = "lin_reg"):
        inputs = _coerce_inputs(predictor, input_vars)
        super().__init__(h, predictor, inputs, stats)
        coefs = np.atleast_2d(predictor.coef_)
        if coefs.shape[0] != 1:
            raise ValueError("Multi-output linear regression is not supported yet.")
        z = _linear_form(coefs[0], np.atleast_1d(predictor.intercept_)[0], inputs)

        if output_var is None:
            lo, hi = z.bounds()
            output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
            self.stats.n_vars += 1
        if z.is_constant():
            h.addConstr(output_var == z.const, name=f"{name}_eq")
        else:
            h.addConstr(output_var == z.to_highspy(), name=f"{name}_eq")
        self.stats.n_constrs += 1
        self.output_var = output_var
        self._z = z

    def _exact_prediction(self, x: np.ndarray) -> float:
        return float(self.predictor.predict(x.reshape(1, -1))[0])


class LogisticRegressionConstr(AbstractPredictorConstr):
    """Embedding of binary ``sklearn.linear_model.LogisticRegression``.

    With ``output_type='probability_1'`` the output is P(class 1) = sigma(z),
    embedded through a certified PWL approximation of the sigmoid.
    With ``output_type='raw'`` the output is the linear score z itself
    (exact linear equality).
    """

    def __init__(self, h, predictor, input_vars, output_var=None,
                 output_type: str = "probability_1", pwl_tol: float = 0.01,
                 stats: Optional[PWLStats] = None, name: str = "log_reg",
                 refinable: bool = False):
        inputs = _coerce_inputs(predictor, input_vars)
        super().__init__(h, predictor, inputs, stats)
        if predictor.coef_.shape[0] != 1:
            raise ValueError("Only binary logistic regression is supported.")
        if output_type not in ("probability_1", "raw"):
            raise ValueError("output_type must be 'probability_1' or 'raw'.")
        self.output_type = output_type

        z = _linear_form(predictor.coef_[0], predictor.intercept_[0], inputs)
        self._z = z

        if output_type == "raw":
            if output_var is None:
                lo, hi = z.bounds()
                output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
                self.stats.n_vars += 1
            if z.is_constant():
                h.addConstr(output_var == z.const, name=f"{name}_eq")
            else:
                h.addConstr(output_var == z.to_highspy(), name=f"{name}_eq")
            self.stats.n_constrs += 1
        elif refinable:
            from ._refine import RefinablePWL
            emb = RefinablePWL(h, sigmoid, z, y=output_var, tol=pwl_tol,
                               name=name, stats=self.stats)
            self.refinable_embeddings.append(emb)
            output_var = emb.y
        else:
            output_var = add_pwl_constr(
                h, sigmoid, z, y=output_var, tol=pwl_tol, name=name, stats=self.stats
            )

        self.output_var = output_var

    def _exact_prediction(self, x: np.ndarray) -> float:
        if self.output_type == "raw":
            return float(self.predictor.decision_function(x.reshape(1, -1))[0])
        return float(self.predictor.predict_proba(x.reshape(1, -1))[0, 1])


class PLSRegressionConstr(AbstractPredictorConstr):
    """Exact embedding of ``sklearn.cross_decomposition.PLSRegression``.

    PLS prediction is linear: ``(X - x_mean) @ coef_.T + intercept_``,
    so it folds into one exact linear equality per output.
    """

    def __init__(self, h, predictor, input_vars, output_var=None,
                 stats: Optional[PWLStats] = None, name: str = "pls"):
        inputs = _coerce_inputs(predictor, input_vars)
        super().__init__(h, predictor, inputs, stats)
        coefs = np.atleast_2d(predictor.coef_)  # (n_targets, n_features)
        if coefs.shape[0] != 1:
            raise ValueError("Multi-target PLS regression is not supported yet.")
        intercept = float(np.atleast_1d(predictor.intercept_)[0]) - float(
            np.asarray(predictor._x_mean) @ coefs[0]
        )
        z = _linear_form(coefs[0], intercept, inputs)

        if output_var is None:
            lo, hi = z.bounds()
            output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
            self.stats.n_vars += 1
        if z.is_constant():
            h.addConstr(output_var == z.const, name=f"{name}_eq")
        else:
            h.addConstr(output_var == z.to_highspy(), name=f"{name}_eq")
        self.stats.n_constrs += 1
        self.output_var = output_var

    def _exact_prediction(self, x: np.ndarray) -> float:
        return float(np.atleast_1d(self.predictor.predict(x.reshape(1, -1)))[0])


class MLPRegressorConstr(AbstractPredictorConstr):
    """Embedding of ``sklearn.neural_network.MLPRegressor``.

    ReLU activations are modelled exactly with big-M constraints (one binary
    per neuron); tanh/logistic activations use certified PWL approximations.
    """

    def __init__(self, h, predictor, input_vars, output_var=None,
                 pwl_tol: float = 0.01, stats: Optional[PWLStats] = None,
                 name: str = "mlp"):
        from ._nn_common import embed_feedforward, link_output, normalize_activation

        inputs = _coerce_inputs(predictor, input_vars)
        super().__init__(h, predictor, inputs, stats)
        activation = predictor.activation
        if normalize_activation(activation) not in ("relu", "linear",
                                                    "sigmoid", "tanh"):
            raise ValueError(f"Unsupported MLP activation: {activation!r}.")

        layers = []
        n_layers = len(predictor.coefs_)
        for li, (W, b) in enumerate(zip(predictor.coefs_, predictor.intercepts_)):
            act = activation if li < n_layers - 1 else predictor.out_activation_
            layers.append((np.asarray(W), np.asarray(b), act))

        out = embed_feedforward(h, layers, inputs, pwl_tol, self.stats, name)
        if len(out) != 1:
            raise ValueError("Multi-output MLPs are not supported yet.")
        self.output_var = link_output(h, out[0], output_var, self.stats, name)

    def _exact_prediction(self, x: np.ndarray) -> float:
        return float(self.predictor.predict(x.reshape(1, -1))[0])


class StandardScalerStep:
    """Expression-level embedding of ``sklearn.preprocessing.StandardScaler``.

    Scaling is affine, so it is folded into the expressions directly --
    no variables or constraints are added.
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, inputs: list[Affine]) -> list[Affine]:
        return [
            (x - float(mu)) * (1.0 / float(sd))
            for x, mu, sd in zip(inputs, self.scaler.mean_, self.scaler.scale_)
        ]


class PipelineConstr(AbstractPredictorConstr):
    """Embedding of an sklearn ``Pipeline`` of supported steps."""

    def __init__(self, h, pipeline, input_vars, output_var=None,
                 output_type: Optional[str] = None, pwl_tol: float = 0.01,
                 stats: Optional[PWLStats] = None, name: str = "pipe",
                 refinable: bool = False):
        from sklearn.compose import ColumnTransformer
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            Ridge,
        )
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.tree import DecisionTreeRegressor

        from ._preprocessing import apply_transform_step

        inputs = _coerce_inputs(pipeline, input_vars)
        super().__init__(h, pipeline, inputs, stats)

        exprs = inputs
        self._final = None
        for step_name, step in pipeline.named_steps.items():
            if isinstance(step, (StandardScaler, PolynomialFeatures,
                                 ColumnTransformer)):
                exprs = apply_transform_step(step, exprs, h=h,
                                             bilinear_tol=pwl_tol)
            elif isinstance(step, (LinearRegression, Ridge)):
                self._final = LinearRegressionConstr(
                    h, step, exprs, output_var=output_var,
                    stats=self.stats, name=f"{name}_{step_name}")
            elif isinstance(step, PLSRegression):
                self._final = PLSRegressionConstr(
                    h, step, exprs, output_var=output_var,
                    stats=self.stats, name=f"{name}_{step_name}")
            elif isinstance(step, LogisticRegression):
                self._final = LogisticRegressionConstr(
                    h, step, exprs, output_var=output_var,
                    output_type=output_type or "probability_1", pwl_tol=pwl_tol,
                    stats=self.stats, name=f"{name}_{step_name}",
                    refinable=refinable)
                self.refinable_embeddings.extend(
                    self._final.refinable_embeddings)
            elif isinstance(step, MLPRegressor):
                self._final = MLPRegressorConstr(
                    h, step, exprs, output_var=output_var, pwl_tol=pwl_tol,
                    stats=self.stats, name=f"{name}_{step_name}")
            elif isinstance(step, (DecisionTreeRegressor, RandomForestRegressor,
                                   GradientBoostingRegressor)):
                from ._trees import (
                    DecisionTreeRegressorConstr,
                    GradientBoostingRegressorConstr,
                    RandomForestRegressorConstr,
                )
                # isinstance (not exact type) so subclasses like
                # ExtraTreeRegressor dispatch to the base embedding.
                cls = next(
                    constr for base, constr in (
                        (DecisionTreeRegressor, DecisionTreeRegressorConstr),
                        (RandomForestRegressor, RandomForestRegressorConstr),
                        (GradientBoostingRegressor,
                         GradientBoostingRegressorConstr),
                    ) if isinstance(step, base)
                )
                self._final = cls(
                    h, step, exprs, output_var=output_var,
                    stats=self.stats, name=f"{name}_{step_name}")
            else:
                raise ValueError(
                    f"Unsupported pipeline step {type(step).__name__!r}. "
                    "Supported transforms: StandardScaler, PolynomialFeatures "
                    "(affine terms), ColumnTransformer. Supported predictors: "
                    "LinearRegression, PLSRegression, LogisticRegression, "
                    "MLPRegressor, DecisionTreeRegressor, "
                    "RandomForestRegressor, GradientBoostingRegressor."
                )
        if self._final is None:
            raise ValueError("Pipeline contains no supported predictor step.")
        self.output_var = self._final.output_var

    def _exact_prediction(self, x: np.ndarray) -> float:
        if hasattr(self.predictor, "predict_proba") and isinstance(
            self._final, LogisticRegressionConstr
        ):
            if self._final.output_type == "raw":
                return float(self.predictor.decision_function(x.reshape(1, -1))[0])
            return float(self.predictor.predict_proba(x.reshape(1, -1))[0, 1])
        return float(self.predictor.predict(x.reshape(1, -1))[0])
