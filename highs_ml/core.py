"""Public entry point: ``add_predictor_constr``."""

from __future__ import annotations

from typing import Optional

from ._pwl import PWLStats
from ._predictors import (
    LinearRegressionConstr,
    LogisticRegressionConstr,
    MLPRegressorConstr,
    PipelineConstr,
)


def add_predictor_constr(
    highs_model,
    predictor,
    input_vars,
    output_var=None,
    output_type: Optional[str] = None,
    pwl_tol: float = 0.01,
    name: Optional[str] = None,
    refinable: bool = False,
):
    """Embed a trained scikit-learn predictor into a HiGHS model.

    Mirrors ``gurobi_ml.add_predictor_constr``. The relation

        output_var = predictor(input_vars)

    is added to ``highs_model`` (a ``highspy.Highs`` instance) using only
    linear constraints, integrality and (where needed) certified
    piecewise-linear approximations -- everything HiGHS can solve exactly.

    Args:
        highs_model: ``highspy.Highs`` model.
        predictor: fitted sklearn predictor. Supported:
            ``LinearRegression``, binary ``LogisticRegression``,
            ``MLPRegressor``, or a ``Pipeline`` of those with
            ``StandardScaler`` preprocessing steps.
        input_vars: mapping of feature name -> ``highs_var`` or constant,
            or a sequence of variables/constants in feature order. Features
            that are decision variables must have finite bounds.
        output_var: existing ``highs_var`` to link to the prediction, or
            ``None`` to create one (with tight bounds where possible).
        output_type: for logistic regression, ``'probability_1'`` (default)
            or ``'raw'`` (the linear score).
        pwl_tol: certified maximum interpolation error for piecewise-linear
            approximations of nonlinear functions (sigmoid, tanh).
        name: prefix for generated variables/constraints.

    Returns:
        A predictor-constraint object with ``output_var``, ``print_stats()``
        and ``get_error()`` (max deviation from the exact predictor at the
        current solution).
    """
    from sklearn.compose import ColumnTransformer  # noqa: F401
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import (
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor

    from ._keras import is_keras_model
    from ._onnx import is_onnx_model
    from ._predictors import PLSRegressionConstr

    try:
        import xgboost as xgb
        _xgb_types = (xgb.XGBRegressor, xgb.XGBClassifier, xgb.Booster)
    except ImportError:
        _xgb_types = ()
    try:
        import lightgbm as lgb
        _lgb_types = (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.Booster)
    except ImportError:
        _lgb_types = ()

    stats = PWLStats()
    label = name or "pred"

    if is_keras_model(predictor):
        from ._keras import KerasNetworkConstr
        return KerasNetworkConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            pwl_tol=pwl_tol, stats=stats, name=label,
        )
    if is_onnx_model(predictor):
        from ._onnx import ONNXNetworkConstr
        return ONNXNetworkConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            pwl_tol=pwl_tol, stats=stats, name=label,
        )
    if _xgb_types and isinstance(predictor, _xgb_types):
        from ._xgboost import XGBoostConstr
        return XGBoostConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            output_type=output_type or "probability_1", pwl_tol=pwl_tol,
            stats=stats, name=label,
        )
    if _lgb_types and isinstance(predictor, _lgb_types):
        from ._lightgbm import LightGBMConstr
        return LightGBMConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            output_type=output_type or "probability_1", pwl_tol=pwl_tol,
            stats=stats, name=label,
        )
    if isinstance(predictor, Pipeline):
        return PipelineConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            output_type=output_type, pwl_tol=pwl_tol, stats=stats, name=label,
            refinable=refinable,
        )
    if isinstance(predictor, (DecisionTreeRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)):
        from ._trees import (
            DecisionTreeRegressorConstr,
            GradientBoostingRegressorConstr,
            RandomForestRegressorConstr,
        )
        # isinstance (not exact type) so subclasses like ExtraTreeRegressor
        # dispatch to the base embedding; the three bases are disjoint.
        cls = next(
            constr for base, constr in (
                (DecisionTreeRegressor, DecisionTreeRegressorConstr),
                (RandomForestRegressor, RandomForestRegressorConstr),
                (GradientBoostingRegressor, GradientBoostingRegressorConstr),
            ) if isinstance(predictor, base)
        )
        return cls(
            highs_model, predictor, input_vars, output_var=output_var,
            stats=stats, name=label,
        )
    if isinstance(predictor, (LinearRegression, Ridge)):
        return LinearRegressionConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            stats=stats, name=label,
        )
    if isinstance(predictor, PLSRegression):
        return PLSRegressionConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            stats=stats, name=label,
        )
    if isinstance(predictor, LogisticRegression):
        return LogisticRegressionConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            output_type=output_type or "probability_1", pwl_tol=pwl_tol,
            stats=stats, name=label, refinable=refinable,
        )
    if isinstance(predictor, MLPRegressor):
        return MLPRegressorConstr(
            highs_model, predictor, input_vars, output_var=output_var,
            pwl_tol=pwl_tol, stats=stats, name=label,
        )
    raise ValueError(
        f"Unsupported predictor type {type(predictor).__name__!r}. Supported: "
        "LinearRegression, PLSRegression, binary LogisticRegression, "
        "MLPRegressor, DecisionTreeRegressor, RandomForestRegressor, "
        "GradientBoostingRegressor, XGBRegressor/XGBClassifier/Booster "
        "(requires xgboost), LGBMRegressor/LGBMClassifier/Booster (requires "
        "lightgbm), Keras dense networks, ONNX ModelProto feedforward "
        "networks, and Pipelines of sklearn predictors with supported "
        "preprocessing."
    )
