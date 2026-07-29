"""highs_ml -- embed trained machine-learning models into HiGHS optimization.

Gurobi Machine Learning-style capabilities
(https://github.com/Gurobi/gurobi-machinelearning) for the MIT-licensed
HiGHS solver (https://github.com/ERGO-Code/HiGHS).

Example
-------
>>> import highspy
>>> from highs_ml import add_predictor_constr
>>> h = highspy.Highs()
>>> x = h.addVariable(lb=0.0, ub=2.5, name="merit")
>>> y = h.addVariable(lb=0.0, ub=1.0, name="prob")
>>> pred = add_predictor_constr(h, pipe, {"merit": x, "SAT": 1300, "GPA": 3.4}, y)
>>> h.maximize(y)
>>> pred.get_error()  # certified approximation error at the solution
"""

from .core import add_predictor_constr
from ._predictors import (
    AbstractPredictorConstr,
    LinearRegressionConstr,
    LogisticRegressionConstr,
    MLPRegressorConstr,
    PipelineConstr,
)
from ._trees import (
    DecisionTreeRegressorConstr,
    GradientBoostingRegressorConstr,
    RandomForestRegressorConstr,
)
from ._xgboost import XGBoostConstr
from ._lightgbm import LightGBMConstr
from ._keras import KerasNetworkConstr
from ._onnx import ONNXNetworkConstr
from ._predictors import PLSRegressionConstr
from .decomp import DecompResult, solve_decomposed
from .dw import DWResult, solve_bp, solve_dw
from .auto import solve_auto
from ._bilinear import (
    PiecewiseBilinear,
    add_adaptive_bilinear,
    add_bilinear_constr,
)
from ._refine import RefinablePWL, solve_adaptive

__version__ = "1.0.0"
__all__ = [
    "add_predictor_constr",
    "add_bilinear_constr",
    "add_adaptive_bilinear",
    "PiecewiseBilinear",
    "RefinablePWL",
    "solve_adaptive",
    "solve_decomposed",
    "DecompResult",
    "solve_dw",
    "solve_bp",
    "solve_auto",
    "DWResult",
    "AbstractPredictorConstr",
    "LinearRegressionConstr",
    "PLSRegressionConstr",
    "LogisticRegressionConstr",
    "MLPRegressorConstr",
    "PipelineConstr",
    "DecisionTreeRegressorConstr",
    "RandomForestRegressorConstr",
    "GradientBoostingRegressorConstr",
    "XGBoostConstr",
    "LightGBMConstr",
    "KerasNetworkConstr",
    "ONNXNetworkConstr",
    "__version__",
]
