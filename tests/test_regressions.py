"""Regression tests for fixed highs_ml findings (one test per finding).

Runnable with pytest or directly: python tests/test_regressions.py
"""

import importlib.util
import inspect
import math
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd
import highspy

try:  # pytest is optional: the file is also runnable directly.
    import pytest
except ImportError:
    pytest = None
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from highs_ml import (  # noqa: E402
    RefinablePWL,
    add_adaptive_bilinear,
    add_bilinear_constr,
    add_predictor_constr,
    solve_adaptive,
    solve_bp,
    solve_decomposed,
    solve_dw,
)
from highs_ml._affine import Affine  # noqa: E402
from highs_ml._predictors import sigmoid  # noqa: E402
from highs_ml._pwl import adaptive_breakpoints  # noqa: E402

kOptimal = highspy.HighsModelStatus.kOptimal


def _quiet_highs():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def _optional_import(name):
    """Import an optional dependency; skip the test if it is missing."""
    if pytest is not None:
        return pytest.importorskip(name)
    try:
        return __import__(name)
    except ImportError as exc:
        raise unittest.SkipTest(f"{name} not available: {exc}") from exc


def _assert_raises(exc_type, fn, match=None):
    """pytest-free raises helper (the file is also runnable directly)."""
    try:
        fn()
    except exc_type as exc:
        if match is not None:
            assert match in str(exc), (match, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


# ----------------------------------------------------------------------
# shared block-angular model builders (findings W1-W9)
# ----------------------------------------------------------------------
def _build_weighted(weights, budget=10.0, offset=None):
    """One block per weight: x_i in [0,10] gated by binary u_i, cost -x_i;
    coupling row sum w_i * x_i <= budget."""
    h = _quiet_highs()
    xs = []
    for i, w in enumerate(weights):
        x = h.addVariable(lb=0.0, ub=10.0, obj=-1.0, name=f"x{i}")
        u = h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                          name=f"u{i}")
        h.addConstr(x <= 10.0 * u)
        xs.append(x)
    h.addConstr(sum(float(w) * x for w, x in zip(weights, xs)) <= budget)
    h.setMinimize()
    if offset is not None:
        h.changeObjectiveOffset(offset)
    return h


def _build_knapsack(offset=None, z_cost=None, z_ub=2.0):
    """Fractional-root knapsack (forces branching): per block x_i in [0,1]
    gated by binary u_i, cost -x_i; coupling sum w_i x_i (+ z) <= 8 with
    weights (6,5,4,3). Optional linking column z (coupling row only)."""
    h = _quiet_highs()
    weights = (6.0, 5.0, 4.0, 3.0)
    xs = []
    for i, w in enumerate(weights):
        x = h.addVariable(lb=0.0, ub=1.0, obj=-1.0, name=f"x{i}")
        u = h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                          name=f"u{i}")
        h.addConstr(x <= u)
        xs.append(x)
    coupling = sum(float(w) * x for w, x in zip(weights, xs))
    if z_cost is not None:
        z = h.addVariable(lb=0.0, ub=z_ub, obj=z_cost,
                          type=highspy.HighsVarType.kInteger, name="z")
        coupling = coupling + z
    h.addConstr(coupling <= 8.0)
    h.setMinimize()
    if offset is not None:
        h.changeObjectiveOffset(offset)
    return h


def _build_cutting_stock(n_rolls=5):
    """Tiny cutting stock: CG needs several iterations to converge, so a
    small max_iterations genuinely truncates column generation."""
    h = _quiet_highs()
    y1s, y2s = [], []
    for r in range(n_rolls):
        used = h.addVariable(lb=0.0, ub=1.0, obj=1.0,
                             type=highspy.HighsVarType.kInteger, name=f"u{r}")
        y1 = h.addVariable(lb=0.0, ub=3.0,
                           type=highspy.HighsVarType.kInteger, name=f"y1_{r}")
        y2 = h.addVariable(lb=0.0, ub=3.0,
                           type=highspy.HighsVarType.kInteger, name=f"y2_{r}")
        h.addConstr(3.0 * y1 + 2.0 * y2 <= 7.0 * used)
        y1s.append(y1)
        y2s.append(y2)
    h.addConstr(sum(y1s) >= 5.0)
    h.addConstr(sum(y2s) >= 4.0)
    h.setMinimize()
    return h


def _direct_optimum(build):
    h = build()
    h.run()
    assert h.getModelStatus() == kOptimal
    return h.getObjectiveValue()


# ----------------------------------------------------------------------
# K1: standalone keras Activation must bind to the PREVIOUS Dense layer
# ----------------------------------------------------------------------
def test_k1_keras_standalone_activation_binding():
    import os
    os.environ.setdefault("KERAS_BACKEND", "jax")
    keras = _optional_import("keras")

    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(3),
        keras.layers.Activation("relu"),
        keras.layers.Dense(1),
    ])
    W0 = np.array([[1.0, -2.0, 0.5], [0.7, 1.5, -1.0]])
    b0 = np.array([0.3, -0.2, 0.1])
    W1 = np.array([[-2.0], [-1.0], [-0.5]])
    b1 = np.array([-0.25])
    model.set_weights([W0, b0, W1, b1])

    target = np.array([1.2, -0.7])
    true_val = float((np.maximum(0.0, target @ W0 + b0) @ W1 + b1)[0])
    wrong_val = max(0.0, true_val)  # Activation mis-bound to the next Dense
    assert true_val < -0.5  # the two semantics are far apart at this point
    exact = float(model.predict(target.reshape(1, -1), verbose=0).ravel()[0])
    assert abs(exact - true_val) < 1e-4, (exact, true_val)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, model, xs, y)
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    got = h.getSolution().col_value[y.index]
    assert abs(got - exact) < 1e-4, (got, exact)
    assert abs(got - wrong_val) > 0.5  # not the pre-fix mis-binding
    assert pc.get_error() < 1e-4

    # Trailing Activation after the last Dense (the most common pattern)
    # previously raised a ValueError; it must embed and match predict.
    model2 = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(3, activation="relu"),
        keras.layers.Dense(1),
        keras.layers.Activation("sigmoid"),
    ])
    model2.set_weights([W0, b0, W1, b1])
    h = _quiet_highs()
    xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=0.0, ub=1.0, name="p")
    pc2 = add_predictor_constr(h, model2, xs, y, pwl_tol=1e-3)
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    exact2 = float(model2.predict(target.reshape(1, -1), verbose=0).ravel()[0])
    got2 = h.getSolution().col_value[y.index]
    assert abs(got2 - exact2) < 2e-3, (got2, exact2)
    assert pc2.get_error() < 2e-3
    print("ok K1 keras standalone Activation binding")


# ----------------------------------------------------------------------
# D1: solve_decomposed must not drop a quadratic (Hessian) objective
# ----------------------------------------------------------------------
def test_d1_decomp_quadratic_objective_not_dropped():
    h = _quiet_highs()
    xs = [h.addVariable(lb=-10.0, ub=10.0, obj=-2.0, name=f"x{i}")
          for i in range(4)]
    for x in xs:
        h.addConstr(x <= 10.0)  # one row per block (independent blocks)
    h.setMinimize()
    hess = highspy.HighsHessian()
    hess.dim_ = 4
    hess.start_ = np.array([0, 1, 2, 3, 4])
    hess.index_ = np.array([0, 1, 2, 3])
    hess.value_ = np.array([2.0, 2.0, 2.0, 2.0])
    h.passHessian(hess)  # objective: sum x_i^2 - 2 x_i, argmin x_i = 1

    res = solve_decomposed(h)
    assert res.status == kOptimal, res.note
    # Pre-fix: blocks re-solved as pure LPs on col_cost_ only -> objective
    # -80 at x = 10. The QP must fall back to a direct solve instead.
    assert not res.decomposed, res.note
    assert abs(res.objective - (-4.0)) < 1e-6, res.objective
    assert np.allclose(res.col_value, 1.0, atol=1e-5)
    print(f"ok D1 quadratic objective (obj {res.objective:.6f})")


# ----------------------------------------------------------------------
# D2: stitched decomposed solution must be installed back into h
# ----------------------------------------------------------------------
def test_d2_decomp_installs_solution_for_get_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 1))
    yv = 2.0 * X[:, 0] + 0.7
    reg = LinearRegression().fit(X, yv)

    h = _quiet_highs()
    pcs = []
    targets = [0.5, 1.0, 1.5]
    for i, t in enumerate(targets):
        x = h.addVariable(lb=-4.0, ub=4.0, name=f"x{i}")
        y = h.addVariable(lb=-100.0, ub=100.0, obj=1.0, name=f"y{i}")
        pcs.append(add_predictor_constr(h, reg, [x], y, name=f"p{i}"))
        h.addConstr(x == t)
    h.setMaximize()

    res = solve_decomposed(h)
    assert res.status == kOptimal, res.note
    assert res.decomposed, res.note
    # Pre-fix: value_valid stayed False and get_error() raised RuntimeError.
    assert h.getSolution().value_valid
    errs = [pc.get_error() for pc in pcs]
    assert max(errs) < 1e-6, errs
    expected = sum(2.0 * t + 0.7 for t in targets)
    assert abs(res.objective - expected) < 1e-6, (res.objective, expected)
    print(f"ok D2 decomp solution installed (max error {max(errs):.2e})")


# ----------------------------------------------------------------------
# O1: ONNX Gemm with constant-first operands and transB
# ----------------------------------------------------------------------
def test_o1_onnx_gemm_constant_first_and_transpose():
    onnx = _optional_import("onnx")
    from onnx import helper, numpy_helper

    rng = np.random.default_rng(11)
    W1 = rng.normal(size=(4, 3))  # non-square: pre-fix crashed on Bm[i, j]
    C1 = rng.normal(size=4)
    W2 = rng.normal(size=(1, 4))  # applied transposed via transB
    C2 = rng.normal(size=1)

    nodes = [
        helper.make_node("Gemm", ["W1", "X", "C1"], ["H1"]),  # constant FIRST
        helper.make_node("Relu", ["H1"], ["A1"]),
        helper.make_node("Gemm", ["A1", "W2", "C2"], ["Y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "const_first_net",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(C1, "C1"),
         numpy_helper.from_array(W2, "W2"), numpy_helper.from_array(C2, "C2")],
    )
    model = helper.make_model(graph)

    worst = 0.0
    for target in [(-2.0, 0.5, 1.0), (0.3, -1.0, 2.0)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(3)]
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        pc = add_predictor_constr(h, model, xs, y)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == kOptimal
        sol = h.getSolution().col_value
        t = np.array(target)
        exact = float((np.maximum(0, W1 @ t + C1) @ W2.T + C2)[0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-5, (target, pc.get_error())
    assert worst < 1e-5, worst
    print(f"ok O1 onnx const-first Gemm (max error {worst:.2e})")


# ----------------------------------------------------------------------
# O2: ONNX input variable count validated against the graph input dim
# ----------------------------------------------------------------------
def test_o2_onnx_input_count_validated():
    onnx = _optional_import("onnx")
    from onnx import helper, numpy_helper

    rng = np.random.default_rng(13)
    B1, C1 = rng.normal(size=(3, 1)), rng.normal(size=1)
    nodes = [helper.make_node("Gemm", ["X", "B1", "C1"], ["Y"])]
    graph = helper.make_graph(
        nodes, "three_input_net",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(B1, "B1"), numpy_helper.from_array(C1, "C1")],
    )
    model = helper.make_model(graph)

    # Too few vars: pre-fix silently used only the first weight rows.
    h = _quiet_highs()
    xs2 = [h.addVariable(lb=-3.0, ub=3.0) for _ in range(2)]
    _assert_raises(ValueError,
                   lambda: add_predictor_constr(h, model, xs2),
                   match="Expected 3 inputs, got 2")
    # Too many vars: pre-fix died with a bare IndexError inside Gemm.
    h = _quiet_highs()
    xs4 = [h.addVariable(lb=-3.0, ub=3.0) for _ in range(4)]
    _assert_raises(ValueError,
                   lambda: add_predictor_constr(h, model, xs4),
                   match="Expected 3 inputs, got 4")
    # The right count still embeds correctly.
    h = _quiet_highs()
    xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(3)]
    y = h.addVariable(lb=-1e4, ub=1e4, name="y")
    pc = add_predictor_constr(h, model, xs, y)
    target = np.array([0.4, -1.0, 2.0])
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    exact = float((target @ B1 + C1)[0])
    got = h.getSolution().col_value[y.index]
    assert abs(got - exact) < 1e-6, (got, exact)
    assert pc.get_error() < 1e-6
    print("ok O2 onnx input count validated")


# ----------------------------------------------------------------------
# X1: XGBoost early stopping honored (best_iteration trees only)
# ----------------------------------------------------------------------
def test_x1_xgboost_early_stopping_embeds_best_iteration():
    xgb = _optional_import("xgboost")
    rng = np.random.default_rng(42)
    X = rng.uniform(-4, 4, size=(400, 2))
    yv = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
          + rng.normal(scale=0.5, size=400))
    model = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.3,
                             early_stopping_rounds=5, random_state=17)
    model.fit(X[:300], yv[:300], eval_set=[(X[300:], yv[300:])], verbose=False)
    booster = model.get_booster()
    assert model.best_iteration + 1 < booster.num_boosted_rounds()

    target = np.array([1.3, -0.8])
    pred = float(model.predict(target.reshape(1, -1))[0])  # best-iter trees
    all_trees = float(booster.predict(xgb.DMatrix(target.reshape(1, -1)),
                                      output_margin=True)[0])
    assert abs(pred - all_trees) > 1e-4  # the two references genuinely differ

    h = _quiet_highs()
    xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=-1e4, ub=1e4, name="y")
    pc = add_predictor_constr(h, model, xs, y, output_type="raw")
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    got = h.getSolution().col_value[y.index]
    # Pre-fix: all 500 trees embedded -> matches all_trees, not predict.
    assert abs(got - pred) < 1e-5, (got, pred)
    assert pc.get_error() < 1e-5
    print(f"ok X1 xgboost early stopping (best_iteration "
          f"{model.best_iteration} of {booster.num_boosted_rounds()})")


# ----------------------------------------------------------------------
# X2: non-identity-link XGBoost objectives rejected
# ----------------------------------------------------------------------
def test_x2_xgboost_unsupported_objective_rejected():
    xgb = _optional_import("xgboost")
    rng = np.random.default_rng(7)
    X = rng.uniform(-4, 4, size=(200, 2))
    yp = rng.poisson(lam=3.0, size=200).astype(float)
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2,
                             objective="count:poisson").fit(X, yp)
    h = _quiet_highs()
    xs = [h.addVariable(lb=-5.0, ub=5.0) for _ in range(2)]
    # Pre-fix: the raw margin was embedded silently (log link ignored).
    _assert_raises(ValueError, lambda: add_predictor_constr(h, model, xs),
                   match="Unsupported XGBoost objective")
    print("ok X2 xgboost objective whitelist")


# ----------------------------------------------------------------------
# X3: DataFrame-fitted XGBoost models embed (DMatrix feature names)
# ----------------------------------------------------------------------
def test_x3_xgboost_dataframe_fitted_model_embeds():
    xgb = _optional_import("xgboost")
    rng = np.random.default_rng(3)
    X = rng.uniform(-4, 4, size=(300, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    df = pd.DataFrame(X, columns=["a", "b"])
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2,
                             random_state=3).fit(df, yv)

    h = _quiet_highs()
    va = h.addVariable(lb=-5.0, ub=5.0, name="a")
    vb = h.addVariable(lb=-5.0, ub=5.0, name="b")
    y = h.addVariable(lb=-1e4, ub=1e4, name="y")
    # Pre-fix: crashed at embedding time with ValueError about training
    # data fields (DMatrix built without feature names).
    pc = add_predictor_constr(h, model, {"a": va, "b": vb}, y,
                              output_type="raw")
    h.addConstr(va == 1.3)
    h.addConstr(vb == -0.8)
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    got = h.getSolution().col_value[y.index]
    pred = float(model.predict(pd.DataFrame([[1.3, -0.8]],
                                            columns=["a", "b"]))[0])
    assert abs(got - pred) < 1e-5, (got, pred)
    assert pc.get_error() < 1e-5
    print(f"ok X3 xgboost DataFrame-fitted model (error {pc.get_error():.2e})")


# ----------------------------------------------------------------------
# T1: multi-output trees raise a curated error instead of output 0 only
# ----------------------------------------------------------------------
def test_t1_multioutput_trees_raise():
    rng = np.random.default_rng(0)
    X = rng.uniform(-4, 4, size=(120, 2))
    Y2 = np.column_stack([np.sin(X[:, 0]), X[:, 1] ** 2])

    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, Y2)
    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0) for _ in range(2)]
    # Pre-fix: embedded output 0 silently, get_error() crashed later.
    _assert_raises(ValueError, lambda: add_predictor_constr(h, tree, xs),
                   match="Multi-output decision trees")

    rf = RandomForestRegressor(n_estimators=3, max_depth=3,
                               random_state=0).fit(X, Y2)
    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0) for _ in range(2)]
    _assert_raises(ValueError, lambda: add_predictor_constr(h, rf, xs),
                   match="Multi-output random forests")
    print("ok T1 multi-output trees rejected")


# ----------------------------------------------------------------------
# T2: constant feature inside the epsilon sliver routes like sklearn
# ----------------------------------------------------------------------
def test_t2_constant_feature_epsilon_sliver():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(-4, 4, size=80),
                         np.repeat([1.5, 2.5], 40)])
    yv = np.where(X[:, 1] <= 2.0, 3.0, 23.0)
    tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, yv)
    # midpoint of {1.5, 2.5}: the split is exactly 2.0 on feature 1
    assert tree.tree_.feature[0] == 1
    assert tree.tree_.threshold[0] == 2.0

    # 2.00005 sits in the (thr, thr+eps] sliver: pre-fix both children were
    # violated and the model went infeasible; sklearn routes it right.
    for const in (2.00005, 2.0, 1.99995):
        h = _quiet_highs()
        xv = h.addVariable(lb=-4.0, ub=4.0, name="x0")
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        add_predictor_constr(h, tree, [xv, const], y)
        h.maximize(y)
        assert h.getModelStatus() == kOptimal, (const, h.getModelStatus())
        got = h.getSolution().col_value[y.index]
        exact = float(tree.predict(np.array([[0.0, const]]))[0])
        assert abs(got - exact) < 1e-9, (const, got, exact)
    print("ok T2 constant-feature routing at the split threshold")


# ----------------------------------------------------------------------
# P1: adaptive_breakpoints raises when tol is unmet at max_points
# ----------------------------------------------------------------------
def test_p1_adaptive_breakpoints_certificate_or_error():
    # Pre-fix: returned 64 points with actual chord error 2.76e-4 silently.
    _assert_raises(
        ValueError,
        lambda: adaptive_breakpoints(sigmoid, -8.0, 8.0, 1e-4, max_points=64),
        match="max_points=64")

    # Default max_points must return a genuine certificate: verify the
    # chord error on a dense grid per segment.
    breaks = adaptive_breakpoints(sigmoid, -8.0, 8.0, 1e-4)
    worst = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        grid = np.linspace(a, b, 200)
        fa, fb = sigmoid(a), sigmoid(b)
        chord = fa + (fb - fa) * (grid - a) / (b - a)
        vals = np.array([sigmoid(float(g)) for g in grid])
        worst = max(worst, float(np.max(np.abs(vals - chord))))
    assert worst <= 1e-4 + 1e-12, worst
    print(f"ok P1 PWL certificate honest ({len(breaks)} points, "
          f"verified err {worst:.2e})")


# ----------------------------------------------------------------------
# P2: fixed-input predictor returns a real output variable
# ----------------------------------------------------------------------
def test_p2_fixed_input_returns_real_output_var():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 3))
    z = X @ np.array([1.0, -1.0, 0.5])
    yv = (1.0 / (1.0 + np.exp(-z)) > rng.random(200)).astype(int)
    logreg = LogisticRegression().fit(X, yv)

    h = _quiet_highs()
    pc = add_predictor_constr(h, logreg, [1.0, 2.0, 3.0])
    # Pre-fix: pc.output_var was a raw float and get_error() raised
    # AttributeError: 'float' object has no attribute 'index'.
    assert not isinstance(pc.output_var, float), type(pc.output_var)
    assert hasattr(pc.output_var, "index")
    h.maximize(pc.output_var)
    assert h.getModelStatus() == kOptimal
    exact = float(logreg.predict_proba(np.array([[1.0, 2.0, 3.0]]))[0, 1])
    got = h.getSolution().col_value[pc.output_var.index]
    assert abs(got - exact) < 1e-9, (got, exact)
    assert pc.get_error() < 1e-9
    print(f"ok P2 fixed-input output var (error {pc.get_error():.2e})")


# ----------------------------------------------------------------------
# A1: mixing variables from different Highs models raises
# ----------------------------------------------------------------------
def test_a1_cross_model_variables_rejected():
    hA = _quiet_highs()
    hB = _quiet_highs()
    a0 = hA.addVariable(lb=0.0, ub=1.0)
    b0 = hB.addVariable(lb=5.0, ub=9.0)  # same column index, other model

    # Pre-fix: terms.get(var) silently merged the two features into one
    # coefficient and bounds() read the wrong model.
    _assert_raises(ValueError,
                   lambda: Affine.coerce(a0) + Affine.coerce(b0),
                   match="different")
    rng = np.random.default_rng(9)
    reg = LinearRegression().fit(rng.normal(size=(50, 2)),
                                 rng.normal(size=50))
    _assert_raises(ValueError,
                   lambda: add_predictor_constr(hA, reg, [a0, b0]),
                   match="different")
    print("ok A1 cross-model variables rejected")


# ----------------------------------------------------------------------
# C1: sklearn subclasses dispatch via isinstance, not exact type
# ----------------------------------------------------------------------
def test_c1_subclass_dispatch_and_curated_error():
    rng = np.random.default_rng(2)
    X = rng.uniform(-4, 4, size=(150, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    et = ExtraTreeRegressor(max_depth=3, random_state=1).fit(X, yv)

    # Pre-fix: bare KeyError from the exact-type dict dispatch.
    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0) for _ in range(2)]
    y = h.addVariable(lb=-1e4, ub=1e4)
    pc = add_predictor_constr(h, et, xs, y)
    h.addConstr(xs[0] == 1.0)
    h.addConstr(xs[1] == -1.0)
    h.maximize(y)
    assert h.getModelStatus() == kOptimal
    got = h.getSolution().col_value[y.index]
    exact = float(et.predict(np.array([[1.0, -1.0]]))[0])
    assert abs(got - exact) < 1e-6, (got, exact)
    assert pc.get_error() < 1e-6

    # Genuinely unsupported types still get the curated ValueError.
    from sklearn.svm import SVR
    svr = SVR().fit(X, yv)
    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0) for _ in range(2)]
    _assert_raises(ValueError, lambda: add_predictor_constr(h, svr, xs),
                   match="Unsupported predictor type")
    print("ok C1 subclass dispatch (ExtraTreeRegressor embeds)")


# ----------------------------------------------------------------------
# S1: the main test module's optional-dep gating works under pytest
# ----------------------------------------------------------------------
def test_s1_optional_dep_gating_no_nameerror():
    path = pathlib.Path(__file__).resolve().parent / "test_highs_ml.py"
    spec = importlib.util.spec_from_file_location("_s1_test_highs_ml", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pre-fix: 'pytest' was referenced but never imported, so the gated
    # tests raised NameError under a pytest run. The name must be bound
    # at module level (module or None) and the helper must exist.
    assert hasattr(mod, "pytest")
    assert callable(getattr(mod, "_optional_import", None))
    # Present dependency: the module is returned.
    assert mod._optional_import("math") is math
    # Missing dependency: a skip-type exception, never NameError.
    skip_types = (unittest.SkipTest,)
    if mod.pytest is not None:
        skip_types = skip_types + (mod.pytest.skip.Exception,)
    try:
        mod._optional_import("_no_such_module_highs_ml_")
    except skip_types:
        pass
    else:
        raise AssertionError("missing optional dep did not raise a skip")
    print("ok S1 optional-dependency gating")


# ----------------------------------------------------------------------
# B1: mixed-sign bilinear products are not clamped to >= 0
# ----------------------------------------------------------------------
def test_b1_bilinear_mixed_sign_product():
    h = _quiet_highs()
    x1 = h.addVariable(lb=0.5, ub=2.0, name="x1")
    x2 = h.addVariable(lb=-2.0, ub=-0.5, name="x2")
    y = add_bilinear_constr(h, x1, x2, tol=0.01)
    h.addConstr(x1 == 1.0)
    h.addConstr(x2 == -1.0)
    h.minimize(y)
    # Pre-fix: segment product vars defaulted to lb=0, cutting off the
    # negative region -> kInfeasible although the true product is -1.
    assert h.getModelStatus() == kOptimal
    got = h.getSolution().col_value[y.index]
    assert abs(got - (-1.0)) <= 0.011, got
    print(f"ok B1 mixed-sign bilinear product (y {got:.4f})")


# ----------------------------------------------------------------------
# B2: piecewise-McCormick segment count is bounded (fast error)
# ----------------------------------------------------------------------
def test_b2_piecewise_mccormick_segment_guard():
    import time
    h = _quiet_highs()
    w1 = h.addVariable(lb=0.0, ub=1600.0, name="w1")
    w2 = h.addVariable(lb=0.0, ub=1600.0, name="w2")
    t0 = time.perf_counter()
    # Pre-fix: n_seg = 64,000,000 segments were attempted (hang/OOM).
    exc = _assert_raises(ValueError,
                         lambda: add_bilinear_constr(h, w1, w2, tol=0.01),
                         match="segments")
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, elapsed
    assert "1024" in str(exc)
    print(f"ok B2 segment-count guard ({elapsed * 1000:.1f} ms)")


# ----------------------------------------------------------------------
# R1: solve_adaptive honors each embedding's OWN tolerance
# ----------------------------------------------------------------------
def test_r1_solve_adaptive_per_embedding_tolerance():
    h = _quiet_highs()
    u = h.addVariable(lb=-2.0, ub=2.0, name="u")
    v = h.addVariable(lb=-2.0, ub=2.0, name="v")
    bil = add_adaptive_bilinear(h, u, v, tol=0.1)  # loose embedding
    z = h.addVariable(lb=-4.0, ub=4.0, name="z")
    h.addConstr(z == 0.7)
    sig = RefinablePWL(h, sigmoid, z, tol=1e-3, name="sig")  # tight one
    h.addConstr(u + v == -1.0)
    h.maximize(sig.y + 0.1 * bil.y)

    status, worst, n_ref = solve_adaptive(h, [bil, sig], max_refines=40)
    assert status == kOptimal
    # Pre-fix: the loop stopped once the global worst error was under the
    # LOOSEST tol (0.1), leaving the sigmoid off by up to 100x its tol.
    assert sig.solution_error() <= 1e-3 + 1e-9, sig.solution_error()
    assert bil.solution_error() <= 0.1 + 1e-9, bil.solution_error()
    print(f"ok R1 per-embedding tolerances (sigmoid err "
          f"{sig.solution_error():.2e}, {n_ref} refinements)")


# ----------------------------------------------------------------------
# W1: pricing classes distinguish coupling-row coefficients
# ----------------------------------------------------------------------
def test_w1_pricing_classes_respect_coupling_coefficients():
    weights = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    ref = _direct_optimum(lambda: _build_weighted(weights))
    assert abs(ref - (-10.0)) < 1e-9  # put everything on the weight-1 block

    res = solve_dw(_build_weighted(weights), max_iterations=60)
    # Pre-fix: all six blocks shared one pricing class -> objective 0.0
    # with bound -1.667.
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    assert res.bound <= res.objective + 1e-6
    assert res.n_unique_classes == 6, res.n_unique_classes
    print(f"ok W1 coupling-aware pricing classes (obj {res.objective:.2f})")


# ----------------------------------------------------------------------
# W2: CG master solved as an LP -> valid duals, fast convergence
# ----------------------------------------------------------------------
def _build_integer_linking():
    h = _quiet_highs()
    xs = []
    for i in range(4):
        x = h.addVariable(lb=0.0, ub=10.0, obj=-1.0, name=f"x{i}")
        u = h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                          name=f"u{i}")
        h.addConstr(x <= 10.0 * u)
        xs.append(x)
    z = h.addVariable(lb=0.0, ub=4.0, obj=-3.0,
                      type=highspy.HighsVarType.kInteger, name="z")
    h.addConstr(sum(xs) + 2.0 * z <= 12.0)  # z only in the coupling row
    h.setMinimize()
    return h


def test_w2_integer_linking_column_cg_converges():
    ref = _direct_optimum(_build_integer_linking)
    res = solve_bp(_build_integer_linking(), max_iterations=40,
                   node_budget=16)
    # Pre-fix: the MIP master yielded zero duals, the same column was
    # re-added until max_iterations solves burnt out.
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    assert res.bound <= res.objective + 1e-6
    assert res.iterations < 40, res.iterations
    assert abs(res.col_value[8] - 4.0) < 1e-6  # linking z recovered
    print(f"ok W2 integer linking column ({res.iterations} CG iterations)")


# ----------------------------------------------------------------------
# W3: objective offset shifts objective AND bound identically
# ----------------------------------------------------------------------
def test_w3_objective_offset_consistent():
    r0 = solve_bp(_build_knapsack(), node_budget=32)
    r1 = solve_bp(_build_knapsack(offset=50.0), node_budget=32)
    assert r0.status == kOptimal, r0.note
    assert r1.status == kOptimal, r1.note
    # Pre-fix: incumbent-vs-bound comparisons mixed shifted and unshifted
    # values, prematurely pruning optimal nodes / accepting worse ones.
    assert abs((r1.objective - r0.objective) - 50.0) < 1e-6
    assert abs((r1.bound - r0.bound) - 50.0) < 1e-6
    assert np.allclose(r0.col_value, r1.col_value, atol=1e-6)
    print(f"ok W3 objective offset (obj {r0.objective:.2f} -> "
          f"{r1.objective:.2f})")


# ----------------------------------------------------------------------
# W4: truncated CG never reports an invalid (too tight) DW bound
# ----------------------------------------------------------------------
def test_w4_truncated_cg_bound_stays_valid():
    ref = _direct_optimum(_build_cutting_stock)
    full = solve_dw(_build_cutting_stock(), max_iterations=60)
    assert full.status == kOptimal, full.note
    assert abs(full.objective - ref) < 1e-6, (full.objective, ref)
    assert full.bound <= ref + 1e-6

    saw_truncated_optimal = False
    for cap in range(1, max(2, full.iterations)):
        res = solve_dw(_build_cutting_stock(), max_iterations=cap)
        if res.status == kOptimal:
            saw_truncated_optimal = True
            # Pre-fix: restricted-master value reported as the bound,
            # overshooting the true optimum. Min sense: bound <= optimum.
            assert res.bound <= ref + 1e-6, (cap, res.bound, ref)
            assert res.bound <= res.objective + 1e-6
    assert saw_truncated_optimal  # the check above actually fired

    # solve_bp with truncated CG must not claim proven optimality falsely.
    for cap in (1, 2, 3):
        res = solve_bp(_build_cutting_stock(), max_iterations=cap,
                       node_budget=6)
        if res.status == kOptimal:
            assert res.bound <= ref + 1e-6, (cap, res.bound, ref)
            if "proven optimal" in res.note:
                assert abs(res.objective - ref) < 1e-6, res.note
    print("ok W4 truncated CG bounds stay valid")


# ----------------------------------------------------------------------
# W5: dual-sign calibration heuristic gone; branched B&P is exact
# ----------------------------------------------------------------------
def test_w5_no_dual_calibration_branched_bp_exact():
    import highs_ml.dw as dw_mod
    # The misfiring _calibrate_dual_sign heuristic must not resurface.
    assert "calibrate" not in inspect.getsource(dw_mod).lower()

    ref = _direct_optimum(_build_knapsack)  # fractional root -> branches
    res = solve_bp(_build_knapsack(), node_budget=32)
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    assert res.bound <= res.objective + 1e-6
    print(f"ok W5 branched B&P exact without calibration "
          f"(obj {res.objective:.2f})")


# ----------------------------------------------------------------------
# W6: best_bound is never the incumbent (gap >= 0, bound valid)
# ----------------------------------------------------------------------
def test_w6_bound_never_falls_back_to_incumbent():
    ref = _direct_optimum(
        lambda: _build_weighted([1.0, 1.0, 1.0, 1.0], budget=10.0))
    for node_budget in (1, 2, 8):
        for lns in (0, 12):
            h = _build_weighted([1.0, 1.0, 1.0, 1.0], budget=10.0)
            res = solve_bp(h, node_budget=node_budget, lns_rounds=lns)
            assert res.status == kOptimal, (node_budget, lns, res.note)
            # Pre-fix: objective -12.0, bound -10.0, gap -0.2 — the
            # solution beat its own claimed bound.
            assert res.gap >= -1e-9, (node_budget, lns, res.gap)
            assert res.bound <= ref + 1e-6, (node_budget, lns, res.bound)
            assert res.objective >= ref - 1e-6, (node_budget, lns,
                                                 res.objective)
    print("ok W6 bound never the incumbent (gap >= 0 in all runs)")


# ----------------------------------------------------------------------
# W7: all-continuous models never branch on continuous variables
# ----------------------------------------------------------------------
def _build_all_continuous():
    h = _quiet_highs()
    xs = []
    for i in range(3):
        x = h.addVariable(lb=0.0, ub=5.0, obj=-1.0, name=f"x{i}")
        yv = h.addVariable(lb=0.0, ub=5.0, obj=-0.5, name=f"y{i}")
        h.addConstr(x + yv <= 5.0)
        xs.append(x)
    h.addConstr(sum(xs) <= 7.5)  # fractional optimum: x_i = 2.5 each
    h.setMinimize()
    return h


def test_w7_all_continuous_lp_solved_at_root():
    ref = _direct_optimum(_build_all_continuous)
    res = solve_bp(_build_all_continuous(), node_budget=8)
    # Pre-fix: floor/ceil branching on a fractional continuous xbar burnt
    # the node budget or returned spurious infeasible results.
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    assert "continuous model" in res.note, res.note
    print(f"ok W7 all-continuous LP at root (obj {res.objective:.2f})")


# ----------------------------------------------------------------------
# W8: linking-column values preserved through branch-and-price
# ----------------------------------------------------------------------
def test_w8_linking_column_not_zeroed():
    build = lambda: _build_knapsack(z_cost=-0.5, z_ub=2.0)  # noqa: E731
    hd = build()
    hd.run()
    ref = hd.getObjectiveValue()
    z_direct = hd.getSolution().col_value[8]
    assert z_direct > 0.5  # the optimum genuinely needs the linking column

    res = solve_bp(build(), node_budget=32)
    # Pre-fix: integral-node recovery zeroed linking columns — wrong
    # objective, failed final verification, kNotset.
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    assert abs(res.col_value[8] - z_direct) < 1e-6, res.col_value[8]
    print(f"ok W8 linking column preserved (z = {res.col_value[8]:.1f})")


# ----------------------------------------------------------------------
# W9: artificial-supported masters never displace feasible incumbents
# ----------------------------------------------------------------------
def _build_tight_integer_blocks():
    h = _quiet_highs()
    xs = []
    for i in range(4):
        x = h.addVariable(lb=0.0, ub=3.0, obj=-1.0,
                          type=highspy.HighsVarType.kInteger, name=f"x{i}")
        u = h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                          name=f"u{i}")
        h.addConstr(x <= 3.0 * u)
        xs.append(x)
    h.addConstr(sum(xs) <= 10.0)  # not all blocks can sit at x = 3
    h.setMinimize()
    return h


def test_w9_artificial_masters_do_not_displace_incumbent():
    ref = _direct_optimum(_build_tight_integer_blocks)
    res = solve_bp(_build_tight_integer_blocks(), node_budget=16)
    # Pre-fix: a coupling-infeasible artificial-supported candidate
    # displaced the feasible incumbent; final verification then failed
    # and solve_bp returned kNotset.
    assert res.status == kOptimal, res.note
    assert abs(res.objective - ref) < 1e-6, (res.objective, ref)
    x_sum = sum(res.col_value[2 * i] for i in range(4))
    assert x_sum <= 10.0 + 1e-6, x_sum  # coupling row respected
    assert "verified" in res.note, res.note
    print(f"ok W9 feasible incumbent kept (obj {res.objective:.2f})")


if __name__ == "__main__":
    _skip_excs = ((unittest.SkipTest,) if pytest is None
                  else (unittest.SkipTest, pytest.skip.Exception))
    for _test in [
        test_k1_keras_standalone_activation_binding,
        test_d1_decomp_quadratic_objective_not_dropped,
        test_d2_decomp_installs_solution_for_get_error,
        test_o1_onnx_gemm_constant_first_and_transpose,
        test_o2_onnx_input_count_validated,
        test_x1_xgboost_early_stopping_embeds_best_iteration,
        test_x2_xgboost_unsupported_objective_rejected,
        test_x3_xgboost_dataframe_fitted_model_embeds,
        test_t1_multioutput_trees_raise,
        test_t2_constant_feature_epsilon_sliver,
        test_p1_adaptive_breakpoints_certificate_or_error,
        test_p2_fixed_input_returns_real_output_var,
        test_a1_cross_model_variables_rejected,
        test_c1_subclass_dispatch_and_curated_error,
        test_s1_optional_dep_gating_no_nameerror,
        test_b1_bilinear_mixed_sign_product,
        test_b2_piecewise_mccormick_segment_guard,
        test_r1_solve_adaptive_per_embedding_tolerance,
        test_w1_pricing_classes_respect_coupling_coefficients,
        test_w2_integer_linking_column_cg_converges,
        test_w3_objective_offset_consistent,
        test_w4_truncated_cg_bound_stays_valid,
        test_w5_no_dual_calibration_branched_bp_exact,
        test_w6_bound_never_falls_back_to_incumbent,
        test_w7_all_continuous_lp_solved_at_root,
        test_w8_linking_column_not_zeroed,
        test_w9_artificial_masters_do_not_displace_incumbent,
    ]:
        try:
            _test()
        except _skip_excs as exc:
            print(f"skip {_test.__name__}: {exc}")
    print("\nAll highs_ml regression tests passed.")
