"""Tests for highs_ml. Runnable with pytest or directly: python tests/test_highs_ml.py"""

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from highs_ml import add_predictor_constr  # noqa: E402

RNG = np.random.default_rng(42)


def _quiet_highs():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def _optional_import(name):
    """Import an optional dependency; skip the test if it is missing.

    Catches OSError too: some wheels (notably LightGBM on macOS without
    libomp) install fine but fail to load their native library at import
    time, raising OSError rather than ImportError.
    """
    if pytest is not None:
        try:
            return pytest.importorskip(name)
        except OSError as exc:
            pytest.skip(f"{name} not loadable: {exc}")
    try:
        return __import__(name)
    except (ImportError, OSError) as exc:
        # Raise a distinct skip exception so the __main__ loop only skips on
        # missing optional deps, not on any ImportError inside a test body.
        raise unittest.SkipTest(f"{name} not available: {exc}") from exc


def test_linear_regression_is_exact():
    X = RNG.normal(size=(200, 3))
    w = np.array([2.0, -1.0, 0.5])
    yv = X @ w + 1.7
    reg = LinearRegression().fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(3)]
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, reg, xs, y)
    # Maximize y subject to x fixed by an equality: pick a point.
    target = np.array([1.0, -2.0, 0.5])
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    sol = h.getSolution().col_value
    expected = float(target @ w + 1.7)
    assert abs(sol[y.index] - expected) < 1e-6, (sol[y.index], expected)
    assert pc.get_error() < 1e-6
    print(f"ok linear regression (error {pc.get_error():.2e})")


def test_logistic_regression_pwl():
    X = RNG.normal(size=(500, 2))
    z = X @ np.array([1.5, -2.0]) + 0.3
    yv = (1.0 / (1.0 + np.exp(-z)) > RNG.random(500)).astype(int)
    reg = LogisticRegression().fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=0.0, ub=1.0, name="p")
    pc = add_predictor_constr(h, reg, xs, y, pwl_tol=1e-3)

    target = np.array([0.8, -0.4])
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    sol = h.getSolution().col_value
    exact = float(reg.predict_proba(target.reshape(1, -1))[0, 1])
    assert abs(sol[y.index] - exact) < 2e-3, (sol[y.index], exact)
    assert pc.get_error() < 2e-3
    print(f"ok logistic regression (error {pc.get_error():.2e})")


def test_mlp_regressor_relu_exact():
    X = RNG.uniform(-3, 3, size=(300, 2))
    yv = np.maximum(0, X[:, 0]) - 2 * np.maximum(0, -X[:, 1]) + 0.5
    mlp = MLPRegressor(hidden_layer_sizes=(8,), activation="relu",
                       max_iter=3000, random_state=7).fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, mlp, xs, y)

    target = np.array([1.2, -0.7])
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    sol = h.getSolution().col_value
    exact = float(mlp.predict(target.reshape(1, -1))[0])
    # ReLU embedding is exact up to solver tolerance.
    assert abs(sol[y.index] - exact) < 1e-5, (sol[y.index], exact)
    assert pc.get_error() < 1e-5
    print(f"ok MLP relu (error {pc.get_error():.2e})")


def test_pipeline_with_scaler():
    X = RNG.normal(loc=[500.0, 3.0], scale=[100.0, 0.5], size=(400, 2))
    z = (X[:, 0] - 500.0) / 100.0 + (X[:, 1] - 3.0) / 0.5
    yv = (1.0 / (1.0 + np.exp(-z)) > RNG.random(400)).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=200.0, ub=800.0, name="sat"),
          h.addVariable(lb=1.5, ub=4.5, name="gpa")]
    y = h.addVariable(lb=0.0, ub=1.0, name="p")
    pc = add_predictor_constr(h, pipe, xs, y, pwl_tol=1e-3)

    h.addConstr(xs[0] == 560.0)
    h.addConstr(xs[1] == 3.4)
    h.maximize(y)
    sol = h.getSolution().col_value
    exact = float(pipe.predict_proba(
        pd.DataFrame([[560.0, 3.4]], columns=["SAT", "GPA"]))[0, 1])
    assert abs(sol[y.index] - exact) < 2e-3, (sol[y.index], exact)
    print(f"ok pipeline (error {pc.get_error():.2e})")


def test_student_enrollment_objective():
    """End-to-end: reproduce the gurobi_ml student enrollment objective."""
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "examples" / "data"
    historical = pd.read_csv(data_dir / "college_student_enroll-s1-1.csv", index_col=0)
    features = ["merit", "SAT", "GPA"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
    pipe.fit(X=historical.loc[:, features], y=historical["enroll"])

    students = pd.read_csv(data_dir / "college_applications6000.csv", index_col=0)
    n = 25
    students = students.sample(n, random_state=1)

    h = _quiet_highs()
    x = {s: h.addVariable(lb=0.0, ub=2.5) for s in students.index}
    y = {s: h.addVariable(lb=0.0, ub=1.0) for s in students.index}
    h.addConstr(sum(x[s] for s in students.index) <= 0.2 * n)
    pcs = [
        add_predictor_constr(
            h, pipe,
            {"merit": x[s], "SAT": float(r["SAT"]), "GPA": float(r["GPA"])},
            y[s], output_type="probability_1", pwl_tol=1e-3, name=f"p{s}",
        )
        for s, r in students.iterrows()
    ]
    h.maximize(sum(y[s] for s in students.index))
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    obj = h.getObjectiveValue()
    # HiGHS-derived reference for this 25-student subsample; PWL tol is
    # 1e-3 per student, so allow n * tol + slack.
    assert abs(obj - 13.7888) < 0.05, obj
    assert max(pc.get_error() for pc in pcs) < 1e-3
    print(f"ok student enrollment (objective {obj:.4f})")


def _check_tree_predictor(predictor, points, tag):
    """Embed a tree-based predictor, evaluate at fixed points, compare exactly."""
    worst = 0.0
    for target in points:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        pc = add_predictor_constr(h, predictor, xs, y)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(predictor.predict(np.array(target).reshape(1, -1))[0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-6, (tag, target, pc.get_error())
    assert worst < 1e-6, (tag, worst)
    print(f"ok {tag} (max error {worst:.2e})")


def test_decision_tree_exact():
    X = RNG.uniform(-4, 4, size=(400, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    tree = DecisionTreeRegressor(max_depth=4, random_state=3).fit(X, yv)
    points = [(-3.0, -3.0), (-3.0, 3.5), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]
    _check_tree_predictor(tree, points, "decision tree")


def test_random_forest_exact():
    X = RNG.uniform(-4, 4, size=(400, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    rf = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=5).fit(X, yv)
    points = [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]
    _check_tree_predictor(rf, points, "random forest")


def test_gradient_boosting_exact():
    X = RNG.uniform(-4, 4, size=(400, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    gbm = GradientBoostingRegressor(n_estimators=10, max_depth=3,
                                    random_state=11).fit(X, yv)
    points = [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]
    _check_tree_predictor(gbm, points, "gradient boosting")


def test_tree_optimization_direction():
    """Sanity: optimizing over an embedded tree finds the true max region."""
    X = RNG.uniform(-4, 4, size=(600, 2))
    yv = -((X[:, 0] - 1.5) ** 2) - ((X[:, 1] + 1.0) ** 2)  # peak at (1.5, -1)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5,
                               random_state=13).fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-4.0, ub=4.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=-1e4, ub=1e4, name="y")
    add_predictor_constr(h, rf, xs, y)
    h.maximize(y)
    sol = h.getSolution().col_value
    xy = np.array([sol[v.index] for v in xs]).reshape(1, -1)
    # Solution value must equal the forest's own prediction there...
    assert abs(sol[y.index] - float(rf.predict(xy)[0])) < 1e-6
    # ...and beat 95% of uniformly sampled inputs (it is the true max).
    grid = RNG.uniform(-4, 4, size=(5000, 2))
    frac_beaten = (rf.predict(grid) <= sol[y.index] + 1e-9).mean()
    assert frac_beaten >= 0.95, frac_beaten
    print(f"ok tree optimization (beats {frac_beaten:.1%} of sampled inputs)")


def test_xgboost_regressor_exact():
    xgb = _optional_import("xgboost")
    X = RNG.uniform(-4, 4, size=(500, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    model = xgb.XGBRegressor(n_estimators=15, max_depth=4,
                             random_state=17).fit(X, yv)

    worst = 0.0
    for target in [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        pc = add_predictor_constr(h, model, xs, y, output_type="raw")
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(model.predict(np.array(target).reshape(1, -1))[0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-5, (target, pc.get_error())
    assert worst < 1e-5, worst
    print(f"ok xgboost regressor (max error {worst:.2e})")


def test_xgboost_classifier_probability():
    xgb = _optional_import("xgboost")
    X = RNG.uniform(-4, 4, size=(600, 2))
    yv = (X[:, 0] + X[:, 1] ** 2 > 1.0).astype(int)
    model = xgb.XGBClassifier(n_estimators=15, max_depth=4,
                              random_state=19).fit(X, yv)

    worst = 0.0
    for target in [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=0.0, ub=1.0, name="p")
        pc = add_predictor_constr(h, model, xs, y,
                                  output_type="probability_1", pwl_tol=1e-3)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(model.predict_proba(np.array(target).reshape(1, -1))[0, 1])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 2e-3, (target, pc.get_error())
    assert worst < 2e-3, worst
    print(f"ok xgboost classifier (max error {worst:.2e})")


def test_pls_regression_exact():
    from sklearn.cross_decomposition import PLSRegression
    X = RNG.normal(size=(300, 3))
    yv = X @ np.array([1.0, -2.0, 0.5]) + 3.0
    pls = PLSRegression(n_components=2).fit(X, yv)

    h = _quiet_highs()
    xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(3)]
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, pls, xs, y)
    target = np.array([0.5, -1.0, 2.0])
    for var, t in zip(xs, target):
        h.addConstr(var == float(t))
    h.maximize(y)
    sol = h.getSolution().col_value
    exact = float(np.atleast_1d(pls.predict(target.reshape(1, -1)))[0])
    assert abs(sol[y.index] - exact) < 1e-6, (sol[y.index], exact)
    assert pc.get_error() < 1e-6
    print(f"ok PLS regression (error {pc.get_error():.2e})")


def test_lightgbm_regressor_exact():
    lgb = _optional_import("lightgbm")
    X = RNG.uniform(-4, 4, size=(500, 2))
    yv = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    model = lgb.LGBMRegressor(n_estimators=15, max_depth=4, verbose=-1,
                              random_state=23).fit(X, yv)
    worst = 0.0
    for target in [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        pc = add_predictor_constr(h, model, xs, y, output_type="raw")
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(model.predict(np.array(target).reshape(1, -1))[0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-5, (target, pc.get_error())
    assert worst < 1e-5, worst
    print(f"ok lightgbm regressor (max error {worst:.2e})")


def test_lightgbm_classifier_probability():
    lgb = _optional_import("lightgbm")
    X = RNG.uniform(-4, 4, size=(600, 2))
    yv = (X[:, 0] + X[:, 1] ** 2 > 1.0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=15, max_depth=4, verbose=-1,
                               random_state=29).fit(X, yv)
    worst = 0.0
    for target in [(-3.0, -3.0), (0.1, -1.0), (2.5, 0.7), (3.9, 3.9)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-5.0, ub=5.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=0.0, ub=1.0, name="p")
        pc = add_predictor_constr(h, model, xs, y,
                                  output_type="probability_1", pwl_tol=1e-3)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(model.predict_proba(np.array(target).reshape(1, -1))[0, 1])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 2e-3, (target, pc.get_error())
    assert worst < 2e-3, worst
    print(f"ok lightgbm classifier (max error {worst:.2e})")


def test_keras_dense_network():
    import os
    os.environ.setdefault("KERAS_BACKEND", "jax")
    keras = _optional_import("keras")

    X = RNG.uniform(-3, 3, size=(400, 2))
    yv = np.maximum(0, X[:, 0]) - 2 * np.maximum(0, -X[:, 1]) + 0.5
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, yv, epochs=60, verbose=0)

    worst = 0.0
    for target in [(-2.5, -2.5), (0.1, -1.0), (1.5, 0.7), (2.9, 2.9)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=-100.0, ub=100.0, name="y")
        pc = add_predictor_constr(h, model, xs, y)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        exact = float(model.predict(np.array(target).reshape(1, -1), verbose=0)[0, 0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-5, (target, pc.get_error())
    assert worst < 1e-5, worst
    print(f"ok keras dense network (max error {worst:.2e})")


def test_onnx_network():
    onnx = _optional_import("onnx")
    from onnx import helper, numpy_helper

    rng = np.random.default_rng(7)
    B1, C1 = rng.normal(size=(3, 4)), rng.normal(size=4)
    B2, C2 = rng.normal(size=(4, 1)), rng.normal(size=1)

    nodes = [
        helper.make_node("Gemm", ["X", "B1", "C1"], ["H1"]),
        helper.make_node("Relu", ["H1"], ["A1"]),
        helper.make_node("Gemm", ["A1", "B2", "C2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "dense_net",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(B1, "B1"), numpy_helper.from_array(C1, "C1"),
         numpy_helper.from_array(B2, "B2"), numpy_helper.from_array(C2, "C2")],
    )
    model = helper.make_model(graph)

    worst = 0.0
    for target in [(-2.0, 0.5, 1.0), (0.3, -1.0, 2.0), (1.5, 1.5, -0.5)]:
        h = _quiet_highs()
        xs = [h.addVariable(lb=-3.0, ub=3.0, name=f"x{j}") for j in range(3)]
        y = h.addVariable(lb=-1e4, ub=1e4, name="y")
        pc = add_predictor_constr(h, model, xs, y)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        h.maximize(y)
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution().col_value
        t = np.array(target)
        exact = float((np.maximum(0, t @ B1 + C1) @ B2 + C2)[0])
        worst = max(worst, abs(sol[y.index] - exact))
        assert pc.get_error() < 1e-5, (target, pc.get_error())
    assert worst < 1e-5, worst
    print(f"ok ONNX network (max error {worst:.2e})")


def test_column_transformer_and_poly_on_fixed_features():
    """Poly-expansion of fixed features + linear decision feature."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    X = RNG.uniform(-4, 4, size=(400, 3))
    # Target depends on merit (linear), SAT*GPA interaction and SAT^2.
    yv = (1.5 * X[:, 0] + 0.01 * X[:, 1] * X[:, 2] + 0.002 * X[:, 1] ** 2
          + 0.3 * X[:, 2])
    ct = ColumnTransformer([
        ("poly", PolynomialFeatures(degree=2, include_bias=False), [1, 2]),
        ("pass", "passthrough", [0]),
    ])
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(ct, Ridge()).fit(X, yv)

    h = _quiet_highs()
    merit = h.addVariable(lb=0.0, ub=2.5, name="merit")
    y = h.addVariable(lb=-1e4, ub=1e4, name="y")
    pc = add_predictor_constr(h, pipe, [merit, 1300.0, 3.5], y)
    h.addConstr(merit == 1.25)
    h.maximize(y)
    sol = h.getSolution().col_value
    exact = float(pipe.predict(np.array([[1.25, 1300.0, 3.5]]))[0])
    assert abs(sol[y.index] - exact) < 1e-5, (sol[y.index], exact)
    assert pc.get_error() < 1e-5
    print(f"ok column transformer + poly (error {pc.get_error():.2e})")


def test_polynomial_features_degree2_two_variables_embeds():
    """Degree-2 products of two decision variables now embed via the
    certified bilinear formulation (was a hard error before v1.3)."""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    X = RNG.uniform(0.5, 2.0, size=(300, 2))
    yv = X[:, 0] * X[:, 1]
    pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                         LinearRegression()).fit(X, yv)
    h = _quiet_highs()
    xs = [h.addVariable(lb=0.5, ub=2.0, name=f"x{j}") for j in range(2)]
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, pipe, xs, y, pwl_tol=0.02)
    h.addConstr(xs[0] == 1.4)
    h.addConstr(xs[1] == 1.1)
    h.maximize(y)
    exact = float(pipe.predict(np.array([[1.4, 1.1]]))[0])
    got = h.getSolution().col_value[y.index]
    assert abs(got - exact) < 0.05, (got, exact)
    print(f"ok poly degree-2 variable product embeds (err {abs(got-exact):.3f})")


if __name__ == "__main__":
    # _optional_import raises unittest.SkipTest (no pytest) or pytest's
    # Skipped (a BaseException) outside a pytest run; treat only those as
    # skips so a genuine ImportError inside a test body still fails.
    _skip_excs = (unittest.SkipTest,) if pytest is None else (unittest.SkipTest, pytest.skip.Exception)
    for _test in [
        test_linear_regression_is_exact,
        test_logistic_regression_pwl,
        test_mlp_regressor_relu_exact,
        test_pipeline_with_scaler,
        test_student_enrollment_objective,
        test_decision_tree_exact,
        test_random_forest_exact,
        test_gradient_boosting_exact,
        test_tree_optimization_direction,
        test_xgboost_regressor_exact,
        test_xgboost_classifier_probability,
        test_pls_regression_exact,
        test_lightgbm_regressor_exact,
        test_lightgbm_classifier_probability,
        test_keras_dense_network,
        test_onnx_network,
        test_column_transformer_and_poly_on_fixed_features,
        test_polynomial_features_degree2_two_variables_embeds,
    ]:
        try:
            _test()
        except _skip_excs as exc:
            print(f"skip {_test.__name__}: {exc}")
    print("\nAll highs_ml tests passed.")
