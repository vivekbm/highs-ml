"""Tests for the bilinear embedding (highs_ml._bilinear)."""

import pathlib
import sys

import numpy as np
import highspy
from highspy import HighsVarType

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from highs_ml._bilinear import add_bilinear_constr  # noqa: E402


def _quiet():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def _value(h, var):
    return h.getSolution().col_value[var.index]


def test_binary_times_continuous_exact():
    h = _quiet()
    b = h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger, name="b")
    x = h.addVariable(lb=-2.0, ub=3.0, name="x")
    y = add_bilinear_constr(h, b, x, name="t1")
    h.addConstr(x == 2.5)
    # force b = 1
    h.addConstr(b == 1.0)
    h.maximize(y)
    assert abs(_value(h, y) - 2.5) < 1e-9
    # force b = 0
    h2 = _quiet()
    b2 = h2.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger, name="b")
    x2 = h2.addVariable(lb=-2.0, ub=3.0, name="x")
    y2 = add_bilinear_constr(h2, b2, x2, name="t1")
    h2.addConstr(x2 == 2.5)
    h2.addConstr(b2 == 0.0)
    h2.maximize(y2)
    assert abs(_value(h2, y2)) < 1e-9
    print("ok binary x continuous exact")


def test_integer_times_continuous_exact():
    h = _quiet()
    i = h.addVariable(lb=0.0, ub=5.0, type=HighsVarType.kInteger, name="i")
    x = h.addVariable(lb=-2.0, ub=3.0, name="x")
    y = add_bilinear_constr(h, i, x, name="t2")
    h.addConstr(x == 1.7)
    h.addConstr(i == 4.0)
    h.maximize(y)
    got = _value(h, y)
    assert abs(got - 6.8) < 1e-6, got
    print("ok integer x continuous exact")


def test_continuous_times_continuous_certified():
    tol = 0.02
    worst = 0.0
    for v1, v2 in ((1.3, 2.4), (2.9, 3.7), (0.5, 4.0), (2.0, 2.0)):
        h = _quiet()
        x1 = h.addVariable(lb=0.0, ub=3.0, name="x1")
        x2 = h.addVariable(lb=0.0, ub=4.0, name="x2")
        y = add_bilinear_constr(h, x1, x2, tol=tol, name="t3")
        h.addConstr(x1 == v1)
        h.addConstr(x2 == v2)
        # envelope: optimizer picks the most favorable edge
        h.maximize(y)
        err_hi = abs(_value(h, y) - v1 * v2)
        h2 = _quiet()
        xa = h2.addVariable(lb=0.0, ub=3.0, name="x1")
        xb = h2.addVariable(lb=0.0, ub=4.0, name="x2")
        ya = add_bilinear_constr(h2, xa, xb, tol=tol, name="t3")
        h2.addConstr(xa == v1)
        h2.addConstr(xb == v2)
        h2.minimize(ya)
        err_lo = abs(_value(h2, ya) - v1 * v2)
        worst = max(worst, err_hi, err_lo)
    assert worst <= tol + 1e-6, worst
    print(f"ok continuous x continuous certified (max error {worst:.4f} <= tol {tol})")


def test_square_certified():
    h = _quiet()
    x = h.addVariable(lb=-3.0, ub=3.0, name="x")
    y = add_bilinear_constr(h, x, x, tol=0.02, name="t4")
    h.addConstr(x == -1.7)
    h.minimize(y)
    got = _value(h, y)
    assert abs(got - 2.89) <= 0.02 + 1e-6, got
    print("ok square certified")


def test_poly_pipeline_with_variable_product():
    """PolynomialFeatures degree-2 with both features variable now embeds."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from highs_ml import add_predictor_constr

    rng = np.random.default_rng(3)
    X = rng.uniform(0.5, 2.0, size=(300, 2))
    yv = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.5 * X[:, 0] * X[:, 1]
    pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=True),
                         LinearRegression()).fit(X, yv)

    h = _quiet()
    x1 = h.addVariable(lb=0.5, ub=2.0, name="f0")
    x2 = h.addVariable(lb=0.5, ub=2.0, name="f1")
    y = h.addVariable(lb=-100.0, ub=100.0, name="y")
    pc = add_predictor_constr(h, pipe, [x1, x2], y, pwl_tol=0.02)
    h.addConstr(x1 == 1.2)
    h.addConstr(x2 == 0.9)
    h.maximize(y)
    exact = float(pipe.predict(np.array([[1.2, 0.9]]))[0])
    got = _value(h, y)
    assert abs(got - exact) < 0.05, (got, exact)
    assert pc.get_error() < 0.05, pc.get_error()
    print(f"ok poly pipeline with variable product (error {pc.get_error():.3f})")


def test_degree_three_still_rejected():
    from sklearn.preprocessing import PolynomialFeatures
    from highs_ml._preprocessing import PolynomialFeaturesStep
    from highs_ml._affine import Affine

    poly = PolynomialFeatures(degree=3, include_bias=False).fit(
        np.zeros((2, 2)))
    h = _quiet()
    v1 = h.addVariable(lb=0.0, ub=1.0)
    v2 = h.addVariable(lb=0.0, ub=1.0)
    inputs = [Affine.coerce(v1), Affine.coerce(v2)]
    try:
        PolynomialFeaturesStep(poly).transform(inputs, h=h)
    except ValueError as e:
        assert "degree > 2" in str(e) or "more than" in str(e)
        print("ok degree-3 terms still rejected with clear error")
        return
    raise AssertionError("expected ValueError for degree-3 variable terms")


def test_adaptive_refinement():
    """Adaptive envelope: solve, refine at incumbent, repeat until certified."""
    from highs_ml import add_adaptive_bilinear

    h = _quiet()
    x = h.addVariable(lb=0.5, ub=2.0, name="x")
    z = h.addVariable(lb=0.5, ub=2.0, name="z")
    h.addConstr(x + z <= 3.0)
    emb = add_adaptive_bilinear(h, x, z, tol=0.02, name="ad", n_initial=4)
    h.maximize(emb.y)
    status, err, n_ref = emb.solve_adaptive()
    assert status == highspy.HighsModelStatus.kOptimal
    assert err <= 0.02, (err, n_ref)
    obj = h.getObjectiveValue()
    # true max of x*z with x+z<=3, x,z in [0.5,2] is 2.25 at (1.5,1.5);
    # certified envelope objective may overshoot by tol
    assert 2.25 - 0.05 <= obj <= 2.25 + 0.05, obj
    print(f"ok adaptive refinement (obj {obj:.4f}, err {err:.1e}, "
          f"{n_ref} refines)")


def test_adaptive_segment_economy():
    """Adaptive path uses far fewer segments than static uniform."""
    from highs_ml import add_adaptive_bilinear
    from highs_ml._bilinear import _piecewise_mccormick
    from highs_ml._pwl import PWLStats

    h = _quiet()
    x = h.addVariable(lb=0.5, ub=2.0, name="x")
    z = h.addVariable(lb=0.5, ub=2.0, name="z")
    h.addConstr(x + z <= 3.0)
    emb = add_adaptive_bilinear(h, x, z, tol=0.02, name="ad2", n_initial=4)
    h.maximize(emb.y)
    emb.solve_adaptive()
    adaptive_segs = sum(1 for s in emb.segments if s["active"])
    # static uniform for the same tolerance would need
    # ceil(1.5 / (4*0.02/1.5)) = 29 segments
    assert adaptive_segs < 29, adaptive_segs
    print(f"ok adaptive segment economy ({adaptive_segs} active vs 29 static)")


if __name__ == "__main__":
    test_binary_times_continuous_exact()
    test_integer_times_continuous_exact()
    test_continuous_times_continuous_certified()
    test_square_certified()
    test_poly_pipeline_with_variable_product()
    test_degree_three_still_rejected()
    test_adaptive_refinement()
    test_adaptive_segment_economy()
    print("\nAll bilinear tests passed.")
