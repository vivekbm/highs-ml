"""Tests for the unified refinement framework (highs_ml._refine)."""

import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import highspy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

from highs_ml import (  # noqa: E402
    RefinablePWL,
    add_adaptive_bilinear,
    add_predictor_constr,
    solve_adaptive,
)
from highs_ml._pwl import add_pwl_constr, PWLStats  # noqa: E402


def _quiet():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def test_refinable_pwl_sigmoid():
    h = _quiet()
    z = h.addVariable(lb=-6.0, ub=6.0, name="z")
    emb = RefinablePWL(h, sigmoid, z, tol=0.01, name="r1", n_initial=2)
    h.maximize(emb.y)
    status, err, n_ref = solve_adaptive(h, [emb])
    assert status == highspy.HighsModelStatus.kOptimal
    assert err <= 0.01, (err, n_ref)
    # max of sigmoid on [-6,6] is ~sigmoid(6) = 0.9975
    assert abs(h.getObjectiveValue() - sigmoid(6.0)) < 0.02
    print(f"ok refinable sigmoid (err {err:.1e}, {n_ref} refines, "
          f"{emb.active_segments()} active segments)")


def test_refinable_segment_economy():
    """Refinable path needs far fewer segments than static at same tol."""
    # static
    from highs_ml._affine import Affine
    h1 = _quiet()
    z1 = h1.addVariable(lb=-6.0, ub=6.0, name="z")
    st = PWLStats()
    add_pwl_constr(h1, sigmoid, Affine.coerce(z1), tol=0.005, name="s",
                   stats=st)
    static_bin = st.n_binaries
    # refinable
    h2 = _quiet()
    z2 = h2.addVariable(lb=-6.0, ub=6.0, name="z")
    emb = RefinablePWL(h2, sigmoid, z2, tol=0.005, name="r", n_initial=2)
    h2.maximize(emb.y)
    solve_adaptive(h2, [emb])
    ref_bin = emb.active_segments()
    assert ref_bin < static_bin, (ref_bin, static_bin)
    print(f"ok refinable segment economy ({ref_bin} active vs "
          f"{static_bin} static)")


def test_mixed_model_round_robin():
    """solve_adaptive drives a bilinear and a sigmoid embedding together."""
    h = _quiet()
    x1 = h.addVariable(lb=0.5, ub=2.0, name="x1")
    x2 = h.addVariable(lb=0.5, ub=2.0, name="x2")
    z = h.addVariable(lb=-4.0, ub=4.0, name="z")
    bil = add_adaptive_bilinear(h, x1, x2, tol=0.02, name="b", n_initial=3)
    pwl = RefinablePWL(h, sigmoid, z, tol=0.02, name="p", n_initial=2)
    h.addConstr(x1 + x2 <= 3.0)
    h.addConstr(z <= 1.0)
    h.maximize(bil.y + pwl.y)
    status, err, n_ref = solve_adaptive(h, [bil, pwl])
    assert status == highspy.HighsModelStatus.kOptimal
    assert err <= 0.02, (err, n_ref)
    print(f"ok mixed round-robin refinement (err {err:.1e}, "
          f"{n_ref} total refines)")


def test_refinable_logistic_end_to_end():
    """Logistic regression with refinable=True matches the static path."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    zv = X @ np.array([1.5, -2.0]) + 0.3
    yv = (1.0 / (1.0 + np.exp(-zv)) > rng.random(500)).astype(int)
    reg = LogisticRegression().fit(X, yv)

    target = np.array([0.8, -0.4])
    results = {}
    for mode, refinable in (("static", False), ("refinable", True)):
        h = _quiet()
        xs = [h.addVariable(lb=-4.0, ub=4.0, name=f"x{j}") for j in range(2)]
        y = h.addVariable(lb=0.0, ub=1.0, name="p")
        pc = add_predictor_constr(h, reg, xs, y, pwl_tol=1e-3,
                                  refinable=refinable)
        for var, t in zip(xs, target):
            h.addConstr(var == float(t))
        if refinable:
            status, err, n_ref = solve_adaptive(
                h, pc.refinable_embeddings)
            assert status == highspy.HighsModelStatus.kOptimal
            assert err <= 1e-3
        else:
            h.maximize(y)
        results[mode] = h.getSolution().col_value[y.index]
    exact = float(reg.predict_proba(target.reshape(1, -1))[0, 1])
    assert abs(results["static"] - exact) < 2e-3
    assert abs(results["refinable"] - exact) < 2e-3
    print(f"ok refinable logistic end-to-end (static {results['static']:.5f}, "
          f"refinable {results['refinable']:.5f}, exact {exact:.5f})")


def test_refinable_student_admission():
    """Student enrollment solved via refinable logistic embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data_dir = pathlib.Path(__file__).resolve().parent.parent / "examples" / "data"
    historical = pd.read_csv(data_dir / "college_student_enroll-s1-1.csv",
                             index_col=0)
    features = ["merit", "SAT", "GPA"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
    pipe.fit(X=historical.loc[:, features], y=historical["enroll"])
    students = pd.read_csv(data_dir / "college_applications6000.csv",
                           index_col=0).sample(25, random_state=1)

    h = _quiet()
    x, y, embs = {}, {}, []
    for sid, row in students.iterrows():
        x[sid] = h.addVariable(lb=0.0, ub=2.5, name=f"merit[{sid}]")
        y[sid] = h.addVariable(lb=0.0, ub=1.0, obj=1.0,
                               name=f"enroll_prob[{sid}]")
    h.addConstr(sum(x.values()) <= 0.2 * 25, name="budget")
    for sid, row in students.iterrows():
        pc = add_predictor_constr(
            h, pipe,
            {"merit": x[sid], "SAT": float(row["SAT"]),
             "GPA": float(row["GPA"])},
            y[sid], output_type="probability_1", pwl_tol=1e-3,
            refinable=True, name=f"p{sid}")
        embs.extend(pc.refinable_embeddings)
    h.setMaximize()
    t0 = time.perf_counter()
    status, err, n_ref = solve_adaptive(h, embs)
    dt = time.perf_counter() - t0
    obj = h.getObjectiveValue()
    assert status == highspy.HighsModelStatus.kOptimal
    assert err <= 1e-3, (err, n_ref)
    # static-path reference objective: 13.7888
    assert abs(obj - 13.7888) < 0.05, obj
    print(f"ok refinable student admission (obj {obj:.4f}, err {err:.1e}, "
          f"{n_ref} refines, {dt:.1f}s)")


if __name__ == "__main__":
    test_refinable_pwl_sigmoid()
    test_refinable_segment_economy()
    test_mixed_model_round_robin()
    test_refinable_logistic_end_to_end()
    test_refinable_student_admission()
    print("\nAll refinement tests passed.")
