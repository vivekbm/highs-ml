"""Tests for the Dantzig-Wolfe solver (highs_ml.dw)."""

import pathlib
import sys
import time

import numpy as np
import highspy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from highs_ml.dw import solve_dw  # noqa: E402

RNG = np.random.default_rng(5)


def _quiet():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def build_coupled(n, budget, y_costs=(-2.0,)):
    """n complementarity blocks + one budget coupling row.

    Block i (class depends on y cost c_i):
      x_i in [0,10], y_i in [-10,0], u_i binary
      x_i <= 10(1-u_i);  -10 u_i <= y_i;  1 <= x_i - y_i <= 15
    Coupling: sum x_i <= budget
    min sum (x_i + c_i y_i)
    """
    h = _quiet()
    xs, ys, us = [], [], []
    for i in range(n):
        c = y_costs[i % len(y_costs)]
        x = h.addVariable(lb=0.0, ub=10.0, obj=1.0, name=f"x{i}")
        y = h.addVariable(lb=-10.0, ub=0.0, obj=c, name=f"y{i}")
        u = h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                          name=f"u{i}")
        h.addConstr(x <= 10.0 * (1.0 - u))
        h.addConstr(-10.0 * u <= y)
        h.addConstr(x - y >= 1.0)
        h.addConstr(x - y <= 15.0)
        xs.append(x)
    h.addConstr(sum(xs) <= budget, name="budget")
    h.setMinimize()
    return h


def test_dw_matches_direct_two_classes():
    n, budget = 200, 100.0
    h = build_coupled(n, budget, y_costs=(-2.0, -3.0))
    t0 = time.perf_counter()
    res = solve_dw(h, max_iterations=60)
    dw_time = time.perf_counter() - t0
    assert res.decomposed, res.note
    assert res.status == highspy.HighsModelStatus.kOptimal, res.note

    h2 = build_coupled(n, budget, y_costs=(-2.0, -3.0))
    t0 = time.perf_counter()
    h2.run()
    direct_time = time.perf_counter() - t0
    direct_obj = h2.getObjectiveValue()

    print(f"  DW: obj {res.objective:.4f}, bound {res.bound:.4f}, "
          f"gap {res.gap:.2%}, {res.iterations} iters, {dw_time:.2f}s")
    print(f"  direct: obj {direct_obj:.4f}, {direct_time:.2f}s")
    assert abs(res.objective - direct_obj) < 1e-4 * max(1, abs(direct_obj)), (
        res.objective, direct_obj)
    assert res.bound <= res.objective + 1e-6  # min: bound <= primal
    print("ok DW matches direct on two-class coupled model")


def test_dw_solution_is_feasible_and_selected():
    n, budget = 60, 20.0
    h = build_coupled(n, budget)
    res = solve_dw(h)
    assert res.status == highspy.HighsModelStatus.kOptimal
    x = res.col_value
    # budget respected (verified internally too, but assert the point)
    xs = x[0::3]
    assert xs.sum() <= 20.0 + 1e-6
    print(f"ok DW solution feasible (budget {xs.sum():.3f} <= 20)")


def test_dw_scaling_vs_direct():
    for n, budget in ((2_000, 1_000.0), (10_000, 5_000.0)):
        h = build_coupled(n, budget)
        t0 = time.perf_counter()
        res = solve_dw(h, max_iterations=40)
        dw_time = time.perf_counter() - t0
        h2 = build_coupled(n, budget)
        t0 = time.perf_counter()
        h2.run()
        direct_time = time.perf_counter() - t0
        direct_obj = h2.getObjectiveValue()
        print(f"  n={n}: DW {dw_time:.2f}s (obj {res.objective:.1f}, "
              f"gap {res.gap:.2%}) vs direct {direct_time:.2f}s "
              f"(obj {direct_obj:.1f})")
        assert abs(res.objective - direct_obj) < 1e-4 * max(1, abs(direct_obj))
    print("ok DW scaling vs direct")


if __name__ == "__main__":
    test_dw_matches_direct_two_classes()
    test_dw_solution_is_feasible_and_selected()
    test_dw_scaling_vs_direct()
    print("\nAll DW tests passed.")
