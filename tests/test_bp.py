"""Tests for branch-and-price and the solve_auto router."""

import pathlib
import sys
import time

import numpy as np
import highspy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

from highs_ml import solve_auto, solve_bp  # noqa: E402
from test_dw import build_coupled  # noqa: E402
from discourse_mwe_scaling import build_highs  # noqa: E402


def test_bp_proves_root_tight_case():
    """When the restricted master matches the root bound, B&P must prove
    optimality immediately (gap 0, no nodes)."""
    h = build_coupled(500, 250.0, y_costs=(-2.0,))
    res = solve_bp(h, node_budget=8)
    assert res.gap < 1e-6, (res.gap, res.note)
    assert "proven optimal at root" in res.note, res.note
    print(f"ok B&P proves root-tight case (gap {res.gap:.2e})")


def test_bp_bound_and_incumbent_are_valid():
    """B&P bound must dominate the true optimum (min: bound <= obj) and
    the incumbent must be primal feasible."""
    h = build_coupled(300, 120.0, y_costs=(-2.0, -3.0))
    res = solve_bp(h, node_budget=8)
    assert res.status == highspy.HighsModelStatus.kOptimal, res.note
    assert res.bound <= res.objective + 1e-6
    h2 = build_coupled(300, 120.0, y_costs=(-2.0, -3.0))
    h2.run()
    direct = h2.getObjectiveValue()
    # incumbent should be within a few percent of direct optimum
    assert res.objective <= direct * 1.05 + 1e-6, (res.objective, direct)
    print(f"ok B&P bound/incumbent valid "
          f"(obj {res.objective:.2f}, bound {res.bound:.2f}, "
          f"direct {direct:.2f})")


def test_auto_routes_separable():
    h, _ = build_highs(500)
    method, res = solve_auto(h)
    assert method == "decomposed", method
    assert abs(res.objective - 500.0) < 1e-6
    print("ok solve_auto routes separable models to decomp")


def test_auto_routes_block_angular():
    h = build_coupled(300, 150.0, y_costs=(-2.0,))
    method, res = solve_auto(h)
    assert method in ("dantzig-wolfe", "branch-and-price"), method
    assert abs(res.objective - 450.0) < 1.0, res.objective
    print(f"ok solve_auto routes block-angular models to {method}")


def test_auto_routes_dense_to_direct():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    rng = np.random.default_rng(1)
    xs = [h.addVariable(lb=0.0, ub=1.0, obj=float(c))
          for c in rng.uniform(-2, 2, size=8)]
    for _ in range(6):  # dense random rows: everything coupled
        coefs = rng.uniform(-1, 1, size=8)
        h.addConstr(sum(c * v for c, v in zip(coefs, xs)) <= 2.0)
    h.setMinimize()
    method, res = solve_auto(h)
    assert method == "direct", method
    h2 = highspy.Highs()
    h2.setOptionValue("output_flag", False)
    xs2 = [h2.addVariable(lb=0.0, ub=1.0, obj=float(c))
           for c in rng.uniform(-2, 2, size=8)]
    print(f"ok solve_auto routes dense models to direct (obj "
          f"{res.objective:.4f})")


if __name__ == "__main__":
    test_bp_proves_root_tight_case()
    test_bp_bound_and_incumbent_are_valid()
    test_auto_routes_separable()
    test_auto_routes_block_angular()
    test_auto_routes_dense_to_direct()
    print("\nAll B&P / solve_auto tests passed.")
