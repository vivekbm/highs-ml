"""Correctness tests for the decomposition presolver (highs_ml.decomp)."""

import pathlib
import sys

import numpy as np
import highspy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

from highs_ml.decomp import solve_decomposed  # noqa: E402
from discourse_mwe_scaling import build_highs  # noqa: E402

RNG = np.random.default_rng(99)


def _quiet():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def _random_block_model(n_blocks, maximize=False, infeasible_block=None):
    """Random independent-block MILP; blocks have varied sizes/data."""
    h = _quiet()
    all_vars = []
    for b in range(n_blocks):
        nx = int(RNG.integers(2, 5))
        xb = [h.addVariable(lb=float(RNG.uniform(-5, 0)),
                            ub=float(RNG.uniform(1, 10)),
                            obj=float(RNG.uniform(-2, 3)),
                            type=(highspy.HighsVarType.kInteger
                                  if RNG.random() < 0.5
                                  else highspy.HighsVarType.kContinuous))
              for _ in range(nx)]
        nr = int(RNG.integers(1, 4))
        for _ in range(nr):
            coefs = RNG.uniform(-3, 3, size=nx)
            lhs = sum(c * v for c, v in zip(coefs, xb))
            rhs = float(RNG.uniform(0, 8))
            if RNG.random() < 0.5:
                h.addConstr(lhs <= rhs)
            else:
                h.addConstr(lhs >= -rhs)
        if infeasible_block is not None and b == infeasible_block:
            h.addConstr(sum(xb) >= 1e6)
        all_vars.append(xb)
    return h, all_vars, maximize


def test_matches_direct_random():
    for trial, maximize in enumerate([False, True, False, True, False]):
        h, vars_, maximize = _random_block_model(12, maximize=maximize)
        if maximize:
            h.setMaximize()
        else:
            h.setMinimize()
        res = solve_decomposed(h)
        # solve_decomposed never mutates h, so h is still pristine; costs
        # were set per-variable at build time.
        h.run()
        direct_obj = h.getObjectiveValue()
        assert res.decomposed, "expected decomposition on random blocks"
        assert res.status == highspy.HighsModelStatus.kOptimal
        assert abs(res.objective - direct_obj) < 1e-6 * max(1, abs(direct_obj)), (
            trial, res.objective, direct_obj)
    print("ok decomp matches direct on random block MILPs (min and max)")


def test_infeasible_block_propagates():
    h, vars_, _ = _random_block_model(6, infeasible_block=3)
    res = solve_decomposed(h)
    assert res.status != highspy.HighsModelStatus.kOptimal
    print(f"ok infeasible block propagates (status {res.status})")


def test_mwe_scaling():
    import time
    for n, ref in ((1000, 1000.0), (10_000, 10_000.0), (50_000, 50_000.0)):
        h, _ = build_highs(n)
        t0 = time.perf_counter()
        res = solve_decomposed(h)
        dt = time.perf_counter() - t0
        assert res.decomposed and res.n_unique_classes == 1
        assert abs(res.objective - ref) < 1e-6, (n, res.objective)
        assert dt < 10.0, (n, dt)  # direct HiGHS needs ~19s at n=50k
        print(f"ok MWE n={n}: {dt:.2f}s, obj {res.objective:.0f}")


def test_coupled_model_falls_back():
    """A model whose blocks share a coupling row must solve directly."""
    h = _quiet()
    x = [h.addVariable(lb=0.0, ub=2.5, obj=-1.0) for _ in range(10)]
    u = [h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger)
         for _ in range(10)]
    for i in range(10):
        h.addConstr(x[i] <= 2.5 * u[i])
    h.addConstr(sum(x) <= 3.0)  # the coupling row
    res = solve_decomposed(h)
    assert not res.decomposed, "coupled model must fall back to direct solve"
    h2 = _quiet()
    x2 = [h2.addVariable(lb=0.0, ub=2.5, obj=-1.0) for _ in range(10)]
    u2 = [h2.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger)
          for _ in range(10)]
    for i in range(10):
        h2.addConstr(x2[i] <= 2.5 * u2[i])
    h2.addConstr(sum(x2) <= 3.0)
    h2.setMinimize()
    h2.run()
    assert abs(res.objective - h2.getObjectiveValue()) < 1e-9
    print(f"ok coupled model falls back to direct solve (obj {res.objective})")


if __name__ == "__main__":
    test_matches_direct_random()
    test_infeasible_block_propagates()
    test_mwe_scaling()
    test_coupled_model_falls_back()
    print("\nAll decomp tests passed.")
