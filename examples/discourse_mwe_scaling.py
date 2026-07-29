"""Reproduce the Julia Discourse MWE (May 2024, HiGHS.jl 1.9.0):
n independent big-M complementarity pairs. The thread reported HiGHS
taking 58.6 s at n = 100_000.

Model per pair i:
  x_i in [0, 10], y_i in [-10, 0], u_i binary
  x_i <= 10 (1 - u_i)        (u=1 forces x=0)
  -10 u_i <= y_i             (u=0 forces y=0)
  1 <= x_i - y_i <= 15
  minimize sum(x_i - 2 y_i)

We time HiGHS 1.15.1 at increasing n, and Gurobi (restricted license,
2000-var cap) at n = 500 for a same-machine calibration point.
"""

import time

import numpy as np
import highspy


def build_highs(n: int) -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    x = h.addVariables(n, lb=0.0, ub=10.0, name_prefix="x")
    y = h.addVariables(n, lb=-10.0, ub=0.0, name_prefix="y")
    u = h.addVariables(n, lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger,
                       name_prefix="u")
    xv = [x[i] for i in range(n)]
    yv = [y[i] for i in range(n)]
    uv = [u[i] for i in range(n)]
    h.addConstrs([xv[i] <= 10.0 * (1.0 - uv[i]) for i in range(n)])
    h.addConstrs([-10.0 * uv[i] <= yv[i] for i in range(n)])
    h.addConstrs([xv[i] - yv[i] >= 1.0 for i in range(n)])
    h.addConstrs([xv[i] - yv[i] <= 15.0 for i in range(n)])
    obj = sum(xv) - 2.0 * sum(yv)
    h.setMinimize()
    h.passModel  # noqa
    h.setObjective(obj) if hasattr(h, "setObjective") else None
    return h, obj


def run_highs(n: int):
    t0 = time.perf_counter()
    h, obj = build_highs(n)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    h.minimize(obj)
    solve = time.perf_counter() - t0
    return build, solve, h.getObjectiveValue(), h.getModelStatus()


def run_gurobi(n: int):
    import gurobipy as gp
    m = gp.Model()
    m.params.OutputFlag = 0
    x = m.addMVar(n, lb=0.0, ub=10.0)
    y = m.addMVar(n, lb=-10.0, ub=0.0)
    u = m.addMVar(n, vtype=gp.GRB.BINARY)
    m.addConstr(x <= 10.0 * (1.0 - u))
    m.addConstr(-10.0 * u <= y)
    m.addConstr(x - y >= 1.0)
    m.addConstr(x - y <= 15.0)
    m.setObjective((x - 2 * y).sum(), gp.GRB.MINIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    return time.perf_counter() - t0, m.ObjVal


if __name__ == "__main__":
    print("same-machine calibration, n = 500 (Gurobi restricted-license cap):")
    gb, gs, gobj, status = run_highs(500)
    print(f"  HiGHS 1.15.1 : build {gb:6.2f}s  solve {gs:7.3f}s  obj {gobj:.1f} ({status})")
    try:
        gsolve, gval = run_gurobi(500)
        print(f"  Gurobi 13    : solve {gsolve:7.3f}s  obj {gval:.1f}")
    except Exception as e:
        print(f"  Gurobi failed: {e}")

    print("\nHiGHS scaling (build = model construction in Python):")
    for n in (1_000, 5_000, 10_000, 50_000, 100_000):
        gb, gs, gobj, status = run_highs(n)
        print(f"  n={n:>7}: build {gb:6.2f}s  solve {gs:8.3f}s  obj {gobj:.1f} ({status})")
