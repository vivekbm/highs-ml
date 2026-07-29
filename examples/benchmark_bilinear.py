"""Bilinear frontier benchmark: highs_ml (uniform / adaptive) vs Gurobi's
non-convex spatial branch-and-bound.

Problem family (n products):
    maximize    sum_i w_i * y_i
    subject to  y_i = x_i * z_i         (bilinear, continuous)
                sum_i x_i <= Bx
                sum_i z_i <= Bz
                x_i, z_i in [0.5, 2.0]

Interior optima (budgets bind mid-range), so the bilinear terms genuinely
matter. Gurobi solves the exact non-convex MIQCP (NonConvex=2, restricted
pip license); highs_ml uses the certified piecewise-McCormick envelope —
once with a uniform partition, once with adaptive refinement.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import highspy
from highs_ml import add_adaptive_bilinear, add_bilinear_constr


def build(n, w, bx, bz):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    xs = [h.addVariable(lb=0.5, ub=2.0, name=f"x{i}") for i in range(n)]
    zs = [h.addVariable(lb=0.5, ub=2.0, name=f"z{i}") for i in range(n)]
    h.addConstr(sum(xs) <= bx, name="bx")
    h.addConstr(sum(zs) <= bz, name="bz")
    return h, xs, zs


def solve_highs_uniform(n, w, bx, bz, tol):
    h, xs, zs = build(n, w, bx, bz)
    ys = [add_bilinear_constr(h, xs[i], zs[i], tol=tol, name=f"b{i}",
                              ) for i in range(n)]
    h.setObjective(sum(w[i] * ys[i] for i in range(n)),
                   highspy.ObjSense.kMaximize)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    values = h.getSolution().col_value
    obj = h.getObjectiveValue()
    err = max(abs(values[ys[i].index]
                  - values[xs[i].index] * values[zs[i].index])
              for i in range(n))
    true = sum(w[i] * values[xs[i].index] * values[zs[i].index]
               for i in range(n))
    return obj, true, err, dt, h.numVariables


def solve_highs_adaptive(n, w, bx, bz, tol):
    h, xs, zs = build(n, w, bx, bz)
    embs = [add_adaptive_bilinear(h, xs[i], zs[i], tol=tol, name=f"b{i}",
                                  n_initial=4) for i in range(n)]
    h.setObjective(sum(w[i] * embs[i].y for i in range(n)),
                   highspy.ObjSense.kMaximize)
    t0 = time.perf_counter()
    total_ref = 0
    # round-robin refinement over the product terms
    for _round in range(30):
        h.run()
        status = h.getModelStatus()
        errs = [e.solution_error() for e in embs]
        if max(errs) <= tol or status != highspy.HighsModelStatus.kOptimal:
            break
        for i, e in enumerate(embs):
            if errs[i] > tol:
                values = h.getSolution().col_value
                if e.refine(float(values[xs[i].index])):
                    total_ref += 1
    dt = time.perf_counter() - t0
    values = h.getSolution().col_value
    obj = h.getObjectiveValue()
    err = max(abs(values[embs[i].y.index]
                  - values[xs[i].index] * values[zs[i].index])
              for i in range(n))
    true = sum(w[i] * values[xs[i].index] * values[zs[i].index]
               for i in range(n))
    return obj, true, err, dt, h.numVariables, total_ref


def solve_gurobi(n, w, bx, bz):
    import gurobipy as gp
    m = gp.Model()
    m.params.OutputFlag = 0
    m.params.NonConvex = 2
    x = m.addMVar(n, lb=0.5, ub=2.0)
    z = m.addMVar(n, lb=0.5, ub=2.0)
    y = m.addMVar(n, lb=-10.0, ub=10.0)
    m.addConstr(x.sum() <= bx)
    m.addConstr(z.sum() <= bz)
    m.addConstr(y == x * z)
    m.setObjective(float(np.sum(w)) * y.sum() if False else
                   gp.quicksum(w[i] * y[i] for i in range(n)),
                   gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    dt = time.perf_counter() - t0
    return m.ObjVal, dt


def main():
    rng = np.random.default_rng(4)
    print(f"{'n':>3} {'method':<18} {'objective':>10} {'true obj':>10} "
          f"{'cert err':>10} {'time':>7} {'vars':>6}")
    for n in (2, 6, 12):
        w = rng.uniform(0.5, 1.5, size=n)
        bx, bz = 1.1 * n, 0.9 * n
        if n <= 6:
            obj, true, err, dt, nv = solve_highs_uniform(n, w, bx, bz, 0.05)
            print(f"{n:>3} {'highs uniform':<18} {obj:10.4f} {true:10.4f} "
                  f"{err:10.2e} {dt:6.2f}s {nv:6d}")
        else:
            print(f"{n:>3} {'highs uniform':<18} (skipped: static uniform "
                  f"partitions scale poorly — see n=6 trend; adaptive is "
                  f"the answer)")
        obj, true, err, dt, nv, nref = solve_highs_adaptive(n, w, bx, bz,
                                                          0.05)
        print(f"{n:>3} {'highs adaptive':<18} {obj:10.4f} {true:10.4f} "
              f"{err:10.2e} {dt:6.2f}s {nv:6d} ({nref} refines)")
        try:
            gobj, gdt = solve_gurobi(n, w, bx, bz)
            print(f"{n:>3} {'gurobi nonconvex':<18} {gobj:10.4f} "
                  f"{gobj:10.4f} {'exact':>10} {gdt:6.2f}s")
        except Exception as e:
            print(f"{n:>3} gurobi failed: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
