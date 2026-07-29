"""Structure-aware solver routing: highs_ml.solve_auto.

Honest dispatch based on what each method is actually good at:

* **fully separable** models (no coupling rows) -> decomposition
  presolver (exact, orders-of-magnitude faster);
* **block-angular with identical/few block classes** -> Dantzig-Wolfe
  (exact bound + restricted master; branch-and-price when the root is
  not already tight);
* **block-angular with many unique classes** -> Dantzig-Wolfe for the
  bound, but direct HiGHS is often the better exact tool here (DW's
  textbook weakness); we run B&P only when the root gap is small;
* **anything else** -> direct HiGHS.
"""

from __future__ import annotations

import time

import numpy as np
import highspy

from .decomp import DecompResult, solve_decomposed
from .dw import DWResult, solve_bp, solve_dw


def solve_auto(h: highspy.Highs, node_budget: int = 32,
               bp_gap_threshold: float = 0.001,
               max_iterations: int = 100, tol: float = 1e-6,
               max_coupling: int = 200):
    """Solve ``h`` with the structure-appropriate method.

    Returns a ``(method, result)`` tuple where method is one of
    'decomposed', 'dantzig-wolfe', 'branch-and-price', 'direct'.
    """
    # 1. fully separable?
    res = solve_decomposed(h)
    if res.decomposed:
        res.note = f"[solve_auto: decomposed] {res.note}"
        return "decomposed", res

    # 2. block-angular?
    dw = solve_dw(h, max_iterations=max_iterations, tol=tol,
                  max_coupling=max_coupling)
    if not dw.decomposed:
        # 3. direct fallback
        t0 = time.perf_counter()
        h.run()
        status = h.getModelStatus()
        sol = h.getSolution()
        direct = DecompResult(
            status=status,
            objective=(h.getObjectiveValue()
                       if status == highspy.HighsModelStatus.kOptimal else 0.0),
            col_value=(np.array(sol.col_value) if sol.value_valid else None),
            decomposed=False,
            note=f"[solve_auto: direct] {dw.note}",
            timings={"direct_solve": time.perf_counter() - t0},
        )
        return "direct", direct

    if dw.gap <= tol:
        dw.note = f"[solve_auto: dantzig-wolfe, proven at root] {dw.note}"
        return "dantzig-wolfe", dw

    # 3. gap remains: branch-and-price when the root is nearly tight or
    #    blocks are aggregated (B&P's strong case).
    if dw.gap <= bp_gap_threshold or dw.n_unique_classes < dw.n_blocks:
        bp = solve_bp(h, max_iterations=max_iterations,
                      node_budget=node_budget, tol=tol,
                      max_coupling=max_coupling)
        if bp.col_value is not None:
            bp.note = f"[solve_auto: branch-and-price] {bp.note}"
            return "branch-and-price", bp

    dw.note = (f"[solve_auto: dantzig-wolfe, gap {dw.gap:.3%} accepted "
               f"(many unique classes; direct solve recommended for "
               f"exactness)] {dw.note}")
    return "dantzig-wolfe", dw
