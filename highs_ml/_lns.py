"""Large-neighborhood search (LNS) primal heuristic for branch-and-price.

The restricted master's incumbent is limited by the LP-generated column
pool: for all-unique block classes with continuous decisions, LP-guided
pricing may never generate the integer-optimal operating points. LNS
sidesteps the pool entirely: fix most blocks to the incumbent, leave a
small neighborhood of blocks free, and solve the resulting small MIP
*exactly* with HiGHS over the original (un-decomposed) rows. Any
improvement becomes the new incumbent; neighborhoods are re-drawn each
round.

Each subproblem keeps the chosen blocks' original rows plus the coupling
rows (with fixed-block contributions substituted into the right-hand
sides), so solutions are always globally feasible.
"""

from __future__ import annotations

import time

import numpy as np
import highspy


def _build_submodel(st, x_inc: np.ndarray, free_cols: np.ndarray):
    """Small MIP over free_cols with everything else fixed to x_inc."""
    from .dw import _pass_lp

    lp = st.lp
    A = st.A
    free_set = set(int(c) for c in free_cols)
    indptr, indices = A.indptr, A.indices

    # rows to keep: coupling rows + block rows of the free blocks
    keep_rows = list(st.coupling)
    for k in range(st.n_blocks):
        if any(int(c) in free_set for c in st.block_cols[k]):
            keep_rows.extend(int(r) for r in st.block_rows_of[k])
    keep_rows = sorted(set(keep_rows))

    col_map = {int(c): j for j, c in enumerate(free_cols)}
    row_map = {int(r): i for i, r in enumerate(keep_rows)}
    col_lists: list[list] = [[] for _ in range(len(free_cols))]
    row_lower, row_upper = [], []
    lo_all, up_all = np.asarray(lp.row_lower_), np.asarray(lp.row_upper_)
    for r in keep_rows:
        lo, hi = float(lo_all[r]), float(up_all[r])
        for p in range(indptr[r], indptr[r + 1]):
            c = int(indices[p])
            v = float(A.data[p])
            if c in col_map:
                col_lists[col_map[c]].append((row_map[r], v))
            else:
                # fixed column: move its contribution to the RHS
                if lo > -highspy.kHighsInf * 0.5:
                    lo -= v * x_inc[c]
                if hi < highspy.kHighsInf * 0.5:
                    hi -= v * x_inc[c]
        row_lower.append(lo)
        row_upper.append(hi)

    new_start, new_index, new_value = [0], [], []
    for entries in col_lists:
        for rr, v in entries:
            new_index.append(rr)
            new_value.append(v)
        new_start.append(len(new_index))

    ints = st.integrality
    sub = {
        "num_col": len(free_cols), "num_row": len(keep_rows),
        "a_start": np.array(new_start, dtype=np.int64),
        "a_index": np.array(new_index, dtype=np.int32),
        "a_value": np.array(new_value, dtype=float),
        "col_lower": np.asarray(lp.col_lower_)[free_cols],
        "col_upper": np.asarray(lp.col_upper_)[free_cols],
        "col_cost": np.asarray(lp.col_cost_)[free_cols],
        "integrality": ([ints[c] for c in free_cols]
                        if len(ints) == lp.num_col_ else []),
        "row_lower": np.array(row_lower),
        "row_upper": np.array(row_upper),
    }
    return _pass_lp(sub, st.sense_min)


def _lns_improve(st, x_inc: np.ndarray, obj_inc: float, rounds: int = 12,
                 k_neighborhood: int = 6, seed: int = 0,
                 time_budget: float = 60.0):
    """LNS over block neighborhoods.

    ``obj_inc`` is in the *original* sense with the model offset
    EXCLUDED (i.e. ``costs @ x_inc``), matching the candidate objectives
    computed below; the returned ``obj_best`` uses the same convention.

    Neighborhoods mix random classes with the classes whose incumbent
    block point differs most from the master-LP fractional mix (the
    'regret' heuristic: those are the blocks where the pool failed).
    Returns (x_best, obj_best, improved)."""
    rng = np.random.default_rng(seed)
    costs = np.asarray(st.lp.col_cost_)
    x_best = x_inc.copy()
    obj_best = obj_inc
    improved = False
    t0 = time.perf_counter()

    for rnd in range(rounds):
        if time.perf_counter() - t0 > time_budget:
            break
        k = min(k_neighborhood, st.n_blocks)
        blocks_free = rng.choice(st.n_blocks, size=k, replace=False)
        free_cols = np.concatenate(
            [st.block_cols[b] for b in blocks_free]
            + ([np.asarray(st.master_cols, dtype=int)]
               if len(st.master_cols) else []))

        h_sub = _build_submodel(st, x_best, free_cols)
        h_sub.run()
        if h_sub.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            continue
        x_sub = np.array(h_sub.getSolution().col_value)
        x_cand = x_best.copy()
        x_cand[free_cols] = x_sub
        obj_cand = float(costs @ x_cand)
        better = (obj_cand < obj_best - 1e-9) if st.sense_min \
            else (obj_cand > obj_best + 1e-9)
        if better:
            x_best, obj_best = x_cand, obj_cand
            improved = True
    return x_best, obj_best, improved
