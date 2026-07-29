"""Decomposition presolver for HiGHS.

Root cause this addresses (see examples/discourse_mwe_scaling.py): HiGHS
solves every model through one global LP. When a MIP consists of many
*independent* blocks (no constraint couples two blocks — block-diagonal
constraint matrix), the root LP still runs a global simplex and pays at
least one global pivot per block: time grows superlinearly in the number
of blocks (measured: 0.09 s at 1k blocks -> 112 s at 100k blocks on the
Julia-Discourse complementarity MWE, while the optimum is found at the
root node).

The fix, implemented at the modeling layer:

1. **Detect** connected components of the row/column incidence graph with
   scipy's C-speed ``connected_components`` on the bipartite graph.
2. **Group** structurally identical blocks by signature, so a family of
   100k equal blocks is solved *once* and its solution replicated.
3. **Solve** each unique block (or class representative) as its own tiny
   HiGHS model. Blocks are independent, so the global optimum is the sum
   of block optima (the objective is linear, hence separable).
4. **Stitch** the full primal solution, verify it numerically against the
   original model (one sparse matvec), and return it.

Rows/columns that couple blocks land in the same component automatically,
so coupled models simply fall back to a direct solve — the presolver only
ever helps, and it reports what it did.

Matrix handling: ``getLp()`` may return the constraint matrix in either
column-wise or row-wise storage (row-wise for models built via
``addRow``/``addConstr``). Everything here goes through
:func:`_csr`, which normalizes both to a scipy CSR matrix.

Limitations (honest scope): block-angular models with a handful of
*coupling* rows (e.g. a shared budget) do not decompose under plain
connectivity — every block touches the coupling row, so everything lands
in one component. That case needs Dantzig-Wolfe/Benders machinery and is
out of scope.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import highspy
from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.csgraph import connected_components


@dataclass
class DecompResult:
    status: highspy.HighsModelStatus
    objective: float
    col_value: np.ndarray | None
    n_components: int = 0
    n_unique_classes: int = 0
    largest_component_cols: int = 0
    decomposed: bool = False
    note: str = ""
    timings: dict = field(default_factory=dict)


def _csr(lp) -> csr_matrix:
    """Constraint matrix as CSR (rows x cols), from either storage format."""
    m = lp.a_matrix_
    index = np.asarray(m.index_, dtype=np.int32)
    value = np.asarray(m.value_, dtype=float)
    indptr = np.asarray(m.start_, dtype=np.int64)
    if m.format_ == highspy.MatrixFormat.kColwise:
        if len(indptr) == lp.num_col_:
            indptr = np.append(indptr, len(index))
        return csc_matrix((value, index, indptr),
                          shape=(lp.num_row_, lp.num_col_)).tocsr()
    # kRowwise (models built via addRow/addConstr)
    if len(indptr) == lp.num_row_:
        indptr = np.append(indptr, len(index))
    return csr_matrix((value, index, indptr),
                      shape=(lp.num_row_, lp.num_col_))


def _integrality(lp) -> list[int]:
    ints = getattr(lp, "integrality_", None) or []
    return [int(t) for t in ints]


def _extract_block(A: csr_matrix, lp, col_ids: np.ndarray,
                   row_ids: np.ndarray, integrality) -> dict:
    sub = A[row_ids][:, col_ids].tocsc()
    return {
        "num_col": len(col_ids),
        "num_row": len(row_ids),
        "a_start": sub.indptr.astype(np.int64),
        "a_index": sub.indices.astype(np.int32),
        "a_value": sub.data.astype(float),
        "col_lower": np.asarray(lp.col_lower_)[col_ids],
        "col_upper": np.asarray(lp.col_upper_)[col_ids],
        "col_cost": np.asarray(lp.col_cost_)[col_ids],
        "integrality": ([integrality[c] for c in col_ids]
                        if len(integrality) == lp.num_col_ else []),
        "row_lower": np.asarray(lp.row_lower_)[row_ids],
        "row_upper": np.asarray(lp.row_upper_)[row_ids],
    }


def _solve_block(sub: dict, sense_min: bool) -> tuple:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_ = sub["num_col"]
    lp.num_row_ = sub["num_row"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = sub["a_start"]
    lp.a_matrix_.index_ = sub["a_index"]
    lp.a_matrix_.value_ = sub["a_value"]
    lp.col_lower_ = sub["col_lower"]
    lp.col_upper_ = sub["col_upper"]
    lp.col_cost_ = sub["col_cost"]
    lp.row_lower_ = sub["row_lower"]
    lp.row_upper_ = sub["row_upper"]
    lp.sense_ = (highspy.ObjSense.kMinimize if sense_min
                 else highspy.ObjSense.kMaximize)
    h.passModel(lp)
    if sub["integrality"]:
        h.changeColsIntegrality(
            len(sub["integrality"]),
            np.arange(len(sub["integrality"]), dtype=np.int32),
            np.array([highspy.HighsVarType(t) for t in sub["integrality"]]),
        )
    h.run()
    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        return status, h.getObjectiveValue(), np.array(h.getSolution().col_value)
    return status, 0.0, None


def _direct_solve(h, timings, n_components, n_classes, note) -> DecompResult:
    t1 = time.perf_counter()
    h.run()
    timings["direct_solve"] = time.perf_counter() - t1
    status = h.getModelStatus()
    optimal = status == highspy.HighsModelStatus.kOptimal
    sol = h.getSolution()
    return DecompResult(
        status=status,
        objective=h.getObjectiveValue() if optimal else 0.0,
        col_value=np.array(sol.col_value) if sol.value_valid else None,
        n_components=n_components, n_unique_classes=n_classes,
        decomposed=False, note=note, timings=timings,
    )


def _classify_blocks(A: csr_matrix, lp, col_labels: np.ndarray,
                     row_labels: np.ndarray, n_components: int,
                     integrality):
    """Group blocks by exact structural signature, vectorized.

    One global argsort puts each block's rows/columns/nonzeros in
    contiguous, canonical order; per-block signatures are then contiguous
    byte slices (no per-block scipy slicing). Returns ``classes`` mapping
    signature -> list of block ids, plus the block column/row id lists.
    """
    n_rows, n_cols = A.shape

    # Contiguous column/row ordering by block label.
    col_order = np.argsort(col_labels, kind="stable")
    row_order = np.argsort(row_labels, kind="stable")
    col_counts = np.bincount(col_labels, minlength=n_components)
    row_counts = np.bincount(row_labels, minlength=n_components)
    col_off = np.concatenate(([0], np.cumsum(col_counts)))
    row_off = np.concatenate(([0], np.cumsum(row_counts)))
    comp_col_ids = [col_order[col_off[k]:col_off[k + 1]] for k in range(n_components)]
    comp_row_ids = [row_order[row_off[k]:row_off[k + 1]] for k in range(n_components)]

    # Rank of each row/column within its own block's canonical order.
    col_rank = np.empty(n_cols, dtype=np.int64)
    col_rank[col_order] = np.arange(n_cols) - col_off[col_labels[col_order]]
    row_rank = np.empty(n_rows, dtype=np.int64)
    row_rank[row_order] = np.arange(n_rows) - row_off[row_labels[row_order]]

    # All nonzeros, sorted by (block, row rank, col rank).
    r_idx = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(A.indptr))
    c_idx = A.indices.astype(np.int64)
    vals = np.round(A.data, 12)
    blk = col_labels[c_idx]
    order = np.lexsort((col_rank[c_idx], row_rank[r_idx], blk))
    blk_s, rr_s = blk[order], row_rank[r_idx][order]
    cc_s, v_s = col_rank[c_idx][order], vals[order]
    nnz_counts = np.bincount(blk_s, minlength=n_components)
    nnz_off = np.concatenate(([0], np.cumsum(nnz_counts)))

    # Sorted bound/cost/integrality arrays (contiguous per block).
    cl_s = np.round(np.asarray(lp.col_lower_)[col_order], 12)
    cu_s = np.round(np.asarray(lp.col_upper_)[col_order], 12)
    ccost_s = np.round(np.asarray(lp.col_cost_)[col_order], 12)
    rl_s = np.round(np.asarray(lp.row_lower_)[row_order], 12)
    ru_s = np.round(np.asarray(lp.row_upper_)[row_order], 12)
    int_s = (np.asarray(integrality, dtype=np.int8)[col_order]
             if len(integrality) == n_cols else None)

    classes: dict[tuple, list[int]] = {}
    for k in range(n_components):
        cb, ce = col_off[k], col_off[k + 1]
        rb, re_ = row_off[k], row_off[k + 1]
        nb, ne = nnz_off[k], nnz_off[k + 1]
        key = (
            cl_s[cb:ce].tobytes(), cu_s[cb:ce].tobytes(),
            ccost_s[cb:ce].tobytes(),
            int_s[cb:ce].tobytes() if int_s is not None else b"",
            rl_s[rb:re_].tobytes(), ru_s[rb:re_].tobytes(),
            rr_s[nb:ne].tobytes(), cc_s[nb:ne].tobytes(),
            v_s[nb:ne].tobytes(),
        )
        classes.setdefault(key, []).append(k)
    return classes, comp_col_ids, comp_row_ids


def solve_decomposed(h: highspy.Highs, max_class_solves: int = 1000,
                     verify: bool = True) -> DecompResult:
    """Solve a HiGHS model via block decomposition when profitable.

    Falls back to a direct solve for single-component models and for
    quadratic objectives. Never changes the *model* in ``h``; on success
    the stitched solution is installed into ``h`` (via ``setSolution``)
    so ``h.getSolution()`` is valid. Returns a :class:`DecompResult`
    with the global solution.
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    if h.getHessianNumNz() > 0:
        # Block solves and verification are linear-only; decomposing
        # would silently drop the quadratic objective.
        return _direct_solve(h, timings, 0, 0,
                             "quadratic objective: solved directly")
    lp = h.getLp()
    sense_min = (getattr(lp, "sense_", highspy.ObjSense.kMinimize)
                 == highspy.ObjSense.kMinimize)
    offset = float(getattr(lp, "offset_", 0.0) or 0.0)
    integrality = _integrality(lp)

    A = _csr(lp)
    # Bipartite incidence graph [0 A; A^T 0]: one node per row and column.
    B = bmat([[None, A], [A.T, None]], format="csr")
    n_components, labels = connected_components(B, directed=False,
                                                return_labels=True)
    col_labels = labels[lp.num_row_:]
    timings["detect"] = time.perf_counter() - t0

    if n_components <= 1:
        return _direct_solve(h, timings, 1, 0,
                             "single component: solved directly")

    # Group components; identical blocks are solved once.
    t1 = time.perf_counter()
    row_labels = labels[:lp.num_row_]
    classes, comp_col_ids, comp_row_ids = _classify_blocks(
        A, lp, col_labels, row_labels, n_components, integrality)
    timings["classify"] = time.perf_counter() - t1

    if len(classes) > max_class_solves:
        return _direct_solve(
            h, timings, n_components, len(classes),
            f"{len(classes)} unique block classes exceed "
            f"max_class_solves={max_class_solves}: solved directly")

    t1 = time.perf_counter()
    solution = np.zeros(lp.num_col_)
    objective = offset
    overall = highspy.HighsModelStatus.kOptimal
    rep_cache: dict[int, tuple] = {}
    for key, members in classes.items():
        rep = members[0]
        if rep not in rep_cache:
            sub = _extract_block(A, lp, comp_col_ids[rep], comp_row_ids[rep],
                                 integrality)
            rep_cache[rep] = _solve_block(sub, sense_min)
        status, obj, sol = rep_cache[rep]
        if status != highspy.HighsModelStatus.kOptimal:
            overall = status  # infeasible/unbounded propagates
            break
        objective += obj * len(members)
        for k in members:
            solution[comp_col_ids[k]] = sol
    timings["block_solves"] = time.perf_counter() - t1

    optimal = overall == highspy.HighsModelStatus.kOptimal
    result = DecompResult(
        status=overall,
        objective=objective if optimal else 0.0,
        col_value=solution if optimal else None,
        n_components=n_components,
        n_unique_classes=len(classes),
        largest_component_cols=max(len(c) for c in comp_col_ids),
        decomposed=True,
        note=f"{n_components} blocks in {len(classes)} equivalence classes",
        timings=timings,
    )

    if verify and result.col_value is not None:
        t1 = time.perf_counter()
        ok, msg = _verify_solution(A, lp, solution, offset, result.objective)
        timings["verify"] = time.perf_counter() - t1
        if not ok:
            return DecompResult(
                status=highspy.HighsModelStatus.kNotset, objective=0.0,
                col_value=None, n_components=n_components,
                n_unique_classes=len(classes), decomposed=False,
                note=f"verification failed, refusing result: {msg}",
                timings=timings)
        result.note += "; stitched solution verified numerically"
    if result.col_value is not None:
        # Install the stitched solution so h.getSolution() is valid and
        # downstream consumers (e.g. predictor-constraint get_error) work.
        stitched = highspy.HighsSolution()
        stitched.col_value = result.col_value
        h.setSolution(stitched)
    return result


def _verify_solution(A: csr_matrix, lp, x: np.ndarray, offset: float,
                     claimed_obj: float) -> tuple[bool, str]:
    """Feasibility + objective check of a stitched solution, one matvec."""
    tol = 1e-6
    cl, cu = np.asarray(lp.col_lower_), np.asarray(lp.col_upper_)
    if (x < cl - tol).any() or (x > cu + tol).any():
        bad = int(np.argmax((x < cl - tol) | (x > cu + tol)))
        return False, f"column bound violated at col {bad}"
    activity = A @ x
    rl, ru = np.asarray(lp.row_lower_), np.asarray(lp.row_upper_)
    if (activity < rl - tol).any() or (activity > ru + tol).any():
        bad = int(np.argmax((activity < rl - tol) | (activity > ru + tol)))
        return False, f"row bound violated at row {bad}"
    obj = offset + float(np.asarray(lp.col_cost_) @ x)
    if abs(obj - claimed_obj) > 1e-4 * max(1.0, abs(claimed_obj)):
        return False, f"objective mismatch ({obj} vs {claimed_obj})"
    return True, "ok"
