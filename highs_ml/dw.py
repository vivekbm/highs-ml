"""Dantzig-Wolfe decomposition for block-angular MILPs with coupling rows.

The decomposition presolver (``decomp.py``) handles *fully separable*
models. Real models — including every ML-embedded one with a shared
budget — are block-angular instead: many independent blocks plus a small
set of coupling rows (and possibly coupling columns).

    min  sum_k c_k x_k
    s.t. sum_k A0_k x_k  (~) b0        <- m0 coupling rows
         B_k x_k         (~) b_k   k   <- independent blocks (may be MIPs)

Algorithm implemented here (the classical workhorse, not full
branch-and-price):

1. **Detect** the bordered block-diagonal structure: rows whose removal
   disconnects the incidence graph become coupling rows (defer-and-
   reinsert heuristic); column components of the remainder are blocks.
2. **Column generation** on the LP master:
       max/min over lambda s.t. convexity + coupling rows
   with per-block MIP pricing problems. Blocks that are structurally
   identical (including their coupling-row coefficients) share one
   pricing problem per iteration. HiGHS's dual convention (verified
   empirically) is: for a min-form LP the reduced cost of a column is
   ``cost - row_dual @ a_col``, with ``row_dual`` used as returned.
3. **Restricted-master heuristic**: solve the final master with the
   convexity variables made binary — each block then picks exactly one of
   its generated operating points, yielding a primal-feasible MIP
   solution. The master LP value is a valid bound, so we report an
   honest optimality gap.

Everything is verified numerically against the original model before
being returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import highspy
from .decomp import _csr, _integrality, _verify_solution


@dataclass
class DWResult:
    status: highspy.HighsModelStatus
    objective: float          # primal (restricted-master) objective, original sense
    bound: float              # master LP bound, original sense
    col_value: np.ndarray | None
    n_blocks: int = 0
    n_coupling_rows: int = 0
    n_master_cols: int = 0
    n_unique_classes: int = 0
    iterations: int = 0
    gap: float = np.nan
    decomposed: bool = False
    note: str = ""
    timings: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# structure detection
# ----------------------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def detect_block_angular(A_csr, max_coupling: int = 200):
    """Detect bordered block-diagonal structure via multi-ordering deferral.

    For each of several row orderings, rows that would merge established
    components are deferred as coupling candidates; the candidate set is
    then *repaired*: blocks are recomputed from scratch with candidates
    removed (healing chain fragments the pass split), and candidates that
    land inside a single block are reabsorbed. The ordering yielding the
    most blocks with the fewest coupling rows wins.

    Returns (coupling_rows, block_of_col, n_blocks, master_cols) or None.
    """
    n_rows, n_cols = A_csr.shape
    indptr, indices = A_csr.indptr, A_csr.indices
    nnz_row = np.diff(indptr)

    def defer_pass(order):
        uf = _UF(n_cols)
        seen = np.zeros(n_cols, dtype=bool)
        deferred = []
        for r in order:
            cols = indices[indptr[r]:indptr[r + 1]]
            if len(cols) == 0:
                continue
            comps = {uf.find(c) for c in cols if seen[c]}
            if len(comps) > 1:
                deferred.append(int(r))
            else:
                first = cols[0]
                for c in cols[1:]:
                    uf.union(first, int(c))
                seen[cols] = True
        return deferred

    def repair(deferred):
        """Blocks = components without candidate rows; shrink candidates."""
        D = set(deferred)
        for _round in range(30):
            uf = _UF(n_cols)
            in_row = np.zeros(n_cols, dtype=bool)
            nontrivial = {}
            for r in range(n_rows):
                if r in D:
                    continue
                cols = indices[indptr[r]:indptr[r + 1]]
                if len(cols) == 0:
                    continue
                first = cols[0]
                for c in cols[1:]:
                    uf.union(first, int(c))
                in_row[cols] = True
                if len(cols) >= 2:
                    for c in cols:
                        nontrivial[uf.find(c)] = True
            shrunk = False
            for d in list(D):
                cols = indices[indptr[d]:indptr[d + 1]]
                comps = {uf.find(c) for c in cols if in_row[c]}
                # 1-nnz fixing rows (e.g. b == 1) create trivial singleton
                # comps that must not block reabsorption.
                nt = {g for g in comps if nontrivial.get(g, False)}
                if len(nt) <= 1:
                    D.remove(d)
                    shrunk = True
            if shrunk:
                continue
            # Island merging: chain fragments (SOS pieces of one block) are
            # comps connected to each other ONLY by degree-2 candidate rows.
            # Merging them is always correctness-safe: over-merged blocks
            # just mean less decomposition, never a wrong model.
            adj2: dict[int, set] = {}
            for d in list(D):
                cols = indices[indptr[d]:indptr[d + 1]]
                comps = {uf.find(c) for c in cols if in_row[c]}
                nt = [g for g in comps if nontrivial.get(g, False)]
                if len(nt) == 2:
                    adj2.setdefault(nt[0], set()).add(nt[1])
                    adj2.setdefault(nt[1], set()).add(nt[0])
            merged = False
            seen_g: set = set()
            for g0 in list(adj2):
                if g0 in seen_g:
                    continue
                group = {g0}
                stack = [g0]
                while stack:
                    u = stack.pop()
                    for v in adj2.get(u, ()):
                        if v not in group:
                            group.add(v)
                            stack.append(v)
                seen_g |= group
                if len(group) >= 2:
                    rep = next(iter(group))
                    for g in group:
                        if g != rep:
                            for c in range(n_cols):
                                if in_row[c] and uf.find(c) == g:
                                    uf.union(rep, c)
                    for d in list(D):
                        cols = indices[indptr[d]:indptr[d + 1]]
                        nt = {uf.find(c) for c in cols if in_row[c]
                              and nontrivial.get(uf.find(c), False)}
                        if len(nt) <= 1:
                            D.remove(d)
                    merged = True
            if not merged:
                break
        # final block assignment
        uf = _UF(n_cols)
        in_row = np.zeros(n_cols, dtype=bool)
        for r in range(n_rows):
            if r in D:
                continue
            cols = indices[indptr[r]:indptr[r + 1]]
            if len(cols) == 0:
                continue
            first = cols[0]
            for c in cols[1:]:
                uf.union(first, int(c))
            in_row[cols] = True
        roots = {}
        block_of_col = np.full(n_cols, -1, dtype=int)
        for c in range(n_cols):
            if in_row[c]:
                root = uf.find(c)
                if root not in roots:
                    roots[root] = len(roots)
                block_of_col[c] = roots[root]
        return sorted(D), block_of_col, len(roots), in_row

    orders = [
        range(n_rows),
        range(n_rows - 1, -1, -1),
        np.argsort(nnz_row, kind="stable"),
        np.argsort(-nnz_row, kind="stable"),
    ]
    best = None
    for order in orders:
        coupling, block_of_col, n_blocks, in_row = repair(defer_pass(order))
        if len(coupling) > max_coupling:
            score = (-1, 0, 0)
        else:
            score = (n_blocks >= 2, n_blocks, -len(coupling))
        if best is None or score > best[0]:
            best = (score, coupling, block_of_col, n_blocks, in_row)

    score, coupling, block_of_col, n_blocks, in_row = best
    coupling = np.array(coupling, dtype=int)
    if n_blocks <= 1 or len(coupling) > max_coupling:
        return None
    master_cols = np.where(~in_row)[0]
    return coupling, block_of_col, n_blocks, master_cols

    coupling = np.array(sorted(deferred), dtype=int)

    # Hub components = articulation points of the comp adjacency graph
    # (comps as nodes, deferred rows as hyperedges). A true linking
    # structure — the budget-style component — is the comp whose removal
    # disconnects the others; block-internal fragments (SOS chain pieces)
    # are never articulation points because they stay connected through
    # the surviving rows.
    comp_ids: dict[int, int] = {}
    for c in range(n_cols):
        root = uf.find(c)
        if root not in comp_ids:
            comp_ids[root] = len(comp_ids)
    n_comp = len(comp_ids)
    adj: list[set] = [set() for _ in range(n_comp + len(deferred))]
    for e, d in enumerate(deferred):
        enode = n_comp + e
        for c in indices[indptr[d]:indptr[d + 1]]:
            g = comp_ids[uf.find(c)]
            adj[g].add(enode)
            adj[enode].add(g)

    def _is_articulation(target: int, total: int) -> bool:
        seen2 = [False] * total
        start = next(i for i in range(n_comp) if i != target)
        stack = [start]
        seen2[start] = True
        seen2[target] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen2[v]:
                    seen2[v] = True
                    stack.append(v)
        return not all(seen2)

    total_nodes = n_comp + len(deferred)
    hubs = {root for root, cid in comp_ids.items()
            if _is_articulation(cid, total_nodes)}
    if not hubs:
        return None  # no linking structure found

    hub_cols = np.array([c for c in range(n_cols) if uf.find(c) in hubs],
                        dtype=int)
    is_hub_col = np.zeros(n_cols, dtype=bool)
    is_hub_col[hub_cols] = True

    # Recompute components with hub columns removed: these are the blocks.
    uf2 = _UF(n_cols)
    in_row = np.zeros(n_cols, dtype=bool)
    row_block_cols = []
    for r in range(n_rows):
        cols = indices[indptr[r]:indptr[r + 1]]
        nb = cols[~is_hub_col[cols]]
        row_block_cols.append(nb)
        if len(nb) == 0:
            continue
        first = nb[0]
        for c in nb[1:]:
            uf2.union(int(first), int(c))
        in_row[nb] = True

    # Assign hub columns to the unique block their rows touch, if any;
    # otherwise they are linking (master) columns.
    block_of_col = np.full(n_cols, -1, dtype=int)
    uniq = {}
    for c in range(n_cols):
        if in_row[c] and not is_hub_col[c]:
            root = uf2.find(c)
            if root not in uniq:
                uniq[root] = len(uniq)
            block_of_col[c] = uniq[root]
    n_blocks = len(uniq)

    # Assign hub columns to the unique block their rows touch, if any;
    # otherwise they are linking (master) columns.
    rows_of_col = [[] for _ in range(n_cols)]
    for r in range(n_rows):
        for c in indices[indptr[r]:indptr[r + 1]]:
            rows_of_col[c].append(r)
    for hc in hub_cols:
        touched = set()
        for r in rows_of_col[hc]:
            for c in indices[indptr[r]:indptr[r + 1]]:
                if block_of_col[c] >= 0:
                    touched.add(block_of_col[c])
        if len(touched) == 1:
            block_of_col[hc] = touched.pop()

    # Row assignment: rows spanning >=2 blocks, or rows containing a
    # still-unassigned (linking) hub column, are coupling rows.
    coupling_list = []
    row_block = np.full(n_rows, -1, dtype=int)
    for r in range(n_rows):
        cols = indices[indptr[r]:indptr[r + 1]]
        blks = {block_of_col[c] for c in cols if block_of_col[c] >= 0}
        has_linking = any(is_hub_col[c] and block_of_col[c] < 0 for c in cols)
        if len(blks) >= 2 or has_linking or len(blks) == 0:
            coupling_list.append(r)
        else:
            row_block[r] = blks.pop()
    coupling = np.array(sorted(coupling_list), dtype=int)
    if len(coupling) > max_coupling or n_blocks <= 1:
        return None

    master_cols = np.where(block_of_col < 0)[0]
    return coupling, block_of_col, n_blocks, master_cols


# ----------------------------------------------------------------------
# HiGHS helpers
# ----------------------------------------------------------------------
def _pass_lp(sub: dict, sense_min: bool) -> highspy.Highs:
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
            np.array([highspy.HighsVarType(t) for t in sub["integrality"]]))
    return h


def _solve_master(rows_bounds, cols, master_var_specs, binary_lam: bool,
                  sense_min: bool):
    """Build and solve the restricted master.

    rows_bounds: list of (lo, hi) for [coupling rows] + [convexity rows]
    cols: list of dicts with keys: kind ('lambda'|'var'), cost,
          coeffs (dict row->coef), var spec (lb,ub,int) for kind='var'
    Returns (status, objective, duals, lam_values, master_var_values).
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    n_rows = len(rows_bounds)
    n_cols = len(cols)
    lp.num_row_ = n_rows
    lp.num_col_ = n_cols
    lp.row_lower_ = np.array([b[0] for b in rows_bounds])
    lp.row_upper_ = np.array([b[1] for b in rows_bounds])
    lp.sense_ = (highspy.ObjSense.kMinimize if sense_min
                 else highspy.ObjSense.kMaximize)

    start, index, value = [0], [], []
    costs, lows, ups, ints = [], [], [], []
    for j, col in enumerate(cols):
        coeffs = col["coeffs"]
        for r in sorted(coeffs):
            index.append(r)
            value.append(coeffs[r])
        start.append(len(index))
        costs.append(col["cost"])
        if col["kind"] == "lambda":
            lo = col.get("lb")
            lows.append(0.0 if lo is None else float(lo))
            ub = col.get("ub")
            if ub is not None:
                ups.append(float(ub))
                ints.append(highspy.HighsVarType.kInteger if binary_lam
                            else highspy.HighsVarType.kContinuous)
            else:
                ups.append(highspy.kHighsInf)
                ints.append(highspy.HighsVarType.kContinuous)
        else:
            lb, ub, is_int = col["var"]
            lows.append(lb)
            ups.append(ub)
            ints.append(highspy.HighsVarType.kInteger if is_int
                        else highspy.HighsVarType.kContinuous)
    lp.col_cost_ = np.array(costs)
    lp.col_lower_ = np.array(lows)
    lp.col_upper_ = np.array(ups)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array(start, dtype=np.int64)
    lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
    lp.a_matrix_.value_ = np.array(value, dtype=float)
    h.passModel(lp)
    # During column generation (binary_lam=False) the master must be a
    # pure LP: applying integrality (even only to integer linking
    # columns) turns it into a MIP, whose solution carries no duals and
    # would silence pricing entirely.
    if binary_lam:
        h.changeColsIntegrality(n_cols, np.arange(n_cols, dtype=np.int32),
                                np.array(ints))
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        return status, 0.0, None, None, None
    sol = h.getSolution()
    if not binary_lam:
        assert bool(sol.dual_valid), "master LP solve returned no duals"
    duals = np.array(sol.row_dual)
    values = np.array(sol.col_value)
    lam_vals = {j: values[j] for j, c in enumerate(cols) if c["kind"] == "lambda"}
    var_vals = {cols[j]["var_index"]: values[j]
                for j, c in enumerate(cols) if c["kind"] == "var"}
    return status, h.getObjectiveValue(), duals, lam_vals, var_vals


# ----------------------------------------------------------------------
# shared state + reusable column-generation / master helpers
# ----------------------------------------------------------------------
class _DWState:
    """Everything branch-and-price nodes share: structure, block pricing
    submodels, the master row scaffold, and the (warm) column pool."""

    def __init__(self, h, max_coupling):
        from types import SimpleNamespace
        from .decomp import _classify_blocks

        lp = h.getLp()
        self.lp = lp
        self.sense_min = (getattr(lp, "sense_", highspy.ObjSense.kMinimize)
                          == highspy.ObjSense.kMinimize)
        self.sign = 1.0 if self.sense_min else -1.0  # internal min form
        self.offset = float(getattr(lp, "offset_", 0.0) or 0.0)
        self.integrality = _integrality(lp)
        self.A = _csr(lp)

        detected = detect_block_angular(self.A, max_coupling=max_coupling)
        if detected is None:
            raise ValueError(f"model is not usefully block-angular "
                             f"(>{max_coupling} coupling rows)")
        (self.coupling, block_of_col, self.n_blocks,
         self.master_cols) = detected
        self.n_coupling = len(self.coupling)
        if self.n_coupling == 0 or self.n_blocks <= 1:
            raise ValueError(f"not block-angular (blocks={self.n_blocks}, "
                             f"coupling rows={self.n_coupling})")
        self.block_of_col = block_of_col

        # block rows: rows whose columns sit in exactly one block
        A = self.A
        row_blk = np.full(A.shape[0], -1, dtype=int)
        for r in range(A.shape[0]):
            cols_r = A.indices[A.indptr[r]:A.indptr[r + 1]]
            blks = np.unique(block_of_col[cols_r])
            if len(blks) == 1 and blks[0] >= 0:
                row_blk[r] = blks[0]

        keep = np.where(row_blk >= 0)[0]
        A_sub = A[keep]
        col_labels = block_of_col.copy()
        col_labels[col_labels < 0] = self.n_blocks  # dummy for master cols
        lp_sub = SimpleNamespace(
            num_col_=lp.num_col_, num_row_=len(keep),
            col_lower_=lp.col_lower_, col_upper_=lp.col_upper_,
            col_cost_=lp.col_cost_,
            row_lower_=np.asarray(lp.row_lower_)[keep],
            row_upper_=np.asarray(lp.row_upper_)[keep],
        )
        classes, comp_col_ids, comp_row_ids_local = _classify_blocks(
            A_sub, lp_sub, col_labels, row_blk[keep],
            self.n_blocks + 1, self.integrality)

        # Blocks identical in their own rows/costs may still differ in
        # their coupling-row coefficients; the representative's A0 column
        # is substituted for every class member, so blocks may only share
        # a class when their A0 columns match too.
        A0_csc = A[self.coupling].tocsc()
        a0_ptr, a0_idx = A0_csc.indptr, A0_csc.indices
        a0_val = np.round(A0_csc.data, 12)

        def _a0_sig(cols_k):
            parts = []
            for c in cols_k:
                b, e = a0_ptr[c], a0_ptr[c + 1]
                parts.append(np.int64(e - b).tobytes())
                parts.append(a0_idx[b:e].tobytes())
                parts.append(a0_val[b:e].tobytes())
            return b"".join(parts)

        refined: dict = {}
        for sig, members in classes.items():
            for k in members:
                a0k = _a0_sig(comp_col_ids[k]) if k < self.n_blocks else b"m"
                refined.setdefault((sig, a0k), []).append(k)
        classes = refined

        self.block_cols = comp_col_ids[:self.n_blocks]
        self.block_rows_of = [keep[comp_row_ids_local[k]]
                              for k in range(self.n_blocks)]

        class_of_block = [-1] * self.n_blocks
        for cl, (_sig, members) in enumerate(classes.items()):
            for k in members:
                if k < self.n_blocks:
                    class_of_block[k] = cl
        uniq_classes = sorted(set(class_of_block))
        remap_c = {old: new for new, old in enumerate(uniq_classes)}
        self.class_of_block = [remap_c[c] for c in class_of_block]
        self.n_classes = len(uniq_classes)
        rep_block = [None] * self.n_classes
        for k in range(self.n_blocks):
            if rep_block[self.class_of_block[k]] is None:
                rep_block[self.class_of_block[k]] = k
        self.rep_block = rep_block

        # per-class pricing submodels
        lp_ = self.lp
        A0 = A[self.coupling]
        self.class_subs, self.class_A0 = [], []
        for cl in range(self.n_classes):
            k = rep_block[cl]
            cols = self.block_cols[k]
            sub = A[self.block_rows_of[k]][:, cols].tocsc()
            self.class_subs.append({
                "num_col": len(cols), "num_row": len(self.block_rows_of[k]),
                "a_start": sub.indptr.astype(np.int64),
                "a_index": sub.indices.astype(np.int32),
                "a_value": sub.data.astype(float),
                "col_lower": np.asarray(lp_.col_lower_)[cols],
                "col_upper": np.asarray(lp_.col_upper_)[cols],
                "col_cost": np.asarray(lp_.col_cost_)[cols] * self.sign,
                "integrality": ([self.integrality[c] for c in cols]
                                if len(self.integrality) == lp_.num_col_
                                else []),
                "row_lower": np.asarray(lp_.row_lower_)[self.block_rows_of[k]],
                "row_upper": np.asarray(lp_.row_upper_)[self.block_rows_of[k]],
            })
            self.class_A0.append(A0[:, cols].toarray())

        # initial columns: one feasible point per class
        self.class_columns: list[list[dict]] = []
        self.class_members: list[list[int]] = []
        for cl in range(self.n_classes):
            self.class_columns.append([])
            self.class_members.append([k for k in range(self.n_blocks)
                                       if self.class_of_block[k] == cl])
        for cl in range(self.n_classes):
            hp = _pass_lp(self.class_subs[cl], True)
            hp.run()
            if hp.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                raise ValueError(f"block class {cl} is infeasible")
            x0 = np.array(hp.getSolution().col_value)
            self.class_columns[cl].append({
                "x": x0,
                "coeffs": self.class_A0[cl] @ x0,
                "cost": float(self.class_subs[cl]["col_cost"] @ x0),
            })

        # master row scaffold: [coupling rows] + [class convexity rows]
        conv = [(float(len(self.class_members[cl])),
                 float(len(self.class_members[cl])))
                for cl in range(self.n_classes)]
        coupling_bounds = [(float(np.asarray(lp_.row_lower_)[r]),
                            float(np.asarray(lp_.row_upper_)[r]))
                           for r in self.coupling]
        self.rows_bounds = coupling_bounds + conv

        self.master_var_cols = []
        for j in self.master_cols:
            coeffs = {i: float(A0[i, j]) for i in range(self.n_coupling)
                      if A0[i, j] != 0.0}
            is_int = (len(self.integrality) == lp_.num_col_
                      and self.integrality[j] != 0)
            self.master_var_cols.append({
                "kind": "var", "cost": float(lp_.col_cost_[j]) * self.sign,
                "coeffs": coeffs, "var_index": int(j),
                "var": (float(lp_.col_lower_[j]), float(lp_.col_upper_[j]),
                        is_int),
            })

    def assemble_cols(self, binary: bool, lam_bounds=None,
                      blk_bounds=None, with_artificials: bool = True):
        """Master columns; ``lam_bounds`` maps (class, q) -> (lo, hi).

        ``blk_bounds`` maps class -> {block column index -> (lo, hi)} from
        original-variable branching: pool columns violating the current
        node's branch are excluded (they remain valid elsewhere).
        """
        lam_bounds = lam_bounds or {}
        blk_bounds = blk_bounds or {}
        cols = list(self.master_var_cols)
        for cl in range(self.n_classes):
            count = len(self.class_members[cl])
            bb = blk_bounds.get(cl)
            for q, col in enumerate(self.class_columns[cl]):
                if bb is not None:
                    xq = col["x"]
                    if any(xq[j] < lo - 1e-9 or xq[j] > hi + 1e-9
                           for j, (lo, hi) in bb.items()):
                        continue  # violates this node's branch
                coeffs = {i: v for i, v in enumerate(col["coeffs"])
                          if v != 0.0}
                coeffs[self.n_coupling + cl] = 1.0  # class convexity row
                lo, hi = lam_bounds.get((cl, q), (None, None))
                cols.append({"kind": "lambda", "cost": col["cost"],
                             "coeffs": coeffs, "class": cl, "q": q,
                             "lb": lo,
                             "ub": (hi if hi is not None
                                    else (float(count) if binary else None))})
        if with_artificials:
            m_cost = 1e5 * (1.0 + max((abs(c["cost"]) for c in cols),
                                      default=0.0))
            for i in range(self.n_coupling):
                for sgn in (1.0, -1.0):
                    cols.append({"kind": "lambda", "cost": m_cost,
                                 "coeffs": {i: sgn}, "class": -1, "q": -1,
                                 "lb": None, "ub": None})
        return cols


def _run_cg(st: _DWState, lam_bounds, max_iterations: int, tol: float,
            blk_bounds=None):
    """Column generation on the LP master (warm-started by st's pool).

    ``blk_bounds`` (class -> {block col -> (lo, hi)}) tightens the pricing
    subproblems at this node — original-variable branching. Columns found
    inside the branch are globally valid block points.

    The reported ``bound`` is the best Lagrangian dual bound
    ``z_RMP + sum_cl M_cl * min(0, rc*_cl)`` seen across iterations,
    which is valid even when CG stops at ``max_iterations`` — the plain
    restricted-master value is only a dual bound after convergence.
    """
    blk_bounds = blk_bounds or {}
    best_lb = -np.inf
    lam_vals = None
    var_vals = None
    cols = None
    artificial_use = 0.0
    iterations = 0
    converged = False
    t0 = time.perf_counter()
    for it in range(max_iterations):
        iterations = it + 1
        cols = st.assemble_cols(binary=False, lam_bounds=lam_bounds,
                                blk_bounds=blk_bounds)
        status, obj, duals, lam_vals, var_vals = _solve_master(
            st.rows_bounds, cols, st.master_var_cols, False, True)
        if status != highspy.HighsModelStatus.kOptimal:
            return {"status": status, "bound": np.nan, "cols": cols,
                    "lam_vals": None, "var_vals": None,
                    "iterations": iterations, "artificial_use": np.inf,
                    "converged": False, "time": time.perf_counter() - t0}
        artificial_use = sum(v for j, v in (lam_vals or {}).items()
                             if cols[j].get("class") == -1)
        # HiGHS convention (verified empirically): min-form reduced cost
        # is cost - row_dual @ a_col, duals used exactly as returned.
        pi = duals
        pi_c, sigma = pi[:st.n_coupling], pi[st.n_coupling:]

        added = 0
        lagr = obj  # z_RMP; corrected below by pricing reduced costs
        lagr_valid = True
        for cl in range(st.n_classes):
            sub = st.class_subs[cl]
            price_cost = sub["col_cost"] - pi_c @ st.class_A0[cl]
            sub2 = dict(sub)
            sub2["col_cost"] = price_cost
            bb = blk_bounds.get(cl)
            if bb:
                lo = sub["col_lower"].copy()
                hi = sub["col_upper"].copy()
                for j, (lb, ub) in bb.items():
                    lo[j] = max(lo[j], lb)
                    hi[j] = min(hi[j], ub)
                sub2["col_lower"] = lo
                sub2["col_upper"] = hi
            hp = _pass_lp(sub2, True)
            if any(int(t) != 0 for t in sub["integrality"]):
                # exact pricing: the Lagrangian bound needs the true
                # pricing optimum, not an incumbent at MIP gap tolerance
                hp.setOptionValue("mip_rel_gap", 0.0)
                hp.setOptionValue("mip_abs_gap", 0.0)
            hp.run()
            if hp.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                lagr_valid = False
                continue
            pobj = hp.getObjectiveValue()
            pobj_lb = pobj  # certified lower bound on the pricing optimum
            if any(int(t) != 0 for t in sub["integrality"]):
                db = float(hp.getInfo().mip_dual_bound)
                if np.isfinite(db):
                    pobj_lb = min(pobj, db)
            x_new = np.array(hp.getSolution().col_value)
            rc = pobj - sigma[cl]
            lagr += len(st.class_members[cl]) * min(0.0, pobj_lb - sigma[cl])
            if rc < -tol:
                st.class_columns[cl].append({
                    "x": x_new,
                    "coeffs": st.class_A0[cl] @ x_new,
                    "cost": float(sub["col_cost"] @ x_new),
                })
                added += 1
        if lagr_valid:
            best_lb = max(best_lb, lagr)
        if added == 0:
            converged = True
            break
    return {"status": highspy.HighsModelStatus.kOptimal, "bound": best_lb,
            "cols": cols, "lam_vals": lam_vals, "var_vals": var_vals,
            "iterations": iterations, "artificial_use": artificial_use,
            "converged": converged, "time": time.perf_counter() - t0}


def _restricted_master(st: _DWState, lam_bounds, blk_bounds=None):
    """Solve the master with integer (count) lambda variables."""
    t0 = time.perf_counter()
    cols = st.assemble_cols(binary=True, lam_bounds=lam_bounds,
                            blk_bounds=blk_bounds)
    status, obj, _, lam_vals, var_vals = _solve_master(
        st.rows_bounds, cols, st.master_var_cols, True, True)
    # _recover silently drops artificial columns, so a solution leaning
    # on them is coupling-infeasible: callers must reject it.
    artificial_use = sum(v for j, v in (lam_vals or {}).items()
                         if cols[j].get("class") == -1)
    return {"status": status, "obj": obj, "cols": cols,
            "lam_vals": lam_vals, "var_vals": var_vals,
            "artificial_use": artificial_use,
            "time": time.perf_counter() - t0}


def _recover(st: _DWState, cols, lam_vals, var_vals) -> np.ndarray:
    """Primal solution from aggregated integer lambda counts."""
    x_full = np.zeros(st.lp.num_col_)
    assigned: dict[int, int] = {cl: 0 for cl in range(st.n_classes)}
    for j, c in enumerate(cols):
        if c["kind"] != "lambda" or c.get("class", -1) < 0:
            continue
        count = int(round(lam_vals.get(j, 0.0)))
        if count <= 0:
            continue
        cl, q = c["class"], c["q"]
        members = st.class_members[cl]
        for t in range(count):
            k = members[assigned[cl] + t]
            x_full[st.block_cols[k]] = st.class_columns[cl][q]["x"]
        assigned[cl] += count
    for cl in range(st.n_classes):
        if assigned[cl] != len(st.class_members[cl]):
            for t in range(assigned[cl], len(st.class_members[cl])):
                k = st.class_members[cl][t]
                x_full[st.block_cols[k]] = st.class_columns[cl][0]["x"]
    for j, v in (var_vals or {}).items():
        x_full[j] = v
    return x_full


def _recover_fractional(st: _DWState, cols, lam_vals, var_vals) -> np.ndarray:
    """Primal solution from a *fractional* master LP (continuous models).

    Each class member gets the lambda-weighted average of the class's
    operating points — a convex combination, feasible for continuous
    blocks — and master columns take their LP values."""
    x_full = np.zeros(st.lp.num_col_)
    agg: dict[int, np.ndarray | None] = {cl: None
                                         for cl in range(st.n_classes)}
    for j, c in enumerate(cols):
        if c["kind"] != "lambda" or c.get("class", -1) < 0:
            continue
        v = (lam_vals or {}).get(j, 0.0)
        if v <= 0.0:
            continue
        cl, q = c["class"], c["q"]
        contrib = v * st.class_columns[cl][q]["x"]
        agg[cl] = contrib if agg[cl] is None else agg[cl] + contrib
    for cl in range(st.n_classes):
        count = float(len(st.class_members[cl]))
        xbar = (agg[cl] / count if agg[cl] is not None
                else st.class_columns[cl][0]["x"])
        for k in st.class_members[cl]:
            x_full[st.block_cols[k]] = xbar
    for j, v in (var_vals or {}).items():
        x_full[j] = v
    return x_full


def _make_result(st, x_full, bound_min, obj_inc, iterations, timings,
                 extra_note="", nodes=0):
    obj_true = st.offset + float(np.asarray(st.lp.col_cost_) @ x_full)
    bound_true = st.offset + st.sign * bound_min
    if st.sense_min:
        gap = (obj_true - bound_true) / max(1.0, abs(bound_true))
    else:
        gap = (bound_true - obj_true) / max(1.0, abs(bound_true))
    note = (f"{st.n_blocks} blocks ({st.n_classes} classes), "
            f"{st.n_coupling} coupling rows, {iterations} CG iterations")
    if nodes:
        note += f", {nodes} B&P nodes"
    if extra_note:
        note += f"; {extra_note}"
    ok, msg = _verify_solution(st.A, st.lp, x_full, st.offset, obj_true)
    if not ok:
        return DWResult(status=highspy.HighsModelStatus.kNotset,
                        objective=0.0, bound=bound_true, col_value=None,
                        n_blocks=st.n_blocks, n_coupling_rows=st.n_coupling,
                        n_master_cols=len(st.master_cols),
                        n_unique_classes=st.n_classes,
                        iterations=iterations, gap=float(gap),
                        decomposed=False,
                        note=f"verification failed: {msg}", timings=timings)
    return DWResult(status=highspy.HighsModelStatus.kOptimal,
                    objective=obj_true, bound=bound_true, col_value=x_full,
                    n_blocks=st.n_blocks, n_coupling_rows=st.n_coupling,
                    n_master_cols=len(st.master_cols),
                    n_unique_classes=st.n_classes, iterations=iterations,
                    gap=float(gap), decomposed=True,
                    note=note + "; solution verified numerically",
                    timings=timings)


# ----------------------------------------------------------------------
# main entries
# ----------------------------------------------------------------------
def solve_dw(h: highspy.Highs, max_iterations: int = 100,
             tol: float = 1e-6, max_coupling: int = 200,
             verify: bool = True) -> DWResult:
    """Solve a block-angular MILP via Dantzig-Wolfe + restricted master."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        st = _DWState(h, max_coupling)
    except ValueError as e:
        return DWResult(status=highspy.HighsModelStatus.kNotset,
                        objective=0.0, bound=0.0, col_value=None,
                        note=str(e), timings=timings)
    timings["setup"] = time.perf_counter() - t0

    cg = _run_cg(st, {}, max_iterations, tol)
    timings["column_generation"] = cg["time"]
    iterations = cg["iterations"]
    if cg["status"] != highspy.HighsModelStatus.kOptimal:
        return DWResult(status=cg["status"], objective=0.0, bound=0.0,
                        col_value=None, note="master LP failed",
                        timings=timings)
    if cg["artificial_use"] > 1e-5:
        return DWResult(status=highspy.HighsModelStatus.kInfeasible,
                        objective=0.0, bound=0.0, col_value=None,
                        n_blocks=st.n_blocks, n_coupling_rows=st.n_coupling,
                        n_unique_classes=st.n_classes, iterations=iterations,
                        decomposed=True,
                        note=("model appears infeasible: Phase-I artificial "
                              f"columns still active ({cg['artificial_use']:.3g})"),
                        timings=timings)

    rm = _restricted_master(st, {})
    timings["restricted_master"] = rm["time"]
    if rm["status"] != highspy.HighsModelStatus.kOptimal:
        return DWResult(status=rm["status"], objective=0.0,
                        bound=float(st.offset + st.sign * cg["bound"]),
                        col_value=None, n_blocks=st.n_blocks,
                        n_coupling_rows=st.n_coupling,
                        n_master_cols=len(st.master_cols),
                        n_unique_classes=st.n_classes, iterations=iterations,
                        decomposed=True, note="restricted master MIP failed",
                        timings=timings)
    if rm["artificial_use"] > 1e-5:
        return DWResult(status=highspy.HighsModelStatus.kNotset,
                        objective=0.0,
                        bound=float(st.offset + st.sign * cg["bound"]),
                        col_value=None, n_blocks=st.n_blocks,
                        n_coupling_rows=st.n_coupling,
                        n_master_cols=len(st.master_cols),
                        n_unique_classes=st.n_classes, iterations=iterations,
                        decomposed=True,
                        note=("restricted master relied on artificial "
                              "columns: no coupling-feasible integer "
                              "combination in the column pool"),
                        timings=timings)

    x_full = _recover(st, rm["cols"], rm["lam_vals"], rm["var_vals"])
    # internal min form, model offset excluded (offset is added back only
    # when populating DWResult fields inside _make_result)
    obj_inc_min = st.sign * float(np.asarray(st.lp.col_cost_) @ x_full)
    return _make_result(st, x_full, cg["bound"], obj_inc_min, iterations,
                        timings)


def solve_bp(h: highspy.Highs, max_iterations: int = 100,
             node_budget: int = 32, tol: float = 1e-6,
             max_coupling: int = 200, lns_rounds: int = 12,
             lns_neighborhood: int = 6, lns_time_budget: float = 60.0) -> DWResult:
    """Exact branch-and-price for block-angular MILPs.

    Branch-and-bound over the master's lambda variables — a branching
    rule that leaves the pricing structure untouched. Every node
    re-runs column generation (warm-started from the shared column
    pool), so node bounds are always valid. When the restricted master
    matches a node bound, optimality is *proven*; if the node budget
    runs out first, the best bound and incumbent are returned with an
    honest residual gap.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        st = _DWState(h, max_coupling)
    except ValueError as e:
        return DWResult(status=highspy.HighsModelStatus.kNotset,
                        objective=0.0, bound=0.0, col_value=None,
                        note=str(e), timings=timings)
    timings["setup"] = time.perf_counter() - t0

    # -- root: CG bound + restricted-master incumbent --------------------
    cg0 = _run_cg(st, {}, max_iterations, tol)
    timings["root_cg"] = cg0["time"]
    if cg0["status"] != highspy.HighsModelStatus.kOptimal:
        return DWResult(status=cg0["status"], objective=0.0, bound=0.0,
                        col_value=None, note="root master LP failed",
                        timings=timings)
    if cg0["artificial_use"] > 1e-5:
        return DWResult(status=highspy.HighsModelStatus.kInfeasible,
                        objective=0.0, bound=0.0, col_value=None,
                        note="model appears infeasible at the root",
                        timings=timings)

    # All-continuous models never branch: the master LP's lambda-weighted
    # block points are convex combinations of feasible block points and
    # hence feasible, so the root LP solution is final.
    if not any(int(t) != 0 for t in st.integrality):
        x_lp = _recover_fractional(st, cg0["cols"], cg0["lam_vals"],
                                   cg0["var_vals"])
        obj_lp = st.sign * float(np.asarray(st.lp.col_cost_) @ x_lp)
        note = ("continuous model: master LP solution is final"
                if cg0["converged"] else
                "continuous model: CG iteration limit reached")
        return _make_result(st, x_lp, cg0["bound"], obj_lp,
                            cg0["iterations"], timings, extra_note=note)

    rm0 = _restricted_master(st, {})
    timings["root_restricted_master"] = rm0["time"]
    total_iters = cg0["iterations"]

    if rm0["status"] != highspy.HighsModelStatus.kOptimal:
        return DWResult(status=rm0["status"], objective=0.0,
                        bound=float(st.offset + st.sign * cg0["bound"]),
                        col_value=None, note="root restricted master failed",
                        timings=timings)

    # All internal comparisons are in min form with the model offset
    # excluded; the offset is added back only inside _make_result.
    x_inc, obj_inc = None, float("inf")
    if rm0["artificial_use"] <= 1e-5:
        x_inc = _recover(st, rm0["cols"], rm0["lam_vals"], rm0["var_vals"])
        obj_inc = st.sign * float(np.asarray(st.lp.col_cost_) @ x_inc)
    bound_root = cg0["bound"]

    if x_inc is not None and \
            obj_inc - bound_root <= tol * max(1.0, abs(bound_root)):
        return _make_result(st, x_inc, bound_root, obj_inc, total_iters,
                            timings, extra_note="proven optimal at root")

    # -- branch-and-price -------------------------------------------------
    t0 = time.perf_counter()
    nodes = [{"lam": {}, "blk": {}, "bound": bound_root, "depth": 0}]
    processed = 0
    # Valid min-form bounds of finished subtrees (pruned or fully
    # explored). Infeasible subtrees contribute +inf, i.e. nothing.
    closed_bounds: list[float] = []
    while nodes and processed < node_budget:
        # best-first: lowest (min-form) bound
        nodes.sort(key=lambda nd: nd["bound"])
        node = nodes.pop(0)
        if node["bound"] >= obj_inc - tol * max(1.0, abs(obj_inc)):
            closed_bounds.append(node["bound"])
            continue  # pruned by incumbent (its bound stays valid)
        cg = _run_cg(st, node["lam"], max_iterations, tol,
                     blk_bounds=node["blk"])
        total_iters += cg["iterations"]
        processed += 1
        if cg["status"] != highspy.HighsModelStatus.kOptimal:
            if cg["status"] != highspy.HighsModelStatus.kInfeasible:
                # unresolved (not proven infeasible): keep its valid bound
                closed_bounds.append(node["bound"])
            continue
        if cg["artificial_use"] > 1e-5:
            continue  # coupling-infeasible node
        # parent bound and this node's Lagrangian bound are both valid
        nd_bound = max(node["bound"], cg["bound"])
        if nd_bound >= obj_inc - tol * max(1.0, abs(obj_inc)):
            closed_bounds.append(nd_bound)
            continue  # pruned

        # try to improve the incumbent at this node
        rm = _restricted_master(st, node["lam"], blk_bounds=node["blk"])
        if rm["status"] == highspy.HighsModelStatus.kOptimal and \
                rm["artificial_use"] <= 1e-5:
            x_cand = _recover(st, rm["cols"], rm["lam_vals"], rm["var_vals"])
            obj_cand = st.sign * float(np.asarray(st.lp.col_cost_) @ x_cand)
            if obj_cand < obj_inc - tol:
                obj_inc, x_inc = obj_cand, x_cand

        # -- choose a branch ----------------------------------------------
        # Preferred: original-variable branching in a single-member class
        # (tightens the pricing polytope directly). Fallback: branching on
        # aggregated lambda counts.
        lam_by_class: dict[int, list] = {}
        for j, c in enumerate(cg["cols"]):
            if c["kind"] == "lambda" and c.get("class", -1) >= 0:
                v = (cg["lam_vals"] or {}).get(j, 0.0)
                if v > 1e-9:
                    lam_by_class.setdefault(c["class"], []).append((c["q"], v))

        branch = None  # ("var", cl, j, xbar_j) or ("lam", cl, q, v)
        best_f = 1e-4
        for cl, qvs in lam_by_class.items():
            if len(st.class_members[cl]) != 1:
                continue
            xbar = sum(v * st.class_columns[cl][q]["x"] for q, v in qvs)
            ints = st.class_subs[cl]["integrality"]
            for j in range(len(xbar)):
                if j >= len(ints) or not ints[j]:
                    continue  # no integrality info means continuous
                f = abs(xbar[j] - round(xbar[j]))
                if f > best_f:
                    best_f = f
                    branch = ("var", cl, j, xbar[j])
        if branch is None:
            for cl, qvs in lam_by_class.items():
                for q, v in qvs:
                    f = abs(v - round(v))
                    if f > best_f:
                        best_f = f
                        branch = ("lam", cl, q, v)

        if branch is None:
            # Lambda-integral LP: a valid primal candidate — including the
            # master (linking) columns at their LP values, which must be
            # integral where the original model demands it.
            var_vals = cg["var_vals"] or {}
            ints_ok = all(abs(v - round(v)) <= 1e-6
                          for j, v in var_vals.items()
                          if (len(st.integrality) == st.lp.num_col_
                              and st.integrality[j] != 0))
            if ints_ok:
                x_lp = _recover(st, cg["cols"], cg["lam_vals"], var_vals)
                obj_lp = st.sign * float(np.asarray(st.lp.col_cost_) @ x_lp)
                if obj_lp < obj_inc - tol:
                    obj_inc, x_inc = obj_lp, x_lp
            closed_bounds.append(nd_bound)  # node fully resolved
            continue

        if branch[0] == "var":
            _, cl, j, v = branch
            sub = st.class_subs[cl]
            lo_b = {"lam": dict(node["lam"]),
                    "blk": {c2: dict(b2) for c2, b2 in node["blk"].items()},
                    "bound": nd_bound, "depth": node["depth"] + 1}
            hi_b = {"lam": dict(node["lam"]),
                    "blk": {c2: dict(b2) for c2, b2 in node["blk"].items()},
                    "bound": nd_bound, "depth": node["depth"] + 1}
            lo_b["blk"].setdefault(cl, {})[j] = (
                float(sub["col_lower"][j]), float(np.floor(v)))
            hi_b["blk"].setdefault(cl, {})[j] = (
                float(np.ceil(v)), float(sub["col_upper"][j]))
            nodes.append(lo_b)
            nodes.append(hi_b)
        else:
            _, cl, q, v = branch
            count = len(st.class_members[cl])
            lo_b = {"lam": dict(node["lam"]),
                    "blk": {c2: dict(b2) for c2, b2 in node["blk"].items()},
                    "bound": nd_bound, "depth": node["depth"] + 1}
            hi_b = {"lam": dict(node["lam"]),
                    "blk": {c2: dict(b2) for c2, b2 in node["blk"].items()},
                    "bound": nd_bound, "depth": node["depth"] + 1}
            lo_b["lam"][(cl, q)] = (None, float(int(np.floor(v))))
            hi_b["lam"][(cl, q)] = (float(int(np.ceil(v))), float(count))
            nodes.append(lo_b)
            nodes.append(hi_b)

    # Global bound: min over the valid bounds of open and finished
    # subtrees. Never the incumbent — that is not a dual bound. The root
    # Lagrangian bound is always a valid fallback.
    all_bounds = closed_bounds + [nd["bound"] for nd in nodes]
    best_bound = min(all_bounds) if all_bounds else bound_root

    # Final incumbent attempt with the fully enriched column pool — columns
    # generated inside branches are globally valid, so the last restricted
    # master sees every operating point found anywhere in the tree.
    rm_final = _restricted_master(st, {})
    if rm_final["status"] == highspy.HighsModelStatus.kOptimal and \
            rm_final["artificial_use"] <= 1e-5:
        x_cand = _recover(st, rm_final["cols"], rm_final["lam_vals"],
                          rm_final["var_vals"])
        obj_cand = st.sign * float(np.asarray(st.lp.col_cost_) @ x_cand)
        if obj_cand < obj_inc - tol:
            obj_inc, x_inc = obj_cand, x_cand

    if x_inc is None:
        timings["branch_and_price"] = time.perf_counter() - t0
        return DWResult(status=highspy.HighsModelStatus.kNotset,
                        objective=0.0,
                        bound=float(st.offset + st.sign * best_bound),
                        col_value=None, n_blocks=st.n_blocks,
                        n_coupling_rows=st.n_coupling,
                        n_master_cols=len(st.master_cols),
                        n_unique_classes=st.n_classes,
                        iterations=total_iters, decomposed=True,
                        note=("no feasible incumbent found (restricted "
                              "masters relied on artificial columns)"),
                        timings=timings)

    # LNS polish: fix most blocks to the incumbent, re-solve small
    # neighborhoods exactly with HiGHS over the original rows.
    if lns_rounds > 0:
        from ._lns import _lns_improve
        obj_orig = st.sign * obj_inc  # original sense, offset excluded
        x_lns, obj_lns, improved = _lns_improve(
            st, x_inc, obj_orig, rounds=lns_rounds,
            k_neighborhood=lns_neighborhood, time_budget=lns_time_budget)
        if improved:
            x_inc = x_lns
            obj_inc = st.sign * obj_lns

    timings["branch_and_price"] = time.perf_counter() - t0
    # optimality is claimed only against the valid bound, re-checked
    # after the rm_final/LNS incumbent improvements
    proven = obj_inc - best_bound <= tol * max(1.0, abs(best_bound))
    if proven:
        note = "proven optimal by branch-and-price"
    elif nodes:
        note = (f"node budget exhausted ({processed} nodes); returning "
                "best bound and incumbent")
    else:
        note = (f"search finished without optimality proof ({processed} "
                "nodes); returning best bound and incumbent")
    return _make_result(st, x_inc, best_bound, obj_inc, total_iters,
                        timings, extra_note=note, nodes=processed)
