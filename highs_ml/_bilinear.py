"""Bilinear terms in HiGHS: y = x1 * x2 via MILP reformulation.

Why HiGHS "can't do bilinear" natively: a bilinear equality is non-convex
(the graph of y = x1*x2 is a saddle surface), and HiGHS has no spatial
branch-and-bound over continuous non-convex quadratics — that is exactly
Gurobi's non-convex quadratic engine. But MILP reformulation theory gives
us three practical routes, all implemented here:

* **binary × continuous  -> EXACT.** The McCormick envelope collapses
  onto the bilinear graph when one factor is binary: four linear
  constraints, zero approximation.

* **integer × continuous (or integer × integer) -> EXACT.** Binary
  expansion of the integer factor turns the product into a sum of
  binary×continuous terms, each embedded exactly. Cost:
  ceil(log2(range)) binaries per integer variable.

* **continuous × continuous -> certified approximation.** Piecewise
  McCormick: partition one factor's range into K segments with segment
  binaries; the McCormick envelope per segment has gap at most
  (segment width) * (other range) / 4. Segments are sized so the gap is
  at most ``tol``, giving a certified envelope: the optimizer may place
  y anywhere inside the envelope, so the *effective* deviation from the
  true product is bounded by ``tol`` — same philosophy as the sigmoid
  PWL embedding (approximation with a certificate, stated precisely).

Factors must be single variables or constants (products of general
affine forms are rejected with a clear message).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from highspy import HighsVarType, kHighsInf

from ._affine import Affine
from ._pwl import PWLStats


def _as_var(x, name: str):
    """Return (var, lo, hi, is_integer) if x is a single bounded variable."""
    a = Affine.coerce(x)
    if a.is_constant():
        return None
    if len(a.terms) != 1:
        raise ValueError(
            f"Bilinear factors must be single variables or constants; "
            f"{name} is a general affine expression. Factor it into "
            "auxiliary variables first."
        )
    (var, coef), = a.terms.items()
    if coef != 1.0 or a.const != 0.0:
        raise ValueError(
            f"Bilinear factors must be plain variables; {name} has a "
            "coefficient or offset. Introduce an auxiliary variable."
        )
    lo, hi = a.bounds()
    return var, lo, hi


def _is_binary(var, lo: float, hi: float) -> bool:
    ints = getattr(var.highs.getLp(), "integrality_", []) or []
    return (lo == 0.0 and hi == 1.0 and var.index < len(ints)
            and int(ints[var.index]) != 0)


def _is_integer(var, lo: float, hi: float) -> bool:
    ints = getattr(var.highs.getLp(), "integrality_", []) or []
    return var.index < len(ints) and int(ints[var.index]) != 0


def add_bilinear_constr(h, x1, x2, y=None, tol: float = 0.01,
                        name: str = "bil", stats: Optional[PWLStats] = None):
    """Add ``y = x1 * x2`` to the HiGHS model; returns the product variable.

    Exact when one factor is constant, binary or integer; certified
    piecewise-McCormick approximation (envelope gap <= tol) when both are
    continuous. ``x1`` and ``x2`` may be the same variable (squares).
    """
    stats = stats if stats is not None else PWLStats()
    a1, a2 = Affine.coerce(x1), Affine.coerce(x2)

    # constant folding
    if a1.is_constant():
        return a2 * a1.const
    if a2.is_constant():
        return a1 * a2.const

    v1 = _as_var(x1, "x1")
    v2 = _as_var(x2, "x2")
    var1, lo1, hi1 = v1
    var2, lo2, hi2 = v2

    same = var1.index == var2.index
    bin1 = _is_binary(var1, lo1, hi1)
    bin2 = (not same) and _is_binary(var2, lo2, hi2)
    int1 = _is_integer(var1, lo1, hi1)
    int2 = (not same) and _is_integer(var2, lo2, hi2)

    if y is None:
        y_lo = min(lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
        y_hi = max(lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
        y = h.addVariable(lb=y_lo, ub=y_hi, name=f"{name}_prod")
        stats.n_vars += 1

    if bin1 or bin2:
        if bin1:
            _mccormick_binary(h, y, var1, var2, lo2, hi2, name, stats)
        else:
            _mccormick_binary(h, y, var2, var1, lo1, hi1, name, stats)
        return y

    if int1 and (hi1 - lo1) <= 1024:
        _integer_expansion(h, y, var1, lo1, hi1, var2, lo2, hi2, name, stats)
        return y
    if int2 and (hi2 - lo2) <= 1024:
        _integer_expansion(h, y, var2, lo2, hi2, var1, lo1, hi1, name, stats)
        return y

    _piecewise_mccormick(h, y, var1, lo1, hi1, var2, lo2, hi2, tol,
                         name, stats)
    return y


def _mccormick_binary(h, y, b, x, lo, hi, name, stats):
    """Exact y = b * x for binary b, x in [lo, hi]."""
    h.addConstr(y >= lo * b, name=f"{name}_mc1")
    h.addConstr(y <= hi * b, name=f"{name}_mc2")
    h.addConstr(y >= x - hi * (1.0 - b), name=f"{name}_mc3")
    h.addConstr(y <= x - lo * (1.0 - b), name=f"{name}_mc4")
    stats.n_constrs += 4


def _integer_expansion(h, y, xi, lo_i, hi_i, x, lo, hi, name, stats):
    """Exact y = xi * x via binary expansion of integer xi in [lo_i, hi_i]."""
    span = int(hi_i - lo_i)
    n_bits = max(1, math.ceil(math.log2(span + 1)))
    bits = [
        h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger,
                      name=f"{name}_bit{i}")
        for i in range(n_bits)
    ]
    # xi = lo_i + sum 2^i b_i   (cap the top bit to keep the range tight)
    h.addConstr(xi == lo_i + sum((2 ** i) * bits[i] for i in range(n_bits)),
                name=f"{name}_expand")
    # y = lo_i * x + sum 2^i (b_i * x)
    expr = lo_i * x
    for i, b in enumerate(bits):
        w = h.addVariable(lb=min(0.0, (2 ** i) * lo), ub=max(0.0, (2 ** i) * hi),
                          name=f"{name}_bx{i}")
        _mccormick_binary(h, w, b, x, lo, hi, f"{name}_bx{i}", stats)
        expr = expr + (2 ** i) * w
        stats.n_vars += 2  # w and bit
    h.addConstr(y == expr, name=f"{name}_sum")
    stats.n_constrs += 2


class PiecewiseBilinear:
    """Refinable piecewise-McCormick embedding of ``y = x1 * x2``.

    Built with a coarse uniform partition; :meth:`refine` splits the
    segment containing the incumbent point (fixing the old segment's
    binary to zero and appending two child segments via ``chgCoeff``), so
    the envelope tightens exactly where the optimizer lands — the same
    dynamic-refinement strategy commercial solvers use for PWL
    approximations of nonlinear functions.
    """

    def __init__(self, h, y, x1, lo1, hi1, x2, lo2, hi2, tol,
                 name, stats, n_initial: int = 8):
        self.h = h
        self.y = y
        self.x1, self.x2 = x1, x2
        self.lo2, self.hi2 = lo2, hi2
        self.tol = tol
        self.name = name
        self.stats = stats
        self.segments: list[dict] = []

        width2 = hi2 - lo2
        self.fixed = width2 <= 0
        if self.fixed:
            h.addConstr(y == hi2 * x1, name=f"{name}_fixed")
            stats.n_constrs += 1
            return

        edges = np.linspace(lo1, hi1, n_initial + 1)
        self.oneseg_row = None
        self.x1sum_row = None
        self.ysum_row = None
        for s in range(n_initial):
            self._add_segment(float(edges[s]), float(edges[s + 1]))

    # -- internal ---------------------------------------------------------
    def _add_segment(self, ls: float, hs: float) -> None:
        h, name, stats = self.h, self.name, self.stats
        s = len(self.segments)
        delta = h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger,
                              name=f"{name}_seg{s}")
        xs = h.addVariable(lb=min(0.0, ls), ub=max(0.0, hs),
                           name=f"{name}_x1_{s}")
        h.addConstr(xs >= ls * delta, name=f"{name}_x1lo{s}")
        h.addConstr(xs <= hs * delta, name=f"{name}_x1hi{s}")
        x2s = h.addVariable(lb=min(0.0, self.lo2), ub=max(0.0, self.hi2),
                            name=f"{name}_x2_{s}")
        _mccormick_binary(h, x2s, delta, self.x2, self.lo2, self.hi2,
                          f"{name}_x2mc{s}", stats)
        lo2, hi2 = self.lo2, self.hi2
        # corner products bound the segment envelope; 0 must stay inside
        # the bounds because inactive segments force yv = 0
        corners = (ls * lo2, ls * hi2, hs * lo2, hs * hi2)
        yv = h.addVariable(lb=min(0.0, *corners), ub=max(0.0, *corners),
                           name=f"{name}_y_{s}")
        h.addConstr(yv >= ls * x2s + lo2 * xs - ls * lo2 * delta,
                    name=f"{name}_pm1_{s}")
        h.addConstr(yv >= hs * x2s + hi2 * xs - hs * hi2 * delta,
                    name=f"{name}_pm2_{s}")
        h.addConstr(yv <= ls * x2s + hi2 * xs - ls * hi2 * delta,
                    name=f"{name}_pm3_{s}")
        h.addConstr(yv <= hs * x2s + lo2 * xs - hs * lo2 * delta,
                    name=f"{name}_pm4_{s}")
        # sum rows: created from the first segment, patched afterwards
        if self.oneseg_row is None:
            self.oneseg_row = h.addConstr(delta == 1.0,
                                          name=f"{name}_oneseg")
            self.x1sum_row = h.addConstr(self.x1 == 1.0 * xs,
                                         name=f"{name}_x1sum")
            self.ysum_row = h.addConstr(self.y == 1.0 * yv,
                                        name=f"{name}_ysum")
            stats.n_constrs += 3
        else:
            h.chgCoeff(self.oneseg_row, delta, 1.0)
            h.chgCoeff(self.x1sum_row, xs, 1.0)
            h.chgCoeff(self.ysum_row, yv, 1.0)
        self.segments.append({"ls": ls, "hs": hs, "delta": delta,
                              "active": True})
        stats.n_vars += 4
        stats.n_binaries += 1
        stats.n_constrs += 6

    # -- public -------------------------------------------------------------
    def solution_error(self) -> float:
        """|y - x1*x2| at the current HiGHS solution (0 if fixed case)."""
        if self.fixed:
            return 0.0
        values = self.h.getSolution().col_value
        v1 = float(values[self.x1.index])
        v2 = float(values[self.x2.index])
        vy = float(values[self.y.index])
        return abs(vy - v1 * v2)

    def refine(self, point: float) -> bool:
        """Split the active segment containing ``point``.

        Boundary-adjacent points bisect the segment instead (guaranteed
        geometric shrinkage; hair-splits at edges wedge the iteration).
        """
        for seg in self.segments:
            if not seg["active"]:
                continue
            if seg["ls"] - 1e-12 <= point <= seg["hs"] + 1e-12:
                if seg["hs"] - seg["ls"] < 2e-9:
                    return False
                if point <= seg["ls"] + 1e-9 or point >= seg["hs"] - 1e-9:
                    mid = 0.5 * (seg["ls"] + seg["hs"])
                else:
                    mid = point
                # deactivate the old segment and split it
                self.h.changeColBounds(seg["delta"].index, 0.0, 0.0)
                seg["active"] = False
                self._add_segment(seg["ls"], mid)
                self._add_segment(mid, seg["hs"])
                return True
        return False

    def solve_adaptive(self, max_refines: int = 25,
                       status_cb=None) -> tuple:
        """Solve, check the certificate at the incumbent, refine, repeat.

        Returns (status, final_error, n_refinements). The HiGHS model's
        objective must already be set by the caller.
        """
        from highspy import HighsModelStatus
        n_ref = 0
        for _ in range(max_refines + 1):
            self.h.run()
            status = self.h.getModelStatus()
            if status != HighsModelStatus.kOptimal:
                return status, float("nan"), n_ref
            err = self.solution_error()
            if err <= self.tol or n_ref == max_refines:
                return status, err, n_ref
            values = self.h.getSolution().col_value
            if not self.refine(float(values[self.x1.index])):
                return status, err, n_ref
            n_ref += 1


def add_adaptive_bilinear(h, x1, x2, y=None, tol: float = 0.01,
                          name: str = "bil", n_initial: int = 8,
                          stats: Optional[PWLStats] = None) -> "PiecewiseBilinear":
    """Adaptive certified embedding of ``y = x1 * x2`` (both continuous).

    Starts with a coarse uniform partition (``n_initial`` segments) and
    refines the envelope at the incumbent point each time
    :meth:`PiecewiseBilinear.solve_adaptive` finds the certificate
    violated. Typically needs far fewer segments than the uniform
    partition a static tolerance would require.

    The caller must set the objective before calling
    ``emb.solve_adaptive()``. For binary/integer factors use
    :func:`add_bilinear_constr` instead (already exact).
    """
    stats = stats if stats is not None else PWLStats()
    v1 = _as_var(x1, "x1")
    v2 = _as_var(x2, "x2")
    var1, lo1, hi1 = v1
    var2, lo2, hi2 = v2
    if _is_binary(var1, lo1, hi1) or _is_binary(var2, lo2, hi2) or \
            _is_integer(var1, lo1, hi1) or _is_integer(var2, lo2, hi2):
        raise ValueError(
            "adaptive refinement is for continuous x continuous products; "
            "binary/integer factors are already exact via "
            "add_bilinear_constr."
        )
    if y is None:
        y_lo = min(lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
        y_hi = max(lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
        y = h.addVariable(lb=y_lo, ub=y_hi, name=f"{name}_prod")
        stats.n_vars += 1
    return PiecewiseBilinear(h, y, var1, lo1, hi1, var2, lo2, hi2, tol,
                             name, stats, n_initial=n_initial)


def _piecewise_mccormick(h, y, x1, lo1, hi1, x2, lo2, hi2, tol,
                         name, stats):
    """Certified piecewise McCormick for y = x1 * x2, both continuous.

    Uniform partition sized so the envelope gap is at most ``tol``
    (gap in a segment of width w is w * (hi2 - lo2) / 4).
    """
    width2 = hi2 - lo2
    if width2 <= 0:  # x2 effectively fixed
        h.addConstr(y == hi2 * x1, name=f"{name}_fixed")
        stats.n_constrs += 1
        return None
    seg_w = 4.0 * tol / width2
    n_seg = max(1, math.ceil((hi1 - lo1) / seg_w))
    if n_seg > 1024:
        raise ValueError(
            f"piecewise McCormick for {name!r} needs {n_seg} segments "
            f"(> 1024) at tol={tol} with x1 in [{lo1}, {hi1}] and "
            f"x2 in [{lo2}, {hi2}]. Tighten the variable bounds or "
            "loosen the tolerance (tol / pwl_tol)."
        )
    emb = PiecewiseBilinear(h, y, x1, lo1, hi1, x2, lo2, hi2, tol,
                            name, stats, n_initial=n_seg)
    return emb
