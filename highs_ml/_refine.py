"""Unified refinement-at-incumbent framework for nonlinear embeddings.

Two embedding families share one protocol:

* ``PiecewiseBilinear`` (``_bilinear.py``): y = x1 * x2
* ``RefinablePWL`` (here): y = f(z) for a univariate nonlinear f
  (sigmoid, tanh, exp, ...)

Both start coarse and refine exactly where the optimizer lands:
:func:`solve_adaptive` solves the model, checks each embedding's
certificate at the incumbent, splits the offending segment, and repeats
until every certificate holds.

RefinablePWL formulation (per segment s = [lo_s, hi_s] of the z-range):

    z = sum_s z_s            y = sum_s y_s          sum_s delta_s = 1
    z_s = z * delta_s        (exact binary x continuous McCormick)
    y_s = slope_s * z_s + (f(lo_s) - slope_s * lo_s) * delta_s

The last row is *linear*: on the active segment it is exactly the chord
of f; on inactive segments it collapses to y_s = z_s = 0 automatically.
Splitting a segment appends two children and fixes the parent's delta to
zero — no row deletions, no SOS2 chains, and it composes with as many
embeddings as the model carries.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from highspy import HighsModelStatus, HighsVarType

from ._affine import Affine
from ._bilinear import _mccormick_binary
from ._pwl import PWLStats


class RefinablePWL:
    """Refinable certified embedding of ``y = f(z)`` (f univariate)."""

    def __init__(self, h, f: Callable[[float], float], z,
                 y=None, tol: float = 0.01, name: str = "pwl",
                 stats: Optional[PWLStats] = None,
                 n_initial: int = 2, envelope: str = "upper"):
        if envelope not in ("upper", "lower", "chord"):
            raise ValueError("envelope must be 'upper', 'lower' or 'chord'.")
        self.envelope = envelope
        stats = stats if stats is not None else PWLStats()
        self.h = h
        self.f = f
        self.tol = tol
        self.name = name
        self.stats = stats
        self.segments: list[dict] = []

        z_aff = Affine.coerce(z)
        self.z_aff = z_aff
        zlo, zhi = z_aff.bounds()
        self.zlo, self.zhi = zlo, zhi

        # z needs a variable for the binary x continuous copies
        self.z_var = h.addVariable(lb=zlo, ub=zhi, name=f"{name}_z")
        if z_aff.is_constant():
            h.addConstr(self.z_var == z_aff.const, name=f"{name}_zeq")
        else:
            h.addConstr(self.z_var == z_aff.to_highspy(), name=f"{name}_zeq")
        stats.n_vars += 1
        stats.n_constrs += 1

        f_lo, f_hi = f(zlo), f(zhi)
        if y is None:
            y = h.addVariable(lb=min(f_lo, f_hi), ub=max(f_lo, f_hi),
                              name=f"{name}_out")
            stats.n_vars += 1
        self.y = y

        self.oneseg_row = None
        self.zsum_row = None
        self.ysum_row = None
        edges = np.linspace(zlo, zhi, n_initial + 1)
        # dedupe in case zlo == zhi (degenerate: handled by caller checks)
        edges = np.unique(edges)
        for s in range(len(edges) - 1):
            self._add_segment(float(edges[s]), float(edges[s + 1]))

    # -- internal ---------------------------------------------------------
    def _add_segment(self, lo: float, hi: float) -> None:
        h, name, stats = self.h, self.name, self.stats
        s = len(self.segments)
        f_lo, f_hi = self.f(lo), self.f(hi)
        slope = (f_hi - f_lo) / (hi - lo)
        intercept = f_lo - slope * lo

        delta = h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger,
                              name=f"{name}_seg{s}")
        zs = h.addVariable(lb=min(0.0, lo, hi), ub=max(0.0, lo, hi),
                           name=f"{name}_zs{s}")
        # z_s is z confined to THIS segment (zero when inactive)
        h.addConstr(zs >= lo * delta, name=f"{name}_zlo{s}")
        h.addConstr(zs <= hi * delta, name=f"{name}_zhi{s}")
        _mccormick_binary(h, zs, delta, self.z_var, self.zlo, self.zhi,
                          f"{name}_zmc{s}", stats)
        ys = h.addVariable(lb=min(0.0, f_lo, f_hi), ub=max(0.0, f_lo, f_hi),
                           name=f"{name}_ys{s}")
        # y_s = slope * z_s + intercept * delta  (the chord, linear).
        # Envelope shift: a bare chord is two-sided around f, so the model's
        # optimum can dodge regions where the chord *underestimates* f.
        # Shifting to a one-sided envelope (default: upper) makes the
        # model optimum converge to the true optimum from a safe side.
        shift = 0.0
        if self.envelope != "chord":
            grid = np.linspace(lo, hi, 33)
            chord_g = f_lo + slope * (grid - lo)
            dev = np.array([self.f(float(g)) for g in grid]) - chord_g
            shift = (float(max(0.0, dev.max())) if self.envelope == "upper"
                     else float(min(0.0, dev.min())))
        intercept_e = intercept + shift
        # Saturated tails produce coefficients below HiGHS's 1e-9
        # rejection threshold; snap them.
        slope_s = slope if abs(slope) >= 1e-9 else 0.0
        intercept_s = intercept_e if abs(intercept_e) >= 1e-9 else 0.0
        if slope_s == 0.0 and intercept_s == 0.0:
            h.addConstr(ys == 0.0, name=f"{name}_chord{s}")
        elif slope_s == 0.0:
            h.addConstr(ys == intercept_s * delta, name=f"{name}_chord{s}")
        elif intercept_s == 0.0:
            h.addConstr(ys == slope_s * zs, name=f"{name}_chord{s}")
        else:
            h.addConstr(ys == slope_s * zs + intercept_s * delta,
                        name=f"{name}_chord{s}")

        if self.oneseg_row is None:
            self.oneseg_row = h.addConstr(delta == 1.0, name=f"{name}_oneseg")
            self.zsum_row = h.addConstr(self.z_var == 1.0 * zs,
                                        name=f"{name}_zsum")
            self.ysum_row = h.addConstr(self.y == 1.0 * ys,
                                        name=f"{name}_ysum")
            stats.n_constrs += 3
        else:
            h.chgCoeff(self.oneseg_row, delta, 1.0)
            h.chgCoeff(self.zsum_row, zs, 1.0)
            h.chgCoeff(self.ysum_row, ys, 1.0)

        self.segments.append({"ls": lo, "hs": hi, "delta": delta,
                              "active": True})
        stats.n_vars += 3
        stats.n_binaries += 1
        stats.n_constrs += 3

    # -- public -----------------------------------------------------------
    def solution_error(self) -> float:
        """|y - f(z)| at the current HiGHS solution."""
        values = self.h.getSolution().col_value
        z_val = self.z_aff.evaluate(values)
        y_val = float(values[self.y.index])
        return abs(y_val - self.f(z_val))

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
                self.h.changeColBounds(seg["delta"].index, 0.0, 0.0)
                seg["active"] = False
                self._add_segment(seg["ls"], mid)
                self._add_segment(mid, seg["hs"])
                return True
        return False

    def active_segments(self) -> int:
        return sum(1 for s in self.segments if s["active"])


def solve_adaptive(h, embeddings, max_refines: int = 60) -> tuple:
    """Solve, check every embedding's certificate, refine, repeat.

    Round-robin over all embeddings in the model. Returns
    (status, max_certificate_error, total_refinements).
    """
    total_ref = 0
    for _round in range(max_refines + 1):
        h.run()
        status = h.getModelStatus()
        if status != HighsModelStatus.kOptimal:
            return status, float("nan"), total_ref
        worst, worst_emb, worst_pt = -1.0, None, None
        for emb in embeddings:
            err = emb.solution_error()
            if err > worst:
                values = h.getSolution().col_value
                worst = err
                worst_emb = emb
                worst_pt = (emb.z_aff.evaluate(values)
                            if hasattr(emb, "z_aff")
                            else float(values[emb.x1.index]))
        if worst_emb is None or worst <= max(e.tol for e in embeddings):
            return status, worst if worst >= 0 else 0.0, total_ref
        if not worst_emb.refine(worst_pt):
            # cannot refine further (point at a boundary): report honestly
            return status, worst, total_ref
        total_ref += 1
    return h.getModelStatus(), worst, total_ref
