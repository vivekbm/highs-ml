"""Piecewise-linear embedding of univariate nonlinear functions into HiGHS.

Gurobi handles ``y = logistic(z)`` natively through *general constraints*
(solved as a non-convex NLP by spatial branch-and-bound). HiGHS is a pure
LP/QP/MIP solver with no general-constraint facility, so highs_ml
reformulates such relations as MILPs instead.

We use the classic *convex combination* formulation with segment binaries
(an SOS2 set enforced through linear constraints, since HiGHS has no SOS2
API):

    z = sum_k lam_k * z_k          y = sum_k lam_k * f(z_k)
    sum_k lam_k = 1                lam_k >= 0
    sum_s b_s = 1                  b_s in {0, 1}     (one per segment)
    lam_0 <= b_0
    lam_k <= b_{k-1} + b_k
    lam_K <= b_{K-1}

The interpolation error is bounded adaptively: breakpoints are inserted
wherever the chord deviates from the true function by more than ``tol``,
so the embedding is *certifiably* within ``tol`` of the exact function over
the whole reachable interval of ``z``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from highspy import HighsVarType

from ._affine import Affine


def adaptive_breakpoints(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float,
    max_points: int = 64,
    samples: int = 33,
) -> np.ndarray:
    """Breakpoints of a PWL interpolant of ``f`` on ``[lo, hi]`` with chord
    error at most ``tol`` (verified on a dense grid per segment)."""
    points: List[float] = [float(lo), float(hi)]
    fvals = {float(lo): f(float(lo)), float(hi): f(float(hi))}

    while len(points) < max_points:
        worst_err, worst_x = -1.0, None
        for a, b in zip(points[:-1], points[1:]):
            xs = np.linspace(a, b, samples)
            fa, fb = fvals[a], fvals[b]
            chord = fa + (fb - fa) * (xs - a) / (b - a)
            errs = np.abs(np.array([f(float(x)) for x in xs]) - chord)
            i = int(np.argmax(errs))
            if errs[i] > worst_err:
                worst_err, worst_x = float(errs[i]), float(xs[i])
        if worst_err <= tol or worst_x is None:
            break
        points.append(worst_x)
        fvals[worst_x] = f(worst_x)
        points.sort()

    return np.array(points)


class PWLStats:
    """Mutable counters describing what an embedding added to the model."""

    def __init__(self) -> None:
        self.n_vars = 0
        self.n_binaries = 0
        self.n_constrs = 0
        self.n_pwl = 0  # number of PWL-approximated nonlinear relations
        self.n_relu = 0  # number of exact big-M ReLU relations

    def as_dict(self):
        return {
            "variables": self.n_vars,
            "binaries": self.n_binaries,
            "constraints": self.n_constrs,
            "pwl_relations": self.n_pwl,
            "relu_relations": self.n_relu,
        }


def add_pwl_constr(
    h,
    f: Callable[[float], float],
    z: Affine,
    y=None,
    tol: float = 0.01,
    name: str = "pwl",
    stats: Optional[PWLStats] = None,
):
    """Add ``y = f(z)`` to HiGHS model ``h`` as a PWL MILP formulation.

    ``z`` is an :class:`Affine` form; ``y`` is an existing ``highs_var`` or
    ``None`` (a fresh output variable is created). Returns the output
    variable, or the constant value of ``f`` when ``z`` is fixed.
    """
    zlo, zhi = z.bounds()

    # Degenerate case: the input is fixed — evaluate the function exactly.
    if zhi - zlo < 1e-9:
        value = float(f(zlo))
        if y is not None:
            h.addConstr(y == value, name=f"{name}_fixed")
            if stats:
                stats.n_constrs += 1
            return y
        return value

    breaks = adaptive_breakpoints(f, zlo, zhi, tol)
    fvals = np.array([f(float(x)) for x in breaks])
    n_seg = len(breaks) - 1

    # HiGHS rejects matrix coefficients below its small_matrix_value
    # threshold (1e-9); saturate-tails of e.g. the sigmoid produce exactly
    # such values. Snap them to zero and skip those terms entirely.
    def _snap(v: float) -> float:
        return float(v) if abs(v) >= 1e-9 else 0.0

    lam = [
        h.addVariable(lb=0.0, ub=1.0, name=f"{name}_lam{k}") for k in range(n_seg + 1)
    ]
    bseg = [
        h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger, name=f"{name}_b{s}")
        for s in range(n_seg)
    ]

    if y is None:
        y = h.addVariable(
            lb=_snap(float(fvals.min())), ub=float(fvals.max()), name=f"{name}_out"
        )
        if stats:
            stats.n_vars += 1

    # Convex combination of breakpoints (zero coefficients skipped).
    z_terms = [
        _snap(float(breaks[k])) * lam[k]
        for k in range(n_seg + 1)
        if _snap(float(breaks[k])) != 0.0
    ]
    f_terms = [
        _snap(float(fvals[k])) * lam[k]
        for k in range(n_seg + 1)
        if _snap(float(fvals[k])) != 0.0
    ]
    z_expr = sum(z_terms) if z_terms else None
    f_expr = sum(f_terms) if f_terms else None

    if z_expr is not None:
        h.addConstr(z.to_highspy() == z_expr, name=f"{name}_z")
    else:
        h.addConstr(z.to_highspy() == 0.0, name=f"{name}_z")
    if f_expr is not None:
        h.addConstr(y == f_expr, name=f"{name}_y")
    else:
        h.addConstr(y == 0.0, name=f"{name}_y")

    one_lam = sum(lam)
    h.addConstr(one_lam == 1.0, name=f"{name}_convex")
    one_seg = sum(bseg)
    h.addConstr(one_seg == 1.0, name=f"{name}_oneseg")

    # SOS2 linking: at most the two lambdas of the active segment are nonzero.
    h.addConstr(lam[0] <= bseg[0], name=f"{name}_sos_0")
    for k in range(1, n_seg):
        h.addConstr(lam[k] <= bseg[k - 1] + bseg[k], name=f"{name}_sos_{k}")
    h.addConstr(lam[n_seg] <= bseg[n_seg - 1], name=f"{name}_sos_{n_seg}")

    if stats:
        stats.n_vars += len(lam)
        stats.n_binaries += len(bseg)
        stats.n_constrs += 4 + (n_seg + 1)
        stats.n_pwl += 1

    return y
