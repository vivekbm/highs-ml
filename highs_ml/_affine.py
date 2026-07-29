"""Structured affine expressions over highspy variables.

highspy's ``highs_linear_expression`` is opaque: once built you cannot
inspect coefficients or compute bounds, both of which are required to
formulate MILP embeddings of ML predictors (big-M values for ReLU
networks, breakpoint ranges for piecewise-linear approximations).

``Affine`` is the internal currency of highs_ml: a sparse linear form
``sum_i coef_i * var_i + const`` that supports arithmetic, interval
bounds, and conversion back into native highspy expressions.
"""

from __future__ import annotations

from typing import Dict, Union

import highspy
from highspy.highs import highs_var

Number = Union[int, float]
AffineLike = Union["Affine", highs_var, Number]


class Affine:
    """A sparse affine form over highspy variables."""

    __slots__ = ("terms", "const")

    def __init__(self, terms: Dict[highs_var, float] | None = None, const: float = 0.0):
        self.terms: Dict[highs_var, float] = dict(terms) if terms else {}
        self.const: float = float(const)

    # ------------------------------------------------------------------
    # construction / coercion
    # ------------------------------------------------------------------
    @staticmethod
    def coerce(value: AffineLike) -> "Affine":
        if isinstance(value, Affine):
            return value
        if isinstance(value, highs_var):
            return Affine({value: 1.0})
        if isinstance(value, (int, float)):
            return Affine(const=float(value))
        raise TypeError(f"Cannot interpret {type(value).__name__} as an affine expression.")

    @staticmethod
    def zero() -> "Affine":
        return Affine()

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: AffineLike) -> "Affine":
        other = Affine.coerce(other)
        terms = dict(self.terms)
        for var, coef in other.terms.items():
            terms[var] = terms.get(var, 0.0) + coef
            if terms[var] == 0.0:
                del terms[var]
        return Affine(terms, self.const + other.const)

    __radd__ = __add__

    def __neg__(self) -> "Affine":
        return Affine({v: -c for v, c in self.terms.items()}, -self.const)

    def __sub__(self, other: AffineLike) -> "Affine":
        return self + (-Affine.coerce(other))

    def __rsub__(self, other: AffineLike) -> "Affine":
        return Affine.coerce(other) + (-self)

    def __mul__(self, scalar: Number) -> "Affine":
        if not isinstance(scalar, (int, float)):
            raise TypeError("Affine expressions only support multiplication by a scalar.")
        return Affine({v: c * scalar for v, c in self.terms.items()}, self.const * scalar)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    def is_constant(self) -> bool:
        return not self.terms

    def bounds(self) -> tuple[float, float]:
        """Interval bounds from the current variable bounds in the HiGHS model."""
        lo, hi = self.const, self.const
        col_bounds: list[tuple[float, float]] | None = None
        for var, coef in self.terms.items():
            if col_bounds is None:
                lp = var.highs.getLp()
                col_bounds = [
                    (float(l), float(u)) for l, u in zip(lp.col_lower_, lp.col_upper_)
                ]
            vlo, vhi = col_bounds[var.index]
            if vlo <= -highspy.kHighsInf * 0.5 or vhi >= highspy.kHighsInf * 0.5:
                raise ValueError(
                    f"Variable {var!r} has (semi-)infinite bounds; ML predictor embeddings "
                    "require finite bounds on all input variables."
                )
            if coef >= 0:
                lo += coef * vlo
                hi += coef * vhi
            else:
                lo += coef * vhi
                hi += coef * vlo
        return lo, hi

    def evaluate(self, values) -> float:
        """Evaluate the form given a column-value array from a HiGHS solution."""
        total = self.const
        for var, coef in self.terms.items():
            total += coef * float(values[var.index])
        return total

    # ------------------------------------------------------------------
    # conversion to native highspy expressions
    # ------------------------------------------------------------------
    def to_highspy(self):
        """Convert to a native highspy linear expression (non-constant forms only).

        Coefficients below HiGHS's small_matrix_value threshold (1e-9)
        are snapped to zero — they arise naturally from fitted models
        (e.g. a linear regression's ~0 coefficient on a quadratic term)
        and would otherwise be rejected by addRow.
        """
        expr = None
        for var, coef in self.terms.items():
            if abs(coef) < 1e-9:
                continue
            term = coef * var
            expr = term if expr is None else expr + term
        if expr is None:
            raise ValueError("Constant affine forms have no native highspy expression.")
        return expr + self.const

    def __repr__(self) -> str:
        parts = [f"{coef:+.4g}*{var!r}" for var, coef in self.terms.items()]
        parts.append(f"{self.const:+.4g}")
        return f"Affine({' '.join(parts)})"
