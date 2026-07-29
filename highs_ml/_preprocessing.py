"""Expression-level embeddings of sklearn preprocessing steps.

These transforms are applied to :class:`Affine` forms before the predictor
step of a pipeline — they add no variables of their own.

* ``StandardScaler`` -- affine rescaling (also handled in ``_predictors``).
* ``PolynomialFeatures`` -- supported only while every expanded term stays
  *affine*: a term may contain at most one decision-variable factor (to the
  first power); all other factors must be constants in the optimization
  model. Genuine products of two decision variables (x_i * x_j) are
  bilinear, and HiGHS — unlike Gurobi — has no non-convex quadratic
  capability, so those terms raise a clear, actionable error.
* ``ColumnTransformer`` -- per-column-subset transforms (StandardScaler,
  PolynomialFeatures, passthrough, or sub-pipelines thereof), concatenated
  in sklearn's output order.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ._affine import Affine


class PolynomialFeaturesStep:
    """Embedding of ``sklearn.preprocessing.PolynomialFeatures``.

    Terms with at most one decision-variable factor stay affine. True
    variable products (degree-2, exactly two variable factors, possibly a
    square) are embedded via ``_bilinear`` — exact when one factor is
    binary/integer, certified piecewise-McCormick otherwise — provided a
    HiGHS model ``h`` is supplied. Anything harder (degree > 2, more
    than two variable factors) raises a clear error.
    """

    def __init__(self, poly):
        self.poly = poly

    def transform(self, inputs: List[Affine], h=None,
                  bilinear_tol: float = 0.01) -> List[Affine]:
        out = []
        for powers in self.poly.powers_:
            coef = 1.0
            var_factors = []
            for p, x in zip(powers, inputs):
                if p == 0:
                    continue
                if x.is_constant():
                    coef *= x.const ** int(p)
                else:
                    var_factors.append((x, int(p)))
            if not var_factors:
                out.append(Affine(const=coef))
            elif len(var_factors) == 1 and var_factors[0][1] == 1:
                out.append(var_factors[0][0] * coef)
            elif (len(var_factors) == 1 and var_factors[0][1] == 2) or \
                    (len(var_factors) == 2 and
                     all(p == 1 for _, p in var_factors)):
                if h is None:
                    raise ValueError(
                        "PolynomialFeatures produced a variable product; "
                        "embedding it requires the HiGHS model (pass h)."
                    )
                from ._bilinear import add_bilinear_constr
                xa = var_factors[0][0]
                xb = var_factors[-1][0]
                if len(xa.terms) != 1 or len(xb.terms) != 1:
                    raise ValueError(
                        "Bilinear embedding requires each factor to be a "
                        "single variable (or constant); introduce "
                        "auxiliary variables for compound factors."
                    )
                y = add_bilinear_constr(h, xa, xb, tol=bilinear_tol,
                                        name="poly_bil")
                out.append(Affine.coerce(y) * coef)
            else:
                raise ValueError(
                    "PolynomialFeatures term has degree > 2 or more than "
                    "two variable factors — beyond the bilinear embedding. "
                    "Keep products quadratic and pairwise, or linearize "
                    "externally."
                )
        return out


class ColumnTransformerStep:
    """Embedding of ``sklearn.compose.ColumnTransformer``."""

    def __init__(self, ct):
        self.ct = ct

    def transform(self, inputs: List[Affine], h=None,
                  bilinear_tol: float = 0.01) -> List[Affine]:
        out: List[Affine] = []
        for name, transformer, columns in self.ct.transformers_:
            idxs = self._resolve_columns(columns, len(inputs), name)
            if transformer == "drop":
                continue
            selected = [inputs[i] for i in idxs]
            if transformer == "passthrough" or self._is_identity(transformer):
                out.extend(selected)
            else:
                out.extend(apply_transform_step(transformer, selected, h=h,
                                                bilinear_tol=bilinear_tol))
        return out

    @staticmethod
    def _is_identity(transformer) -> bool:
        # sklearn >= 1.6 materializes 'passthrough' as an identity
        # FunctionTransformer in fitted transformers_.
        from sklearn.preprocessing import FunctionTransformer
        return (isinstance(transformer, FunctionTransformer)
                and transformer.func is None)

    @staticmethod
    def _resolve_columns(columns, n_inputs: int, name: str) -> List[int]:
        if isinstance(columns, slice):
            return list(range(n_inputs))[columns]
        cols = list(columns)
        if not cols:
            return []
        if all(isinstance(c, (bool, np.bool_)) for c in cols):
            return [i for i, c in enumerate(cols) if c]
        if all(isinstance(c, (int, np.integer)) for c in cols):
            return [int(c) for c in cols]
        raise ValueError(
            f"ColumnTransformer entry {name!r} uses column names; highs_ml "
            "needs positional inputs — pass inputs as a sequence and fit the "
            "ColumnTransformer with integer column selectors."
        )


def apply_transform_step(step, exprs: List[Affine], h=None,
                         bilinear_tol: float = 0.01) -> List[Affine]:
    """Apply one supported preprocessing step to affine expressions.

    ``h`` (the HiGHS model) is required when the step can produce
    variable products (PolynomialFeatures of degree >= 2 with more than
    one optimization-variable feature); they are embedded exactly or with
    a certified bilinear envelope.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if isinstance(step, StandardScaler):
        return [
            (x - float(mu)) * (1.0 / float(sd))
            for x, mu, sd in zip(exprs, step.mean_, step.scale_)
        ]
    if isinstance(step, PolynomialFeatures):
        return PolynomialFeaturesStep(step).transform(exprs, h=h,
                                                      bilinear_tol=bilinear_tol)
    if isinstance(step, ColumnTransformer):
        return ColumnTransformerStep(step).transform(exprs, h=h,
                                                     bilinear_tol=bilinear_tol)
    if isinstance(step, Pipeline):
        for _, sub in step.named_steps.items():
            exprs = apply_transform_step(sub, exprs, h=h,
                                         bilinear_tol=bilinear_tol)
        return exprs
    raise ValueError(
        f"Unsupported preprocessing step {type(step).__name__!r}. Supported: "
        "StandardScaler, PolynomialFeatures, ColumnTransformer, and "
        "sub-pipelines of these."
    )
