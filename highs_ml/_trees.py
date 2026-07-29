"""Exact MILP embeddings of tree-based scikit-learn regressors.

Unlike the sigmoid, trees need no approximation: each leaf defines a
polyhedral region of feature space and a constant prediction, so the
embedding is exact.

Formulation (leaf selection, one binary per leaf):

    sum_l lam_l = 1,  lam_l in {0, 1}
    y = sum_l v_l * lam_l                     (v_l = leaf value)

    for every internal node n on the path to leaf l:
      left branch  (x_j <= t):  x_j + (hi_j - t)     * lam_l <= hi_j
      right branch (x_j >  t):  x_j + (lo_j - t - e) * lam_l >= lo_j

where [lo_j, hi_j] are the interval bounds of feature expression j and
``e`` is a small epsilon modelling the strict inequality of the right
branch (sklearn routes ``x_j > threshold`` to the right child). Big-M
coefficients come from the actual variable bounds, and vacuous
constraints (zero coefficient) are skipped.

A random forest averages its trees; gradient boosting sums them scaled by
the learning rate plus the initial constant. Both are linear combinations
of the per-leaf binaries, so the embeddings compose exactly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from highspy import HighsVarType

from ._affine import Affine
from ._pwl import PWLStats
from ._predictors import AbstractPredictorConstr, _coerce_inputs

_EPS_DEFAULT = 1e-4
_SNAP = 1e-9  # HiGHS rejects matrix coefficients below ~1e-9


def _leaf_paths(tree) -> List[Tuple[float, List[Tuple[int, float, bool]]]]:
    """Extract (value, path) per leaf from a fitted sklearn tree.

    ``path`` is a list of (feature_index, threshold, went_left) tuples from
    the root to the leaf.
    """
    t = tree.tree_
    leaves: List[Tuple[float, List[Tuple[int, float, bool]]]] = []

    def walk(node: int, path: List[Tuple[int, float, bool]]) -> None:
        left = t.children_left[node]
        if left == -1:  # leaf
            leaves.append((float(t.value[node].ravel()[0]), path))
            return
        feat = int(t.feature[node])
        thr = float(t.threshold[node])
        walk(left, path + [(feat, thr, True)])
        walk(int(t.children_right[node]), path + [(feat, thr, False)])

    walk(0, [])
    return leaves


def embed_tree(
    h,
    tree,
    inputs: List[Affine],
    coef: float = 1.0,
    epsilon: float = _EPS_DEFAULT,
    name: str = "tree",
    stats: Optional[PWLStats] = None,
    split_style: str = "sklearn",
    leaves=None,
) -> Affine:
    """Embed one decision tree; returns its output as an :class:`Affine`
    form over the per-leaf binaries (scaled by ``coef``).

    Ensembles call this once per member tree and combine the results
    linearly, so the composition stays exact.

    ``split_style`` selects the routing convention: sklearn trees send
    ``x <= t`` left (right is strict), XGBoost sends ``x < t`` left (right
    is inclusive). ``leaves`` may supply pre-extracted
    ``(value, path)`` pairs (used by the XGBoost adapter).
    """
    if leaves is None:
        leaves = _leaf_paths(tree)
    bounds = [x.bounds() for x in inputs]

    lam = [
        h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger,
                      name=f"{name}_leaf{l}")
        for l in range(len(leaves))
    ]
    h.addConstr(sum(lam) == 1.0, name=f"{name}_pick_one")

    n_constrs = 1
    out = Affine.zero()
    for l, (value, path) in enumerate(leaves):
        out = out + Affine.coerce(lam[l]) * (coef * value)
        for feat, thr, went_left in path:
            lo, hi = bounds[feat]
            x = inputs[feat]
            # Effective routing threshold under each convention.
            if went_left:
                t_eff = thr if split_style == "sklearn" else thr - epsilon
                violates_if = lambda v, t=t_eff: v > t
            else:
                t_eff = thr + epsilon if split_style == "sklearn" else thr
                violates_if = lambda v, t=t_eff: v < t
            if x.is_constant():
                # Fixed feature: the routing decision is decidable now.
                if violates_if(x.const):
                    h.addConstr(lam[l] <= 0.0, name=f"{name}_l{l}_f{feat}_fix")
                    n_constrs += 1
                continue
            if went_left:
                m = hi - t_eff  # x <= t_eff enforced when lam_l = 1
                if m <= _SNAP:
                    continue  # holds on the whole reachable interval
                h.addConstr(x.to_highspy() + m * lam[l] <= hi,
                            name=f"{name}_l{l}_f{feat}_le")
            else:
                m = lo - t_eff  # x >= t_eff enforced when lam_l = 1
                if m >= -_SNAP:
                    continue  # holds on the whole reachable interval
                h.addConstr(x.to_highspy() + m * lam[l] >= lo,
                            name=f"{name}_l{l}_f{feat}_gt")
            n_constrs += 1

    if stats is not None:
        stats.n_binaries += len(lam)
        stats.n_constrs += n_constrs
        stats.n_vars += len(lam)
    return out


class _TreeEnsembleConstr(AbstractPredictorConstr):
    """Shared driver for single trees and ensembles of trees."""

    def __init__(self, h, predictor, input_vars, output_var=None,
                 epsilon: float = _EPS_DEFAULT,
                 stats: Optional[PWLStats] = None, name: str = "tree_ens"):
        inputs = _coerce_inputs(predictor, input_vars)
        super().__init__(h, predictor, inputs, stats)

        terms = self._ensemble_terms()  # list of (tree, coef)
        constant = self._ensemble_constant()

        out = Affine(const=constant)
        for k, (tree, coef) in enumerate(terms):
            out = out + embed_tree(
                h, tree, inputs, coef=coef, epsilon=epsilon,
                name=f"{name}_{k}", stats=self.stats,
            )

        if output_var is None:
            lo, hi = self._output_bounds(terms, constant)
            output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
            self.stats.n_vars += 1
        if out.is_constant():
            h.addConstr(output_var == out.const, name=f"{name}_eq")
        else:
            h.addConstr(output_var == out.to_highspy(), name=f"{name}_eq")
        self.stats.n_constrs += 1
        self.output_var = output_var

    # -- subclass hooks --------------------------------------------------
    def _ensemble_terms(self):  # pragma: no cover
        raise NotImplementedError

    def _ensemble_constant(self) -> float:
        return 0.0

    def _output_bounds(self, terms, constant) -> tuple[float, float]:
        lo, hi = constant, constant
        for tree, coef in terms:
            vals = [v for v, _ in _leaf_paths(tree)]
            if coef >= 0:
                lo += coef * min(vals)
                hi += coef * max(vals)
            else:
                lo += coef * max(vals)
                hi += coef * min(vals)
        return lo, hi

    def _exact_prediction(self, x) -> float:
        return float(self.predictor.predict(x.reshape(1, -1))[0])


class DecisionTreeRegressorConstr(_TreeEnsembleConstr):
    """Exact embedding of ``sklearn.tree.DecisionTreeRegressor``."""

    def _ensemble_terms(self):
        return [(self.predictor, 1.0)]


class RandomForestRegressorConstr(_TreeEnsembleConstr):
    """Exact embedding of ``sklearn.ensemble.RandomForestRegressor``."""

    def _ensemble_terms(self):
        trees = self.predictor.estimators_
        return [(tree, 1.0 / len(trees)) for tree in trees]


class GradientBoostingRegressorConstr(_TreeEnsembleConstr):
    """Exact embedding of ``sklearn.ensemble.GradientBoostingRegressor``.

    prediction = init_constant + learning_rate * sum(stage predictions).
    """

    def _ensemble_terms(self):
        est = self.predictor.estimators_
        if est.shape[1] != 1:
            raise ValueError("Multi-output gradient boosting is not supported.")
        lr = float(self.predictor.learning_rate)
        return [(est[k, 0], lr) for k in range(est.shape[0])]

    def _ensemble_constant(self) -> float:
        init = self.predictor.init_
        if isinstance(init, str):  # 'zero'
            return 0.0
        if hasattr(init, "constant_"):
            return float(init.constant_.ravel()[0])
        raise ValueError(
            f"Unsupported GradientBoostingRegressor init estimator: {type(init).__name__!r}."
        )
