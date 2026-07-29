"""Exact MILP embedding of LightGBM models (regressor and binary classifier).

Like XGBoost, LightGBM is an additive ensemble of regression trees whose
margin is a linear function of per-tree leaf binaries. Differences handled
here:

* parsing uses ``Booster.dump_model()`` (nested dicts, integer feature
  indices, ``decision_type == '<='`` — the same convention as sklearn, so
  ``split_style='sklearn'`` applies);
* the model constant (init score) is recovered empirically with a probe
  point and validated by a random-point self-check, as in ``_xgboost``;
* ``binary`` objective: probability = sigmoid(margin) via certified PWL.

Categorical splits (``decision_type == '=='``) and multiclass objectives
are not supported. Missing-value routing is ignored — optimization
variables are always defined.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._affine import Affine
from ._pwl import PWLStats, add_pwl_constr
from ._predictors import AbstractPredictorConstr, sigmoid
from ._trees import embed_tree


def _as_booster(model):
    import lightgbm as lgb

    if isinstance(model, lgb.Booster):
        return model
    if hasattr(model, "booster_"):
        return model.booster_
    raise TypeError(f"Cannot extract a lightgbm.Booster from {type(model).__name__}.")


def _lgb_leaf_paths(node: dict, path=None) -> List[Tuple[float, list]]:
    """(leaf_value, path) pairs from a LightGBM tree_structure dict.

    Path entries are (feature_index, threshold, went_left) with routing
    ``x <= threshold`` -> left child.
    """
    path = path or []
    if "leaf_value" in node:
        return [(float(node["leaf_value"]), path)]
    decision = node.get("decision_type", "<=")
    if decision != "<=":
        raise ValueError(
            f"Unsupported LightGBM split type {decision!r} "
            "(categorical splits are not supported)."
        )
    feat = int(node["split_feature"])
    thr = float(node["threshold"])
    return _lgb_leaf_paths(node["left_child"], path + [(feat, thr, True)]) + \
        _lgb_leaf_paths(node["right_child"], path + [(feat, thr, False)])


def _eval_margin_from_paths(all_paths, x: np.ndarray) -> float:
    total = 0.0
    for leaves in all_paths:
        for value, path in leaves:
            if all((x[f] <= t) if left else (x[f] > t) for f, t, left in path):
                total += value
                break
    return total


class LightGBMConstr(AbstractPredictorConstr):
    """Exact embedding of a LightGBM booster (or sklearn wrapper).

    Regression objectives expose the raw score exactly. For the ``binary``
    objective, ``output_type='probability_1'`` (default) applies the
    certified PWL sigmoid; ``output_type='raw'`` exposes the exact margin.
    """

    def __init__(self, h, model, input_vars, output_var=None,
                 output_type: str = "probability_1", pwl_tol: float = 0.01,
                 stats: Optional[PWLStats] = None, name: str = "lgbm"):
        booster = _as_booster(model)
        self.booster = booster
        dump = booster.dump_model()
        objective = str(dump.get("objective", "regression")).split()[0]
        if objective not in ("regression", "regression_l2", "regression_l1",
                             "huber", "binary"):
            raise ValueError(f"Unsupported LightGBM objective {objective!r}.")
        is_classifier = objective == "binary"
        if output_type not in ("probability_1", "raw"):
            raise ValueError("output_type must be 'probability_1' or 'raw'.")
        self.output_type = "raw" if not is_classifier else output_type

        n_features = booster.num_feature()
        if isinstance(input_vars, dict):
            names = getattr(model, "feature_names_in_", None)
            if names is None:
                raise ValueError(
                    "Mapping inputs require an sklearn wrapper fitted with "
                    "feature names; pass a sequence for a raw Booster."
                )
            inputs = [Affine.coerce(input_vars[str(nm)]) for nm in names]
        else:
            inputs = [Affine.coerce(v) for v in input_vars]
        if len(inputs) != n_features:
            raise ValueError(f"Expected {n_features} inputs, got {len(inputs)}.")
        super().__init__(h, model, inputs, stats)

        all_paths = [
            _lgb_leaf_paths(tree["tree_structure"])
            for tree in dump["tree_info"]
        ]

        # Recover the additive constant empirically and self-check the parse.
        probe = np.zeros((1, n_features))
        margin0 = float(np.atleast_1d(
            booster.predict(probe, raw_score=True))[0])
        constant = margin0 - _eval_margin_from_paths(all_paths, probe[0])
        rng = np.random.default_rng(0)
        for _ in range(3):
            pt = rng.normal(size=(1, n_features))
            ref = float(np.atleast_1d(booster.predict(pt, raw_score=True))[0])
            mine = constant + _eval_margin_from_paths(all_paths, pt[0])
            if abs(ref - mine) > 1e-6:
                raise RuntimeError(
                    f"LightGBM parse self-check failed (ref {ref}, parsed {mine})."
                )

        margin = Affine(const=constant)
        for k, leaves in enumerate(all_paths):
            margin = margin + embed_tree(
                h, None, inputs, coef=1.0, name=f"{name}_t{k}",
                stats=self.stats, split_style="sklearn", leaves=leaves,
            )
        self._margin = margin

        if self.output_type == "raw" or not is_classifier:
            if output_var is None:
                lo = constant + sum(min(v for v, _ in L) for L in all_paths)
                hi = constant + sum(max(v for v, _ in L) for L in all_paths)
                output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
                self.stats.n_vars += 1
            if margin.is_constant():
                h.addConstr(output_var == margin.const, name=f"{name}_eq")
            else:
                h.addConstr(output_var == margin.to_highspy(), name=f"{name}_eq")
            self.stats.n_constrs += 1
        else:
            output_var = add_pwl_constr(
                h, sigmoid, margin, y=output_var, tol=pwl_tol,
                name=f"{name}_sig", stats=self.stats,
            )

        self.output_var = output_var
        self._is_classifier = is_classifier

    def _exact_prediction(self, x: np.ndarray) -> float:
        if self._is_classifier and self.output_type == "probability_1":
            return float(np.atleast_1d(self.booster.predict(x.reshape(1, -1)))[0])
        return float(np.atleast_1d(
            self.booster.predict(x.reshape(1, -1), raw_score=True))[0])
