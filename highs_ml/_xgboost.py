"""Exact MILP embedding of XGBoost models (regressor and binary classifier).

XGBoost is an additive ensemble of regression trees, so the margin

    margin(x) = base_score + sum_trees leaf_value(x)

is a linear function of per-tree leaf binaries — the exact same machinery
as ``_trees.embed_tree``, with XGBoost's routing convention (condition
``x < split`` goes to the *Yes* child). For ``binary:logistic`` the
probability ``sigmoid(margin)`` is added on top via the certified PWL
embedding, reusing ``_pwl.add_pwl_constr``.

Notes and limitations
---------------------
* Missing values: optimization variables are always defined, so XGBoost's
  default-direction (missing) routing is ignored — inputs must be complete.
* The model constant (base_score, whose storage format varies across
  XGBoost versions) is recovered *empirically*: we evaluate the booster at
  a probe point and subtract the sum of leaf values from our own tree
  traversal. A self-check on random points validates the round-trip at
  embedding time and raises if the parsed model disagrees with the booster.
* Multiclass objectives are not supported.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from ._affine import Affine
from ._pwl import PWLStats, add_pwl_constr
from ._predictors import AbstractPredictorConstr, sigmoid
from ._trees import embed_tree


def _as_booster(model):
    import xgboost as xgb

    if isinstance(model, xgb.Booster):
        return model
    if hasattr(model, "get_booster"):
        return model.get_booster()
    raise TypeError(f"Cannot extract an xgboost.Booster from {type(model).__name__}.")


def _objective_name(booster) -> str:
    cfg = json.loads(booster.save_config())
    return cfg["learner"]["objective"]["name"]


def _feature_index_map(booster) -> dict[str, int]:
    """Map split feature names in the tree dump to input column indices."""
    names = booster.feature_names
    n = booster.num_features()
    if names is None:
        return {f"f{i}": i for i in range(n)}
    if len(names) != n:
        return {f"f{i}": i for i in range(n)}
    return {name: i for i, name in enumerate(names)}


def _xgb_leaf_paths(df_tree, fmap) -> List[Tuple[float, List[Tuple[int, float, bool]]]]:
    """(value, path) per leaf for one tree of a trees_to_dataframe() dump.

    Path entries are (feature_index, split, went_yes); routing is
    ``x < split`` -> Yes child.
    """
    rows = {row.ID: row for row in df_tree.itertuples()}

    def walk(node_id: str, path: List[Tuple[int, float, bool]]):
        row = rows[node_id]
        if row.Feature == "Leaf":
            return [(float(row.Gain), path)]
        feat = fmap[row.Feature]
        thr = float(row.Split)
        return walk(row.Yes, path + [(feat, thr, True)]) + \
            walk(row.No, path + [(feat, thr, False)])

    root = df_tree.iloc[0].ID
    return walk(root, [])


def _eval_margin_from_paths(all_paths, x: np.ndarray) -> float:
    """Sum of leaf values at x using our own parsed structure (no constant)."""
    total = 0.0
    for leaves in all_paths:
        for value, path in leaves:
            if all((x[f] < t) if yes else (x[f] >= t) for f, t, yes in path):
                total += value
                break
    return total


class XGBoostConstr(AbstractPredictorConstr):
    """Exact embedding of an XGBoost booster (or sklearn wrapper).

    For ``XGBRegressor``/regression objectives the output is the margin
    (identity link) — exact. For ``XGBClassifier``/``binary:logistic``,
    ``output_type='probability_1'`` (default) applies the certified PWL
    sigmoid to the margin; ``output_type='raw'`` exposes the exact margin.
    """

    def __init__(self, h, model, input_vars, output_var=None,
                 output_type: str = "probability_1", pwl_tol: float = 0.01,
                 stats: Optional[PWLStats] = None, name: str = "xgb"):
        import xgboost as xgb

        self._sklearn_model = model if hasattr(model, "get_booster") else None
        booster = _as_booster(model)
        self.booster = booster
        objective = _objective_name(booster)
        is_classifier = objective.startswith("binary:") or objective == "reg:logistic"
        if objective.startswith("multi:"):
            raise ValueError("Multiclass XGBoost objectives are not supported.")
        if output_type not in ("probability_1", "raw"):
            raise ValueError("output_type must be 'probability_1' or 'raw'.")
        self.output_type = "raw" if not is_classifier else output_type

        # Coerce inputs (sklearn wrappers carry feature metadata; raw
        # boosters require a sequence input).
        n_features = booster.num_features()
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

        # Parse all trees.
        df = booster.trees_to_dataframe()
        fmap = _feature_index_map(booster)
        all_paths = [
            _xgb_leaf_paths(df[df["Tree"] == k], fmap)
            for k in sorted(df["Tree"].unique())
        ]

        # Recover the additive constant empirically and self-check the parse.
        probe = np.zeros((1, n_features))
        margin0 = float(
            booster.predict(xgb.DMatrix(probe), output_margin=True)[0]
        )
        constant = margin0 - _eval_margin_from_paths(all_paths, probe[0])
        rng = np.random.default_rng(0)
        for trial in range(3):
            pt = rng.normal(size=(1, n_features))
            ref = float(booster.predict(xgb.DMatrix(pt), output_margin=True)[0])
            mine = constant + _eval_margin_from_paths(all_paths, pt[0])
            if abs(ref - mine) > 1e-6:
                raise RuntimeError(
                    f"XGBoost parse self-check failed (ref {ref}, parsed {mine})."
                )

        # Embed every tree exactly and sum.
        margin = Affine(const=constant)
        for k, leaves in enumerate(all_paths):
            margin = margin + embed_tree(
                h, None, inputs, coef=1.0, name=f"{name}_t{k}",
                stats=self.stats, split_style="xgboost", leaves=leaves,
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
        import xgboost as xgb

        dm = xgb.DMatrix(x.reshape(1, -1))
        if self._is_classifier and self.output_type == "probability_1":
            return float(self.booster.predict(dm)[0])
        return float(self.booster.predict(dm, output_margin=True)[0])
