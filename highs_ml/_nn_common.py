"""Shared embedding machinery for feedforward neural networks.

A feedforward net is a sequence of dense layers ``pre = W^T h + b``
followed by an activation. ReLU activations embed exactly with big-M
constraints (one binary per neuron); sigmoid/tanh activations use the
certified PWL embedding from ``_pwl``. Used by the sklearn MLP, Keras and
ONNX embeddings — only the model parsing differs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from highspy import HighsVarType

from ._affine import Affine
from ._pwl import PWLStats, add_pwl_constr
from ._predictors import _linear_form, sigmoid

_ACT_FNS = {"sigmoid": sigmoid, "tanh": math.tanh}


def normalize_activation(name: Optional[str]) -> str:
    name = "linear" if name is None else str(name).lower()
    aliases = {"identity": "linear", "logistic": "sigmoid"}
    return aliases.get(name, name)


def embed_relu(h, pre: Affine, name: str, stats: PWLStats) -> Affine:
    """Exact big-M embedding of ``out = max(0, pre)``."""
    lo, hi = pre.bounds()
    if hi <= 0.0:
        return Affine.zero()
    if lo >= 0.0:
        return pre
    out = h.addVariable(lb=0.0, ub=hi, name=f"{name}_relu")
    delta = h.addVariable(lb=0.0, ub=1.0, type=HighsVarType.kInteger,
                          name=f"{name}_delta")
    pre_expr = pre.to_highspy()
    h.addConstr(out >= pre_expr, name=f"{name}_ge")
    h.addConstr(out <= pre_expr - lo * (1.0 - delta), name=f"{name}_le_pre")
    h.addConstr(out <= hi * delta, name=f"{name}_le_m")
    stats.n_vars += 1
    stats.n_binaries += 1
    stats.n_constrs += 3
    stats.n_relu += 1
    return Affine.coerce(out)


LayerSpec = Tuple[np.ndarray, np.ndarray, str]  # (W (in,out), b (out,), activation)


def embed_feedforward(
    h,
    layers: List[LayerSpec],
    inputs: List[Affine],
    pwl_tol: float,
    stats: PWLStats,
    name: str,
) -> List[Affine]:
    """Propagate affine inputs through dense layers; returns output affines."""
    layer_out: List[Affine] = list(inputs)
    for li, (W, b, activation) in enumerate(layers):
        act = normalize_activation(activation)
        next_out: List[Affine] = []
        for j in range(W.shape[1]):
            pre = _linear_form(W[:, j], b[j], layer_out)
            lname = f"{name}_l{li}n{j}"
            if act == "linear":
                next_out.append(pre)
            elif act == "relu":
                next_out.append(embed_relu(h, pre, lname, stats))
            elif act in _ACT_FNS:
                var = add_pwl_constr(h, _ACT_FNS[act], pre, tol=pwl_tol,
                                     name=lname, stats=stats)
                next_out.append(Affine.coerce(var) if not isinstance(var, float)
                                else Affine(const=var))
            else:
                raise ValueError(
                    f"Unsupported activation {activation!r} in layer {li}. "
                    "Supported: linear/identity, relu, sigmoid/logistic, tanh."
                )
        layer_out = next_out
    return layer_out


def link_output(h, out: Affine, output_var, stats: PWLStats, name: str):
    """Create (if needed) and link the scalar output variable."""
    if output_var is None:
        lo, hi = out.bounds()
        output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
        stats.n_vars += 1
    if out.is_constant():
        h.addConstr(output_var == out.const, name=f"{name}_eq")
    else:
        h.addConstr(output_var == out.to_highspy(), name=f"{name}_eq")
    stats.n_constrs += 1
    return output_var
