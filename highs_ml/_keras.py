"""Exact/certified MILP embedding of Keras feedforward networks.

Supports Keras models (any backend — the embedding only reads topology and
weights through the backend-agnostic ``model.layers`` API) composed of
``Dense`` layers with relu/sigmoid/tanh/linear activations, plus
``Input``, ``Flatten`` and standalone ``Activation`` layers. Convolutional,
recurrent, normalization and attention layers are not supported — HiGHS
would need bilinear or attention-specific formulations that do not exist in
an LP/MIP solver.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._affine import Affine
from ._nn_common import embed_feedforward, link_output, normalize_activation
from ._pwl import PWLStats
from ._predictors import AbstractPredictorConstr, _coerce_inputs

_SKIP_LAYERS = {"InputLayer", "Input", "Flatten", "Dropout", "Reshape"}


def _keras_layer_specs(model):
    """Extract (W, b, activation) triples from a Keras model's layers."""
    specs = []
    pending_activation: Optional[str] = None
    for layer in model.layers:
        lname = type(layer).__name__
        if lname in _SKIP_LAYERS:
            continue
        if lname == "Activation":
            if pending_activation is not None:
                raise ValueError("Two consecutive Activation layers are not supported.")
            pending_activation = layer.activation.__name__
            continue
        if lname != "Dense":
            raise ValueError(
                f"Unsupported Keras layer {lname!r}. Supported: Dense, "
                "Activation, Input/Flatten/Dropout/Reshape."
            )
        weights = layer.get_weights()
        W = np.asarray(weights[0], dtype=float)
        b = (np.asarray(weights[1], dtype=float) if layer.use_bias
             else np.zeros(W.shape[1]))
        act = pending_activation or layer.activation.__name__
        specs.append((W, b, normalize_activation(act)))
        pending_activation = None
    if pending_activation is not None:
        raise ValueError("A trailing Activation layer is not supported.")
    if not specs:
        raise ValueError("Keras model contains no Dense layers.")
    return specs


class KerasNetworkConstr(AbstractPredictorConstr):
    """Embedding of a Keras Sequential/Functional dense network."""

    def __init__(self, h, model, input_vars, output_var=None,
                 pwl_tol: float = 0.01, stats: Optional[PWLStats] = None,
                 name: str = "keras"):
        specs = _keras_layer_specs(model)
        n_in = specs[0][0].shape[0]
        if isinstance(input_vars, dict):
            raise ValueError("Keras inputs must be a sequence in feature order.")
        inputs = [Affine.coerce(v) for v in input_vars]
        if len(inputs) != n_in:
            raise ValueError(f"Expected {n_in} inputs, got {len(inputs)}.")
        super().__init__(h, model, inputs, stats)

        out = embed_feedforward(h, specs, inputs, pwl_tol, self.stats, name)
        if len(out) != 1:
            raise ValueError("Multi-output Keras models are not supported yet.")
        self.output_var = link_output(h, out[0], output_var, self.stats, name)

    def _exact_prediction(self, x: np.ndarray) -> float:
        return float(np.atleast_1d(
            self.predictor.predict(x.reshape(1, -1), verbose=0)).ravel()[0])


def is_keras_model(obj) -> bool:
    """Duck-typed detection that works for both tf.keras and Keras 3."""
    return (
        hasattr(obj, "layers")
        and callable(getattr(obj, "predict", None))
        and "keras" in type(obj).__module__.lower()
    )
