"""Exact/certified MILP embedding of ONNX feedforward networks.

Parses an ``onnx.ModelProto`` directly (no ONNX runtime needed) and maps
the supported operator chain onto the shared feedforward machinery:

* ``Gemm`` (``Y = alpha*A@B + beta*C``, honoring ``transA``/``transB``)
* ``MatMul`` + ``Add`` (weights must be graph initializers, i.e. constants)
* ``Relu`` (exact big-M), ``Sigmoid``/``Tanh`` (certified PWL), ``Identity``,
  ``Flatten``

Anything else (Conv, BatchNorm, Softmax, ...) raises a clear error. Only
single-input, single-output feedforward graphs are supported.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ._affine import Affine
from ._nn_common import embed_feedforward, embed_relu, normalize_activation
from ._pwl import PWLStats, add_pwl_constr
from ._predictors import AbstractPredictorConstr, sigmoid

import math

_ACT_FNS = {"Sigmoid": sigmoid, "Tanh": math.tanh}

Tensor = Union[np.ndarray, list]  # constant array or list[Affine]


class ONNXNetworkConstr(AbstractPredictorConstr):
    """Embedding of an ONNX dense feedforward network (``onnx.ModelProto``)."""

    def __init__(self, h, model_proto, input_vars, output_var=None,
                 pwl_tol: float = 0.01, stats: Optional[PWLStats] = None,
                 name: str = "onnx"):
        from onnx import numpy_helper

        graph = model_proto.graph
        initializers = {init.name: numpy_helper.to_array(init)
                        for init in graph.initializer}
        input_names = [i.name for i in graph.input
                       if i.name not in initializers]
        if len(input_names) != 1:
            raise ValueError("Only single-input ONNX graphs are supported.")
        if len(graph.output) != 1:
            raise ValueError("Only single-output ONNX graphs are supported.")

        inputs = [Affine.coerce(v) for v in input_vars]
        super().__init__(h, model_proto, inputs, stats)

        tensors: dict[str, Tensor] = dict(initializers)
        tensors[input_names[0]] = inputs

        for node in graph.node:
            tensors[node.output[0]] = self._apply_node(
                node, tensors, h, pwl_tol, name, len(tensors))

        out = tensors[graph.output[0].name]
        if not isinstance(out, list) or len(out) != 1:
            raise ValueError("ONNX graph output must be a scalar vector.")
        self.output_var = self._link(h, out[0], output_var, name)

    # ------------------------------------------------------------------
    def _link(self, h, out: Affine, output_var, name: str):
        if output_var is None:
            lo, hi = out.bounds()
            output_var = h.addVariable(lb=lo, ub=hi, name=f"{name}_out")
            self.stats.n_vars += 1
        if out.is_constant():
            h.addConstr(output_var == out.const, name=f"{name}_eq")
        else:
            h.addConstr(output_var == out.to_highspy(), name=f"{name}_eq")
        self.stats.n_constrs += 1
        return output_var

    def _apply_node(self, node, tensors, h, pwl_tol, prefix, seq) -> Tensor:
        op = node.op_type
        attrs = {a.name: a for a in node.attribute}

        def get(name):
            if name not in tensors:
                raise ValueError(f"ONNX tensor {name!r} not found "
                                 f"(node {node.name or op}).")
            return tensors[name]

        if op == "Gemm":
            A, B = get(node.input[0]), get(node.input[1])
            if isinstance(A, np.ndarray) == isinstance(B, np.ndarray):
                raise ValueError("Gemm requires exactly one constant operand.")
            if isinstance(A, np.ndarray):
                A, B = B, A  # ensure A is the affine vector
            alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
            beta = float(attrs["beta"].f) if "beta" in attrs else 1.0
            Bm = B.T if "transB" in attrs and attrs["transB"].i else B
            C = (get(node.input[2]) if len(node.input) > 2
                 else np.zeros(Bm.shape[1]))
            out = []
            for j in range(Bm.shape[1]):
                z = Affine(const=beta * float(np.atleast_1d(C)[j]
                                           if np.ndim(C) else C))
                for i, x in enumerate(A):
                    coef = alpha * float(Bm[i, j])
                    if coef != 0.0:
                        z = z + x * coef
                out.append(z)
            return out

        if op == "MatMul":
            A, B = get(node.input[0]), get(node.input[1])
            if not isinstance(B, np.ndarray):
                raise ValueError("MatMul requires a constant second operand "
                                 "(weights as initializer).")
            Bm = np.atleast_2d(B)
            return [sum((x * float(Bm[i, j]) for i, x in enumerate(A)),
                        Affine.zero())
                    for j in range(Bm.shape[1])]

        if op == "Add":
            a, b = get(node.input[0]), get(node.input[1])
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return a + b
            vec, const = (a, b) if not isinstance(a, np.ndarray) else (b, a)
            flat = np.atleast_1d(const)
            return [x + float(flat[i]) for i, x in enumerate(vec)]

        if op in ("Relu", "Sigmoid", "Tanh", "Identity", "Flatten"):
            x = get(node.input[0])
            if op in ("Identity", "Flatten"):
                return x
            if isinstance(x, np.ndarray):
                fn = (np.maximum(0, x) if op == "Relu"
                      else np.vectorize(_ACT_FNS[op])(x))
                return fn
            out = []
            for i, xi in enumerate(x):
                lname = f"{prefix}_{op.lower()}{seq}_{i}"
                if op == "Relu":
                    out.append(embed_relu(h, xi, lname, self.stats))
                else:
                    var = add_pwl_constr(h, _ACT_FNS[op], xi, tol=pwl_tol,
                                         name=lname, stats=self.stats)
                    out.append(Affine.coerce(var) if not isinstance(var, float)
                               else Affine(const=var))
            return out

        raise ValueError(
            f"Unsupported ONNX operator {op!r}. Supported: Gemm, MatMul, "
            "Add, Relu, Sigmoid, Tanh, Identity, Flatten."
        )

    def _exact_prediction(self, x: np.ndarray) -> float:
        # Evaluate the ONNX graph numerically without onnxruntime: reuse the
        # parsed tensors by re-running with constant inputs.
        # Simplest robust check: forward-pass through the node list again.
        from onnx import numpy_helper

        graph = self.predictor.graph
        initializers = {init.name: numpy_helper.to_array(init)
                        for init in graph.initializer}
        input_names = [i.name for i in graph.input
                       if i.name not in initializers]
        values: dict[str, np.ndarray] = dict(initializers)
        values[input_names[0]] = x.reshape(-1)
        for node in graph.node:
            values[node.output[0]] = self._eval_node(node, values)
        return float(np.atleast_1d(values[graph.output[0].name]).ravel()[0])

    @staticmethod
    def _eval_node(node, values) -> np.ndarray:
        op = node.op_type
        attrs = {a.name: a for a in node.attribute}
        if op == "Gemm":
            A = values[node.input[0]]
            B = values[node.input[1]]
            C = values[node.input[2]] if len(node.input) > 2 else 0.0
            alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
            beta = float(attrs["beta"].f) if "beta" in attrs else 1.0
            if "transA" in attrs and attrs["transA"].i:
                A = A.T
            if "transB" in attrs and attrs["transB"].i:
                B = B.T
            return alpha * (A @ B) + beta * C
        if op == "MatMul":
            return values[node.input[0]] @ values[node.input[1]]
        if op == "Add":
            return values[node.input[0]] + values[node.input[1]]
        x = values[node.input[0]]
        if op == "Relu":
            return np.maximum(0, x)
        if op == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if op == "Tanh":
            return np.tanh(x)
        if op in ("Identity", "Flatten"):
            return x
        raise ValueError(f"Unsupported ONNX operator {op!r}.")


def is_onnx_model(obj) -> bool:
    return type(obj).__name__ == "ModelProto" and "onnx" in type(obj).__module__
