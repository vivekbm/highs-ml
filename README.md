# highs_ml

**Embed trained machine-learning models as constraints in [HiGHS](https://github.com/ERGO-Code/HiGHS) optimization models — with certified accuracy.**

Train a model with scikit-learn, XGBoost, LightGBM, Keras or ONNX, then
*optimize over its predictions*: find the feature values that maximize
(or bound) the model's output, exactly or to a certified tolerance,
inside a plain MILP. MIT-licensed end to end.

```python
import highspy
from highs_ml import add_predictor_constr

h = highspy.Highs()
x = h.addVariable(lb=0.0, ub=2.5, name="merit")   # decision variable
y = h.addVariable(lb=0.0, ub=1.0, name="prob")    # predicted probability

pred = add_predictor_constr(
    h, pipe,                               # fitted sklearn pipeline
    {"merit": x, "SAT": 1300, "GPA": 3.4}, # inputs: variables or constants
    y,
    output_type="probability_1",
)
h.maximize(y)
print(pred.get_error())                    # certified approximation error
```

## Install

```
pip install git+https://github.com/vivekbm/highs-ml
```

Requires `highspy >= 1.10`, `scikit-learn`, `numpy`, `scipy`. The
optional predictor backends are ordinary pip installs (`xgboost`,
`lightgbm`, `keras` with any backend, `onnx`; also available as extras,
e.g. `[all-predictors]`) — none are required by `highs_ml` itself.
`pandas` is used by the examples and tests, and `gurobipy` +
`gurobi-machinelearning` only by the optional comparison script.

## Supported predictors

| Predictor                        | Embedding in HiGHS                                   | Accuracy                     |
|----------------------------------|------------------------------------------------------|------------------------------|
| `LinearRegression` / `Ridge`     | linear equality                                      | exact                        |
| `PLSRegression`                  | linear equality (PLS prediction is linear)           | exact                        |
| `LogisticRegression` (binary)    | sigmoid via adaptive piecewise-linear interpolation, SOS2 enforced with binaries | certified ≤ `pwl_tol`        |
| `MLPRegressor`                   | ReLU: exact big-M (1 binary/neuron); tanh/logistic activations: certified PWL    | ReLU exact; others ≤ `pwl_tol` |
| `DecisionTreeRegressor`          | leaf-selection MILP (1 binary/leaf, big-M routing)   | exact                        |
| `RandomForestRegressor`          | linear average of per-tree leaf binaries             | exact                        |
| `GradientBoostingRegressor`      | learning-rate-weighted sum of per-tree leaf binaries + init constant | exact          |
| `XGBRegressor` / `Booster`       | additive trees over leaf binaries; base_score recovered empirically with a parse self-check | exact (margin) |
| `XGBClassifier` (binary)         | exact margin + certified PWL sigmoid                   | certified ≤ `pwl_tol` |
| `LGBMRegressor` / `Booster`      | additive trees over leaf binaries; empirical init score + parse self-check | exact (raw score) |
| `LGBMClassifier` (binary)        | exact margin + certified PWL sigmoid                   | certified ≤ `pwl_tol` |
| Keras dense networks             | Dense/Activation layers → shared big-M/PWL machinery (any backend: TF, JAX, torch) | ReLU exact; sigmoid/tanh ≤ `pwl_tol` |
| ONNX `ModelProto` (dense nets)   | Gemm/MatMul+Add/Relu/Sigmoid/Tanh/Identity → shared machinery, no ONNX runtime needed | ReLU exact; others ≤ `pwl_tol` |
| `Pipeline` + preprocessing       | `StandardScaler`, `ColumnTransformer`, `PolynomialFeatures` (affine terms only) folded into expressions — no variables | — |

HiGHS is a pure LP/QP/MIP solver with no general or nonlinear
constraints, so every relation is reformulated: linear models become
equalities, trees become leaf-selection binaries, ReLU networks become
big-M constraints, and smooth activations become adaptive
piecewise-linear interpolations whose maximum error is *certified*
`≤ pwl_tol` over the whole reachable interval.

Bilinear terms `y = x1*x2` (e.g. from `PolynomialFeatures`) embed via
`add_bilinear_constr`:

| factor types | embedding | accuracy |
|---|---|---|
| binary × continuous | McCormick envelope (collapses exactly) | **exact** |
| integer × continuous / integer × integer | binary expansion + exact binary×continuous terms | **exact** (range ≤ 1024) |
| continuous × continuous | piecewise McCormick with segment binaries; envelope gap ≤ `tol` | **certified ≤ tol** |

## Solving structured models

Embedded predictors produce models with heavy repeated structure (one
block per sample, tree, or neuron). `solve_auto` detects the structure
and routes to the right method:

```python
from highs_ml import solve_auto
method, result = solve_auto(h)   # method in {'decomposed',
                                 #  'dantzig-wolfe', 'branch-and-price',
                                 #  'direct'}
```

```
fully separable            -> solve_decomposed (exact)
block-angular, few classes -> solve_dw (+ solve_bp when not root-tight)
block-angular, many unique -> solve_dw bound; direct HiGHS recommended
anything else              -> direct HiGHS
```

* **`solve_decomposed`** — splits fully separable models into
  independent blocks, solves each unique block once, and numerically
  verifies the stitched solution (a 100,000-block model solves in under
  2 s where the direct solve takes nearly two minutes).
* **`solve_dw`** — Dantzig-Wolfe column generation for block-angular
  models (independent blocks plus coupling rows), returning a primal
  solution with an honest optimality-gap certificate.
* **`solve_bp`** — exact branch-and-price on top of `solve_dw`, with a
  large-neighborhood-search polish for the incumbent.
* **`solve_adaptive`** — refines nonlinear embeddings (PWL and
  bilinear) exactly where the optimizer lands, until every embedding
  meets its own certified tolerance.

Formulations, design decisions, and measured performance:
**[docs/DESIGN.md](docs/DESIGN.md)**.

## API

```python
add_predictor_constr(highs_model, predictor, input_vars, output_var=None,
                     output_type=None, pwl_tol=0.01, name=None,
                     refinable=False)
```

* `input_vars`: mapping of feature name → `highs_var`/constant (uses
  `predictor.feature_names_in_`), or a sequence in feature order.
* `output_var`: existing variable to link, or `None` to create one with
  tight bounds.
* `output_type`: `'probability_1'` (default for logistic) or `'raw'`
  (linear score, exact).
* `refinable`: build PWL embeddings as refinable handles for
  `solve_adaptive`.
* Returns an object with `.output_var`, `.print_stats()`, `.get_error()`.

## Example

```
cd highs-ml
PYTHONPATH=. python examples/student_admission.py
```

Trains a `StandardScaler + LogisticRegression` pipeline on the Janos
college-enrollment dataset (Bergman et al., 2020 — the same problem
gurobi-ml uses as its flagship example), embeds it once per applicant,
and maximizes expected enrollments under a scholarship budget:

| Metric                 | HiGHS 1.15 + highs_ml (PWL MILP, tol 1e-3) |
|------------------------|--------------------------------------------|
| Expected enrollments   | **13.7888**                                |
| Scholarship recipients | students 5909, 3054, 3772, 277, 2712, 5462 |
| Max embedding error    | 3.19e-04 (certified ≤ 1e-3)                |

Tightening `pwl_tol` trades binaries for accuracy (e.g. `1e-4` cuts the
certified error bound by roughly an order of magnitude at about double
the segment count).

## Tests

```
cd highs-ml
PYTHONPATH=. python tests/test_highs_ml.py     # or: pytest tests/
```

Covers exactness of every exact embedding, certified error of every
approximate one, optimization-direction sanity checks, the structured
solvers (`tests/test_decomp.py`, `test_dw.py`, `test_bp.py`,
`test_bilinear.py`, `test_refine.py`), and a regression suite
(`tests/test_regressions.py`).

## Related work

[gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)
pioneered the embed-a-predictor API for the Gurobi solver; `highs_ml`
brings the same modeling pattern to the open-source HiGHS solver using
classical MILP reformulations. For users who want to compare the two
stacks, `examples/benchmark_vs_gurobi.py` embeds identical predictors
on both and solves the same budget-constrained maximization (tree
models use `epsilon=1e-4` on both sides for like-for-like boundary
semantics; results are written to `benchmark_results.csv`).
Comparative results are not published in this repository — Gurobi's
EULA restricts publishing solver benchmark results — so run it under
your own license.

The datasets in `examples/data/` are from the
[Janos repository](https://github.com/INFORMSJoC/2020.1023) (Bergman et
al., 2020); see that repository for their terms.

*Gurobi is a registered trademark of Gurobi Optimization, LLC. highs-ml
is not affiliated with, sponsored by, or endorsed by Gurobi
Optimization, LLC.*

## License

MIT, same as HiGHS.
