# highs_ml — design notes and measured performance

How the embeddings and structured-model solvers work, why they are
built the way they are, and what they measure. All timings in this
document are HiGHS-only self-measurements taken on one development
machine with HiGHS 1.15.1; every number is reproducible from the
scripts in `examples/` and the suites in `tests/`.

## Reformulation approach

Modern solver-native ML integrations handle nonlinear pieces (e.g. the
logistic sigmoid) through general constraints solved by a global
non-convex NLP algorithm. HiGHS is a pure LP/QP/MIP solver: it has no
general or nonlinear constraints. `highs_ml` therefore reformulates
every relation as an *exact or certified* MILP — the classical
approach, and the one gurobi-ml itself used before Gurobi 11
introduced solver-native nonlinear constraints.

## How the PWL embedding works

For `y = f(z)` with `z` an affine expression bounded on `[z_lo, z_hi]`:

1. **Adaptive breakpoints.** Breakpoints are inserted recursively wherever
   the chord deviates from `f` by more than `pwl_tol` — the interpolation
   error is *certified* over the whole reachable interval, not just sampled.
2. **Convex-combination formulation.** `K+1` lambda weights plus `K` segment
   binaries enforce an SOS2 set through plain linear constraints (HiGHS has
   no SOS2 API):

   ```
   z = Σ λ_k z_k,   y = Σ λ_k f(z_k),   Σ λ_k = 1,   Σ b_s = 1
   λ_0 ≤ b_0,   λ_k ≤ b_{k-1} + b_k,   λ_K ≤ b_{K-1}
   ```

3. **Coefficient hygiene.** Function values in saturated tails (e.g.
   `σ(-39) ≈ 1e-17`) fall below HiGHS's `small_matrix_value` rejection
   threshold (1e-9) and are snapped to zero.

All big-M values and breakpoint ranges come from *interval arithmetic over
the actual variable bounds*, so input decision variables must have finite
bounds.

## Bilinear embeddings

A bilinear equality `y = x1*x2` is non-convex and HiGHS has no spatial
branch-and-bound for it. MILP reformulation theory gives three
practical routes, all implemented in `highs_ml.add_bilinear_constr`
and used automatically by `PolynomialFeatures`:

| factor types | embedding | accuracy |
|---|---|---|
| binary × continuous | McCormick envelope (collapses exactly) | **exact** |
| integer × continuous / integer × integer | binary expansion + exact binary×continuous terms | **exact** (range ≤ 1024) |
| continuous × continuous | piecewise McCormick with segment binaries; envelope gap ≤ `tol` | **certified ≤ tol** |

For continuous × continuous, `add_adaptive_bilinear` starts with a
*coarse* partition and refines the envelope exactly where the optimizer
lands (segment binaries fixed off, child segments appended via
`chgCoeff` — no model rebuild), looping until the certificate holds.
Measured on the frontier family in `examples/benchmark_bilinear.py`
(maximization with interior optima):

| n | highs uniform | **highs adaptive** |
|---|---|---|
| 2 | obj 3.500 (overshoot), 102 vars, 0.3 s | obj **3.4420** (certified exact), err 4e-15, 46 vars, 1 refine, 0.1 s |
| 6 | obj exact, 306 vars, 5.7 s | obj **10.6923** (certified exact), 114 vars, 0 refines, 0.3 s |
| 12 | (static partitions scale poorly) | obj **22.6621** (certified exact), 228 vars, 0 refines, 0.25 s |

The adaptive path reaches the certified-exact objective at every size,
with far fewer segments than a static tolerance-uniform partition needs
(12 active vs 29 at tol 0.02 in the segment-economy test,
`tests/test_bilinear.py`).

Degree-2 polynomial expansions with variable products (including squares)
embed directly; terms beyond degree 2 or with more than two variable
factors raise a clear, actionable error.

Tree ensembles compose exactly: each tree contributes one set of leaf
binaries and the ensemble output is a linear combination of them, so a
100-tree forest is exact but adds ~100 × (leaves per tree) binaries.

## Split-boundary safety

Tree splits are ambiguous at `x == threshold`: an `epsilon=0` encoding
lets the optimizer claim leaf values at the boundary that sklearn's own
`predict` would never produce. highs_ml therefore defaults to
`epsilon=1e-4`, which routes boundary points the sklearn way (left at
`x == t`). (For like-for-like comparisons, note that gurobi_ml's
default is `epsilon=0` — see its documentation.)

## The decomposition presolver (`solve_decomposed`)

Embedding ML models produces exactly the structure HiGHS struggles with:
many independent big-M blocks (per sample, neuron, or tree). Measured on
the [Julia Discourse complementarity MWE](https://discourse.julialang.org/t/114905)
(n independent big-M pairs), HiGHS 1.15.1 scales *superlinearly* — 112 s at
n = 100k — because presolve doesn't collapse independent blocks and the
global simplex pays one pivot per block (the optimum is found at the root;
the time is pure LP grind).

`solve_decomposed(h)` fixes this at the modeling layer:

1. detects connected components of the row/column incidence graph
   (C-speed, scipy `connected_components`);
2. groups structurally identical blocks by an exact vectorized signature,
   so 100k equal blocks are solved **once** and the solution replicated;
3. solves each unique block as its own tiny HiGHS model (blocks are
   independent, so the global optimum is the sum of block optima);
4. stitches and **numerically verifies** the full solution (one matvec).

Measured results (same machine, HiGHS 1.15.1):

| n (independent blocks) | HiGHS direct | `solve_decomposed` | speedup |
|---|---|---|---|
| 1,000 | 0.09 s | 0.06 s | 1.5× |
| 10,000 | 0.96 s | 0.11 s | 9× |
| 50,000 | 18.8 s | 0.98 s | 19× |
| 100,000 | 112.2 s | 1.77 s | **63×** |

The presolver removes the superlinear blow-up entirely without touching
HiGHS internals. Coupled models (blocks sharing rows, e.g. a budget
constraint) automatically fall back to a direct solve — the presolver
only ever helps. See `tests/test_decomp.py` for the correctness suite
(random block MILPs vs direct solves, min/max, infeasibility
propagation, coupled fallback).

## Dantzig-Wolfe for coupled models (`solve_dw`)

Real models — including every ML-embedded one with a shared budget — are
**block-angular**: many independent blocks plus a few coupling rows.
`solve_dw(h)` implements the classical workhorse for exactly this:

1. **Bordered block-diagonal detection** (multi-ordering deferral with
   chain-fragment repair and island merging — over-merging is
   correctness-safe by construction);
2. **Column generation** on the LP master with per-block-class MIP
   pricing, identical blocks aggregated into one convexity row (integer
   counts in the MIP phase), Phase-I artificial columns for master
   feasibility, and a valid Lagrangian dual bound at every iteration;
3. **Restricted-master heuristic** for a primal MIP solution, reported
   with an honest optimality gap against the master bound;
4. numerical verification of the recovered solution.

Measured (same machine, HiGHS 1.15.1):

| Model | Direct HiGHS | `solve_dw` | Result |
|---|---|---|---|
| coupled complementarity blocks, n = 10,000 | 40.5 s | 1.74 s | **23× faster**, gap 0.00% |
| Student Enrollment, 250 PWL-logistic blocks + budget row | 9.67 s (obj 159.3884) | 9.26 s (obj 159.3803) | parity in time, **certified gap 0.017%** |

The student case shows the other reason DW matters: it produces a *bound
certificate* for the structure, and its advantage grows with the number
of blocks (pricing subproblems stay tiny; the direct root-LP grows
superlinearly). See `examples/student_admission_dw.py` and
`tests/test_dw.py`.

## Exact branch-and-price (`solve_bp`)

`solve_bp(h)` upgrades the restricted-master heuristic to an exact
method: branch-and-bound over the master, branching on **original block
variables** (Ryan-Foster style — enforced in the pricing subproblems,
which both tightens the node bound and generates exactly the columns the
branch needs) with lambda-count branching for aggregated classes. Every
node re-runs column generation warm-started from the shared pool, so
node bounds are always valid; columns found inside branches are globally
valid and all feed the final restricted master.

Measured behavior (honest):

* **Aggregated/few-class models**: proves optimality at the root
  (gap 0.00%, zero nodes) — this is DW's textbook strong case.
* **Student model, 25 unique blocks**: the node bound tightens to the
  exact optimum (13.7888, matching direct HiGHS) within 32 nodes —
  but the restricted-master incumbent can lag slightly (0.015%), because
  LP-guided pricing doesn't always generate the integer-optimal
  operating points for all-unique classes with continuous decisions.
  Direct HiGHS is the better exact tool for that family; solve_bp still
  returns a valid bound + incumbent with an honest residual gap.

## The LNS primal polish (inside `solve_bp`)

The restricted-master incumbent is pool-limited; the **large-neighborhood
search** (`_lns.py`) sidesteps the pool: fix most blocks to the
incumbent, leave a small neighborhood free, and solve that tiny MIP
*exactly* with HiGHS over the original rows — always globally feasible,
any improvement becomes the new incumbent. Measured on the student model
(32 B&P nodes + 12 LNS rounds of 6 blocks):

| n | gap before LNS | gap after LNS |
|---|---|---|
| 25 | 0.015% | **0.002%** |
| 50 | 0.841% | **0.369%** |
| 100 | 0.005% | **0.001%** (incumbent *beats* the direct HiGHS solution) |

## One refinement framework for every nonlinear embedding

`_refine.py` unifies the refinement-at-incumbent loop across both
nonlinear embedding families — `PiecewiseBilinear` (y = x1·x2) and
`RefinablePWL` (y = f(z): sigmoid, tanh, ...). Both start coarse and
split the segment where the optimizer lands until the certificate holds;
`solve_adaptive(h, embeddings)` drives any mix of them round-robin.

Two hard-won design points:

* the PWL chord is written *per segment* as the linear form
  `y_s = slope·z_s + intercept·δ_s` with `z_s = z·δ_s` (exact
  binary×continuous McCormick) — refinement is append-only, no row
  deletions, no SOS2 chains;
* the chord is shifted to a **one-sided envelope** (default upper). A
  bare chord is two-sided, so the model optimum can dodge regions where
  the chord underestimates f; the envelope converges to the true optimum
  from a safe side. Boundary-adjacent splits bisect geometrically.

Measured: the student enrollment problem solved with 25 refinable
logistic embeddings gives **obj 13.7934** (static path 13.7888) with
certified error 7.7e-04 in 5.0 s — the tightest certificate of any of
our paths, with 3 active segments vs 15 static for the same tolerance.
Use `add_predictor_constr(..., refinable=True)` and
`solve_adaptive(h, constr.refinable_embeddings)` (handles are also
collected on pipeline predictor-constraints).

## Repository layout

```
highs_ml/
  __init__.py       public API
  core.py           add_predictor_constr dispatch
  _affine.py        structured affine forms (bounds, evaluation) over highs_var
  _pwl.py           adaptive breakpoints + convex-combination PWL embedding
  _predictors.py    Linear / Ridge / PLS / Logistic / MLP / Pipeline constrs
  _trees.py         exact leaf-selection MILP for trees, forests, boosting
  _xgboost.py       exact XGBoost embedding (regressor + binary classifier)
  _lightgbm.py      exact LightGBM embedding (regressor + binary classifier)
  _nn_common.py     shared big-M/PWL feedforward network machinery
  _keras.py         Keras dense-network embedding (backend-agnostic)
  _onnx.py          ONNX ModelProto embedding (no runtime needed)
  _preprocessing.py StandardScaler / PolynomialFeatures / ColumnTransformer
  _bilinear.py      exact & certified/adaptive MILP embeddings of y = x1*x2
  _refine.py        unified refinement-at-incumbent framework (PWL + bilinear)
  _lns.py           large-neighborhood-search primal heuristic
  decomp.py         decomposition presolver for independent-block models
  dw.py             Dantzig-Wolfe + exact branch-and-price (solve_dw/solve_bp)
  auto.py           solve_auto: structure-aware solver routing
examples/
  student_admission.py      the README example (Janos scholarship problem)
  student_admission_dw.py   the same model via solve_dw / solve_bp
  benchmark_bilinear.py     bilinear frontier study (uniform vs adaptive)
  benchmark_vs_gurobi.py    cross-stack comparison script (results unpublished)
  bp_gap_study.py           branch-and-price gap study
  discourse_mwe_scaling.py  decomposition scaling MWE
  data/             Janos college enrollment data (Bergman et al., 2020)
tests/
  test_highs_ml.py  exactness, certified error, optimization sanity
  test_regressions.py  regression suite
  test_decomp.py    decomposition presolver correctness suite
  test_dw.py        Dantzig-Wolfe correctness and scaling suite
  test_bp.py        branch-and-price and solve_auto routing tests
  test_bilinear.py  bilinear embedding tests (exact + certified paths)
  test_refine.py    unified refinement framework tests
```
