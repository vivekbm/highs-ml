"""Benchmark: highs_ml + HiGHS  vs  gurobi_ml + Gurobi on identical models.

For each trained predictor we solve the same optimization problem twice:

    maximize    prediction(x)
    subject to  sum(x_j) <= 0.5 * d          (budget)
                -4 <= x_j <= 4

once embedded with highs_ml and solved by HiGHS (MIT license, no size
limits), and once embedded with gurobi_ml and solved by Gurobi.

Note on the Gurobi side: this machine has no paid Gurobi license, so the
Gurobi runs use the free *restricted* pip license (size-capped). If a model
exceeds the cap it is reported as 'license-limited' — which is itself part
of the story this project tells.

Results are printed as a table and written to benchmark_results.csv.
"""

import pathlib
import time

import numpy as np
import pandas as pd

import highs_ml

RNG = np.random.default_rng(0)
D = 4  # input dimension


# ----------------------------------------------------------------------
# problem setup
# ----------------------------------------------------------------------
def make_data(kind="regression", n=800):
    X = RNG.uniform(-4, 4, size=(n, D))
    signal = (
        np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        - 0.3 * X[:, 2] * X[:, 3]
        + 0.8 * np.maximum(0, X[:, 3])
    )
    if kind == "regression":
        return X, signal
    p = 1.0 / (1.0 + np.exp(-(signal - signal.mean())))
    return X, (p > RNG.random(n)).astype(int)


def train_predictors():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    Xr, yr = make_data("regression")
    Xc, yc = make_data("classification")

    preds = {
        "LinearRegression": (LinearRegression().fit(Xr, yr), "raw"),
        "LogisticRegression": (LogisticRegression(max_iter=1000).fit(Xc, yc),
                               "probability_1"),
        "Pipeline(Scaler+MLP)": (
            make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(6,), max_iter=3000,
                                       random_state=1)).fit(Xr, yr), "raw"),
        "DecisionTree": (DecisionTreeRegressor(max_depth=4, random_state=1).fit(Xr, yr),
                         "raw"),
        "RandomForest(15)": (RandomForestRegressor(n_estimators=15, max_depth=4,
                                                   random_state=1).fit(Xr, yr), "raw"),
        "GradBoosting(15)": (GradientBoostingRegressor(n_estimators=15, max_depth=3,
                                                       random_state=1).fit(Xr, yr), "raw"),
    }
    try:
        import xgboost as xgb
        preds["XGBoost(15)"] = (xgb.XGBRegressor(n_estimators=15, max_depth=4,
                                                 random_state=1).fit(Xr, yr), "raw")
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        preds["LightGBM(15)"] = (lgb.LGBMRegressor(n_estimators=15, max_depth=4,
                                                   verbose=-1,
                                                   random_state=1).fit(Xr, yr), "raw")
    except ImportError:
        pass
    return preds


# ----------------------------------------------------------------------
# highs_ml / HiGHS side
# ----------------------------------------------------------------------
def run_highs(predictor, output_type):
    import highspy

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    xs = [h.addVariable(lb=-4.0, ub=4.0, name=f"x{j}") for j in range(D)]
    y = h.addVariable(lb=-1e6, ub=1e6, name="pred")
    h.addConstr(sum(xs) <= 0.5 * D)
    pc = highs_ml.add_predictor_constr(h, predictor, xs, y,
                                       output_type=output_type, pwl_tol=1e-3)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    h.maximize(y)
    solve_s = time.perf_counter() - t0

    return {
        "highs_obj": h.getObjectiveValue(),
        "highs_err": pc.get_error(),
        "highs_build_s": build_s,
        "highs_solve_s": solve_s,
        "highs_vars": h.numVariables,
        "highs_bin": sum(
            1 for i in range(h.numVariables)
            if h.getColIntegrality(i)[1] == highspy.HighsVarType.kInteger
        ),
        "highs_cons": h.numConstrs,
    }


# ----------------------------------------------------------------------
# gurobi_ml / Gurobi side (optional; restricted license)
# ----------------------------------------------------------------------
def run_gurobi(predictor, output_type):
    import gurobipy as gp
    from gurobi_ml import add_predictor_constr

    t0 = time.perf_counter()
    m = gp.Model()
    m.params.OutputFlag = 0
    x = m.addMVar((1, D), lb=-4.0, ub=4.0)
    y = m.addMVar((1, 1), lb=-gp.GRB.INFINITY)
    m.addConstr(x.sum() <= 0.5 * D)
    kwargs = {"output_type": output_type} if output_type == "probability_1" else {}
    # Like-for-like with highs_ml: epsilon=1e-4 makes tree split boundaries
    # route the sklearn way (left at x == t). gurobi_ml's default epsilon=0
    # leaves boundaries ambiguous, letting the optimizer claim leaf values
    # sklearn would never produce at the optimum.
    if any(k in type(predictor).__name__ for k in
           ("Tree", "Forest", "Boosting", "XGB", "LGBM")):
        kwargs["epsilon"] = 1e-4
    pc = add_predictor_constr(m, predictor, x, y, **kwargs)
    build_s = time.perf_counter() - t0

    m.setObjective(y.sum(), gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    solve_s = time.perf_counter() - t0

    return {
        "gurobi_obj": float(y.X.sum()),
        "gurobi_err": float(np.max(np.abs(pc.get_error()))),
        "gurobi_build_s": build_s,
        "gurobi_solve_s": solve_s,
        "gurobi_vars": m.NumVars,
        "gurobi_bin": m.NumBinVars,
        "gurobi_cons": m.NumConstrs,
    }


# ----------------------------------------------------------------------
def main():
    preds = train_predictors()
    try:
        import gurobipy  # noqa: F401
        import gurobi_ml  # noqa: F401
        have_gurobi = True
    except ImportError:
        have_gurobi = False

    rows = []
    for name, (predictor, otype) in preds.items():
        row = {"predictor": name}
        row.update(run_highs(predictor, otype))
        if have_gurobi:
            try:
                row.update(run_gurobi(predictor, otype))
            except Exception as e:  # restricted-license cap, unsupported, ...
                row["gurobi_obj"] = f"n/a ({type(e).__name__})"
                print(f"  [gurobi side failed for {name}: {e}]")
        rows.append(row)
        print(f"done: {name}")

    df = pd.DataFrame(rows).set_index("predictor")
    out = pathlib.Path(__file__).resolve().parent / "benchmark_results.csv"
    df.to_csv(out)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n" + "=" * 100)
    print("BENCHMARK  highs_ml+HiGHS (MIT, unlimited)  vs  gurobi_ml+Gurobi (restricted license)")
    print("=" * 100)
    cols = [c for c in df.columns]
    print(df[cols].to_string())
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
