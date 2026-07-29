"""Student Enrollment at scale: direct HiGHS vs Dantzig-Wolfe.

The student model is block-angular: one independent PWL-logistic block
per student, coupled by a single scholarship-budget row. This is exactly
the structure solve_decomposed cannot touch (coupling row) and where
plain HiGHS pays the superlinear price as the number of blocks grows.

Here we scale to 250 applicants, embed the trained logistic pipeline once
per student with highs_ml, and compare:
  * direct HiGHS solve
  * highs_ml.dw.solve_dw  (Dantzig-Wolfe: CG + aggregated restricted master)
"""

import pathlib
import sys
import time

import pandas as pd
import highspy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from highs_ml import add_predictor_constr, solve_bp, solve_dw

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
NSTUDENTS = 250


def build_model(pipe, studentsdata, features):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    x, y = {}, {}
    for sid, row in studentsdata.iterrows():
        x[sid] = h.addVariable(lb=0.0, ub=2.5, name=f"merit[{sid}]")
        y[sid] = h.addVariable(lb=0.0, ub=1.0, obj=1.0,
                               name=f"enroll_prob[{sid}]")
    n = len(studentsdata)
    h.addConstr(sum(x.values()) <= 0.2 * n, name="budget")
    for sid, row in studentsdata.iterrows():
        add_predictor_constr(
            h, pipe,
            {"merit": x[sid], "SAT": float(row["SAT"]),
             "GPA": float(row["GPA"])},
            y[sid], output_type="probability_1", pwl_tol=1e-3,
            name=f"p{sid}",
        )
    h.setMaximize()
    return h, x, y


def main():
    historical = pd.read_csv(DATA_DIR / "college_student_enroll-s1-1.csv",
                             index_col=0)
    features = ["merit", "SAT", "GPA"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
    pipe.fit(X=historical.loc[:, features], y=historical["enroll"])

    students = pd.read_csv(DATA_DIR / "college_applications6000.csv",
                           index_col=0).sample(NSTUDENTS, random_state=1)

    # -- direct solve ----------------------------------------------------
    h, x, y = build_model(pipe, students, features)
    t0 = time.perf_counter()
    h.run()
    t_direct = time.perf_counter() - t0
    obj_direct = h.getObjectiveValue()
    print(f"direct HiGHS : obj {obj_direct:.4f}  time {t_direct:.2f}s "
          f"({h.numVariables} vars, {h.numConstrs} cons)")

    # -- Dantzig-Wolfe ---------------------------------------------------
    h2, _, _ = build_model(pipe, students, features)
    t0 = time.perf_counter()
    res = solve_dw(h2, max_iterations=200, tol=1e-6)
    t_dw = time.perf_counter() - t0
    print(f"Dantzig-Wolfe: obj {res.objective:.4f}  bound {res.bound:.4f}  "
          f"gap {res.gap:.3%}  time {t_dw:.2f}s")
    print(f"  structure  : {res.n_blocks} blocks ({res.n_unique_classes} "
          f"classes), {res.n_coupling_rows} coupling row(s), "
          f"{res.iterations} CG iterations")
    print(f"  note       : {res.note}")

    # -- branch-and-price (exact) ----------------------------------------
    h3, _, _ = build_model(pipe, students, features)
    t0 = time.perf_counter()
    res_bp = solve_bp(h3, max_iterations=200, node_budget=32, tol=1e-6)
    t_bp = time.perf_counter() - t0
    print(f"Branch&Price : obj {res_bp.objective:.4f}  bound "
          f"{res_bp.bound:.4f}  gap {res_bp.gap:.3%}  time {t_bp:.2f}s")
    print(f"  note       : {res_bp.note}")
    print(f"  timings    : "
          f"{ {k: round(v, 2) for k, v in res_bp.timings.items()} }")
    print(f"\n  direct obj : {obj_direct:.4f}")
    print(f"  DW vs direct  : {abs(res.objective - obj_direct):.2e}")
    print(f"  B&P vs direct : {abs(res_bp.objective - obj_direct):.2e}")


if __name__ == "__main__":
    main()
