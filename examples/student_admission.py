"""Student Enrollment with HiGHS + highs_ml.

Reproduces the Gurobi Machine Learning "Student Enrollment" example
(Bergman et al., 2020, Janos dataset) using only open-source software:

    * scikit-learn  -- trains StandardScaler + LogisticRegression pipeline
    * highs_ml      -- embeds the trained pipeline into a MILP
    * HiGHS (MIT)   -- solves the MILP; no Gurobi license required

Problem: choose scholarships x_i in [0, 2.5] K$ for n admitted students to
maximize the expected number of enrollments sum_i P(enroll_i), subject to a
total budget of 0.2*n K$.
"""

import pathlib

import numpy as np
import pandas as pd
import highspy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from highs_ml import add_predictor_constr

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------
# 1. Historical data and regression (identical to the Gurobi example)
# ----------------------------------------------------------------------
historical_data = pd.read_csv(DATA_DIR / "college_student_enroll-s1-1.csv", index_col=0)

features = ["merit", "SAT", "GPA"]
target = "enroll"

pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
pipe.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])

# ----------------------------------------------------------------------
# 2. Applicant data
# ----------------------------------------------------------------------
studentsdata = pd.read_csv(DATA_DIR / "college_applications6000.csv", index_col=0)

nstudents = 25
studentsdata = studentsdata.sample(nstudents, random_state=1)

# ----------------------------------------------------------------------
# 3. Optimization model with HiGHS
# ----------------------------------------------------------------------
h = highspy.Highs()
h.setOptionValue("output_flag", False)

# Scholarship decision variables, one per student, 0 <= x_i <= 2.5 (K$)
x = {
    sid: h.addVariable(lb=0.0, ub=2.5, name=f"merit[{sid}]")
    for sid in studentsdata.index
}
# Enrollment probability variables
y = {
    sid: h.addVariable(lb=0.0, ub=1.0, name=f"enroll_prob[{sid}]")
    for sid in studentsdata.index
}

# Budget: sum of scholarships <= 0.2 * n
h.addConstr(sum(x[sid] for sid in studentsdata.index) <= 0.2 * nstudents, name="budget")

# Embed the trained pipeline once per student. SAT and GPA are fixed
# features; "merit" is the decision variable.
pred_constrs = []
for sid, row in studentsdata.iterrows():
    pc = add_predictor_constr(
        h,
        pipe,
        {"merit": x[sid], "SAT": float(row["SAT"]), "GPA": float(row["GPA"])},
        y[sid],
        output_type="probability_1",
        pwl_tol=1e-3,
        name=f"pipe_{sid}",
    )
    pred_constrs.append(pc)

# ----------------------------------------------------------------------
# 4. Maximize expected enrollments
# ----------------------------------------------------------------------
h.maximize(sum(y[sid] for sid in studentsdata.index))

sol = h.getSolution()
objective = h.getObjectiveValue()

print("=" * 64)
print("Student Enrollment -- solved with HiGHS (open source, no Gurobi)")
print("=" * 64)
print(f"HiGHS version        : {h.version()}")
print(f"Students             : {nstudents}")
print(f"Model status         : {h.getModelStatus()}")
print(f"Expected enrollments : {objective:.4f}")
print(f"Scholarship budget   : {sum(sol.col_value[x[s].index] for s in studentsdata.index):.4f}"
      f" / {0.2 * nstudents:.1f} K$")

max_error = max(pc.get_error() for pc in pred_constrs)
print(f"Max approximation error in the embedded regression: {max_error:.3e}")

print("\nScholarships offered (K$):")
awards = {
    sid: sol.col_value[x[sid].index]
    for sid in studentsdata.index
    if sol.col_value[x[sid].index] > 1e-6
}
for sid, amount in sorted(awards.items(), key=lambda kv: -kv[1]):
    row = studentsdata.loc[sid]
    p = sol.col_value[y[sid].index]
    print(f"  student {sid:>5}: {amount:6.3f} K$  "
          f"(SAT {row['SAT']:.0f}, GPA {row['GPA']:.2f}) -> P(enroll) = {p:.3f}")
if not awards:
    print("  none")

print("\nCross-check against sklearn's own predict_proba:")
errors = []
for sid, row in studentsdata.iterrows():
    exact = pipe.predict_proba(
        pd.DataFrame(
            [[sol.col_value[x[sid].index], row["SAT"], row["GPA"]]],
            columns=features,
        )
    )[0, 1]
    errors.append(abs(exact - sol.col_value[y[sid].index]))
print(f"  max |HiGHS y_i - sklearn predict_proba| = {max(errors):.3e}")
