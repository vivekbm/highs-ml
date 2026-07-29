"""B&P gap-closure study at student counts where nodes are affordable."""

import pathlib
import sys
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from student_admission_dw import build_model, DATA_DIR
from highs_ml import solve_bp

historical = pd.read_csv(DATA_DIR / "college_student_enroll-s1-1.csv", index_col=0)
features = ["merit", "SAT", "GPA"]
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
pipe.fit(X=historical.loc[:, features], y=historical["enroll"])
students_all = pd.read_csv(DATA_DIR / "college_applications6000.csv", index_col=0)

for n in (25, 50, 100):
    students = students_all.sample(n, random_state=1)
    h, x, y = build_model(pipe, students, features)
    t0 = time.perf_counter()
    h.run()
    t_direct = time.perf_counter() - t0
    obj_direct = h.getObjectiveValue()

    h2, _, _ = build_model(pipe, students, features)
    t0 = time.perf_counter()
    res = solve_bp(h2, max_iterations=200, node_budget=32, tol=1e-6)
    t_bp = time.perf_counter() - t0
    print(f"n={n:>3}: direct {obj_direct:.4f} ({t_direct:.1f}s) | "
          f"B&P obj {res.objective:.4f} bound {res.bound:.4f} "
          f"gap {res.gap:.3%} ({t_bp:.1f}s)")
    print(f"        {res.note}")
