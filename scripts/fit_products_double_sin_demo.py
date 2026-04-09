"""Demo script: fit double-sinusoid trend per product using synthetic sample.

Run with:
$env:PYTHONPATH='D:\Associate\Repo-Module5-FidZulu-PythonML-Practice'; python scripts/fit_products_double_sin_demo.py
"""
import sys, os
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
from src.fidzulu.business.train_test_splitter import TrainTestSplitter
from src.fidzulu.business.fitting import fit_double_sin_per_product

# build a synthetic dataset with two products and clear seasonal structure
sample = pd.DataFrame({
    "prod_id": [1]*8 + [2]*8,
    "start_date": pd.to_datetime([
        "2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01","2020-06-01","2020-07-01","2020-08-01",
        "2020-01-15","2020-02-15","2020-03-15","2020-04-15","2020-05-15","2020-06-15","2020-07-15","2020-08-15",
    ]),
    "base_price": [10,11,12,11,13,12,12,13, 20,21,22,21,23,22,22,21]
})

splitter = TrainTestSplitter(sample, test_ratio=0.0)
train_splits, test_splits = splitter.split()

# ensure 't' present (split already creates t in our implementation)
results = fit_double_sin_per_product(train_splits)

for pid, r in results.items():
    if r["success"]:
        p = r["params"]
        print(f"Product {pid}: fitted params: intercept={p[0]:.3f}, slope={p[1]:.6f}, A1={p[2]:.3f}, freq1={p[3]:.6f}, A2={p[5]:.3f}, freq2={p[6]:.6f}")
    else:
        print(f"Product {pid}: fit failed: {r['message']}")
