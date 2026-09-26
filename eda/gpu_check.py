"""Quick GPU sanity check for XGBoost (CUDA) and LightGBM (OpenCL) on a synthetic problem."""
import time

import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(2_000_000, 40)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(size=len(X)) > 0).astype(np.int8)

import xgboost as xgb

print("xgboost", xgb.__version__)
d = xgb.DMatrix(X, label=y)
for dev in ("cuda", "cpu"):
    t = time.time()
    try:
        xgb.train({"tree_method": "hist", "device": dev, "max_depth": 8, "objective": "binary:logistic",
                   "nthread": 12}, d, num_boost_round=200)
        print(f"  xgboost {dev}: {time.time() - t:.1f}s")
    except Exception as e:
        print(f"  xgboost {dev} FAILED: {e}")

import lightgbm as lgb

print("lightgbm", lgb.__version__)
for dev in ("gpu", "cpu"):
    t = time.time()
    try:
        lgb.train({"objective": "binary", "device_type": dev, "num_leaves": 255, "verbose": -1, "num_threads": 12},
                  lgb.Dataset(X, label=y), num_boost_round=200)
        print(f"  lightgbm {dev}: {time.time() - t:.1f}s")
    except Exception as e:
        print(f"  lightgbm {dev} FAILED: {str(e)[:200]}")
