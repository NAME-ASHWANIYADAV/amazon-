"""XGBoost pair classifier (CUDA by default; ER_XGB_DEVICE=cpu when the GPU is busy with kNN)."""
import os

import numpy as np
import xgboost as xgb

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
          "device": os.environ.get("ER_XGB_DEVICE", "cuda"), "nthread": max(1, (os.cpu_count() or 4) - 2),
          "max_depth": 9, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
          "lambda": 1.0, "max_bin": 256}


def train(X, y, n_eval, rounds=3000):
    """Last `n_eval` rows form the early-stopping slice."""
    cut = len(y) - n_eval
    dtr = xgb.QuantileDMatrix(X[:cut], y[:cut])
    dev = xgb.QuantileDMatrix(X[cut:], y[cut:], ref=dtr)
    return xgb.train(PARAMS, dtr, rounds, evals=[(dev, "eval")], early_stopping_rounds=60, verbose_eval=100)


def predict(booster, X, chunk=2_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for b in range(0, len(X), chunk):
        out[b:b + chunk] = booster.inplace_predict(np.asarray(X[b:b + chunk]),
                                                   iteration_range=(0, booster.best_iteration + 1))
    return out
