"""
Adds a single per-allocation calibration-bias feature learned from the
out-of-fold predictions of the stacked model. The intuition is simple:
the model systematically under- or over-shoots certain allocations by
a few basis points; feeding that per-allocation bias back in as a
feature lets the next-round MLP learn non-linear interactions around
the correction instead of leaving it to a linear post-processing step.

Produced column:
  ALLOC_CALIB_BIAS  :  mean(y) - mean(oof_pred) at allocation level

Train and test rows of the same allocation receive the same value.
All out-of-fold predictions are produced by a plain Ridge stack over
the five base-model OOFs; using Ridge (not the row itself) is what keeps
this leakage-free.
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from config import CV_N_SPLITS, features_parquet, oof_path, test_preds_path


BASE_MODELS : list[str] = [
    "mlp-bag8",
    "cat-deep",
    "xgb-cls-v2",
    "seq-cnn",
    "lgbm-classification-v3",
]


def main() -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet("v2", "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet("v2", "test"))
    y     : np.ndarray   = (train["TARGET"].values > 0).astype(int)

    oof_matrix : np.ndarray = np.stack([np.load(oof_path(m))          for m in BASE_MODELS], axis=1)
    test_matrix: np.ndarray = np.stack([np.load(test_preds_path(m))   for m in BASE_MODELS], axis=1)

    # Regenerate Ridge stack OOF predictions. Using fold-wise refits rather
    # than a single full-train fit keeps the bias estimate honest.
    oof_stack : np.ndarray = np.zeros(len(y), dtype=np.float32)
    gkf = GroupKFold(CV_N_SPLITS)
    for tr_idx, va_idx in gkf.split(np.arange(len(y)), groups=train["TS"].values):
        rd = Ridge(alpha=1.0).fit(oof_matrix[tr_idx], y[tr_idx].astype(float))
        oof_stack[va_idx] = rd.predict(oof_matrix[va_idx])

    df = pd.DataFrame({"alloc": train["ALLOC_ID"].values, "y": y, "p": oof_stack})
    agg = df.groupby("alloc").agg(mean_pred=("p", "mean"), mean_true=("y", "mean"))
    agg["bias"] = agg["mean_true"] - agg["mean_pred"]
    bias_lut : dict = agg["bias"].to_dict()

    train["ALLOC_CALIB_BIAS"] = pd.Series(train["ALLOC_ID"].values).map(bias_lut).fillna(0.0).astype(np.float32).values
    test ["ALLOC_CALIB_BIAS"] = pd.Series(test ["ALLOC_ID"].values).map(bias_lut).fillna(0.0).astype(np.float32).values

    train.to_parquet(features_parquet("v7", "train"), index=False)
    test .to_parquet(features_parquet("v7", "test"),  index=False)
    print(
        f"[save] total {time.time() - t0:.1f}s  "
        f"bias std {train['ALLOC_CALIB_BIAS'].std():.4e}  "
        f"train {train.shape}  test {test.shape}"
    )


if __name__ == "__main__":
    main()
