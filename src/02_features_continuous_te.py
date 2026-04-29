"""
Adds three out-of-fold target-encoding features built on the continuous
target (next-day return), rather than on its sign. The continuous target
is richer and typically more informative for the gradient boosting model
than a plain binary win-rate encoding.

Produced columns:
  TE_ALLOC_CONT          per-allocation OOF mean of continuous target
  TE_AG_CONT             per (ALLOC, GROUP) OOF mean of continuous target
  ALLOC_TARGET_STD_OOF   per-allocation OOF std of continuous target

All encodings use empirical-Bayes shrinkage toward the fold's global prior
so low-support keys collapse to the mean.
"""
import time
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import CV_N_SPLITS, features_parquet


Splits = Iterable[Tuple[np.ndarray, np.ndarray]]


def _oof_mean_encoding(
    train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
    key: str, splits: list, smoothing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OOF mean encoding with additive Beta-style smoothing. Train rows receive
    the LUT built from the OTHER folds only. Test rows use a full-train LUT
    since there is no fold structure on the test side.
    """
    prior_global : float = float(y.mean())
    train_encoded : np.ndarray = np.full(len(train), prior_global, dtype=np.float32)

    for tr_idx, va_idx in splits:
        prior_fold : float = float(y[tr_idx].mean())
        fold = pd.DataFrame({"k": train[key].values[tr_idx], "y": y[tr_idx]})
        agg  = fold.groupby("k")["y"].agg(["sum", "count"])
        agg["enc"] = (agg["sum"] + prior_fold * smoothing) / (agg["count"] + smoothing)

        mapped = pd.Series(train[key].values[va_idx]).map(agg["enc"]).fillna(prior_fold)
        train_encoded[va_idx] = mapped.astype(np.float32).values

    full = pd.DataFrame({"k": train[key].values, "y": y})
    agg  = full.groupby("k")["y"].agg(["sum", "count"])
    agg["enc"] = (agg["sum"] + prior_global * smoothing) / (agg["count"] + smoothing)
    test_encoded : np.ndarray = (
        pd.Series(test[key].values).map(agg["enc"]).fillna(prior_global)
          .astype(np.float32).values
    )
    return train_encoded, test_encoded


def _oof_std_encoding(
    train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
    key: str, splits: list, smoothing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same idea as `_oof_mean_encoding`, but on within-group std."""
    prior_global : float = float(y.std())
    train_encoded : np.ndarray = np.full(len(train), prior_global, dtype=np.float32)

    for tr_idx, va_idx in splits:
        prior_fold : float = float(y[tr_idx].std())
        fold = pd.DataFrame({"k": train[key].values[tr_idx], "y": y[tr_idx]})
        agg  = fold.groupby("k")["y"].agg(["std", "count"]).fillna(prior_fold)
        agg["enc"] = (agg["std"] * agg["count"] + prior_fold * smoothing) / (agg["count"] + smoothing)
        mapped = pd.Series(train[key].values[va_idx]).map(agg["enc"]).fillna(prior_fold)
        train_encoded[va_idx] = mapped.astype(np.float32).values

    full = pd.DataFrame({"k": train[key].values, "y": y})
    agg  = full.groupby("k")["y"].agg(["std", "count"]).fillna(prior_global)
    agg["enc"] = (agg["std"] * agg["count"] + prior_global * smoothing) / (agg["count"] + smoothing)
    test_encoded : np.ndarray = (
        pd.Series(test[key].values).map(agg["enc"]).fillna(prior_global)
          .astype(np.float32).values
    )
    return train_encoded, test_encoded


def main() -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet("v2", "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet("v2", "test"))
    y     : np.ndarray  = train["TARGET"].values.astype(np.float32)

    # Group by TS so entire days live on one side of each split — this is
    # the single most important thing in time-series CV for this dataset.
    splits : list = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(train)), groups=train["TS"].values))

    smoothing : float = 50.0

    for key, out_col in (("ALLOC_ID", "TE_ALLOC_CONT"), ("AG_ID", "TE_AG_CONT")):
        tr_enc, te_enc = _oof_mean_encoding(train, test, y, key, splits, smoothing)
        train[out_col] = tr_enc
        test [out_col] = te_enc

    tr_enc, te_enc = _oof_std_encoding(train, test, y, "ALLOC_ID", splits, smoothing)
    train["ALLOC_TARGET_STD_OOF"] = tr_enc
    test ["ALLOC_TARGET_STD_OOF"] = te_enc

    train.to_parquet(features_parquet("v3", "train"), index=False)
    test .to_parquet(features_parquet("v3", "test"),  index=False)
    print(f"[save] total {time.time() - t0:.1f}s  train {train.shape}  test {test.shape}")


if __name__ == "__main__":
    main()
