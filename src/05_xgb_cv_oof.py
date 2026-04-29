"""
XGBoost sign-classifier with GroupKFold-by-TS cross-validation.
Adds per-fold target encoding for ALLOC_ID / AG_ID / GROUP on the fly.
Saves OOF and averaged test probabilities for the stacker.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from config import (
    CV_N_SPLITS, RANDOM_SEED,
    features_parquet, oof_path, score_path, test_preds_path,
)


META_COLUMNS : list[str] = ["TS", "TS_INT", "ROW_ID", "TARGET"]


def _fold_target_encoding(
    keys_train_fold: np.ndarray, keys_val: np.ndarray, keys_test: np.ndarray,
    y_train_fold: np.ndarray, smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior : float = float(y_train_fold.mean())
    lut = pd.DataFrame({"k": keys_train_fold, "y": y_train_fold}).groupby("k")["y"].agg(["sum", "count"])
    lut["enc"] = (lut["sum"] + prior * smoothing) / (lut["count"] + smoothing)
    enc = lut["enc"]
    encoded_train = pd.Series(keys_train_fold).map(enc).fillna(prior).astype(np.float32).values
    encoded_val   = pd.Series(keys_val).map(enc).fillna(prior).astype(np.float32).values
    encoded_test  = pd.Series(keys_test).map(enc).fillna(prior).astype(np.float32).values
    return encoded_train, encoded_val, encoded_test


def run(version: str, num_boost_round: int, tag: str) -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet(version, "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet(version, "test"))
    feature_cols : list[str] = [c for c in train.columns if c not in META_COLUMNS]
    y_sign : np.ndarray = (train["TARGET"].values > 0).astype(np.int32)

    splits = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(train)), groups=train["TS"].values))

    params : dict = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "tree_method":      "hist",
        "max_depth":        6,
        "learning_rate":    0.03,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_lambda":       1.0,
        "reg_alpha":        0.05,
        "nthread":          -1,
        "seed":             RANDOM_SEED,
    }

    oof_prob  : np.ndarray = np.zeros(len(train), dtype=np.float32)
    test_prob : np.ndarray = np.zeros(len(test),  dtype=np.float32)
    fold_scores : list[float] = []

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        base_train = train[feature_cols].iloc[tr_idx].values.astype(np.float32)
        base_val   = train[feature_cols].iloc[va_idx].values.astype(np.float32)
        base_test  = test [feature_cols].values.astype(np.float32)

        extras_train : list[np.ndarray] = []
        extras_val   : list[np.ndarray] = []
        extras_test  : list[np.ndarray] = []
        for cat_key in ("ALLOC_ID", "AG_ID", "GROUP"):
            enc_tr, enc_va, enc_te = _fold_target_encoding(
                train[cat_key].values[tr_idx],
                train[cat_key].values[va_idx],
                test [cat_key].values,
                y_sign[tr_idx],
                smoothing=50.0,
            )
            extras_train.append(enc_tr[:, None])
            extras_val  .append(enc_va[:, None])
            extras_test .append(enc_te[:, None])

        X_train : np.ndarray = np.concatenate([base_train] + extras_train, axis=1)
        X_val   : np.ndarray = np.concatenate([base_val]   + extras_val,   axis=1)
        X_test  : np.ndarray = np.concatenate([base_test]  + extras_test,  axis=1)

        dtrain = xgb.DMatrix(X_train, label=y_sign[tr_idx])
        dvalid = xgb.DMatrix(X_val,   label=y_sign[va_idx])
        dtest  = xgb.DMatrix(X_test)

        booster = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "val")],
            early_stopping_rounds=80,
            verbose_eval=0,
        )

        p_val  : np.ndarray = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        p_test : np.ndarray = booster.predict(dtest,  iteration_range=(0, booster.best_iteration + 1))

        oof_prob [va_idx] = p_val
        test_prob        += p_test / CV_N_SPLITS

        acc : float = accuracy_score(y_sign[va_idx], (p_val > 0.5).astype(int))
        fold_scores.append(acc)
        print(f"  fold {fold_i + 1}: val_acc {acc:.4f}  best_iter {booster.best_iteration}")

    mean : float = float(np.mean(fold_scores))
    std  : float = float(np.std (fold_scores))
    print(f"CV mean {mean:.4f}  std {std:.4f}  elapsed {time.time() - t0:.1f}s")

    np.save(oof_path(tag),        oof_prob)
    np.save(test_preds_path(tag), test_prob)
    with open(score_path(tag), "w") as fp:
        json.dump({"mean": mean, "std": std, "version": version, "model": "xgb"}, fp, indent=2)
    print(f"saved tag {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",         default="v2", choices=["v2", "v3"])
    parser.add_argument("--num_boost_round", type=int, default=2000)
    parser.add_argument("--tag",             default="xgb-cls-v2")
    args = parser.parse_args()
    run(args.version, args.num_boost_round, args.tag)


if __name__ == "__main__":
    main()
