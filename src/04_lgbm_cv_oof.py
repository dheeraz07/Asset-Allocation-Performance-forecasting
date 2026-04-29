"""
LightGBM sign-classification trained with GroupKFold-by-TS cross-validation.
Produces out-of-fold probabilities and averaged test probabilities used as
one of the base models in the stacked ensemble.

Per-fold categorical target encoding is added for ALLOC_ID / AG_ID / GROUP,
computed on the training folds only and applied to the held-out fold.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from config import (
    CV_N_SPLITS, RANDOM_SEED,
    features_parquet, oof_path, score_path, test_preds_path,
)


CATEGORICAL_COLUMNS : list[str] = ["ALLOC_ID", "AG_ID", "GROUP"]
META_COLUMNS        : list[str] = ["TS", "TS_INT", "ROW_ID", "TARGET"]


def _fold_target_encoding(
    key_train: np.ndarray, key_val: np.ndarray, key_test: np.ndarray,
    y_train: np.ndarray, smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrinkage target encoding restricted to `key_train` / `y_train`."""
    prior : float = float(y_train.mean())
    lut = pd.DataFrame({"k": key_train, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    lut["enc"] = (lut["sum"] + prior * smoothing) / (lut["count"] + smoothing)

    enc_val  : np.ndarray = pd.Series(key_val ).map(lut["enc"]).fillna(prior).astype(np.float32).values
    enc_test : np.ndarray = pd.Series(key_test).map(lut["enc"]).fillna(prior).astype(np.float32).values
    return enc_val, enc_test


def _build_params(seed: int) -> dict:
    return {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "learning_rate":    0.02,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples":200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     1,
        "lambda_l2":        1.0,
        "verbose":          -1,
        "seed":             seed,
        "num_threads":      0,
    }


def run(version: str, num_boost_round: int, tag: str) -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet(version, "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet(version, "test"))
    feature_cols : list[str] = [c for c in train.columns if c not in META_COLUMNS]

    y_sign : np.ndarray = (train["TARGET"].values > 0).astype(np.int32)

    splits = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(train)), groups=train["TS"].values))
    print(f"train {train.shape}  test {test.shape}  feats {len(feature_cols)}")

    oof_prob  : np.ndarray = np.zeros(len(train), dtype=np.float32)
    test_prob : np.ndarray = np.zeros(len(test),  dtype=np.float32)
    fold_scores : list[float] = []
    total_importance : np.ndarray | None = None

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        # Rebuild the three target-encoding columns for this fold.
        oof_enc   : dict[str, np.ndarray] = {}
        test_enc  : dict[str, np.ndarray] = {}
        for cat_key, out_name in (
            ("ALLOC_ID", "TE_ALLOC_SIGN"),
            ("AG_ID",    "TE_AG_SIGN"),
            ("GROUP",    "TE_GROUP_SIGN"),
        ):
            oof_enc[out_name], test_enc[out_name] = _fold_target_encoding(
                train[cat_key].values[tr_idx],
                train[cat_key].values[va_idx],
                test [cat_key].values,
                y_sign[tr_idx],
                smoothing=50.0,
            )

        X_train : pd.DataFrame = train[feature_cols].iloc[tr_idx].reset_index(drop=True)
        X_val   : pd.DataFrame = train[feature_cols].iloc[va_idx].reset_index(drop=True).copy()
        X_test  : pd.DataFrame = test [feature_cols].reset_index(drop=True).copy()

        # Train rows do not need a TE column at train time since LightGBM is
        # trained on these folds. We still pass zeros so column layout matches.
        for name, arr in oof_enc.items():
            X_train[name] = 0.0
            X_val  [name] = arr
            X_test [name] = test_enc[name]
        for name in oof_enc:
            X_train[name] = pd.Series(np.zeros(len(X_train), dtype=np.float32))

        # Rebuild categorical index after adding TE columns.
        cat_idx : list[int] = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]

        params  : dict = _build_params(RANDOM_SEED)
        dtrain  = lgb.Dataset(X_train.values, label=y_sign[tr_idx], categorical_feature=cat_idx, free_raw_data=False)
        dvalid  = lgb.Dataset(X_val  .values, label=y_sign[va_idx], categorical_feature=cat_idx, free_raw_data=False)

        model = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        p_val  : np.ndarray = model.predict(X_val .values)
        p_test : np.ndarray = model.predict(X_test.values)
        oof_prob [va_idx] = p_val
        test_prob        += p_test / CV_N_SPLITS

        fold_acc : float = accuracy_score(y_sign[va_idx], (p_val > 0.5).astype(int))
        fold_scores.append(fold_acc)
        imp = model.feature_importance(importance_type="gain")
        total_importance = imp if total_importance is None else total_importance + imp
        print(f"  fold {fold_i + 1}: val_acc {fold_acc:.4f}  best_iter {model.best_iteration}")

    mean : float = float(np.mean(fold_scores))
    std  : float = float(np.std (fold_scores))
    print(f"CV mean {mean:.4f}  std {std:.4f}  elapsed {time.time() - t0:.1f}s")

    np.save(oof_path(tag),         oof_prob)
    np.save(test_preds_path(tag),  test_prob)
    with open(score_path(tag), "w") as fp:
        json.dump({"mean": mean, "std": std, "version": version, "model": "lgbm"}, fp, indent=2)
    print(f"saved tag {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", choices=["v2", "v3", "v7"])
    parser.add_argument("--num_boost_round", type=int, default=3000)
    parser.add_argument("--tag", default="lgbm-classification-v3")
    args = parser.parse_args()

    run(args.version, args.num_boost_round, args.tag)


if __name__ == "__main__":
    main()
