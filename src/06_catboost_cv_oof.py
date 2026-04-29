"""
CatBoost sign-classifier with GroupKFold-by-TS cross-validation. CatBoost
handles ALLOC_ID / AG_ID / GROUP natively as categorical features, so no
manual target encoding is needed here.

The defaults are tuned ('deep' configuration) to the final version used in
the stacked ensemble: depth 8, lr 0.02, l2 3.
"""
import argparse
import json
import time

import catboost as cb
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


def run(version: str, num_boost_round: int, depth: int, learning_rate: float,
        l2_leaf_reg: float, tag: str) -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet(version, "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet(version, "test"))
    feature_cols : list[str] = [c for c in train.columns if c not in META_COLUMNS]
    cat_idx : list[int] = [feature_cols.index(c) for c in CATEGORICAL_COLUMNS if c in feature_cols]

    y_sign : np.ndarray = (train["TARGET"].values > 0).astype(np.int32)

    splits = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(train)), groups=train["TS"].values))

    oof_prob  : np.ndarray = np.zeros(len(train), dtype=np.float32)
    test_prob : np.ndarray = np.zeros(len(test),  dtype=np.float32)
    fold_scores : list[float] = []

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        pool_train = cb.Pool(train[feature_cols].iloc[tr_idx], label=y_sign[tr_idx], cat_features=cat_idx)
        pool_val   = cb.Pool(train[feature_cols].iloc[va_idx], label=y_sign[va_idx], cat_features=cat_idx)
        pool_test  = cb.Pool(test [feature_cols],              cat_features=cat_idx)

        model = cb.CatBoostClassifier(
            iterations=num_boost_round,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_SEED,
            early_stopping_rounds=150,
            verbose=0,
        )
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)

        p_val  : np.ndarray = model.predict_proba(pool_val )[:, 1]
        p_test : np.ndarray = model.predict_proba(pool_test)[:, 1]
        oof_prob [va_idx] = p_val
        test_prob        += p_test / CV_N_SPLITS

        acc : float = accuracy_score(y_sign[va_idx], (p_val > 0.5).astype(int))
        fold_scores.append(acc)
        print(f"  fold {fold_i + 1}: val_acc {acc:.4f}  best_iter {model.get_best_iteration()}")

    mean : float = float(np.mean(fold_scores))
    std  : float = float(np.std (fold_scores))
    print(f"CV mean {mean:.4f}  std {std:.4f}  elapsed {time.time() - t0:.1f}s")

    np.save(oof_path(tag),        oof_prob)
    np.save(test_preds_path(tag), test_prob)
    with open(score_path(tag), "w") as fp:
        json.dump({"mean": mean, "std": std, "version": version, "model": "catboost",
                   "depth": depth, "lr": learning_rate, "l2": l2_leaf_reg}, fp, indent=2)
    print(f"saved tag {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",         default="v2",  choices=["v2", "v3"])
    parser.add_argument("--num_boost_round", type=int, default=6000)
    parser.add_argument("--depth",           type=int, default=8)
    parser.add_argument("--learning_rate",   type=float, default=0.02)
    parser.add_argument("--l2_leaf_reg",     type=float, default=3.0)
    parser.add_argument("--tag",             default="cat-deep")
    args = parser.parse_args()

    run(args.version, args.num_boost_round, args.depth, args.learning_rate, args.l2_leaf_reg, args.tag)


if __name__ == "__main__":
    main()
