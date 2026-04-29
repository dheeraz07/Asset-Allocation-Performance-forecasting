"""
Final stacker. Reads every base model's OOF and test probability files,
fits a Ridge meta-learner on the OOFs with honest GroupKFold evaluation,
applies a per-allocation additive bias correction (post-processing) and
finally converts probabilities to class labels with a per-GROUP optimum
threshold. Writes the CSV submissions at project root.

Two artefacts are produced:
  submission_v5_final.csv         - per-GROUP optimal threshold (best OOF)
  submission_v5_conservative.csv  - per-GROUP threshold that forces the
                                    predicted positive fraction to match
                                    each group's in-sample positive rate.
                                    More robust to a drifted test regime.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from config import (
    CV_N_SPLITS, DATA_DIR, SAMPLE_SUB_CSV, WORK_DIR,
    features_parquet, oof_path, test_preds_path,
)


BASE_MODELS : list[str] = [
    "mlp-bag8-biasfeature",
    "mlp-bag8",
    "cat-deep",
    "xgb-cls-v2",
    "seq-cnn",
    "lgbm-classification-v3",
]


def _best_threshold(probs: np.ndarray, y: np.ndarray,
                    lo: float = 0.35, hi: float = 0.65, step: float = 0.001) -> tuple[float, float]:
    """Grid search for the classification threshold that maximises accuracy."""
    best_threshold : float = 0.5
    best_accuracy  : float = 0.0
    for threshold in np.arange(lo, hi + 1e-9, step):
        accuracy = accuracy_score(y, (probs > threshold).astype(int))
        if accuracy > best_accuracy:
            best_accuracy  = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def main() -> None:
    train : pd.DataFrame = pd.read_parquet(features_parquet("v2", "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet("v2", "test"))
    y     : np.ndarray   = (train["TARGET"].values > 0).astype(int)
    groups_ts : np.ndarray = train["TS"].values
    alloc_train : np.ndarray = train["ALLOC_ID"].values
    alloc_test  : np.ndarray = test ["ALLOC_ID"].values
    group_train : np.ndarray = train["GROUP"].values
    group_test  : np.ndarray = test ["GROUP"].values

    oof_matrix  : np.ndarray = np.stack([np.load(oof_path(m))        for m in BASE_MODELS], axis=1)
    test_matrix : np.ndarray = np.stack([np.load(test_preds_path(m)) for m in BASE_MODELS], axis=1)

    # Ridge meta fitted on the full OOF for test-time scoring, plus a
    # fold-wise refit so the train-side OOF stays honest.
    ridge_full = Ridge(alpha=1.0).fit(oof_matrix, y.astype(float))
    test_stack : np.ndarray = ridge_full.predict(test_matrix).astype(np.float32)

    gkf = GroupKFold(CV_N_SPLITS)
    oof_stack : np.ndarray = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in gkf.split(np.arange(len(y)), groups=groups_ts):
        fold_ridge = Ridge(alpha=1.0).fit(oof_matrix[tr_idx], y[tr_idx].astype(float))
        oof_stack[va_idx] = fold_ridge.predict(oof_matrix[va_idx])

    threshold_raw, acc_raw = _best_threshold(oof_stack, y)
    print(f"Raw stack OOF accuracy {acc_raw:.4f} at threshold {threshold_raw:.3f}")
    print("Ridge coefficients: " + ", ".join(
        f"{name} {coef:+.3f}" for name, coef in zip(BASE_MODELS, ridge_full.coef_)
    ))

    # Per-allocation additive bias correction. Uses the honest OOF stack
    # only, so test-side predictions only see the per-allocation LUT.
    alloc_frame = pd.DataFrame({"alloc": alloc_train, "y": y, "p": oof_stack})
    alloc_agg = alloc_frame.groupby("alloc").agg(mean_pred=("p", "mean"), mean_true=("y", "mean"))
    alloc_agg["bias"] = alloc_agg["mean_true"] - alloc_agg["mean_pred"]
    bias_lut : dict = alloc_agg["bias"].to_dict()

    bias_train = pd.Series(alloc_train).map(bias_lut).fillna(0.0).astype(np.float32).values
    bias_test  = pd.Series(alloc_test ).map(bias_lut).fillna(0.0).astype(np.float32).values
    oof_calibrated  : np.ndarray = np.clip(oof_stack   + bias_train, 0.01, 0.99).astype(np.float32)
    test_calibrated : np.ndarray = np.clip(test_stack  + bias_test,  0.01, 0.99).astype(np.float32)

    threshold_cal, acc_cal = _best_threshold(oof_calibrated, y)
    print(f"Calibrated OOF accuracy {acc_cal:.4f} at threshold {threshold_cal:.3f}  (delta {acc_cal - acc_raw:+.4f})")

    # Per-GROUP optimal threshold on the calibrated probabilities.
    pred_test_optimal : np.ndarray = np.zeros(len(test_calibrated), dtype=np.int8)
    pred_oof_optimal  : np.ndarray = np.zeros(len(y),               dtype=np.int8)
    per_group : dict = {}
    for group_id in np.unique(group_train):
        tr_mask = group_train == group_id
        te_mask = group_test  == group_id
        t_g, acc_g = _best_threshold(oof_calibrated[tr_mask], y[tr_mask])
        per_group[int(group_id)] = {"threshold": t_g, "oof_acc": acc_g}
        pred_test_optimal[te_mask] = (test_calibrated[te_mask] > t_g).astype(np.int8)
        pred_oof_optimal [tr_mask] = (oof_calibrated [tr_mask] > t_g).astype(np.int8)

    overall_optimal : float = accuracy_score(y, pred_oof_optimal)
    print(f"Per-GROUP optimal OOF accuracy {overall_optimal:.4f}")
    for group_id, info in per_group.items():
        print(f"  GROUP {group_id}: threshold {info['threshold']:.3f}  OOF accuracy {info['oof_acc']:.4f}")

    # Per-GROUP quantile-matched threshold. Forces the predicted positive
    # fraction on test to equal the train positive fraction within that
    # group — a cheap but effective defence against distribution drift.
    pred_test_conservative : np.ndarray = np.zeros(len(test_calibrated), dtype=np.int8)
    for group_id in np.unique(group_train):
        tr_mask = group_train == group_id
        te_mask = group_test  == group_id
        pos_rate = float(y[tr_mask].mean())
        if te_mask.sum() == 0:
            continue
        t_quant = float(np.quantile(test_calibrated[te_mask], 1 - pos_rate))
        pred_test_conservative[te_mask] = (test_calibrated[te_mask] > t_quant).astype(np.int8)

    sample = pd.read_csv(SAMPLE_SUB_CSV)
    assert (sample["ROW_ID"].values == test["ROW_ID"].values).all(), "ROW_ID alignment mismatch"

    final_path        : str = os.path.join(DATA_DIR, "submission_v5_final.csv")
    conservative_path : str = os.path.join(DATA_DIR, "submission_v5_conservative.csv")

    sample_final = sample.copy();         sample_final       ["prediction"] = pred_test_optimal
    sample_cons  = sample.copy();         sample_cons        ["prediction"] = pred_test_conservative
    sample_final.to_csv(final_path,        index=False)
    sample_cons .to_csv(conservative_path, index=False)

    print(f"wrote {os.path.basename(final_path)}  pos_rate {sample_final.prediction.mean():.4f}")
    print(f"wrote {os.path.basename(conservative_path)}  pos_rate {sample_cons.prediction.mean():.4f}")

    # Persist intermediate arrays in case they are useful for further probing.
    np.save(os.path.join(WORK_DIR, "oof_calibrated_v5.npy"),  oof_calibrated)
    np.save(os.path.join(WORK_DIR, "test_calibrated_v5.npy"), test_calibrated)
    with open(os.path.join(WORK_DIR, "final_v5_report.json"), "w") as fp:
        json.dump({
            "models":                         BASE_MODELS,
            "raw_oof_best_acc":               float(acc_raw),
            "calibrated_oof_best_acc":        float(acc_cal),
            "per_group_optimal_overall_acc":  float(overall_optimal),
            "per_group_thresholds":           per_group,
            "submission_pos_rate_optimal":    float(sample_final.prediction.mean()),
            "submission_pos_rate_conservative": float(sample_cons.prediction.mean()),
        }, fp, indent=2)


if __name__ == "__main__":
    main()
