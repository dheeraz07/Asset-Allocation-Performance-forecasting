#!/usr/bin/env bash
#
# End-to-end pipeline. Run from the project root.
#
# Each stage writes artefacts into ./work and can be skipped if its output
# already exists. The final stage writes the submission CSVs into the
# project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

echo "[1/9]  base features  (v2)"
$PYTHON src/01_features_engineering.py

echo "[2/9]  continuous-TE features  (v3)"
$PYTHON src/02_features_continuous_te.py

echo "[3/9]  LightGBM base model"
$PYTHON src/04_lgbm_cv_oof.py  --version v3 --tag lgbm-classification-v3

echo "[4/9]  XGBoost base model"
$PYTHON src/05_xgb_cv_oof.py   --version v2 --tag xgb-cls-v2

echo "[5/9]  CatBoost deep base model"
$PYTHON src/06_catboost_cv_oof.py --version v2 --tag cat-deep

echo "[6/9]  sequence CNN + GRU base model"
$PYTHON src/08_sequence_cnn_gru_cv_oof.py

echo "[7/9]  bagged MLP base model  (v2 features)"
$PYTHON src/07_mlp_bagged_cv_oof.py --version v2 --seeds 8 --tag mlp-bag8

echo "[8/9]  calibration-bias features  (v7)  and bias-feature MLP"
$PYTHON src/03_features_calibration_bias.py
$PYTHON src/07_mlp_bagged_cv_oof.py --version v7 --seeds 8 --tag mlp-bag8-biasfeature

echo "[9/9]  final stack and submission"
$PYTHON src/09_final_stack_and_submit.py

echo "done"
