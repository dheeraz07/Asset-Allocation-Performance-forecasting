"""
Project paths and constants. Every script imports from here so there are no
hard-coded paths anywhere and the repo runs unchanged on any machine.
"""
import os


ROOT_DIR   : str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   : str = ROOT_DIR
WORK_DIR   : str = os.path.join(ROOT_DIR, "work")

os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_X_CSV      : str = os.path.join(DATA_DIR, "X_train_9xQjqvZ.csv")
TEST_X_CSV       : str = os.path.join(DATA_DIR, "X_test_1zTtEnD.csv")
TRAIN_Y_CSV      : str = os.path.join(DATA_DIR, "y_train_Ppwhaz8.csv")
SAMPLE_SUB_CSV   : str = os.path.join(DATA_DIR, "sample_submission_SpGVFuH.csv")

RANDOM_SEED      : int = 42
CV_N_SPLITS      : int = 5

RETURN_COLUMNS        : list = [f"RET_{i}" for i in range(1, 21)]
SIGNED_VOLUME_COLUMNS : list = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def features_parquet(name: str, split: str) -> str:
    """
    Resolve the parquet path for a feature matrix.

    name  :  v2 / v3 / v7  (stage of the feature pipeline)
    split :  train / test
    """
    assert split in {"train", "test"}
    return os.path.join(WORK_DIR, f"{split}_feats_{name}.parquet")


def oof_path(tag: str) -> str:
    return os.path.join(WORK_DIR, f"oof_{tag}.npy")


def test_preds_path(tag: str) -> str:
    return os.path.join(WORK_DIR, f"test_{tag}.npy")


def score_path(tag: str) -> str:
    return os.path.join(WORK_DIR, f"score_{tag}.json")
