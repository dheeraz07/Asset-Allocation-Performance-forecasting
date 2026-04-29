"""
Sequence model that reads the raw 20-day window directly: a small 1D-CNN
followed by a one-way GRU, concatenated with learned embeddings of
ALLOCATION and GROUP plus a handful of hand-crafted aux features. Trained
with GroupKFold-by-TS and saves OOF / test probabilities for the stacker.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from config import (
    CV_N_SPLITS, RANDOM_SEED,
    TRAIN_X_CSV, TEST_X_CSV, TRAIN_Y_CSV,
    RETURN_COLUMNS, SIGNED_VOLUME_COLUMNS,
    oof_path, score_path, test_preds_path,
)


DEVICE : str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class SequenceModel(nn.Module):
    """
    Two-channel input (returns, signed volumes) over 20 chronological days,
    run through a small Conv1d stack, then a GRU that compresses to a single
    state vector, concatenated with categorical embeddings and aux features.
    """

    def __init__(self, n_alloc: int, n_group: int, n_aux: int,
                 emb_alloc: int = 16, emb_group: int = 4,
                 conv_channels: int = 32, gru_hidden: int = 32) -> None:
        super().__init__()
        self.emb_alloc = nn.Embedding(n_alloc, emb_alloc)
        self.emb_group = nn.Embedding(n_group, emb_group)

        self.conv = nn.Sequential(
            nn.Conv1d(2, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.gru    = nn.GRU(conv_channels, gru_hidden, batch_first=True)
        self.aux_bn = nn.BatchNorm1d(n_aux)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + emb_alloc + emb_group + n_aux, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, seq: torch.Tensor, aux: torch.Tensor,
                alloc_id: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(seq).transpose(1, 2)
        _, hidden = self.gru(conv_out)
        h = hidden.squeeze(0)
        z = torch.cat([h, self.emb_alloc(alloc_id), self.emb_group(group_id), self.aux_bn(aux)], dim=1)
        return self.head(z).squeeze(-1)


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_csv(TRAIN_X_CSV)
    X_test  = pd.read_csv(TEST_X_CSV)
    y_train = pd.read_csv(TRAIN_Y_CSV)

    # Shared category dictionary so train and test share integer codes.
    alloc = pd.concat([X_train["ALLOCATION"], X_test["ALLOCATION"]]).astype("category").cat.codes.astype(np.int32)
    X_train["ALLOC_ID"] = alloc[:len(X_train)].values
    X_test ["ALLOC_ID"] = alloc[len(X_train):].values
    return X_train, X_test, y_train


def _build_sequence(df: pd.DataFrame) -> np.ndarray:
    """
    Returns: array of shape (N, 2, 20) with channel 0 returns, channel 1 signed volumes.
    Columns are reversed to chronological order so day 0 is the oldest day.
    Each row is normalised by its own std / mean-abs to remove scale.
    """
    R = np.nan_to_num(df[RETURN_COLUMNS       ].values.astype(np.float32), nan=0.0)[:, ::-1]
    V = np.nan_to_num(df[SIGNED_VOLUME_COLUMNS].values.astype(np.float32), nan=0.0)[:, ::-1]

    R = R / (R.std(axis=1, keepdims=True) + 1e-6)
    V = V / (np.abs(V).mean(axis=1, keepdims=True) + 1e-6)
    return np.stack([R, V], axis=1).astype(np.float32)


def _build_aux(df: pd.DataFrame) -> np.ndarray:
    """Handful of window summary stats that don't come out of the sequence path."""
    R = np.nan_to_num(df[RETURN_COLUMNS       ].values.astype(np.float32), nan=0.0)
    V = np.nan_to_num(df[SIGNED_VOLUME_COLUMNS].values.astype(np.float32), nan=0.0)
    mdt = df["MEDIAN_DAILY_TURNOVER"].fillna(df["MEDIAN_DAILY_TURNOVER"].median()).values.astype(np.float32)

    features : dict[str, np.ndarray] = {
        "RET_1":        R[:, 0],
        "RET_SUM_5":    R[:, :5].sum(1),
        "RET_SUM_20":   R[:, :20].sum(1),
        "RET_STD_20":   R[:, :20].std(1),
        "SV_1":         V[:, 0],
        "SV_STD_20":    V[:, :20].std(1),
        "SV1_MISSING":  df["SIGNED_VOLUME_1"].isna().astype(np.float32).values,
        "MDT":          mdt,
        "MDT_LOG":      np.log1p(np.clip(mdt, 0, None)),
    }
    return pd.DataFrame(features).values.astype(np.float32)


def _predict_batched(
    model: SequenceModel,
    seq: torch.Tensor, aux: torch.Tensor, alloc: torch.Tensor, group: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks : list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, seq.shape[0], batch_size):
            logits = model(
                seq  [i : i + batch_size].to(DEVICE),
                aux  [i : i + batch_size].to(DEVICE),
                alloc[i : i + batch_size].to(DEVICE),
                group[i : i + batch_size].to(DEVICE),
            )
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 : float = time.time()
    X_train, X_test, y_train = _load_raw()
    y_sign : np.ndarray = (y_train["target"].values > 0).astype(np.float32)

    seq_train  : np.ndarray = _build_sequence(X_train)
    seq_test   : np.ndarray = _build_sequence(X_test)
    aux_train  : np.ndarray = _build_aux(X_train)
    aux_test   : np.ndarray = _build_aux(X_test)

    n_alloc : int = int(max(X_train["ALLOC_ID"].max(), X_test["ALLOC_ID"].max())) + 1
    n_group : int = int(max(X_train["GROUP"].max(),    X_test["GROUP"].max()))    + 1

    splits = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(X_train)), groups=X_train["TS"].values))

    oof_prob  : np.ndarray = np.zeros(len(X_train), dtype=np.float32)
    test_prob : np.ndarray = np.zeros(len(X_test),  dtype=np.float32)
    fold_scores : list[float] = []

    batch_size : int = 8192
    print(f"device {DEVICE}")

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        mu = np.nanmean(aux_train[tr_idx], axis=0, keepdims=True)
        sd = np.nanstd (aux_train[tr_idx], axis=0, keepdims=True) + 1e-6
        aux_tr_n = np.clip(np.nan_to_num((aux_train - mu) / sd, nan=0.0), -8, 8).astype(np.float32)
        aux_te_n = np.clip(np.nan_to_num((aux_test  - mu) / sd, nan=0.0), -8, 8).astype(np.float32)

        seq_tr_t = torch.from_numpy(seq_train[tr_idx])
        seq_va_t = torch.from_numpy(seq_train[va_idx])
        seq_te_t = torch.from_numpy(seq_test)
        aux_tr_t = torch.from_numpy(aux_tr_n[tr_idx])
        aux_va_t = torch.from_numpy(aux_tr_n[va_idx])
        aux_te_t = torch.from_numpy(aux_te_n)
        alloc_tr_t = torch.from_numpy(X_train["ALLOC_ID"].values[tr_idx].astype(np.int64))
        alloc_va_t = torch.from_numpy(X_train["ALLOC_ID"].values[va_idx].astype(np.int64))
        alloc_te_t = torch.from_numpy(X_test ["ALLOC_ID"].values.astype(np.int64))
        group_tr_t = torch.from_numpy(X_train["GROUP"].values[tr_idx].astype(np.int64))
        group_va_t = torch.from_numpy(X_train["GROUP"].values[va_idx].astype(np.int64))
        group_te_t = torch.from_numpy(X_test ["GROUP"].values.astype(np.int64))
        y_tr_t     = torch.from_numpy(y_sign[tr_idx])

        model = SequenceModel(n_alloc, n_group, aux_tr_n.shape[1]).to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)
        loss_fn   = nn.BCEWithLogitsLoss()

        best_val_acc    : float = 0.0
        best_state_dict : dict | None = None
        epochs_no_improve : int = 0

        for epoch in range(15):
            model.train()
            permutation = torch.randperm(seq_tr_t.shape[0])
            for i in range(0, seq_tr_t.shape[0], batch_size):
                idx = permutation[i : i + batch_size]
                logits = model(
                    seq_tr_t  [idx].to(DEVICE),
                    aux_tr_t  [idx].to(DEVICE),
                    alloc_tr_t[idx].to(DEVICE),
                    group_tr_t[idx].to(DEVICE),
                )
                loss = loss_fn(logits, y_tr_t[idx].to(DEVICE))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            scheduler.step()

            val_probs = _predict_batched(model, seq_va_t, aux_va_t, alloc_va_t, group_va_t, batch_size)
            val_acc   = accuracy_score(y_sign[va_idx], (val_probs > 0.5).astype(int))

            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print(f"  fold {fold_i + 1}  epoch {epoch + 1}  val_acc {val_acc:.4f}  best {best_val_acc:.4f}")
            if epochs_no_improve >= 3:
                break

        model.load_state_dict(best_state_dict)
        val_probs  = _predict_batched(model, seq_va_t, aux_va_t, alloc_va_t, group_va_t, batch_size)
        test_probs = _predict_batched(model, seq_te_t, aux_te_t, alloc_te_t, group_te_t, batch_size)
        oof_prob [va_idx] = val_probs
        test_prob        += test_probs / CV_N_SPLITS
        fold_scores.append(best_val_acc)

    mean : float = float(np.mean(fold_scores))
    std  : float = float(np.std (fold_scores))
    print(f"CV mean {mean:.4f}  std {std:.4f}  elapsed {time.time() - t0:.1f}s")

    tag : str = "seq-cnn"
    np.save(oof_path(tag),        oof_prob)
    np.save(test_preds_path(tag), test_prob)
    with open(score_path(tag), "w") as fp:
        json.dump({"mean": mean, "std": std, "model": "seq_cnn_gru"}, fp, indent=2)
    print(f"saved tag {tag}")


if __name__ == "__main__":
    main()
