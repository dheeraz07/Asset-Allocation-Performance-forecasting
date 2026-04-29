"""
Seed-bagged embedding MLP trained with GroupKFold-by-TS cross-validation.

Architecture: BatchNorm on continuous features, learned embeddings for
ALLOC_ID and GROUP, 2-layer trunk with dropout, single sigmoid head.
For each fold we train `seeds` independent replicas and average their
validation / test probabilities. Bagging materially reduces fold variance
and is the single biggest-impact trick for neural nets on low-SNR data.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from config import (
    CV_N_SPLITS,
    features_parquet, oof_path, score_path, test_preds_path,
)


CATEGORICAL_COLUMNS : list[str] = ["ALLOC_ID", "AG_ID", "GROUP"]
META_COLUMNS        : list[str] = ["TS", "TS_INT", "ROW_ID", "TARGET"]

DEVICE : str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class EmbeddingMLP(nn.Module):
    """Continuous features passed through BN, concatenated with learned
    embeddings of ALLOC_ID and GROUP, fed into an MLP trunk with dropout."""

    def __init__(
        self, n_alloc: int, n_group: int, n_continuous: int,
        emb_alloc: int = 16, emb_group: int = 4,
        hidden: tuple[int, ...] = (128, 64), dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.emb_alloc = nn.Embedding(n_alloc, emb_alloc)
        self.emb_group = nn.Embedding(n_group, emb_group)
        self.bn        = nn.BatchNorm1d(n_continuous)

        trunk : list[nn.Module] = []
        prev_dim : int = emb_alloc + emb_group + n_continuous
        for hidden_dim in hidden:
            trunk += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)
        self.head  = nn.Linear(prev_dim, 1)

    def forward(self, x_cont: torch.Tensor, alloc_id: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.emb_alloc(alloc_id), self.emb_group(group_id), self.bn(x_cont)], dim=1)
        return self.head(self.trunk(z)).squeeze(-1)


def _standardise(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-column mean/std on the training-fold rows only and apply to the
    held-out validation and test rows. Winsorise at ±10σ to protect the
    batch-norm statistics from extreme outliers.
    """
    mu = np.nanmean(X_train, axis=0, keepdims=True)
    sd = np.nanstd (X_train, axis=0, keepdims=True) + 1e-6

    def _fit(x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((x - mu) / sd, nan=0.0), -10, 10).astype(np.float32)

    return _fit(X_train), _fit(X_val), _fit(X_test)


def _as_long(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long)


def _as_float(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def _predict_batched(
    model: EmbeddingMLP,
    x_cont: torch.Tensor, alloc: torch.Tensor, group: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        chunks : list[np.ndarray] = []
        for i in range(0, x_cont.shape[0], batch_size):
            logits = model(
                x_cont[i : i + batch_size].to(DEVICE),
                alloc [i : i + batch_size].to(DEVICE),
                group [i : i + batch_size].to(DEVICE),
            )
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks)


def run(version: str, n_seeds: int, wide: bool, tag: str) -> None:
    t0 : float = time.time()

    train : pd.DataFrame = pd.read_parquet(features_parquet(version, "train"))
    test  : pd.DataFrame = pd.read_parquet(features_parquet(version, "test"))

    continuous_cols : list[str] = [c for c in train.columns if c not in META_COLUMNS + CATEGORICAL_COLUMNS]
    print(f"device {DEVICE}  continuous features {len(continuous_cols)}")

    y_sign : np.ndarray = (train["TARGET"].values > 0).astype(np.float32)
    n_alloc : int = int(max(train["ALLOC_ID"].max(), test["ALLOC_ID"].max())) + 1
    n_group : int = int(max(train["GROUP"].max(),    test["GROUP"].max()))    + 1

    splits = list(GroupKFold(CV_N_SPLITS).split(np.arange(len(train)), groups=train["TS"].values))

    oof_prob  : np.ndarray = np.zeros(len(train), dtype=np.float32)
    test_prob : np.ndarray = np.zeros(len(test),  dtype=np.float32)
    fold_scores : list[float] = []

    batch_size : int = 8192
    hidden : tuple[int, ...] = (256, 128, 64) if wide else (128, 64)

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        X_train_cont, X_val_cont, X_test_cont = _standardise(
            train[continuous_cols].iloc[tr_idx].values,
            train[continuous_cols].iloc[va_idx].values,
            test [continuous_cols].values,
        )

        xt_tr = _as_float(X_train_cont)
        xt_va = _as_float(X_val_cont)
        xt_te = _as_float(X_test_cont)
        alloc_tr = _as_long(train["ALLOC_ID"].iloc[tr_idx].values)
        alloc_va = _as_long(train["ALLOC_ID"].iloc[va_idx].values)
        alloc_te = _as_long(test ["ALLOC_ID"].values)
        group_tr = _as_long(train["GROUP"].iloc[tr_idx].values)
        group_va = _as_long(train["GROUP"].iloc[va_idx].values)
        group_te = _as_long(test ["GROUP"].values)
        y_tr = _as_float(y_sign[tr_idx])

        fold_val_prob  : np.ndarray = np.zeros(len(va_idx), dtype=np.float32)
        fold_test_prob : np.ndarray = np.zeros(len(test),   dtype=np.float32)

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Varying dropout slightly across seeds improves bag diversity.
            model = EmbeddingMLP(
                n_alloc, n_group, xt_tr.shape[1],
                hidden=hidden, dropout=0.20 + 0.05 * (seed % 3),
            ).to(DEVICE)

            optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)
            loss_fn   = nn.BCEWithLogitsLoss()

            best_val_acc    : float = 0.0
            best_state_dict : dict | None = None
            epochs_no_improve : int = 0

            for epoch in range(12):
                model.train()
                permutation = torch.randperm(xt_tr.shape[0])
                for i in range(0, xt_tr.shape[0], batch_size):
                    idx = permutation[i : i + batch_size]
                    logits = model(
                        xt_tr[idx].to(DEVICE),
                        alloc_tr[idx].to(DEVICE),
                        group_tr[idx].to(DEVICE),
                    )
                    loss = loss_fn(logits, y_tr[idx].to(DEVICE))
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                scheduler.step()

                val_probs = _predict_batched(model, xt_va, alloc_va, group_va, batch_size)
                val_acc   = accuracy_score(y_sign[va_idx], (val_probs > 0.5).astype(int))

                if val_acc > best_val_acc:
                    best_val_acc    = val_acc
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= 3:
                    break

            model.load_state_dict(best_state_dict)
            val_probs  = _predict_batched(model, xt_va, alloc_va, group_va, batch_size)
            test_probs = _predict_batched(model, xt_te, alloc_te, group_te, batch_size)
            fold_val_prob  += val_probs  / n_seeds
            fold_test_prob += test_probs / n_seeds
            print(f"  fold {fold_i + 1}  seed {seed}  best_val {best_val_acc:.4f}")

        oof_prob [va_idx] = fold_val_prob
        test_prob        += fold_test_prob / CV_N_SPLITS

        fold_acc : float = accuracy_score(y_sign[va_idx], (fold_val_prob > 0.5).astype(int))
        fold_scores.append(fold_acc)
        print(f"  fold {fold_i + 1}  bag acc {fold_acc:.4f}")

    mean : float = float(np.mean(fold_scores))
    std  : float = float(np.std (fold_scores))
    print(f"CV mean {mean:.4f}  std {std:.4f}  elapsed {time.time() - t0:.1f}s")

    np.save(oof_path(tag),        oof_prob)
    np.save(test_preds_path(tag), test_prob)
    with open(score_path(tag), "w") as fp:
        json.dump({"mean": mean, "std": std, "seeds": n_seeds, "version": version,
                   "wide": wide, "model": "embedding_mlp"}, fp, indent=2)
    print(f"saved tag {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", choices=["v2", "v3", "v7"])
    parser.add_argument("--seeds",   type=int, default=8)
    parser.add_argument("--wide",    action="store_true")
    parser.add_argument("--tag",     default="mlp-bag8")
    args = parser.parse_args()
    run(args.version, args.seeds, args.wide, args.tag)


if __name__ == "__main__":
    main()
