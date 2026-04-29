"""
Row-level and cross-sectional features built from the raw 20-day windows of
returns and signed volumes, plus per-allocation long-run stats and
(TS, GROUP) cross-sectional ranks.

The output is a single parquet per split used by every downstream model.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

from config import (
    TRAIN_X_CSV, TEST_X_CSV, TRAIN_Y_CSV,
    RETURN_COLUMNS, SIGNED_VOLUME_COLUMNS,
    features_parquet,
)


warnings.filterwarnings("ignore")


def _build_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pointwise statistics of the 20-day return and signed-volume windows.
    Also keeps the raw missingness pattern since it carries information
    (notably SIGNED_VOLUME_1 missing for a large fraction of rows).
    """
    out: dict[str, np.ndarray] = {}
    R = df[RETURN_COLUMNS].values.astype(np.float32)
    V = df[SIGNED_VOLUME_COLUMNS].values.astype(np.float32)

    out["SV1_MISSING"]   = df["SIGNED_VOLUME_1"].isna().astype(np.int8).values
    out["SV20_MISSING"]  = df["SIGNED_VOLUME_20"].isna().astype(np.int8).values
    out["MDT_MISSING"]   = df["MEDIAN_DAILY_TURNOVER"].isna().astype(np.int8).values
    out["RET_NAN_COUNT"] = np.isnan(R).sum(1).astype(np.int8)
    out["SV_NAN_COUNT"]  = np.isnan(V).sum(1).astype(np.int8)

    R = np.nan_to_num(R, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    # Raw lags and simple transforms of RET_1, which is the single most
    # informative lag for next-day sign.
    out["RET_1"]      : np.ndarray = R[:, 0]
    out["RET_2"]      : np.ndarray = R[:, 1]
    out["RET_3"]      : np.ndarray = R[:, 2]
    out["RET_5"]      : np.ndarray = R[:, 4]
    out["RET_10"]     : np.ndarray = R[:, 9]
    out["RET_20"]     : np.ndarray = R[:, 19]
    out["ABS_RET_1"]  : np.ndarray = np.abs(R[:, 0])
    out["SIGN_RET_1"] : np.ndarray = np.sign(R[:, 0])

    # Momentum / reversal over expanding horizons plus their realised Sharpe.
    for k in (2, 3, 5, 10, 15, 20):
        m  = R[:, :k].mean(1)
        s  = R[:, :k].sum(1)
        st = R[:, :k].std(1)
        out[f"RET_MEAN_{k}"]      = m
        out[f"RET_SUM_{k}"]       = s
        out[f"RET_STD_{k}"]       = st
        out[f"RET_CUM_SIGN_{k}"]  = (s > 0).astype(np.int8)
        out[f"POS_FRAC_{k}"]      = (R[:, :k] > 0).astype(np.float32).mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"RET_SHARPE_{k}"] = np.where(st > 1e-12, m / st, 0.0)

    # Momentum acceleration: recent minus older window. Separates fresh
    # momentum from stale momentum and is often more informative than either.
    out["MOM_ACC_5_10"]   = R[:, :5].sum(1)  - R[:, 5:10].sum(1)
    out["MOM_ACC_10_20"]  = R[:, :10].sum(1) - R[:, 10:20].sum(1)
    out["MOM_MINUS_REV"]  = R[:, :20].sum(1) - R[:, 0]

    out["RET_1_Z"]         = np.where(out["RET_STD_20"] > 1e-12, R[:, 0] / out["RET_STD_20"], 0.0)
    out["VOL_RATIO_5_20"]  = np.where(out["RET_STD_20"] > 1e-12, out["RET_STD_5"]  / out["RET_STD_20"], 1.0)
    out["VOL_RATIO_10_20"] = np.where(out["RET_STD_20"] > 1e-12, out["RET_STD_10"] / out["RET_STD_20"], 1.0)

    out["RET_MIN_20"]   = R[:, :20].min(1)
    out["RET_MAX_20"]   = R[:, :20].max(1)
    out["RET_RANGE_20"] = out["RET_MAX_20"] - out["RET_MIN_20"]

    mu = R.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True) + 1e-12
    z  = (R - mu) / sd
    out["RET_SKEW_20"] = (z ** 3).mean(1)
    out["RET_KURT_20"] = (z ** 4).mean(1) - 3.0

    # Sign flips inside the window proxy for short-horizon mean-reversion.
    signs = np.sign(R)
    out["SIGN_FLIPS_20"] = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / 2.0

    # Reversed to chronological order so cumulative sums make economic sense.
    R_chrono = R[:, ::-1]
    cum  = np.cumsum(R_chrono, axis=1)
    peak = np.maximum.accumulate(cum, axis=1)
    dd   = cum - peak

    out["MAX_DD_20"]   = dd.min(1)
    out["DD_END"]      = dd[:, -1]
    out["CUM_RET_END"] = cum[:, -1]

    x     = np.arange(20, dtype=np.float32)
    x_c   = x - x.mean()
    denom = (x_c ** 2).sum()
    # OLS slope of returns on time. Signals trend vs chop.
    out["RET_TREND_SLOPE"] = (R_chrono * x_c).sum(1) / denom

    # Lag-1 autocorrelation within the window. Small but persistent edge
    # in financial series.
    a  = R_chrono[:, 1:]
    b  = R_chrono[:, :-1]
    am = a.mean(1, keepdims=True)
    bm = b.mean(1, keepdims=True)
    num = ((a - am) * (b - bm)).sum(1)
    den = np.sqrt(((a - am) ** 2).sum(1) * ((b - bm) ** 2).sum(1)) + 1e-12
    out["RET_AC1"] = num / den

    # Volume statistics. Use absolute value because sign is already captured
    # in SIGN_AGREE features below.
    for k in (5, 10, 20):
        out[f"SV_MEAN_{k}"]    = V[:, :k].mean(1)
        out[f"SV_STD_{k}"]     = V[:, :k].std(1)
        out[f"SV_ABSMEAN_{k}"] = np.abs(V[:, :k]).mean(1)

    out["SV_1"]             = V[:, 0]
    out["SV_POS_FRAC_20"]   = (V[:, :20] > 0).astype(np.float32).mean(1)
    # Fraction of days the sign of volume matches the sign of return. A
    # classic momentum-confirmation feature.
    out["SV_SIGN_AGREE_20"] = (np.sign(R) == np.sign(V)).astype(np.float32).mean(1)
    out["SV_RATIO_5_20"]    = np.where(out["SV_ABSMEAN_20"] > 1e-12,
                                        out["SV_ABSMEAN_5"] / out["SV_ABSMEAN_20"], 1.0)
    out["SV_Z_1"]           = np.where(out["SV_STD_20"] > 1e-12, V[:, 0] / out["SV_STD_20"], 0.0)

    V_chrono = V[:, ::-1]
    out["SV_TREND_SLOPE"] = (V_chrono * x_c).sum(1) / denom

    # Rolling vol-return correlation — picks up the "bad news on high volume"
    # anti-momentum effect on short horizons.
    for k in (5, 10):
        Rk  = R[:, :k]; Vk = V[:, :k]
        Rm  = Rk.mean(1, keepdims=True); Vm = Vk.mean(1, keepdims=True)
        num = ((Rk - Rm) * (Vk - Vm)).sum(1)
        den = np.sqrt(((Rk - Rm) ** 2).sum(1) * ((Vk - Vm) ** 2).sum(1)) + 1e-12
        out[f"VOL_RET_CORR_{k}"] = num / den

    # Turnover with log transform since it spans orders of magnitude.
    mdt = df["MEDIAN_DAILY_TURNOVER"].fillna(df["MEDIAN_DAILY_TURNOVER"].median()).values.astype(np.float32)
    out["MDT"]     = mdt
    out["MDT_LOG"] = np.log1p(np.clip(mdt, 0, None))

    return pd.DataFrame(out, index=df.index)


def _add_allocation_long_run_stats(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    feats_train: pd.DataFrame, feats_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-allocation aggregates over the concatenation of train and test. These
    use only X-side values, so there is no label leakage. Gives the model
    a stable per-allocation baseline to z-score against.
    """
    concat = pd.concat(
        [
            pd.DataFrame({
                "ALLOC_ID":   X_train["ALLOC_ID"].values,
                "RET_1":      feats_train["RET_1"].values,
                "RET_STD_20": feats_train["RET_STD_20"].values,
                "MDT":        feats_train["MDT"].values,
            }),
            pd.DataFrame({
                "ALLOC_ID":   X_test["ALLOC_ID"].values,
                "RET_1":      feats_test["RET_1"].values,
                "RET_STD_20": feats_test["RET_STD_20"].values,
                "MDT":        feats_test["MDT"].values,
            }),
        ],
        ignore_index=True,
    )

    grouped = concat.groupby("ALLOC_ID")
    stats = pd.DataFrame({
        "ALLOC_RET1_MEAN": grouped["RET_1"].transform("mean"),
        "ALLOC_RET1_STD":  grouped["RET_1"].transform("std"),
        "ALLOC_VOL_MEAN":  grouped["RET_STD_20"].transform("mean"),
        "ALLOC_MDT_MEAN":  grouped["MDT"].transform("mean"),
    })

    n_train : int = len(feats_train)
    feats_train = feats_train.copy()
    feats_test  = feats_test.copy()
    feats_train[stats.columns.tolist()] = stats.iloc[:n_train].values
    feats_test [stats.columns.tolist()] = stats.iloc[n_train:].values

    # Z-score RET_1 inside its allocation's historical distribution.
    for c in ("RET_1",):
        feats_train[f"{c}_ZALLOC"] = (
            (feats_train[c] - feats_train["ALLOC_RET1_MEAN"])
            / feats_train["ALLOC_RET1_STD"].replace(0, np.nan)
        ).fillna(0).astype(np.float32)
        feats_test[f"{c}_ZALLOC"] = (
            (feats_test[c] - feats_test["ALLOC_RET1_MEAN"])
            / feats_test["ALLOC_RET1_STD"].replace(0, np.nan)
        ).fillna(0).astype(np.float32)

    return feats_train, feats_test


def _cross_sectional(df: pd.DataFrame, feats: pd.DataFrame, base_cols: list[str], key: str) -> pd.DataFrame:
    """
    Within-`key` pct-rank, deviation from median and z-score. `key` is
    either the full TS or the (TS, GROUP) pair.
    """
    out : dict[str, np.ndarray] = {}
    big = feats[base_cols].copy()
    if key == "TS":
        big[key] = df["TS"].values
    else:
        big[key] = df["TS"].astype(str) + "_" + df["GROUP"].astype(str)

    suffix : str = {"TS": "TS", "TS_GROUP": "TG"}[key]

    for c in base_cols:
        grouped = big.groupby(key)[c]
        out[f"{c}_{suffix}_RANK"] = grouped.rank(pct=True).astype(np.float32).values
        mu = grouped.transform("mean").astype(np.float32).values
        sd = grouped.transform("std").astype(np.float32).values + 1e-12
        out[f"{c}_{suffix}_Z"]   = ((big[c].values - mu) / sd).astype(np.float32)
        if key == "TS":
            med = grouped.transform("median").astype(np.float32).values
            out[f"{c}_{suffix}_MED"] = (big[c].values - med).astype(np.float32)

    return pd.DataFrame(out, index=df.index)


def _encode_identifiers(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Shared category dictionary so train and test get the same integer codes.
    alloc_codes = pd.concat([train["ALLOCATION"], test["ALLOCATION"]]).astype("category").cat.codes.astype(np.int32)
    train["ALLOC_ID"] : pd.Series = alloc_codes[:len(train)].values
    test ["ALLOC_ID"] : pd.Series = alloc_codes[len(train):].values
    # Interaction of allocation and its GROUP; lets tree models pick up
    # within-allocation regime differences without a hard split.
    train["AG_ID"] = (train["ALLOC_ID"].astype(np.int64) * 10 + train["GROUP"]).astype(np.int64)
    test ["AG_ID"] = (test ["ALLOC_ID"].astype(np.int64) * 10 + test ["GROUP"]).astype(np.int64)
    return train, test


def main() -> None:
    t_start : float = time.time()

    X_train : pd.DataFrame = pd.read_csv(TRAIN_X_CSV)
    X_test  : pd.DataFrame = pd.read_csv(TEST_X_CSV)
    y_train : pd.DataFrame = pd.read_csv(TRAIN_Y_CSV)
    print(f"[load] {time.time() - t_start:.1f}s  train {X_train.shape}  test {X_test.shape}")

    # The TS labels look like "DATE_0001"; keep only the integer part for
    # ordering and downstream GroupKFold splits.
    X_train["TS_INT"] : pd.Series = X_train["TS"].str.slice(5).astype(int)
    X_test ["TS_INT"] : pd.Series = X_test ["TS"].str.slice(5).astype(int)
    X_train, X_test = _encode_identifiers(X_train, X_test)

    t : float = time.time()
    feats_train : pd.DataFrame = _build_row_features(X_train)
    feats_test  : pd.DataFrame = _build_row_features(X_test)
    print(f"[row features] {time.time() - t:.1f}s  {feats_train.shape[1]} columns")

    t = time.time()
    feats_train, feats_test = _add_allocation_long_run_stats(X_train, X_test, feats_train, feats_test)
    print(f"[allocation stats] {time.time() - t:.1f}s")

    # Two layers of cross-sectional features: across all allocations on a
    # given day, and within each (TS, GROUP) cell.
    cs_base : list[str] = [
        "RET_1", "ABS_RET_1", "RET_SUM_5", "RET_SUM_20", "RET_STD_20",
        "VOL_RATIO_5_20", "SV_ABSMEAN_20", "MDT_LOG", "RET_1_Z",
        "RET_SHARPE_5", "RET_SHARPE_10", "RET_1_ZALLOC",
    ]

    t = time.time()
    feats_train = pd.concat([feats_train, _cross_sectional(X_train, feats_train, cs_base, "TS")], axis=1)
    feats_test  = pd.concat([feats_test,  _cross_sectional(X_test,  feats_test,  cs_base, "TS")], axis=1)
    print(f"[cross-sectional TS] {time.time() - t:.1f}s")

    cs_tg_base : list[str] = ["RET_1", "RET_SUM_5", "RET_STD_20", "MDT_LOG"]
    t = time.time()
    feats_train = pd.concat([feats_train, _cross_sectional(X_train, feats_train, cs_tg_base, "TS_GROUP")], axis=1)
    feats_test  = pd.concat([feats_test,  _cross_sectional(X_test,  feats_test,  cs_tg_base, "TS_GROUP")], axis=1)
    print(f"[cross-sectional TS-GROUP] {time.time() - t:.1f}s")

    # Carry forward identifiers, the date key, and the target.
    for col in ("ALLOC_ID", "AG_ID", "GROUP", "TS_INT"):
        feats_train[col] = X_train[col].values
        feats_test [col] = X_test [col].values
    feats_train["TS"]     = X_train["TS"].values
    feats_test ["TS"]     = X_test ["TS"].values
    feats_train["ROW_ID"] = X_train["ROW_ID"].values
    feats_test ["ROW_ID"] = X_test ["ROW_ID"].values
    feats_train["TARGET"] = y_train["target"].values

    feats_train.to_parquet(features_parquet("v2", "train"), index=False)
    feats_test .to_parquet(features_parquet("v2", "test"),  index=False)
    print(f"[save] total {time.time() - t_start:.1f}s  train {feats_train.shape}  test {feats_test.shape}")


if __name__ == "__main__":
    main()
