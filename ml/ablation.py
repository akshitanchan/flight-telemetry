"""
ml/ablation.py — Feature-group ablation table + HistGBR challenger.
====================================================================

Objectives (W4.2 / ml-04)
--------------------------
1. Feature-group ablation: quantify each group's contribution to per-interval
   RMSE (kg) by dropping one group at a time and measuring the delta vs the
   full-feature baseline.

2. HistGradientBoostingRegressor challenger: head-to-head comparison against
   the MLP baseline, scored through the same chronological grouped CV folds for
   an apples-to-apples per-interval RMSE-kg comparison.

Both are evaluated using the same expanding-window chronological CV harness
defined in ml/cv.py (build_folds, _rmse) so results are directly comparable to
the official CV RMSE 442.65 ± 36.36 kg baseline (real PRC-2025 data).

Feature groups
--------------
The 38 feature columns from ml/features.py (FEATURE_COLUMNS) are organised into
six natural groups:

  G1  duration         : ["duration_s"]
                         Single most important feature — total interval length
                         directly proportional to fuel burned.

  G2  altitude         : ["alt_change", "avg_altitude", "max_altitude", "alt_std"]
                         Vertical profile: climb/descent profile, cruise altitude,
                         altitude variability.

  G3  speed_vrate      : ["avg_speed", "max_vrate"]
                         Groundspeed and peak vertical rate; proxies for flight
                         phase and aerodynamic drag.

  G4  track            : ["avg_track_change"]
                         Turning intensity — captures non-direct routing and
                         holding patterns.

  G5  mach_tas_cas     : ["avg_mach", "avg_tas", "avg_cas"]
                         Airspeed / Mach metrics.  NOTE: these are 0-filled for
                         ~100% of real ADS-B tracks (see ml/features.py); their
                         contribution is expected to be negligible on real data.
                         On mock data (where they are also 0.0) the expectation
                         is the same.

  G6  aircraft_type    : all 27 "ac_*" one-hot columns
                         Aircraft type identity — captures airframe-level fuel
                         burn differences.

Ablation protocol
-----------------
For each group G:
  - Remove G's columns from the feature matrix (set them to constant 0.0 OR
    slice them out).  We ZERO-OUT the columns rather than slicing so the MLP
    input_dim stays fixed at INPUT_DIM=38 (required because model.py's weights
    are fixed-size).  For HistGBR the columns are dropped entirely (sklearn is
    not input-dim-sensitive).

  - Run chronological grouped CV (same build_folds logic) on the ablated feature
    set.

  - Record RMSE mean/std vs full-feature baseline.

Delta is defined as: ablated_RMSE - baseline_RMSE
  Positive delta  → group removal HURTS performance (group is important).
  Negative delta  → removing the group HELPS (redundant or noisy feature).

How to run
----------
# On mock data (offline, no real PRC data needed):
python -m ml.ablation --mock --epochs 5 --output-dir outputs/ablation

# On real data:
python -m ml.ablation --data-dir data/raw/prc_2025 --epochs 10 --output-dir outputs/ablation

Outputs
-------
  outputs/ablation/ablation_results.json  — machine-readable full results
  outputs/ablation/ablation_table.csv     — ablation table (CSV)
  outputs/ablation/ablation_table.md      — ablation table (Markdown, stdout-ready)
  outputs/ablation/challenger_results.json — MLP vs HistGBR comparison

Authoritative numbers
---------------------
On MOCK data: absolute RMSE values are meaningless as reference — mock fuel_kg
is generated with a different distribution than real PRC-2025 data.  The
ablation DELTAS and rank ordering of groups is the informative signal even on
mock data.

On REAL PRC-2025 data via `ml/cv.py`: the authoritative MLP baseline is
CV RMSE 442.65 ± 36.36 kg (ml-03).  The HistGBR challenger result and the
ablation deltas reported against that baseline are the canonical numbers.

All outputs from this script clearly label whether they come from mock or real
data.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, Subset

from ml.cv import build_folds, _rmse
from ml.features import FEATURE_COLUMNS, INPUT_DIM, AIRCRAFT_TYPES, encode_aircraft_type
from ml.model import FuelBurnMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO  [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("ml.ablation")


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

#: Ordered list of (group_name, group_description, list_of_columns) triples.
#: Every column must appear in FEATURE_COLUMNS.
FEATURE_GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "G1_duration",
        "Interval duration (duration_s)",
        ["duration_s"],
    ),
    (
        "G2_altitude",
        "Altitude profile (alt_change, avg_altitude, max_altitude, alt_std)",
        ["alt_change", "avg_altitude", "max_altitude", "alt_std"],
    ),
    (
        "G3_speed_vrate",
        "Speed & vertical rate (avg_speed, max_vrate)",
        ["avg_speed", "max_vrate"],
    ),
    (
        "G4_track",
        "Track / turning intensity (avg_track_change)",
        ["avg_track_change"],
    ),
    (
        "G5_mach_tas_cas",
        "Airspeed metrics — 0-filled in ADS-B (avg_mach, avg_tas, avg_cas)",
        ["avg_mach", "avg_tas", "avg_cas"],
    ),
    (
        "G6_aircraft_type",
        "Aircraft-type one-hot (all 27 ac_* columns)",
        [f"ac_{t}" for t in AIRCRAFT_TYPES],
    ),
]

# Validate group membership at import time: every named column must be in
# FEATURE_COLUMNS and the union of all groups must cover FEATURE_COLUMNS.
_ALL_GROUP_COLS: set[str] = set()
for _gname, _gdesc, _gcols in FEATURE_GROUPS:
    for _c in _gcols:
        assert _c in FEATURE_COLUMNS, (
            f"Column '{_c}' in group '{_gname}' is not in FEATURE_COLUMNS"
        )
        _ALL_GROUP_COLS.add(_c)

_UNCOVERED = set(FEATURE_COLUMNS) - _ALL_GROUP_COLS
assert len(_UNCOVERED) == 0, (
    f"Feature columns not covered by any group: {sorted(_UNCOVERED)}"
)


# ---------------------------------------------------------------------------
# Inline mock-data fixture (multi-date, no filesystem dependency)
# ---------------------------------------------------------------------------

def _make_mock_df_merged(seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal but structurally valid df_merged suitable for CV.

    Generates synthetic features + fuel_kg for 30 flights spread across
    3 flight_dates so that build_folds produces 2 expanding-window folds.
    All values are plausible (not zero-only) so that models can learn a
    non-trivial signal.
    """
    import datetime

    rng = np.random.default_rng(seed)
    dates = [
        datetime.date(2025, 4, 13),
        datetime.date(2025, 4, 14),
        datetime.date(2025, 4, 15),
    ]
    ac_types = ["A320", "B738", "A359", "B77W"]

    rows = []
    flight_counter = 0
    for d in dates:
        for _ in range(10):  # 10 flights per date
            flight_id = f"mock_{flight_counter:04d}"
            flight_counter += 1
            ac_type = rng.choice(ac_types)
            ac_onehot = encode_aircraft_type(ac_type)
            ac_onehot_dict = {f"ac_{t}": v for t, v in zip(AIRCRAFT_TYPES, ac_onehot)}

            # Generate 2-4 intervals per flight
            n_intervals = rng.integers(2, 5)
            base_duration = rng.uniform(800, 2000)
            for _i in range(n_intervals):
                duration_s = base_duration + rng.normal(0, 100)
                avg_alt = rng.uniform(10000, 38000)
                row = {
                    "flight_id": flight_id,
                    "flight_date": d,
                    "aircraft_type": ac_type,
                    "duration_s": max(300.0, duration_s),
                    "alt_change": rng.uniform(-5000, 5000),
                    "avg_speed": rng.uniform(150, 500),
                    "max_vrate": rng.uniform(0, 3000),
                    "avg_altitude": avg_alt,
                    "max_altitude": avg_alt + rng.uniform(0, 2000),
                    "alt_std": rng.uniform(0, 3000),
                    "avg_track_change": rng.uniform(0, 5),
                    "avg_mach": 0.0,  # always 0.0 (NaN-filled)
                    "avg_tas": 0.0,
                    "avg_cas": 0.0,
                    # fuel is a linear combination of features + noise
                    # this gives the model a learnable signal
                    "fuel_kg": max(
                        50.0,
                        duration_s * 0.6
                        + avg_alt * 0.002
                        + rng.normal(0, 30),
                    ),
                }
                row.update(ac_onehot_dict)
                rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Shared dataset wrapper (mirrors _MergedDataset in cv.py)
# ---------------------------------------------------------------------------

class _AblationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that reads from an in-memory DataFrame and supports
    per-column zeroing for ablation experiments.

    Parameters
    ----------
    df            : merged DataFrame with all FEATURE_COLUMNS + fuel_kg
    zeroed_cols   : set of column names to zero out (ablation mode)
    """

    def __init__(self, df: pd.DataFrame, zeroed_cols: set[str] | None = None):
        self._df = df.reset_index(drop=True)
        self._zeroed = zeroed_cols or set()
        # Pre-fill missing feature columns with 0.0
        for col in FEATURE_COLUMNS:
            if col not in self._df.columns:
                self._df[col] = 0.0

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        feat_vals = []
        for col in FEATURE_COLUMNS:
            feat_vals.append(0.0 if col in self._zeroed else float(row[col]))
        features = torch.tensor(feat_vals, dtype=torch.float32)
        target = torch.tensor(float(row["fuel_kg"]), dtype=torch.float32)
        return features, target


# ---------------------------------------------------------------------------
# MLP training helpers (self-contained, no MLflow)
# ---------------------------------------------------------------------------

def _train_mlp_fold(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    zeroed_cols: set[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
) -> float:
    """Train one MLP fold and return the best validation RMSE."""
    torch.manual_seed(seed)
    device = torch.device("cpu")

    full_ds = _AblationDataset(df, zeroed_cols)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size)

    model = FuelBurnMLP(input_dim=INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None

    for _epoch in range(1, epochs + 1):
        # Train
        model.train()
        for feats, tgt in train_loader:
            feats, tgt = feats.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), tgt)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        preds_list, tgt_list = [], []
        with torch.no_grad():
            for feats, tgt in val_loader:
                feats = feats.to(device)
                preds_list.append(model(feats).cpu().numpy())
                tgt_list.append(tgt.numpy())
        preds = np.concatenate(preds_list)
        targets = np.concatenate(tgt_list)
        fold_rmse = _rmse(preds, targets)
        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_state = copy.deepcopy(model.state_dict())

    return best_rmse


def run_mlp_cv(
    df: pd.DataFrame,
    zeroed_cols: set[str] | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_folds: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run chronological expanding-window CV for the MLP on df_merged.

    Parameters
    ----------
    df            : merged DataFrame with FEATURE_COLUMNS + flight_id +
                    flight_date + fuel_kg
    zeroed_cols   : columns to zero out (ablation mode); None = full features
    epochs        : training epochs per fold
    batch_size    : mini-batch size
    lr            : Adam learning rate
    n_folds       : cap on number of folds (None = all D-1 folds)
    seed          : random seed for reproducibility

    Returns
    -------
    dict with keys: fold_rmses, mean_rmse_kg, std_rmse_kg
    """
    torch.manual_seed(seed)
    folds = build_folds(df, n_folds=n_folds)
    fold_rmses = []

    for fold_info in folds:
        rmse = _train_mlp_fold(
            df=df,
            train_idx=fold_info["train_idx"],
            val_idx=fold_info["val_idx"],
            zeroed_cols=zeroed_cols or set(),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
        fold_rmses.append(rmse)
        logger.debug(f"  MLP fold {fold_info['fold']}: RMSE = {rmse:.2f} kg")

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses, ddof=0))
    return {
        "fold_rmses": fold_rmses,
        "mean_rmse_kg": mean_rmse,
        "std_rmse_kg": std_rmse,
    }


# ---------------------------------------------------------------------------
# HistGBR CV (sklearn, no MLflow, supports column-drop ablation)
# ---------------------------------------------------------------------------

def _get_feature_matrix(
    df: pd.DataFrame,
    dropped_cols: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract the feature matrix from df, optionally dropping specified columns.

    Unlike MLP ablation (which zeros columns), HistGBR ablation DROPS columns
    entirely because sklearn estimators are not input-dim-sensitive.

    Returns (X, active_feature_names).
    """
    dropped = dropped_cols or set()
    active_cols = [c for c in FEATURE_COLUMNS if c not in dropped]

    # Fill missing columns with 0.0 (schema transition safety)
    X_df = df.reindex(columns=active_cols, fill_value=0.0)
    return X_df.values.astype(np.float64), active_cols


def run_histgbr_cv(
    df: pd.DataFrame,
    dropped_cols: set[str] | None = None,
    n_folds: int | None = None,
    seed: int = 42,
    **histgbr_kwargs,
) -> dict[str, Any]:
    """
    Run chronological expanding-window CV for HistGradientBoostingRegressor.

    Parameters
    ----------
    df            : merged DataFrame with FEATURE_COLUMNS + flight_id +
                    flight_date + fuel_kg
    dropped_cols  : columns to drop entirely (ablation mode); None = full features
    n_folds       : cap on number of folds
    seed          : random seed for HistGBR
    **histgbr_kwargs : extra keyword args passed to HistGradientBoostingRegressor

    Returns
    -------
    dict with keys: fold_rmses, mean_rmse_kg, std_rmse_kg, active_features
    """
    folds = build_folds(df, n_folds=n_folds)
    X, active_cols = _get_feature_matrix(df, dropped_cols)
    y = df["fuel_kg"].values.astype(np.float64)

    fold_rmses = []
    for fold_info in folds:
        train_idx = fold_info["train_idx"]
        val_idx = fold_info["val_idx"]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        gbr = HistGradientBoostingRegressor(
            random_state=seed,
            **histgbr_kwargs,
        )
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_val)
        fold_rmse = _rmse(preds, y_val)
        fold_rmses.append(fold_rmse)
        logger.debug(f"  HistGBR fold {fold_info['fold']}: RMSE = {fold_rmse:.2f} kg")

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses, ddof=0))
    return {
        "fold_rmses": fold_rmses,
        "mean_rmse_kg": mean_rmse,
        "std_rmse_kg": std_rmse,
        "active_features": active_cols,
    }


# ---------------------------------------------------------------------------
# Load df_merged from real data directory
# ---------------------------------------------------------------------------

def _load_df_merged(data_dir: str) -> pd.DataFrame:
    """
    Load and merge features_train.parquet + flightlist_train.parquet into
    the same df_merged format used by ml/cv.py.
    """
    data_path = Path(data_dir)
    features_path = data_path / "features_train.parquet"
    flightlist_path = data_path / "flightlist_train.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features_train.parquet not found at {features_path}. "
            "Run ml-extract-features first."
        )
    if not flightlist_path.exists():
        raise FileNotFoundError(
            f"flightlist_train.parquet not found at {flightlist_path}."
        )

    df_feat = pd.read_parquet(features_path)
    df_fl = pd.read_parquet(flightlist_path, columns=["flight_id", "flight_date", "aircraft_type"])

    # Drop aircraft_type from features if present (canon comes from flightlist)
    if "aircraft_type" in df_feat.columns:
        df_feat = df_feat.drop(columns=["aircraft_type"])

    df_merged = df_feat.merge(df_fl, on="flight_id", how="left")
    missing_date = df_merged["flight_date"].isna().sum()
    if missing_date > 0:
        logger.warning(f"{missing_date} intervals have no flight_date — dropping.")
        df_merged = df_merged.dropna(subset=["flight_date"])

    return df_merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    df: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_folds: int | None = None,
    seed: int = 42,
    model_type: str = "mlp",
) -> dict[str, Any]:
    """
    Run feature-group ablation for the specified model type.

    For each group in FEATURE_GROUPS, removes that group and measures the
    RMSE delta relative to the full-feature baseline.

    Parameters
    ----------
    df         : merged DataFrame (FEATURE_COLUMNS + flight_id + flight_date +
                 fuel_kg)
    epochs     : training epochs per fold (MLP only)
    batch_size : mini-batch size (MLP only)
    lr         : learning rate (MLP only)
    n_folds    : cap on number of folds
    seed       : random seed
    model_type : "mlp" or "histgbr"

    Returns
    -------
    dict with keys:
      baseline    : full-feature result dict (fold_rmses, mean_rmse_kg, std_rmse_kg)
      groups      : list of per-group dicts with ablation results + deltas
    """
    logger.info(f"Running ablation for model_type={model_type}")
    logger.info(
        f"Dataset: {len(df)} intervals, {df['flight_id'].nunique()} flights, "
        f"{df['flight_date'].nunique()} date(s)"
    )

    # --- Baseline (full features) ---
    logger.info("Computing baseline (full features)...")
    if model_type == "mlp":
        baseline = run_mlp_cv(
            df, zeroed_cols=None, epochs=epochs, batch_size=batch_size,
            lr=lr, n_folds=n_folds, seed=seed,
        )
    else:
        baseline = run_histgbr_cv(df, dropped_cols=None, n_folds=n_folds, seed=seed)

    logger.info(
        f"Baseline {model_type.upper()} RMSE: "
        f"{baseline['mean_rmse_kg']:.2f} ± {baseline['std_rmse_kg']:.2f} kg"
    )

    # --- Per-group ablation ---
    group_results = []
    for group_name, group_desc, group_cols in FEATURE_GROUPS:
        logger.info(f"Ablating group {group_name}: {group_cols}")
        if model_type == "mlp":
            result = run_mlp_cv(
                df, zeroed_cols=set(group_cols), epochs=epochs, batch_size=batch_size,
                lr=lr, n_folds=n_folds, seed=seed,
            )
        else:
            result = run_histgbr_cv(df, dropped_cols=set(group_cols), n_folds=n_folds, seed=seed)

        delta = result["mean_rmse_kg"] - baseline["mean_rmse_kg"]
        logger.info(
            f"  {group_name}: RMSE {result['mean_rmse_kg']:.2f} ± {result['std_rmse_kg']:.2f} kg "
            f"(delta: {delta:+.2f} kg)"
        )

        group_results.append({
            "group_name": group_name,
            "group_description": group_desc,
            "columns_removed": group_cols,
            "n_cols_removed": len(group_cols),
            "mean_rmse_kg": result["mean_rmse_kg"],
            "std_rmse_kg": result["std_rmse_kg"],
            "fold_rmses": result["fold_rmses"],
            "delta_rmse_kg": delta,
        })

    return {"baseline": baseline, "groups": group_results}


# ---------------------------------------------------------------------------
# Challenger: MLP vs HistGBR head-to-head
# ---------------------------------------------------------------------------

def run_challenger_comparison(
    df: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_folds: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Head-to-head comparison of MLP vs HistGradientBoostingRegressor.

    Both models use the same chronological expanding-window CV folds,
    full feature set (no ablation), and shared random seed.

    Returns
    -------
    dict with keys: mlp, histgbr, winner, delta_rmse_kg
    """
    logger.info("Running MLP vs HistGBR challenger comparison...")

    mlp_result = run_mlp_cv(
        df, zeroed_cols=None, epochs=epochs, batch_size=batch_size,
        lr=lr, n_folds=n_folds, seed=seed,
    )
    logger.info(
        f"MLP CV RMSE: {mlp_result['mean_rmse_kg']:.2f} ± {mlp_result['std_rmse_kg']:.2f} kg"
    )

    histgbr_result = run_histgbr_cv(df, dropped_cols=None, n_folds=n_folds, seed=seed)
    logger.info(
        f"HistGBR CV RMSE: {histgbr_result['mean_rmse_kg']:.2f} ± {histgbr_result['std_rmse_kg']:.2f} kg"
    )

    delta = histgbr_result["mean_rmse_kg"] - mlp_result["mean_rmse_kg"]
    if delta < 0:
        winner = "HistGBR"
        winner_margin = f"HistGBR better by {abs(delta):.2f} kg RMSE"
    elif delta > 0:
        winner = "MLP"
        winner_margin = f"MLP better by {abs(delta):.2f} kg RMSE"
    else:
        winner = "tie"
        winner_margin = "identical RMSE"

    return {
        "mlp": mlp_result,
        "histgbr": histgbr_result,
        "winner": winner,
        "delta_rmse_kg": delta,
        "winner_margin": winner_margin,
    }


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def format_ablation_table_md(
    ablation_result: dict[str, Any],
    model_type: str,
    data_label: str,
) -> str:
    """
    Render the ablation results as a Markdown table.

    Parameters
    ----------
    ablation_result : output of run_ablation()
    model_type      : "mlp" or "histgbr"
    data_label      : "mock" or "real"
    """
    baseline = ablation_result["baseline"]
    groups = ablation_result["groups"]

    # Sort by delta descending (most important group first)
    sorted_groups = sorted(groups, key=lambda r: r["delta_rmse_kg"], reverse=True)

    lines = [
        f"## Feature-Group Ablation — {model_type.upper()} ({data_label} data)",
        "",
        f"Baseline (full features): **{baseline['mean_rmse_kg']:.2f} ± {baseline['std_rmse_kg']:.2f} kg RMSE**",
        "",
        "NOTE: On mock data, absolute RMSE values reflect synthetic fuel_kg distribution,",
        "not the real PRC-2025 distribution. Authoritative numbers require real data.",
        "Delta direction: positive = removing group hurts (group is important).",
        "",
        "| # | Group | Description | Cols Removed | RMSE (kg) | Std (kg) | Delta vs Baseline |",
        "|---|-------|-------------|:---:|-----------|----------|-------------------|",
    ]

    for i, row in enumerate(sorted_groups, 1):
        delta_str = f"{row['delta_rmse_kg']:+.2f}"
        lines.append(
            f"| {i} | `{row['group_name']}` | {row['group_description']} "
            f"| {row['n_cols_removed']} "
            f"| {row['mean_rmse_kg']:.2f} "
            f"| {row['std_rmse_kg']:.2f} "
            f"| **{delta_str}** |"
        )

    return "\n".join(lines)


def format_challenger_table_md(
    comparison: dict[str, Any],
    data_label: str,
) -> str:
    """Render the MLP vs HistGBR challenger comparison as Markdown."""
    mlp = comparison["mlp"]
    histgbr = comparison["histgbr"]

    per_fold_mlp = " / ".join(f"{r:.2f}" for r in mlp["fold_rmses"])
    per_fold_gbr = " / ".join(f"{r:.2f}" for r in histgbr["fold_rmses"])

    lines = [
        f"## MLP vs HistGBR Challenger Comparison ({data_label} data)",
        "",
        f"NOTE: On mock data, absolute RMSE values reflect synthetic distribution.",
        "Authoritative comparison requires real PRC-2025 data via `ml/cv.py`.",
        "",
        "| Model | CV RMSE (kg) | Std (kg) | Per-fold RMSE (kg) |",
        "|-------|-------------|----------|-------------------|",
        f"| MLP (baseline) | {mlp['mean_rmse_kg']:.2f} | {mlp['std_rmse_kg']:.2f} | {per_fold_mlp} |",
        f"| HistGBR (challenger) | {histgbr['mean_rmse_kg']:.2f} | {histgbr['std_rmse_kg']:.2f} | {per_fold_gbr} |",
        "",
        f"**Winner: {comparison['winner']}** — {comparison['winner_margin']}",
        f"(Delta HistGBR - MLP: {comparison['delta_rmse_kg']:+.2f} kg RMSE; "
        "negative = HistGBR wins)",
    ]
    return "\n".join(lines)


def ablation_to_df(ablation_result: dict[str, Any]) -> pd.DataFrame:
    """Convert ablation results to a tidy DataFrame for CSV export."""
    baseline = ablation_result["baseline"]
    rows = []
    for row in ablation_result["groups"]:
        rows.append({
            "group_name": row["group_name"],
            "group_description": row["group_description"],
            "n_cols_removed": row["n_cols_removed"],
            "columns_removed": ", ".join(row["columns_removed"]),
            "baseline_mean_rmse_kg": baseline["mean_rmse_kg"],
            "baseline_std_rmse_kg": baseline["std_rmse_kg"],
            "ablated_mean_rmse_kg": row["mean_rmse_kg"],
            "ablated_std_rmse_kg": row["std_rmse_kg"],
            "delta_rmse_kg": row["delta_rmse_kg"],
        })
    return pd.DataFrame(rows).sort_values("delta_rmse_kg", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Feature-group ablation + HistGBR challenger for the fuel-burn model. "
            "Runs offline on mock data with --mock."
        )
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--mock",
        action="store_true",
        help="Use built-in multi-date mock fixture (no real data needed).",
    )
    src_group.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory containing features_train.parquet + "
        "flightlist_train.parquet (real PRC-2025 data).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="MLP training epochs per fold (default: 5 for offline speed).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="MLP mini-batch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="MLP Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Cap the number of CV folds (default: all D-1 folds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation",
        help="Directory for output files (default: outputs/ablation).",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "histgbr", "both"],
        default="both",
        help="Which model to use for ablation (default: both).",
    )
    args = parser.parse_args()

    # --- Load data ---
    if args.mock:
        logger.info("Using built-in mock fixture (--mock)")
        df = _make_mock_df_merged(seed=args.seed)
        data_label = "mock"
        logger.info(
            f"Mock dataset: {len(df)} intervals, {df['flight_id'].nunique()} flights, "
            f"{df['flight_date'].nunique()} dates"
        )
    else:
        logger.info(f"Loading real data from {args.data_dir}")
        df = _load_df_merged(args.data_dir)
        data_label = "real"
        logger.info(
            f"Real dataset: {len(df)} intervals, {df['flight_id'].nunique()} flights, "
            f"{df['flight_date'].nunique()} dates"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Challenger comparison (MLP vs HistGBR) ---
    logger.info("\n" + "=" * 60)
    logger.info("HEAD-TO-HEAD: MLP vs HistGBR")
    logger.info("=" * 60)

    comparison = run_challenger_comparison(
        df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    challenger_md = format_challenger_table_md(comparison, data_label)
    print("\n" + challenger_md)

    # Save challenger results
    challenger_json_path = out_dir / "challenger_results.json"
    with open(challenger_json_path, "w") as f:
        json.dump(
            {
                "data_label": data_label,
                "mlp": {
                    k: v for k, v in comparison["mlp"].items()
                    if k != "active_features"
                },
                "histgbr": {
                    k: v for k, v in comparison["histgbr"].items()
                    if k != "active_features"
                },
                "winner": comparison["winner"],
                "delta_rmse_kg": comparison["delta_rmse_kg"],
                "winner_margin": comparison["winner_margin"],
                "note": (
                    "Mock data only — authoritative comparison requires real PRC-2025 "
                    "data via `python -m ml.ablation --data-dir <path>`"
                    if data_label == "mock" else
                    "Real PRC-2025 data. Compare MLP baseline against cv.py RMSE 442.65 ± 36.36 kg."
                ),
            },
            f,
            indent=2,
        )
    logger.info(f"Challenger results saved to {challenger_json_path}")

    # --- Ablation ---
    model_types_to_run = []
    if args.model == "both":
        model_types_to_run = ["mlp", "histgbr"]
    else:
        model_types_to_run = [args.model]

    all_ablation_results = {}
    for model_type in model_types_to_run:
        logger.info("\n" + "=" * 60)
        logger.info(f"ABLATION: {model_type.upper()}")
        logger.info("=" * 60)

        abl_result = run_ablation(
            df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_folds=args.n_folds,
            seed=args.seed,
            model_type=model_type,
        )
        all_ablation_results[model_type] = abl_result

        # Print markdown table
        md_table = format_ablation_table_md(abl_result, model_type, data_label)
        print("\n" + md_table)

        # Save CSV
        df_abl = ablation_to_df(abl_result)
        csv_path = out_dir / f"ablation_table_{model_type}.csv"
        df_abl.to_csv(csv_path, index=False)
        logger.info(f"Ablation CSV saved to {csv_path}")

        # Save Markdown
        md_path = out_dir / f"ablation_table_{model_type}.md"
        with open(md_path, "w") as f:
            f.write(md_table + "\n")
        logger.info(f"Ablation Markdown saved to {md_path}")

    # Save consolidated JSON
    json_out = {
        "data_label": data_label,
        "feature_groups": [
            {
                "name": g[0],
                "description": g[1],
                "columns": g[2],
            }
            for g in FEATURE_GROUPS
        ],
    }
    for model_type, abl_result in all_ablation_results.items():
        json_out[f"{model_type}_ablation"] = {
            "baseline": {
                "mean_rmse_kg": abl_result["baseline"]["mean_rmse_kg"],
                "std_rmse_kg": abl_result["baseline"]["std_rmse_kg"],
                "fold_rmses": abl_result["baseline"]["fold_rmses"],
            },
            "groups": abl_result["groups"],
        }

    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"Ablation JSON saved to {json_path}")

    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)
    if data_label == "mock":
        print(
            "NOTE: These numbers are from MOCK data. "
            "Absolute RMSE values do not match the PRC-2025 baseline of 442.65 ± 36.36 kg. "
            "Run with --data-dir <real-data-path> for authoritative numbers."
        )
    else:
        print(
            "Authoritative comparison against: MLP baseline 442.65 ± 36.36 kg "
            "(ml-03 / cv.py on real PRC-2025 data)"
        )
    print(f"Outputs written to: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
