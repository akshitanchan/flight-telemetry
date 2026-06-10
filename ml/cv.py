"""
Chronological cross-validation harness for the fuel-burn MLP.

Protocol — expanding-window, grouped by flight_id, ordered by flight_date
--------------------------------------------------------------------------
Given D unique flight dates sorted chronologically, produce (D-1) folds:

  Fold 1 : train = dates[0..0],       val = dates[1]
  Fold 2 : train = dates[0..1],       val = dates[2]
  ...
  Fold k : train = dates[0..k-1],     val = dates[k]

"Expanding window" means each successive training set is strictly a superset
of the previous one, matching how a production system accumulates history.

Leakage guarantee
-----------------
Every interval is assigned to exactly one date bucket (the flight_date of its
parent flight_id).  Because flight_date comes from `flightlist_train.parquet`
and is joined on `flight_id`, a flight never crosses a date boundary.  An
assertion at fold construction time verifies that no flight_id appears in both
the training set and the validation set of any fold.

Metrics
-------
Primary : RMSE in kg  (official PRC-2025 metric)
Reported : per-fold RMSE, aggregate mean ± std
Slices   : per aircraft_type, per duration bucket
All results are logged to MLflow (experiment FuelBurn_Baseline).
"""

import argparse
import copy
import logging
import math
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ml.dataset import FuelBurnDataset
from ml.model import FuelBurnMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO  [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("ml.cv")


# ---------------------------------------------------------------------------
# Training helpers (mirrored from ml/train.py — kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            total_loss += criterion(model(features), target).item() * features.size(0)
    return total_loss / len(loader.dataset)


def _collect_predictions(model, loader, device):
    """Return (preds, targets) as numpy arrays."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            preds.append(model(features).cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def build_folds(df_merged: pd.DataFrame, n_folds: int | None = None):
    """
    Return a list of (train_indices, val_indices, val_date) tuples.

    `df_merged` must have columns: interval index (row position),
    flight_id, flight_date (date objects, sortable).

    `n_folds` caps the number of folds from the end; None means use all
    (D-1) possible expanding folds.
    """
    dates = sorted(df_merged["flight_date"].unique())
    if len(dates) < 2:
        raise ValueError(
            f"Need at least 2 distinct flight_dates to build CV folds; got {len(dates)}."
        )

    # Maximum folds = len(dates) - 1 (first date is always training-only)
    max_folds = len(dates) - 1
    if n_folds is not None:
        if n_folds < 1 or n_folds > max_folds:
            raise ValueError(
                f"n_folds must be between 1 and {max_folds}; got {n_folds}."
            )
        # Use the last `n_folds` splits (largest training sets)
        start_fold = max_folds - n_folds
    else:
        start_fold = 0

    folds = []
    for k in range(start_fold, max_folds):
        train_dates = dates[: k + 1]
        val_date = dates[k + 1]

        train_mask = df_merged["flight_date"].isin(set(train_dates))
        val_mask = df_merged["flight_date"] == val_date

        train_idx = df_merged.index[train_mask].tolist()
        val_idx = df_merged.index[val_mask].tolist()

        # --- Leakage assertion ---
        train_flights = set(df_merged.loc[train_mask, "flight_id"])
        val_flights = set(df_merged.loc[val_mask, "flight_id"])
        leaked = train_flights & val_flights
        assert len(leaked) == 0, (
            f"Fold {k+1}: {len(leaked)} flight_id(s) appear in both train and val: "
            f"{list(leaked)[:5]}"
        )

        folds.append(
            {
                "fold": k - start_fold + 1,
                "train_dates": [str(d) for d in train_dates],
                "val_date": str(val_date),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_train_intervals": len(train_idx),
                "n_val_intervals": len(val_idx),
                "n_train_flights": len(train_flights),
                "n_val_flights": len(val_flights),
            }
        )

    return folds


# ---------------------------------------------------------------------------
# Per-slice RMSE
# ---------------------------------------------------------------------------

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_slices(preds, targets, df_val_meta):
    """
    Compute per-slice RMSE.

    Parameters
    ----------
    preds, targets : np.ndarray  (N,)
    df_val_meta    : DataFrame aligned with preds/targets, containing
                     'aircraft_type' and 'duration_s'.

    Returns
    -------
    dict with keys 'by_aircraft_type' and 'by_duration_bucket', each a
    DataFrame with columns [slice_label, n, rmse_kg].
    """
    df = df_val_meta.copy().reset_index(drop=True)
    df["pred"] = preds
    df["target"] = targets

    # --- Aircraft type slice ---
    ac_rows = []
    for ac_type, grp in df.groupby("aircraft_type"):
        ac_rows.append(
            {
                "aircraft_type": ac_type,
                "n": len(grp),
                "rmse_kg": _rmse(grp["pred"].values, grp["target"].values),
            }
        )
    df_ac = pd.DataFrame(ac_rows).sort_values("rmse_kg", ascending=False).reset_index(drop=True)

    # --- Duration bucket slice ---
    bins = [0, 1800, 3600, 7200, 18000, np.inf]
    labels = ["<30min", "30-60min", "1-2h", "2-5h", ">5h"]
    df["duration_bucket"] = pd.cut(
        df["duration_s"], bins=bins, labels=labels, right=False
    )
    dur_rows = []
    for bucket, grp in df.groupby("duration_bucket", observed=True):
        if len(grp) == 0:
            continue
        dur_rows.append(
            {
                "duration_bucket": str(bucket),
                "n": len(grp),
                "rmse_kg": _rmse(grp["pred"].values, grp["target"].values),
            }
        )
    df_dur = (
        pd.DataFrame(dur_rows)
        .sort_values("duration_bucket")
        .reset_index(drop=True)
    )

    return {"by_aircraft_type": df_ac, "by_duration_bucket": df_dur}


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_cv(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_folds: int | None = None,
):
    torch.manual_seed(42)

    data_path = Path(data_dir)
    features_path = data_path / "features_train.parquet"
    flightlist_path = data_path / "flightlist_train.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found at {features_path}. "
            "Run `make ml-baseline-real` (or ml-extract-features) first."
        )
    if not flightlist_path.exists():
        raise FileNotFoundError(
            f"flightlist_train.parquet not found at {flightlist_path}."
        )

    logger.info("Loading features and flightlist …")
    df_feat = pd.read_parquet(features_path)
    df_fl = pd.read_parquet(
        flightlist_path,
        columns=["flight_id", "flight_date", "aircraft_type"],
    )

    # Join chronological metadata onto the interval rows.
    # flight_date is a Python date object (object dtype in parquet); keep as-is
    # for sorting — comparison operators work correctly.
    df_merged = df_feat.merge(df_fl, on="flight_id", how="left")

    missing_date = df_merged["flight_date"].isna().sum()
    if missing_date > 0:
        logger.warning(
            f"{missing_date} intervals have no matching flight_date in flightlist; "
            "they will be dropped."
        )
        df_merged = df_merged.dropna(subset=["flight_date"])

    # Reset integer index so .iloc lookup in Subset is contiguous.
    df_merged = df_merged.reset_index(drop=True)

    logger.info(
        f"Dataset: {len(df_merged)} intervals, "
        f"{df_merged['flight_id'].nunique()} flights, "
        f"{df_merged['flight_date'].nunique()} date(s)"
    )

    folds = build_folds(df_merged, n_folds=n_folds)
    logger.info(
        f"CV protocol: expanding-window, {len(folds)} fold(s) over "
        f"{df_merged['flight_date'].nunique()} date blocks"
    )
    for f in folds:
        logger.info(
            f"  Fold {f['fold']}: train={f['train_dates']} "
            f"({f['n_train_intervals']} intervals, {f['n_train_flights']} flights) "
            f"→ val={f['val_date']} "
            f"({f['n_val_intervals']} intervals, {f['n_val_flights']} flights)"
        )

    device = torch.device("cpu")

    # We wrap the entire dataset in a FuelBurnDataset for __getitem__ but index
    # into it with per-fold Subset objects.  The dataset reads df_merged via
    # path; we need a dataset whose internal DataFrame is df_merged (already
    # merged).  We subclass temporarily to avoid touching dataset.py.
    class _MergedDataset(torch.utils.data.Dataset):
        """Thin wrapper so fold Subsets can index df_merged directly."""
        def __init__(self, df):
            self._df = df.reset_index(drop=True)

        def __len__(self):
            return len(self._df)

        def __getitem__(self, idx):
            row = self._df.iloc[idx]
            features = torch.tensor(
                [row["duration_s"], row["alt_change"], row["avg_speed"], row["max_vrate"]],
                dtype=torch.float32,
            )
            target = torch.tensor(row["fuel_kg"], dtype=torch.float32)
            return features, target

    full_ds = _MergedDataset(df_merged)

    criterion = nn.MSELoss()
    fold_rmses = []

    # Aggregate slices across all folds (append then average at the end)
    all_preds = []
    all_targets = []
    all_val_meta = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("FuelBurn_Baseline")

    with mlflow.start_run(run_name="ChronologicalCV") as parent_run:
        mlflow.log_params(
            {
                "cv_protocol": "expanding_window",
                "n_folds": len(folds),
                "epochs_per_fold": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "model_type": "MLP_Baseline",
                "torch_seed": 42,
                "date_range": f"{folds[0]['train_dates'][0]} – {folds[-1]['val_date']}",
            }
        )

        for fold_info in folds:
            fold_num = fold_info["fold"]
            logger.info(f"\n{'='*60}")
            logger.info(f"  Fold {fold_num}/{len(folds)}")
            logger.info(f"{'='*60}")

            train_idx = fold_info["train_idx"]
            val_idx = fold_info["val_idx"]

            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            # Fresh model per fold — each fold is an independent experiment.
            # torch.manual_seed(42) at top of run_cv ensures identical weight
            # initialisation across folds so any RMSE difference is due to
            # data, not random weight variation.
            torch.manual_seed(42)
            model = FuelBurnMLP().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_rmse = float("inf")
            best_state = None

            with mlflow.start_run(
                run_name=f"fold_{fold_num}", nested=True
            ) as fold_run:
                mlflow.log_params(
                    {
                        "fold": fold_num,
                        "train_dates": str(fold_info["train_dates"]),
                        "val_date": fold_info["val_date"],
                        "n_train_intervals": fold_info["n_train_intervals"],
                        "n_val_intervals": fold_info["n_val_intervals"],
                        "n_train_flights": fold_info["n_train_flights"],
                        "n_val_flights": fold_info["n_val_flights"],
                    }
                )

                for epoch in range(1, epochs + 1):
                    train_mse = _train_epoch(model, train_loader, criterion, optimizer, device)
                    val_mse = _evaluate(model, val_loader, criterion, device)
                    t_rmse = math.sqrt(train_mse)
                    v_rmse = math.sqrt(val_mse)

                    mlflow.log_metrics(
                        {"train_rmse": t_rmse, "val_rmse": v_rmse}, step=epoch
                    )

                    if v_rmse < best_val_rmse:
                        best_val_rmse = v_rmse
                        best_state = copy.deepcopy(model.state_dict())

                    if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                        logger.info(
                            f"  [Fold {fold_num}] Epoch {epoch:03d} | "
                            f"Train RMSE: {t_rmse:.2f} | Val RMSE: {v_rmse:.2f}"
                        )

                mlflow.log_metric("best_val_rmse_kg", best_val_rmse)
                logger.info(
                    f"  [Fold {fold_num}] Best Val RMSE: {best_val_rmse:.2f} kg"
                )

            # Restore best weights then collect predictions for slice analysis
            if best_state is not None:
                model.load_state_dict(best_state)

            fold_preds, fold_targets = _collect_predictions(model, val_loader, device)
            fold_rmse = _rmse(fold_preds, fold_targets)
            fold_rmses.append(fold_rmse)

            # Accumulate for aggregate slice computation
            all_preds.append(fold_preds)
            all_targets.append(fold_targets)
            val_meta = df_merged.iloc[val_idx][
                ["aircraft_type", "duration_s"]
            ].reset_index(drop=True)
            all_val_meta.append(val_meta)

        # ---------------------------------------------------------------
        # Aggregate statistics
        # ---------------------------------------------------------------
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses, ddof=0))  # population std across folds

        logger.info(f"\n{'='*60}")
        logger.info("  CHRONOLOGICAL CV RESULTS")
        logger.info(f"{'='*60}")
        for i, rmse in enumerate(fold_rmses, 1):
            logger.info(f"  Fold {i}: {rmse:.2f} kg")
        logger.info(f"  Aggregate: {mean_rmse:.2f} ± {std_rmse:.2f} kg (mean ± std)")

        mlflow.log_metrics(
            {
                "cv_mean_rmse_kg": mean_rmse,
                "cv_std_rmse_kg": std_rmse,
                **{f"cv_fold_{i+1}_rmse_kg": r for i, r in enumerate(fold_rmses)},
            }
        )

        # ---------------------------------------------------------------
        # Per-slice RMSE across all folds combined
        # ---------------------------------------------------------------
        all_preds_cat = np.concatenate(all_preds)
        all_targets_cat = np.concatenate(all_targets)
        all_meta_cat = pd.concat(all_val_meta, ignore_index=True)

        slices = compute_slices(all_preds_cat, all_targets_cat, all_meta_cat)

        df_ac = slices["by_aircraft_type"]
        df_dur = slices["by_duration_bucket"]

        logger.info("\n  Per-aircraft-type RMSE (all folds combined):")
        logger.info(df_ac.to_string(index=False))

        logger.info("\n  Per-duration-bucket RMSE (all folds combined):")
        logger.info(df_dur.to_string(index=False))

        # Log slice tables as MLflow artifacts (CSV text)
        # mlflow.log_text expects (text, artifact_file)
        mlflow.log_text(df_ac.to_csv(index=False), "slice_by_aircraft_type.csv")
        mlflow.log_text(df_dur.to_csv(index=False), "slice_by_duration_bucket.csv")

        # Also log per-slice RMSE as individual metrics so they surface in the
        # MLflow UI comparison pane.
        for _, row in df_ac.iterrows():
            safe_name = row["aircraft_type"].replace("-", "_").replace("/", "_")
            mlflow.log_metric(f"slice_rmse_{safe_name}_kg", row["rmse_kg"])
        for _, row in df_dur.iterrows():
            safe_name = (
                row["duration_bucket"]
                .replace("<", "lt")
                .replace(">", "gt")
                .replace("-", "_")
                .replace(" ", "")
            )
            mlflow.log_metric(f"slice_dur_{safe_name}_rmse_kg", row["rmse_kg"])

        logger.info(f"\n  Run ID: {parent_run.info.run_id}")

    return {
        "fold_rmses": fold_rmses,
        "mean_rmse_kg": mean_rmse,
        "std_rmse_kg": std_rmse,
        "slices": slices,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chronological CV for the fuel-burn MLP (PRC-2025 RMSE metric)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the data directory that contains features_train.parquet "
        "and flightlist_train.parquet",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per fold (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Cap the number of CV folds (default: all D-1 possible folds). "
        "When specified, the last N folds (largest training sets) are used.",
    )
    args = parser.parse_args()

    results = run_cv(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.n_folds,
    )

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL CV SUMMARY")
    print("=" * 60)
    for i, rmse in enumerate(results["fold_rmses"], 1):
        print(f"  Fold {i}: {rmse:.2f} kg")
    print(f"\n  CV RMSE: {results['mean_rmse_kg']:.2f} ± {results['std_rmse_kg']:.2f} kg")
    print()
    print("  Per-aircraft-type RMSE (all folds):")
    print(results["slices"]["by_aircraft_type"].to_string(index=False))
    print()
    print("  Per-duration-bucket RMSE (all folds):")
    print(results["slices"]["by_duration_bucket"].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
