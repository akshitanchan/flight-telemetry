"""
ml/score_rank.py — PRC-2025 rank-phase scoring harness.
=====================================================================

Loads the production fuel-burn model, predicts on the PRC-2025 rank phase,
computes the official per-interval RMSE in kg against the TRUE rank labels
(fuel_rank.parquet), and optionally compares it with a user-supplied reference
measured on the same rank split.

Official metric
---------------
Per-interval RMSE in kg  (the PRC-2025 challenge metric).
  RMSE = sqrt( mean( (predicted_kg - true_kg)^2 ) )
Lower is better.

Data contract
-------------
The TRUE rank labels live in:
  <data-dir>/fuel_rank.parquet   columns: idx, flight_id, start, end, fuel_kg
Pre-extracted features are read from:
  <data-dir>/features_rank.parquet
If features_rank.parquet does not exist, the harness will call
ml.extract_features.extract_features() with split="rank" to generate it first.
Run python -m ml.extract_features --data-dir <dir> --split rank beforehand
for full control over extraction (checkpointing, --limit, etc.).

Feature-assembly guarantee (no train/score skew)
-------------------------------------------------
The prediction tensor is built from FEATURE_COLUMNS (imported from
ml/features.py) in the EXACT same order used by FuelBurnDataset.__getitem__
during training.  The encode_aircraft_type() call is also identical to the
one made by extract_features.py at extraction time (aircraft_type column in
features_rank.parquet carries the raw string; the harness does NOT re-encode
at score time — it reads the pre-encoded one-hot columns directly from the
parquet to avoid any double-encoding risk).  FEATURE_COLUMNS is the single
source of truth for column ordering.

Model loading (mirrors serve.py priority)
-----------------------------------------
1. --model-path  → torch.load (offline-safe, no tracking server needed)
2. --model-uri   → mlflow.pytorch.load_model (default: models:/FuelBurn@production)
   The MLflow 3.x alias form is used; the stage form is deprecated.

Usage
-----
# Real run (owner, against production registry):
python -m ml.score_rank \\
    --data-dir data/raw/prc_2025 \\
    --model-uri models:/FuelBurn@production

# Offline test (local checkpoint, mock data):
python -m ml.score_rank \\
    --data-dir data/ml/prc_2025_mock \\
    --model-path /tmp/model.pth \\
    --baseline-rmse 120.0

Output
------
Prints a result block to stdout and writes two artefacts:
  <data-dir>/rank_score_result.json     — machine-readable
  <data-dir>/rank_score_result.md       — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ml.features import FEATURE_COLUMNS, INPUT_DIM
from ml.model import FuelBurnMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO  [%(name)s] %(message)s",
)
logger = logging.getLogger("ml.score_rank")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPLIT = "rank"
_RESULT_JSON = "rank_score_result.json"
_RESULT_MD = "rank_score_result.md"


# ---------------------------------------------------------------------------
# Model loading  (mirrors serve.py _load_model priority)
# ---------------------------------------------------------------------------

def _load_model(
    model_path: Optional[str] = None,
    model_uri: Optional[str] = None,
) -> FuelBurnMLP:
    """Load FuelBurnMLP from a local checkpoint or MLflow URI.

    Priority:
      1. model_path  → torch.load  (offline-safe)
      2. model_uri   → mlflow.pytorch.load_model
      3. Neither set → raises RuntimeError
    """
    if model_path:
        logger.info("Loading model from local path: %s", model_path)
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(obj, FuelBurnMLP):
            model = obj
        else:
            # State-dict checkpoint produced by some training runs.
            model = FuelBurnMLP(input_dim=INPUT_DIM)
            model.load_state_dict(obj)
        model.eval()
        logger.info("Model loaded from local path.")
        return model

    if model_uri:
        logger.info("Loading model from MLflow URI: %s", model_uri)
        try:
            import mlflow.pytorch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mlflow is not installed.  Install it or use --model-path instead."
            ) from exc
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        logger.info("Model loaded from MLflow URI.")
        return model  # type: ignore[return-value]

    raise RuntimeError(
        "No model source specified.  Provide --model-path or --model-uri."
    )


# ---------------------------------------------------------------------------
# Feature loading / extraction
# ---------------------------------------------------------------------------

def _ensure_features(data_dir: Path, split: str = _SPLIT) -> pd.DataFrame:
    """Return the features DataFrame for *split*, extracting if necessary.

    Reads features_{split}.parquet if it exists; otherwise calls
    ml.extract_features.extract_features() to generate it.
    """
    feat_path = data_dir / f"features_{split}.parquet"
    if feat_path.exists():
        logger.info("Loading pre-extracted features from %s", feat_path)
        return pd.read_parquet(feat_path)

    logger.info(
        "features_%s.parquet not found — running extract_features (split=%s) ...",
        split,
        split,
    )
    from ml.extract_features import extract_features  # noqa: PLC0415

    extract_features(str(data_dir), split=split)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Extraction completed but {feat_path} not found.  "
            "Check extraction logs above."
        )
    return pd.read_parquet(feat_path)


# ---------------------------------------------------------------------------
# Per-interval RMSE
# ---------------------------------------------------------------------------

def _rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute per-interval RMSE in kg."""
    return float(math.sqrt(np.mean((predictions - targets) ** 2)))


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def score_rank(
    data_dir: str,
    model_path: Optional[str] = None,
    model_uri: Optional[str] = "models:/FuelBurn@production",
    baseline_rmse: Optional[float] = None,
    batch_size: int = 512,
) -> dict:
    """Run the rank-phase scoring harness.

    Parameters
    ----------
    data_dir      : directory containing fuel_rank.parquet (TRUE labels)
                    and optionally features_rank.parquet.
    model_path    : path to a local .pth checkpoint (takes priority over URI).
    model_uri     : MLflow model URI (default: models:/FuelBurn@production).
                    Ignored when model_path is set.
    baseline_rmse : optional RMSE measured on the same rank labels.
                    When None, reference comparison is skipped.
    batch_size    : number of intervals per inference batch.

    Returns
    -------
    dict with keys: our_rmse, baseline_rmse, delta, verdict,
                    n_flights, n_intervals, model_source.
    """
    data_path = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. Load TRUE rank labels
    # ------------------------------------------------------------------
    rank_label_path = data_path / "fuel_rank.parquet"
    if not rank_label_path.exists():
        raise FileNotFoundError(
            f"TRUE rank labels not found at {rank_label_path}.  "
            "This file must contain columns: idx, flight_id, start, end, fuel_kg."
        )
    logger.info("Loading TRUE rank labels from %s", rank_label_path)
    df_labels = pd.read_parquet(rank_label_path)

    required_label_cols = {"idx", "flight_id", "fuel_kg"}
    missing_label_cols = required_label_cols - set(df_labels.columns)
    if missing_label_cols:
        raise ValueError(
            f"fuel_rank.parquet is missing required columns: {missing_label_cols}"
        )

    n_intervals = len(df_labels)
    n_flights = df_labels["flight_id"].nunique()
    logger.info(
        "Rank phase: %d intervals across %d flights.", n_intervals, n_flights
    )

    # ------------------------------------------------------------------
    # 2. Load / extract rank features
    # ------------------------------------------------------------------
    df_features = _ensure_features(data_path, split=_SPLIT)

    # Validate that FEATURE_COLUMNS are all present in the features parquet.
    # Missing columns are filled with 0.0 (same degradation policy as dataset.py).
    missing_feat_cols = [c for c in FEATURE_COLUMNS if c not in df_features.columns]
    if missing_feat_cols:
        logger.warning(
            "%d feature column(s) missing from features_rank.parquet — filling with 0.0: %s",
            len(missing_feat_cols),
            missing_feat_cols,
        )
        for col in missing_feat_cols:
            df_features[col] = 0.0

    # ------------------------------------------------------------------
    # 3. Align features with labels on idx
    # ------------------------------------------------------------------
    # Both dataframes carry an `idx` column (row identity within the fuel table).
    # Merge on idx so predictions align with TRUE labels regardless of row order.
    df_merged = df_labels[["idx", "fuel_kg"]].merge(
        df_features[["idx"] + FEATURE_COLUMNS],
        on="idx",
        how="left",
        validate="1:1",
    )

    # Any intervals with no matching feature row get zero features (safe fallback).
    unmatched = df_merged[FEATURE_COLUMNS].isnull().any(axis=1).sum()
    if unmatched > 0:
        logger.warning(
            "%d interval(s) had no matching features — filling with 0.0.  "
            "These will degrade prediction quality.",
            unmatched,
        )
        df_merged[FEATURE_COLUMNS] = df_merged[FEATURE_COLUMNS].fillna(0.0)

    true_fuel = df_merged["fuel_kg"].to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Build feature matrix in FEATURE_COLUMNS order
    #    (identical ordering to FuelBurnDataset.__getitem__ — no skew)
    # ------------------------------------------------------------------
    X = df_merged[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    logger.info(
        "Feature matrix shape: %s  (expected: [n_intervals, %d])", X.shape, INPUT_DIM
    )
    assert X.shape[1] == INPUT_DIM, (
        f"Feature matrix width {X.shape[1]} != INPUT_DIM {INPUT_DIM}.  "
        "Re-run ml.extract_features --split rank to regenerate."
    )

    # ------------------------------------------------------------------
    # 5. Load model
    # ------------------------------------------------------------------
    # model_path takes priority; model_uri is used when path is None.
    model_source = model_path or model_uri or "unknown"
    model = _load_model(model_path=model_path, model_uri=model_uri)

    # ------------------------------------------------------------------
    # 6. Predict in batches (no OOM on large rank set)
    # ------------------------------------------------------------------
    logger.info("Running inference on %d intervals (batch_size=%d) ...", n_intervals, batch_size)
    predictions: list[float] = []
    X_tensor = torch.from_numpy(X)  # [n_intervals, INPUT_DIM]

    with torch.no_grad():
        for start_idx in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start_idx : start_idx + batch_size]  # [B, INPUT_DIM]
            preds = model(batch)  # [B] or [B, 1]
            preds = preds.reshape(-1)
            predictions.extend(preds.cpu().tolist())

    pred_array = np.array(predictions, dtype=np.float32)

    # ------------------------------------------------------------------
    # 7. Compute per-interval RMSE in kg
    # ------------------------------------------------------------------
    our_rmse = _rmse(pred_array, true_fuel)
    logger.info("Our RMSE (rank phase): %.4f kg", our_rmse)

    # ------------------------------------------------------------------
    # 8. Baseline comparison
    # ------------------------------------------------------------------
    delta: Optional[float] = None
    verdict: str = "N/A (no baseline supplied)"

    if baseline_rmse is not None:
        delta = our_rmse - baseline_rmse
        if delta < 0:
            verdict = "BEATS baseline"
        elif delta == 0:
            verdict = "MATCHES baseline"
        else:
            verdict = "BELOW baseline"

    # ------------------------------------------------------------------
    # 9. Report
    # ------------------------------------------------------------------
    _print_report(
        our_rmse=our_rmse,
        baseline_rmse=baseline_rmse,
        delta=delta,
        verdict=verdict,
        n_flights=n_flights,
        n_intervals=n_intervals,
        model_source=model_source,
    )

    result = {
        "our_rmse": our_rmse,
        "baseline_rmse": baseline_rmse,
        "delta": delta,
        "verdict": verdict,
        "n_flights": n_flights,
        "n_intervals": n_intervals,
        "model_source": model_source,
    }

    # ------------------------------------------------------------------
    # 10. Write artefacts
    # ------------------------------------------------------------------
    _write_artefacts(data_path, result)

    return result


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_report(
    our_rmse: float,
    baseline_rmse: Optional[float],
    delta: Optional[float],
    verdict: str,
    n_flights: int,
    n_intervals: int,
    model_source: str,
) -> None:
    sep = "=" * 60
    print(sep)
    print("PRC-2025 RANK PHASE — SCORING RESULT")
    print(sep)
    print(f"  Intervals evaluated : {n_intervals:,}")
    print(f"  Flights evaluated   : {n_flights:,}")
    print(f"  Model source        : {model_source}")
    print()
    print(f"  Our RMSE (kg)       : {our_rmse:.4f}")
    if baseline_rmse is not None:
        print(f"  Reference RMSE (kg) : {baseline_rmse:.4f}  [same split]")
        sign = "-" if (delta or 0) < 0 else "+"
        print(f"  Delta (ours - base) : {sign}{abs(delta or 0):.4f} kg")
        print()
        print(f"  VERDICT: {verdict}")
    else:
        print()
        print("  VERDICT: N/A (no same-split reference supplied)")
    print(sep)


def _write_artefacts(data_path: Path, result: dict) -> None:
    """Write JSON and Markdown result artefacts to data_path."""
    json_path = data_path / _RESULT_JSON
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("JSON result written to %s", json_path)

    md_lines = [
        "# PRC-2025 Rank Phase — Scoring Result",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Intervals evaluated | {result['n_intervals']:,} |",
        f"| Flights evaluated | {result['n_flights']:,} |",
        f"| Model source | `{result['model_source']}` |",
        f"| **Our RMSE (kg)** | **{result['our_rmse']:.4f}** |",
    ]
    if result["baseline_rmse"] is not None:
        md_lines += [
            f"| Same-split reference RMSE (kg) | {result['baseline_rmse']:.4f} |",
            f"| Delta (ours − baseline) | {result['delta']:+.4f} |",
            f"| **Verdict** | **{result['verdict']}** |",
        ]
    else:
        md_lines.append("| Same-split reference | not supplied |")

    md_path = data_path / _RESULT_MD
    with open(md_path, "w") as fh:
        fh.write("\n".join(md_lines) + "\n")
    logger.info("Markdown result written to %s", md_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the PRC-2025 rank phase: compute per-interval RMSE-kg "
            "against TRUE rank labels."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Directory containing fuel_rank.parquet (TRUE labels) "
            "and optionally features_rank.parquet."
        ),
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-uri",
        type=str,
        default="models:/FuelBurn@production",
        help=(
            "MLflow model URI (MLflow 3.x alias form).  "
            "Default: models:/FuelBurn@production.  "
            "Ignored when --model-path is set."
        ),
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to a local .pth checkpoint saved with torch.save(model, path).  "
            "Takes priority over --model-uri.  Use for offline tests."
        ),
    )

    parser.add_argument(
        "--baseline-rmse",
        type=float,
        default=None,
        help=(
            "Optional RMSE from another model evaluated on these exact rank labels. "
            "Omit when no apples-to-apples reference exists."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=_SPLIT,
        choices=[_SPLIT],
        help=f"Dataset split to score.  Only '{_SPLIT}' is supported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (default: 512).",
    )

    args = parser.parse_args()

    try:
        score_rank(
            data_dir=args.data_dir,
            model_path=args.model_path,
            model_uri=args.model_uri if args.model_path is None else None,
            baseline_rmse=args.baseline_rmse,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
