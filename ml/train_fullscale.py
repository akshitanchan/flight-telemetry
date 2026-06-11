"""
ml/train_fullscale.py — Databricks-ready full-scale training entrypoint.
========================================================================

Design
------
This module is a WRAPPER around the existing ml/train.py and
ml/extract_features.py.  It does NOT edit either of those files.

It:
  1. Reads ml/configs/fullscale.yaml (or a path you supply via --config).
  2. Applies CLI overrides (--override key=value pairs, dot-notation supported).
  3. Runs ml.extract_features.extract_features() for the resumable feature
     extraction step (skips if features_{split}.parquet already exists and
     --force-extract is not set).
  4. Calls ml.train.train(args) directly, passing an argparse.Namespace built
     from the resolved config.  The sys.argv-patching hack previously used to
     drive ml.train.main() has been REMOVED (rem-ml-02).
  5. Supports a --smoke flag: generates 20-flight mock data and runs 2 epochs
     to verify the pipeline end-to-end without any real PRC data or Databricks
     connection.

train.py direct-call contract (rem-ml-02)
-----------------------------------------
train.py now exposes train(args: argparse.Namespace) as a proper callable.
This wrapper builds a Namespace from the resolved config and calls it
directly.  The sys.argv save/patch/restore block has been removed entirely.

The Namespace passed to train() carries:
  data_dir    str   — path to the directory with features_train.parquet
  epochs      int   — training epochs
  batch_size  int   — mini-batch size
  lr          float — Adam learning rate
  device      str   — "auto" | "cpu" | "cuda" | "mps"

Note for rem-ml-03: a training.checkpoint_dir config key will be added to
the training section of fullscale.yaml by the next task.  Leave the
training section open for that addition.

Usage
-----
Offline smoke (no real data, no Databricks):
  python -m ml.train_fullscale --smoke

Full scale on Databricks (owner runs this):
  python -m ml.train_fullscale \\
      --config ml/configs/fullscale.yaml \\
      --override data.data_dir=/Volumes/<catalog>/<schema>/<vol>/prc_2025 \\
      --override mlflow.tracking_uri=databricks \\
      --override training.epochs=50

Dry-run (real local data, 3 epochs, no Databricks):
  python -m ml.train_fullscale \\
      --config ml/configs/fullscale.yaml \\
      --override training.epochs=3 \\
      --override data.extract_limit=100
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO  [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("ml.train_fullscale")


# ---------------------------------------------------------------------------
# Config helpers (pure stdlib — no PyYAML import at module level so the
# module is importable even if PyYAML is absent; we raise a clear error on use)
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """Load a YAML config file.  Requires PyYAML (pyyaml in pip)."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        ) from exc
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _apply_override(cfg: dict, key: str, value: str) -> None:
    """
    Apply a dot-notation override to a nested config dict in-place.

    Example: key="training.epochs", value="50"
    Attempts int -> float -> str coercion in that order.
    """
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    leaf = parts[-1]
    # Type coercion: try int, then float, then keep as string
    for cast in (int, float):
        try:
            node[leaf] = cast(value)
            return
        except ValueError:
            pass
    # Handle YAML null / boolean strings
    if value.lower() in ("null", "none", "~"):
        node[leaf] = None
    elif value.lower() == "true":
        node[leaf] = True
    elif value.lower() == "false":
        node[leaf] = False
    else:
        node[leaf] = value


def _get(cfg: dict, dotkey: str, default: Any = None) -> Any:
    """Read a dot-notation key from a nested config dict."""
    parts = dotkey.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Default config (mirrors fullscale.yaml — used when YAML is unavailable)
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "data": {
        "data_dir": "data/raw/prc_2025",
        "split": "train",
        "checkpoint_dir": None,
        "extract_limit": None,
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "seed": 42,
    },
    "model": {
        "type": "mlp",
        "hidden_dim": 64,
    },
    "mlflow": {
        "experiment_name": "FuelBurn_Baseline",
        "tracking_uri": "sqlite:///mlflow.db",
        "run_name": "fullscale_prc2025",
        "log_every_n_epochs": 5,
    },
}


# ---------------------------------------------------------------------------
# Feature extraction step
# ---------------------------------------------------------------------------

def _run_extract(cfg: dict, data_dir: str) -> None:
    """
    Run ml.extract_features.extract_features() if needed.

    Skips extraction if features_{split}.parquet already exists and
    --force-extract was not requested (idempotent / resumable).
    """
    from ml.extract_features import extract_features  # noqa: PLC0415

    split = _get(cfg, "data.split", "train")
    out_parquet = Path(data_dir) / f"features_{split}.parquet"

    if out_parquet.exists():
        logger.info(
            f"features_{split}.parquet already exists at {out_parquet}. "
            "Skipping extraction (use --force-extract to re-run)."
        )
        return

    checkpoint_dir = _get(cfg, "data.checkpoint_dir", None)
    extract_limit = _get(cfg, "data.extract_limit", None)

    logger.info(
        f"Starting feature extraction: data_dir={data_dir}, split={split}, "
        f"limit={extract_limit}, checkpoint_dir={checkpoint_dir}"
    )
    extract_features(
        data_dir=data_dir,
        split=split,
        limit=extract_limit,
        checkpoint_dir=checkpoint_dir,
    )
    logger.info("Feature extraction complete.")


# ---------------------------------------------------------------------------
# Training step (direct call to ml.train.train)
# ---------------------------------------------------------------------------

def _run_train(cfg: dict, data_dir: str) -> None:
    """
    Build an argparse.Namespace from cfg and call ml.train.train() directly.

    The sys.argv save/patch/restore block has been removed (rem-ml-02).
    ml.train now exposes train(args: argparse.Namespace) as a proper callable;
    this function constructs that Namespace from the resolved config and calls
    it without touching sys.argv.

    Device routing (production path to CUDA):
        training.device in cfg (default "auto") is passed straight through to
        train(), which delegates to ml.train.resolve_device().  On a GPU
        Databricks cluster, set training.device="cuda" (or leave "auto") via
        an --override flag or directly in fullscale.yaml.
    """
    import mlflow  # noqa: PLC0415
    from ml.train import train as _train  # noqa: PLC0415

    epochs = _get(cfg, "training.epochs", 50)
    batch_size = _get(cfg, "training.batch_size", 16)
    lr = _get(cfg, "training.lr", 0.001)
    device = _get(cfg, "training.device", "auto")
    # training.checkpoint_dir — added by rem-ml-03.  None (default) = off;
    # set to a durable directory path to enable epoch checkpointing + resume.
    checkpoint_dir = _get(cfg, "training.checkpoint_dir", None)
    tracking_uri = _get(cfg, "mlflow.tracking_uri", "sqlite:///mlflow.db")
    experiment_name = _get(cfg, "mlflow.experiment_name", "FuelBurn_Baseline")

    # Set MLflow tracking URI before train() calls mlflow.set_experiment()
    # internally — the URI must be configured first so the experiment lands
    # in the right backend (local SQLite vs Databricks workspace MLflow).
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")

    ns = argparse.Namespace(
        data_dir=str(data_dir),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        device=str(device),
        # checkpoint_dir is passed through as-is (None or a path string).
        # train() reads it via getattr(args, "checkpoint_dir", None) so the
        # field is optional — existing callers that don't set it are unaffected.
        checkpoint_dir=checkpoint_dir if checkpoint_dir is None else str(checkpoint_dir),
    )

    logger.info(
        f"Invoking ml.train.train() with: "
        f"data_dir={ns.data_dir}, epochs={ns.epochs}, "
        f"batch_size={ns.batch_size}, lr={ns.lr}, device={ns.device!r}, "
        f"checkpoint_dir={ns.checkpoint_dir!r}"
    )

    _train(ns)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Smoke-test path (offline, mock data, no Databricks)
# ---------------------------------------------------------------------------

def _run_smoke(config_path: str, overrides: list[str]) -> None:
    """
    End-to-end smoke test using 20-flight mock data and 2 training epochs.

    Generates mock PRC-shaped data in data/ml/prc_2025_mock, extracts features,
    and runs 2 epochs of training.  No real PRC data or Databricks connection
    required.

    This path is used by:
      python -m ml.train_fullscale --smoke
      pytest ml/test_train_fullscale.py  (which also calls this path)
    """
    import tempfile  # noqa: PLC0415
    from ml.mock_data import generate_mock_eurocontrol_data  # noqa: PLC0415

    logger.info("=== SMOKE MODE: generating 20-flight mock dataset ===")

    # Use a temp directory so smoke runs are isolated and leave no artefacts.
    with tempfile.TemporaryDirectory(prefix="fullscale_smoke_") as tmpdir:
        mock_dir = os.path.join(tmpdir, "mock_prc_2025")
        generate_mock_eurocontrol_data(mock_dir, num_flights=20)

        # Build a minimal config for the smoke run.
        cfg: dict = {
            "data": {
                "data_dir": mock_dir,
                "split": "train",
                "checkpoint_dir": None,
                "extract_limit": None,
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "lr": 0.001,
                "seed": 42,
            },
            "model": {"type": "mlp", "hidden_dim": 64},
            "mlflow": {
                "experiment_name": "FuelBurn_Baseline_Smoke",
                "tracking_uri": f"sqlite:///{os.path.join(tmpdir, 'smoke_mlflow.db')}",
                "run_name": "smoke_run",
                "log_every_n_epochs": 1,
            },
        }

        # Apply any CLI overrides on top of the smoke config.
        for ov in overrides:
            if "=" not in ov:
                logger.warning(f"Skipping malformed override (no '='): {ov}")
                continue
            k, v = ov.split("=", 1)
            _apply_override(cfg, k.strip(), v.strip())

        logger.info("Running extraction on mock data...")
        _run_extract(cfg, cfg["data"]["data_dir"])

        logger.info("Running 2-epoch training on mock data...")
        _run_train(cfg, cfg["data"]["data_dir"])

    logger.info("=== SMOKE PASSED ===")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full-scale training entrypoint for PRC-2025 (Databricks-ready). "
            "Reads ml/configs/fullscale.yaml, runs resumable feature extraction, "
            "then trains the FuelBurnMLP via ml.train."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file. "
            "Defaults to ml/configs/fullscale.yaml next to this script."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config key using dot-notation. "
            "May be repeated. Example: --override training.epochs=50 "
            "--override mlflow.tracking_uri=databricks"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Run an offline smoke test: generate 20-flight mock data, "
            "extract features, train for 2 epochs.  No real data required."
        ),
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        default=False,
        help=(
            "Force re-extraction even if features_{split}.parquet already exists."
        ),
    )

    args = parser.parse_args()

    t0 = time.time()

    # --- Smoke path ---
    if args.smoke:
        _run_smoke(
            config_path=args.config or "",
            overrides=args.override,
        )
        logger.info(f"Smoke completed in {time.time() - t0:.1f}s")
        return

    # --- Full / dry-run path ---
    # Locate config file
    if args.config is None:
        # Default: ml/configs/fullscale.yaml relative to this file's directory
        args.config = str(
            Path(__file__).parent / "configs" / "fullscale.yaml"
        )

    if not Path(args.config).exists():
        logger.warning(
            f"Config file not found at {args.config}. "
            "Using built-in defaults."
        )
        cfg = dict(_DEFAULTS)
    else:
        logger.info(f"Loading config from {args.config}")
        cfg = _load_yaml(args.config)

    # Apply CLI overrides
    for ov in args.override:
        if "=" not in ov:
            logger.warning(f"Skipping malformed override (no '='): {ov}")
            continue
        k, v = ov.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
        logger.info(f"Config override applied: {k.strip()} = {v.strip()}")

    # Log effective config
    logger.info("Effective configuration:")
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                logger.info(f"  {section}.{k} = {v!r}")

    data_dir = _get(cfg, "data.data_dir", "data/raw/prc_2025")

    # Model type guard
    model_type = _get(cfg, "model.type", "mlp")
    if model_type != "mlp":
        raise NotImplementedError(
            f"model.type={model_type!r} is not yet wired into the full-scale "
            "entrypoint.  Only 'mlp' is supported (FuelBurnMLP, as validated in "
            "the CV re-baseline).  To use HistGBR, a future ml-trainer task must "
            "add a train_histgbr() path."
        )

    # Set seed early (before any torch imports in sub-calls)
    seed = _get(cfg, "training.seed", 42)
    _set_seeds(seed)

    # Force-extract: delete the existing parquet so _run_extract will re-run
    if args.force_extract:
        split = _get(cfg, "data.split", "train")
        stale = Path(data_dir) / f"features_{split}.parquet"
        if stale.exists():
            stale.unlink()
            logger.info(f"--force-extract: removed {stale}")

    # Step 1: Feature extraction (resumable)
    _run_extract(cfg, data_dir)

    # Step 2: Training
    _run_train(cfg, data_dir)

    elapsed = time.time() - t0
    logger.info(f"Full pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")


def _set_seeds(seed: int) -> None:
    """Fix seeds across Python, NumPy, and PyTorch for reproducibility."""
    import random  # noqa: PLC0415
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
