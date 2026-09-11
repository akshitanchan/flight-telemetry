import argparse
import os
import random
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import mlflow
import mlflow.pytorch
import math
import copy
import logging
import numpy as np
from pathlib import Path

from ml.dataset import FuelBurnDataset
from ml.model import FuelBurnMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s", force=True)  # force: override MLflow's root-logger config so epoch RMSE logs show
logger = logging.getLogger("ml.train")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

#: Valid explicit device tokens accepted by --device.
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Mapping:
      ``"auto"``  → cuda if available, else mps if available, else cpu.
      ``"cpu"``   → torch.device("cpu")  (always available)
      ``"cuda"``  → torch.device("cuda") (requires CUDA-capable GPU)
      ``"mps"``   → torch.device("mps")  (requires Apple Silicon / Metal)

    The function does NOT consume any torch RNG state — availability checks
    are pure hardware probes.  It is therefore safe to call this after
    ``torch.manual_seed()`` and before model instantiation without affecting
    the weight-initialization RNG sequence.

    Parameters
    ----------
    device_str:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    ValueError
        If ``device_str`` is not one of the accepted tokens.
    """
    device_str = device_str.strip().lower()
    if device_str not in _VALID_DEVICES:
        raise ValueError(
            f"--device must be one of {_VALID_DEVICES}; got {device_str!r}."
        )

    if device_str == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device_str)

    logger.info(f"Device: --device={device_str!r} resolved to {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for features, target in dataloader:
        features, target = features.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(features)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            loss = criterion(output, target)
            total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


# ---------------------------------------------------------------------------
# Core training callable
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run one full training job given a parsed :class:`argparse.Namespace`.

    This is the primary callable entry point.  ``main()`` is a thin shim that
    parses sys.argv into a Namespace and delegates here, so callers such as
    ``ml.train_fullscale`` can drive training directly without patching
    ``sys.argv``.

    Parameters
    ----------
    args:
        Namespace with the following attributes:

        data_dir : str
            Path to the directory containing ``features_train.parquet``.
        epochs : int
            Number of training epochs (default: 5).
        batch_size : int
            Mini-batch size (default: 16).
        lr : float
            Adam learning rate (default: 0.001).
        device : str
            Device token — one of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``
            (default: ``"auto"``).  ``"auto"`` resolves to cuda → mps → cpu in
            that priority order at runtime.
        checkpoint_dir : str or None
            Directory in which to write ``checkpoint.pt`` after each epoch.
            ``None`` (default) disables checkpointing entirely — behaviour is
            bit-identical to a run without this argument.
        hidden_dim : int
            Width of the first MLP hidden layer (default: 64).
        experiment_name : str
            MLflow experiment name (default: ``FuelBurn_Baseline``).
        run_name : str or None
            Optional MLflow run name.

    RNG / reproducibility contract
    --------------------------------
    The call sequence that determines numerical outputs is:

      1. ``torch.manual_seed(args.seed)`` — fixes weight-init and training RNG
      2. ``FuelBurnMLP()``                — consumes the torch RNG for init
      3. ``GroupShuffleSplit(…, random_state=0)``  — independent sklearn RNG

    This order MUST NOT be changed without re-baselining the CV numbers.
    The device resolution probe between steps 1 and 2 does not touch the RNG.

    Checkpoint / resume contract
    ----------------------------
    When ``checkpoint_dir`` is set a checkpoint is written atomically after
    every epoch via a temp-file + ``os.replace`` pattern.  The checkpoint
    stores ``torch_rng_state`` (the full Mersenne Twister state at the end of
    the epoch).  On resume, restoring that state before the next epoch
    guarantees that ``DataLoader(shuffle=True)`` produces the same shuffle
    sequence and that any dropout / other stochastic ops produce identical
    draws, making ``resumed == straight-through`` bit-for-bit on CPU.

    If ``checkpoint_dir=None`` no files are written and no extra branches are
    taken — the code path is structurally identical to the pre-checkpoint
    version.

    MLflow contract (frozen — do NOT rename)
    -----------------------------------------
    * Experiment : ``"FuelBurn_Baseline"``
    * Per-epoch  : ``mlflow.log_metrics({"train_rmse", "val_rmse"}, step=epoch)``
    * Final      : ``mlflow.log_metric("best_val_rmse", ...)``
    * Artifact   : ``mlflow.pytorch.log_model(model, "model")``
    """
    # --- Reproducibility: fix torch RNG before any weight init ---
    # IMPORTANT: do not reorder the three steps in the contract above.
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.data_dir == "data/ml/prc_2025_mock" and not Path(args.data_dir).exists():
        raise FileNotFoundError("Mock data missing. Run generate_mock in extract script.")

    # --- Device resolution (no RNG side-effect) ---
    device_attr = getattr(args, "device", "auto")
    device = resolve_device(device_attr)

    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = FuelBurnDataset(data_dir=args.data_dir, split="train")

    # --- Flight-level group split (no flight leaks across train/val) ---
    # Use GroupShuffleSplit so the 80/20 boundary is drawn at the flight level,
    # not the interval level.  random_state=0 is a fixed seed independent of
    # torch.manual_seed so the split is identical across runs.
    from sklearn.model_selection import GroupShuffleSplit

    flight_ids = dataset.flight_ids            # shape (N,) — one entry per interval
    all_indices = np.arange(len(dataset))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(gss.split(all_indices, groups=flight_ids))

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    # Verify that no flight appears on both sides (correctness guarantee).
    train_flights = set(flight_ids[train_idx])
    val_flights   = set(flight_ids[val_idx])
    assert train_flights.isdisjoint(val_flights), (
        f"Data leakage: {len(train_flights & val_flights)} flights appear in both "
        "train and val splits."
    )
    logger.info(
        f"Flight-level split — train flights: {len(train_flights)}, "
        f"val flights: {len(val_flights)}, "
        f"train intervals: {len(train_idx)}, val intervals: {len(val_idx)} — "
        "train/val flight sets are DISJOINT (no leakage)."
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)

    hidden_dim = int(getattr(args, "hidden_dim", 64))
    model = FuelBurnMLP(hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Checkpoint / resume setup ---
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    start_epoch = 1
    best_val_rmse = float("inf")
    best_state = None

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
        if ckpt_path.exists():
            logger.info(f"Checkpoint found at {ckpt_path}. Resuming...")
            # weights_only=False is required because the checkpoint contains
            # numpy RNG state (a numpy array object), which PyTorch 2.6+
            # rejects under the new weights_only=True default.  The checkpoint
            # is written by this same process, so it is a trusted source.
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_rmse = ckpt["best_val_rmse"]
            best_state = ckpt["best_state"]
            # Restore full torch RNG state so the next epoch's shuffle and
            # dropout draws are identical to what an uninterrupted run would
            # have produced — this is the guarantee that resumed==straight-through.
            torch.set_rng_state(ckpt["torch_rng_state"])
            # Restore Python and NumPy RNG states (advanced during data loading
            # if workers use them; included for completeness).
            random.setstate(ckpt["python_rng_state"])
            np.random.set_state(ckpt["numpy_rng_state"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(
                f"Resumed from epoch {ckpt['epoch']}. "
                f"best_val_rmse so far: {best_val_rmse:.4f}. "
                f"Continuing from epoch {start_epoch}."
            )
        else:
            logger.info(
                f"checkpoint_dir={checkpoint_dir!r} — no existing checkpoint; "
                "starting fresh run with checkpointing enabled."
            )
    else:
        ckpt_path = None

    # Initialize MLflow tracking
    experiment_name = getattr(args, "experiment_name", "FuelBurn_Baseline")
    run_name = getattr(args, "run_name", None)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model_type": "MLP_Baseline",
            "hidden_dim": hidden_dim,
            "seed": seed,
            "device": str(device),
        })
        mlflow.set_tag("data.leakage_fix", "post")

        logger.info(f"Starting training for {args.epochs} epochs (from epoch {start_epoch})...")

        for epoch in range(start_epoch, args.epochs + 1):
            train_mse = train_epoch(model, train_loader, criterion, optimizer, device)
            val_mse = evaluate(model, val_loader, criterion, device)

            train_rmse = math.sqrt(train_mse)
            val_rmse = math.sqrt(val_mse)

            mlflow.log_metrics({
                "train_rmse": train_rmse,
                "val_rmse": val_rmse
            }, step=epoch)

            logger.info(f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())

            # --- Atomic epoch checkpoint ---
            # Written AFTER best_state update so the checkpoint always reflects
            # the best model seen up to and including this epoch.
            if ckpt_path is not None:
                ckpt_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "best_state": best_state,
                    # Full torch RNG state — Mersenne Twister internal state
                    # captured at end-of-epoch.  Restoring this before epoch N+1
                    # ensures DataLoader shuffle and dropout are identical to an
                    # uninterrupted run (the core of the resume==straight-through
                    # guarantee on CPU).
                    "torch_rng_state": torch.get_rng_state(),
                    # Python and NumPy RNG states for completeness.
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                }
                # Atomic write: write to a sibling temp file then os.replace so
                # a crash during the write never leaves a corrupt checkpoint.
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(ckpt_path.parent), suffix=".tmp"
                )
                try:
                    os.close(tmp_fd)
                    torch.save(ckpt_data, tmp_path)
                    os.replace(tmp_path, str(ckpt_path))
                except Exception:
                    # Clean up the temp file if anything goes wrong before
                    # os.replace; do not silently swallow the exception.
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
                logger.info(f"Checkpoint saved after epoch {epoch} → {ckpt_path}")

        logger.info(f"Training complete. Best Val RMSE: {best_val_rmse:.2f}")
        mlflow.log_metric("best_val_rmse", best_val_rmse)

        # Restore the best-performing weights so we register the best model, not
        # the (possibly overfit) final-epoch model (audit M4).
        if best_state is not None:
            model.load_state_dict(best_state)

        # Log the PyTorch model.
        # pickle keeps the artifact loadable as an nn.Module by serve.py and
        # score_rank.py; mlflow >= 3.14 defaults to the pt2 traced-graph format.
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")


# ---------------------------------------------------------------------------
# CLI shim — thin argparse wrapper around train()
# ---------------------------------------------------------------------------

def main():
    """CLI entry point.  Parses sys.argv and delegates to :func:`train`."""
    parser = argparse.ArgumentParser(description="Eurocontrol Fuel Burn Baseline Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to Eurocontrol dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Compute device: 'auto' (default) selects cuda > mps > cpu, "
            "or specify 'cpu', 'cuda', or 'mps' explicitly."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        dest="checkpoint_dir",
        help=(
            "Directory for epoch checkpoints.  After each epoch, "
            "checkpoint.pt is written atomically (temp + os.replace).  "
            "On restart, training resumes from the last completed epoch.  "
            "Default: None (no checkpointing; behaviour bit-identical to "
            "a run without this flag)."
        ),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
