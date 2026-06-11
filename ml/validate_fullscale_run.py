"""
ml/validate_fullscale_run.py — Post-run validation + registry handoff for the
full-scale PRC-2025 Databricks training job.
============================================================================

Purpose
-------
This script is the agent-side GLUE that bridges the owner's Databricks full-
scale run (ml-05) and the model registry (ml-09).  It is run AFTER the
Databricks job completes, before anything touches the registry.

It:
  1. Loads the specified MLflow run (via --run-id + --tracking-uri).
  2. Validates the run against the expected-output contract defined in ml-05's
     runbook (docs/runbooks/ml-fullscale-databricks.md).
  3. Prints a clear PASS / FAIL line per check — all checks must pass for the
     run to be considered valid.
  4. On ALL-PASS: hands the run_id to ml-09's compare_and_promote() so the
     winning full-scale model becomes models:/FuelBurn@production.
  5. On ANY FAIL: exits with code 1 and does NOT touch the registry.

Expected-Output Contract (from ml-05)
--------------------------------------
These are the exact checks performed.  All thresholds match the ml-05 runbook:

  CHECK 1 — best_val_rmse PRESENT
    The run must have logged the scalar metric "best_val_rmse".
    Threshold: metric must exist (not None).

  CHECK 2 — best_val_rmse BELOW REGRESSION THRESHOLD
    best_val_rmse < REGRESSION_THRESHOLD_KG (500 kg).
    Rationale: if RMSE > 500 kg after 50 epochs on 11,037 flights the run
    has regressed and must not be promoted.

  CHECK 3 — best_val_rmse AT OR BELOW TARGET
    best_val_rmse <= TARGET_RMSE_KG (442.65 kg).
    Rationale: the CV re-baseline on the bounded 500-flight real dataset
    achieves 442.65 kg.  Training on 5x more data should match or beat this.
    This check emits WARN (not FAIL) by default; pass --strict to make it FAIL.

  CHECK 4 — model/ ARTIFACT PRESENT
    The run's artifact store must contain the "model/" directory logged by
    mlflow.pytorch.log_model(model, "model") in train.py.
    This is required for register_run() to succeed downstream.

  CHECK 5 — train_rmse METRIC PRESENT
    The run must have logged the per-epoch training RMSE history as "train_rmse"
    (step-indexed).  This confirms the training loop ran to completion.

  CHECK 6 — FIRST-TO-LAST TRAIN RMSE IMPROVEMENT
    The final logged train_rmse must be <= the first logged train_rmse. This is
    an endpoint sanity check, not a claim that every epoch is monotonic.

Registry Handoff (ml-09)
------------------------
On ALL-PASS the script calls:
  1. ml.registry.compare_and_promote(run_id)  — registers the run's model as a
     new version, compares it against the current production champion by RMSE
     (lower wins), and promotes it if it wins.
  2. Prints the resulting decision dict and, on promotion, confirms the
     production URI: models:/FuelBurn@production

Usage
-----
  # Validate + promote from Databricks run (on owner's machine with creds):
  python -m ml.validate_fullscale_run \\
      --run-id <mlflow-run-id> \\
      --tracking-uri databricks

  # Validate + promote from a local MLflow store:
  python -m ml.validate_fullscale_run \\
      --run-id <mlflow-run-id> \\
      --tracking-uri sqlite:///mlflow.db

  # Validate only — dry-run (do not touch registry):
  python -m ml.validate_fullscale_run \\
      --run-id <mlflow-run-id> \\
      --tracking-uri sqlite:///mlflow.db \\
      --dry-run

  # Strict mode (TARGET check is FAIL instead of WARN):
  python -m ml.validate_fullscale_run \\
      --run-id <mlflow-run-id> \\
      --tracking-uri databricks \\
      --strict

Exit codes
----------
  0 — All checks passed (and, unless --dry-run, registry handoff completed).
  1 — One or more checks FAILED; registry not touched.
  2 — Argument error / MLflow connection error before validation could begin.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import mlflow
from mlflow import MlflowClient

# Import ml-09 registry functions at module level so tests can patch them via
# the canonical "ml.validate_fullscale_run.compare_and_promote" target.
from ml.registry import compare_and_promote, PRODUCTION_URI  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ml.validate_fullscale_run")

# ---------------------------------------------------------------------------
# Contract thresholds (mirror ml-05 runbook — do NOT change without updating
# docs/runbooks/ml-fullscale-databricks.md as well)
# ---------------------------------------------------------------------------

#: Hard upper bound on best_val_rmse.  Exceeding this is a regression.
REGRESSION_THRESHOLD_KG: float = 500.0

#: Soft target derived from the CV re-baseline (500 flights, 10 epochs, 4 folds).
#: Used for CHECK 3.  Warn by default; --strict turns this into a hard FAIL.
TARGET_RMSE_KG: float = 442.65

#: Artifact path that train.py logs via mlflow.pytorch.log_model(model, "model").
EXPECTED_ARTIFACT_PATH: str = "model"

#: Metric keys used by train.py.
METRIC_BEST_VAL_RMSE: str = "best_val_rmse"
METRIC_TRAIN_RMSE: str = "train_rmse"
METRIC_VAL_RMSE: str = "val_rmse"

# ---------------------------------------------------------------------------
# Validation result dataclass-like dict helpers
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"


def _result(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_best_val_rmse_present(metrics: dict) -> dict:
    """CHECK 1 — best_val_rmse must be logged."""
    if METRIC_BEST_VAL_RMSE in metrics:
        val = metrics[METRIC_BEST_VAL_RMSE]
        return _result(
            "best_val_rmse_present",
            _PASS,
            f"{METRIC_BEST_VAL_RMSE} = {val:.2f} kg",
        )
    return _result(
        "best_val_rmse_present",
        _FAIL,
        f"Metric '{METRIC_BEST_VAL_RMSE}' not found in run. "
        "Ensure train.py logged mlflow.log_metric('best_val_rmse', ...).",
    )


def _check_regression_threshold(metrics: dict) -> dict:
    """CHECK 2 — best_val_rmse must be < REGRESSION_THRESHOLD_KG."""
    val = metrics.get(METRIC_BEST_VAL_RMSE)
    if val is None:
        return _result(
            "best_val_rmse_below_regression_threshold",
            _FAIL,
            f"Cannot check threshold: '{METRIC_BEST_VAL_RMSE}' is absent.",
        )
    if val < REGRESSION_THRESHOLD_KG:
        return _result(
            "best_val_rmse_below_regression_threshold",
            _PASS,
            f"{val:.2f} kg < {REGRESSION_THRESHOLD_KG} kg (regression threshold).",
        )
    return _result(
        "best_val_rmse_below_regression_threshold",
        _FAIL,
        f"{val:.2f} kg >= {REGRESSION_THRESHOLD_KG} kg (regression threshold). "
        "Do NOT promote. Flag the run to ml-trainer for investigation.",
    )


def _check_target_rmse(metrics: dict, strict: bool = False) -> dict:
    """CHECK 3 — best_val_rmse should be <= TARGET_RMSE_KG.

    In default mode this emits WARN (not FAIL) so the owner can still promote
    if there is a reasonable explanation.  In --strict mode it becomes a FAIL.
    """
    val = metrics.get(METRIC_BEST_VAL_RMSE)
    if val is None:
        return _result(
            "best_val_rmse_at_or_below_target",
            _FAIL if strict else _WARN,
            f"Cannot check target: '{METRIC_BEST_VAL_RMSE}' is absent.",
        )
    if val <= TARGET_RMSE_KG:
        return _result(
            "best_val_rmse_at_or_below_target",
            _PASS,
            f"{val:.2f} kg <= {TARGET_RMSE_KG} kg (CV re-baseline target).",
        )
    severity = _FAIL if strict else _WARN
    return _result(
        "best_val_rmse_at_or_below_target",
        severity,
        f"{val:.2f} kg > {TARGET_RMSE_KG} kg (CV re-baseline). "
        f"More data should match or beat the bounded baseline. "
        f"{'FAILING in strict mode. ' if strict else 'Investigate before promoting. '}"
        "Check for data loading or split bugs.",
    )


def _check_model_artifact(client: MlflowClient, run_id: str) -> dict:
    """CHECK 4 — model/ artifact must exist in the run's artifact store.

    MLflow 3.x stores PyTorch models via a "logged model" path.  In SQLite-
    backed stores the root list_artifacts() call may return an empty list even
    when the model is present; calling list_artifacts(run_id, path='model')
    directly is the reliable probe.  If that call returns at least one entry
    (e.g. MLmodel, conda.yaml, data/) the artifact exists.
    """
    try:
        # Primary probe: list the model/ sub-directory directly.
        sub_artifacts = client.list_artifacts(run_id, path=EXPECTED_ARTIFACT_PATH)
    except mlflow.exceptions.MlflowException as exc:
        # Some backends raise when the path doesn't exist; treat as missing.
        return _result(
            "model_artifact_present",
            _FAIL,
            f"Artifact '{EXPECTED_ARTIFACT_PATH}/' NOT found "
            f"(list_artifacts raised: {exc}). "
            "Ensure train.py called mlflow.pytorch.log_model(model, 'model').",
        )

    if sub_artifacts:
        return _result(
            "model_artifact_present",
            _PASS,
            f"Artifact '{EXPECTED_ARTIFACT_PATH}/' found in run artifact store "
            f"({len(sub_artifacts)} file(s) inside).",
        )

    # sub_artifacts is empty — the path exists as a directory but is empty,
    # OR the path does not exist at all (MLflow returns [] for both cases).
    # Fall back to the root listing to produce a better error message.
    try:
        root_artifacts = client.list_artifacts(run_id, path="")
        root_paths = sorted(a.path for a in root_artifacts)
    except mlflow.exceptions.MlflowException:
        root_paths = []

    return _result(
        "model_artifact_present",
        _FAIL,
        f"Artifact '{EXPECTED_ARTIFACT_PATH}/' NOT found (directory empty or absent). "
        f"Top-level artifacts: {root_paths or '(none)'}. "
        "Ensure train.py called mlflow.pytorch.log_model(model, 'model').",
    )


def _check_train_rmse_present(client: MlflowClient, run_id: str, metrics: dict) -> dict:
    """CHECK 5 — train_rmse metric (per-step history) must be present."""
    if METRIC_TRAIN_RMSE in metrics:
        val = metrics[METRIC_TRAIN_RMSE]
        return _result(
            "train_rmse_history_present",
            _PASS,
            f"Scalar '{METRIC_TRAIN_RMSE}' present (last logged value = {val:.2f} kg).",
        )
    return _result(
        "train_rmse_history_present",
        _FAIL,
        f"Metric '{METRIC_TRAIN_RMSE}' not found in run. "
        "The training loop must log per-epoch train_rmse for the endpoint check.",
    )


def _check_decreasing_train_rmse(client: MlflowClient, run_id: str) -> dict:
    """CHECK 6 — final train_rmse must be no higher than the first.

    Fetches the full step-indexed metric history.  If only one step is present,
    or the history is unavailable, the check is skipped (WARN).
    """
    try:
        history = client.get_metric_history(run_id, METRIC_TRAIN_RMSE)
    except mlflow.exceptions.MlflowException as exc:
        return _result(
            "train_rmse_decreasing",
            _WARN,
            f"Could not fetch train_rmse history: {exc}. Skipping endpoint check.",
        )

    if not history:
        return _result(
            "train_rmse_decreasing",
            _WARN,
            "train_rmse history is empty. Skipping endpoint check.",
        )

    # Sort by step to get chronological order.
    sorted_history = sorted(history, key=lambda m: m.step)

    if len(sorted_history) == 1:
        return _result(
            "train_rmse_decreasing",
            _WARN,
            "Only one train_rmse data point logged. Cannot compare endpoints.",
        )

    first_val = sorted_history[0].value
    last_val = sorted_history[-1].value
    n_steps = len(sorted_history)

    if last_val <= first_val:
        return _result(
            "train_rmse_decreasing",
            _PASS,
            f"train_rmse decreased from {first_val:.2f} kg (step {sorted_history[0].step}) "
            f"to {last_val:.2f} kg (step {sorted_history[-1].step}) over {n_steps} logged steps.",
        )
    return _result(
        "train_rmse_decreasing",
        _FAIL,
        f"train_rmse DID NOT decrease: {first_val:.2f} kg (step {sorted_history[0].step}) "
        f"-> {last_val:.2f} kg (step {sorted_history[-1].step}) over {n_steps} steps. "
        "Check for NaN loss or a broken optimiser — trigger ml-debugger if needed.",
    )


# ---------------------------------------------------------------------------
# Main validation orchestrator
# ---------------------------------------------------------------------------

def validate_run(
    run_id: str,
    strict: bool = False,
) -> tuple[list[dict], bool]:
    """Run all validation checks against an MLflow run.

    Parameters
    ----------
    run_id:
        MLflow run_id to validate.  The tracking URI must already be set via
        mlflow.set_tracking_uri() before calling this function.
    strict:
        If True, CHECK 3 (target RMSE) becomes a hard FAIL instead of WARN.

    Returns
    -------
    results : list[dict]
        One result dict per check, each with keys: check, status, detail.
    all_passed : bool
        True iff every check status is PASS or WARN (i.e. no FAIL).
    """
    client = MlflowClient()

    logger.info("Fetching run %s ...", run_id)
    try:
        run = client.get_run(run_id)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"Could not fetch run '{run_id}' from MLflow: {exc}. "
            "Verify --run-id and --tracking-uri."
        ) from exc

    metrics = run.data.metrics

    results: list[dict] = []

    # CHECK 1 — metric present
    results.append(_check_best_val_rmse_present(metrics))

    # CHECK 2 — regression threshold
    results.append(_check_regression_threshold(metrics))

    # CHECK 3 — target RMSE (warn by default, fail in strict mode)
    results.append(_check_target_rmse(metrics, strict=strict))

    # CHECK 4 — model/ artifact
    results.append(_check_model_artifact(client, run_id))

    # CHECK 5 — train_rmse history present
    results.append(_check_train_rmse_present(client, run_id, metrics))

    # CHECK 6 — decreasing train RMSE
    results.append(_check_decreasing_train_rmse(client, run_id))

    all_passed = all(r["status"] != _FAIL for r in results)
    return results, all_passed


def print_validation_report(results: list[dict]) -> None:
    """Print a human-readable validation report to stdout."""
    print()
    print("=" * 68)
    print("  Full-Scale Run Validation Report")
    print("=" * 68)
    for r in results:
        status = r["status"]
        indicator = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, f"[{status}]")
        print(f"  {indicator:7s}  {r['check']}")
        print(f"           {r['detail']}")
    print("=" * 68)

    n_pass = sum(1 for r in results if r["status"] == _PASS)
    n_warn = sum(1 for r in results if r["status"] == _WARN)
    n_fail = sum(1 for r in results if r["status"] == _FAIL)
    print(f"  Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Registry handoff
# ---------------------------------------------------------------------------

def handoff_to_registry(run_id: str) -> dict:
    """Hand the validated run_id to ml-09's compare_and_promote().

    This is the ONLY registry interaction in this module.  All champion/
    challenger logic lives in ml/registry.py (ml-09).

    Parameters
    ----------
    run_id:
        MLflow run_id that passed all validation checks.

    Returns
    -------
    result : dict
        The dict returned by compare_and_promote() with keys:
          promoted, candidate_version, candidate_rmse,
          champion_version, champion_rmse, reason.
    """
    # compare_and_promote and PRODUCTION_URI are imported at module level
    # (see top of file) so they can be patched in tests via
    # `patch("ml.validate_fullscale_run.compare_and_promote")`.

    logger.info("Handing off run %s to registry (ml-09: compare_and_promote)...", run_id)
    result = compare_and_promote(candidate_run_id=run_id)

    print()
    print("=" * 68)
    print("  Registry Handoff Result (ml-09: compare_and_promote)")
    print("=" * 68)
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result.get("promoted"):
        print()
        print(f"  Production URI: {PRODUCTION_URI}")
        print(f"  Set ML_MODEL_URI={PRODUCTION_URI} in serving (ml-10).")
    else:
        print()
        print("  Champion retained. Production URI unchanged.")
    print("=" * 68)
    print()
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.validate_fullscale_run",
        description=(
            "Validate a full-scale PRC-2025 MLflow run against the ml-05 "
            "expected-output contract and hand it to the ml-09 registry on PASS."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run_id of the full-scale training run to validate.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI. "
            "Use 'databricks' on Databricks, 'sqlite:///mlflow.db' locally. "
            "Defaults to MLFLOW_TRACKING_URI env var."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate only — do NOT call compare_and_promote(). "
            "Useful for pre-check without registry side-effects."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Strict mode: make CHECK 3 (best_val_rmse <= target) a hard FAIL "
            "instead of a warning.  Use when the owner requires the full-scale "
            "run to beat the CV re-baseline."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        logger.info("MLflow tracking URI set to: %s", args.tracking_uri)

    # --- Validate ---
    try:
        results, all_passed = validate_run(run_id=args.run_id, strict=args.strict)
    except RuntimeError as exc:
        logger.error("Validation aborted: %s", exc)
        sys.exit(2)

    print_validation_report(results)

    if not all_passed:
        print("VALIDATION FAILED — registry handoff skipped.")
        print("Fix the issues above before promoting to models:/FuelBurn@production.")
        print()
        sys.exit(1)

    print("VALIDATION PASSED.")

    if args.dry_run:
        print("--dry-run set: registry handoff skipped.")
        print()
        sys.exit(0)

    # --- Registry handoff (ml-09) ---
    try:
        handoff_to_registry(args.run_id)
    except Exception as exc:  # noqa: BLE001 — surface any registry error clearly
        logger.error("Registry handoff failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
