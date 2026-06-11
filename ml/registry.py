"""
ml/registry.py — MLflow Model Registry promotion workflow for FuelBurn.
========================================================================

Design
------
This module is a STANDALONE promotion step.  It reads existing logged runs
from an MLflow tracking store and promotes their model artifacts into the
Model Registry — train.py is never touched.

Registry name
-------------
All models are registered under the name **FuelBurn** so that serving can
load them via the URI ``models:/FuelBurn@production`` (see "URI note" below).

API choice: ALIASES over STAGES
--------------------------------
MLflow >= 2.9 deprecated stages (Staging / Production) in favour of named
aliases.  As of MLflow 3.x (this project uses 3.13.0) stages still function
but carry a FutureWarning and will be removed in a future major release.
We therefore use the alias API exclusively:

  client.set_registered_model_alias(name, alias, version)
  client.get_model_version_by_alias(name, alias)
  client.delete_registered_model_alias(name, alias)

Alias convention
----------------
  "champion"   — the model currently serving in production.
  "challenger" — a candidate being evaluated before promotion.
  "production" — primary production alias (mirrors the serving hook).

The aliases "champion" and "production" always point to the same version.
When a challenger wins it is promoted:
  1. Its version gets the "champion" and "production" aliases.
  2. The former champion's aliases are deleted (it falls back to version
     number only — still accessible, just no longer "live").

Production URI (exact)
----------------------
  models:/FuelBurn@production

This is what ml/serve.py reads via ``ML_MODEL_URI``.  Both the alias name
and the registered model name must be lower-case as shown for the URI to
resolve correctly with mlflow.pytorch.load_model.

Note for ml-10 (serve.py hook)
-------------------------------
serve.py already has:
    MODEL_URI: Optional[str] = os.environ.get("ML_MODEL_URI")
    ...
    model = mlflow.pytorch.load_model(MODEL_URI)

Set the env var at deploy time:
    ML_MODEL_URI=models:/FuelBurn@production

If you also point MLFLOW_TRACKING_URI at the shared tracking server, serve.py
will resolve the alias and load the correct champion version automatically.

Stale-run tagging strategy
--------------------------
Pre-leakage runs are identified only by explicit provenance tags:
``data.leakage_fix = "pre"`` or ``registry.stale = "true"``. Metric values
are never used as a proxy for lineage; a genuinely improved model can score
below an old leaky run and must not be rejected for being better.

For each stale run the function sets these MLflow run tags:
  registry.stale = "true"
  registry.stale_reason = "pre_leakage_fix_ml01"
  registry.stale_since = <ISO timestamp>

These tags surface in the MLflow UI and are checked by compare_and_promote()
so that stale runs are never registered as champions.

Usage (CLI / notebook)
----------------------
  python -m ml.registry promote \\
      --run-id <run-id-of-candidate> \\
      --tracking-uri sqlite:///mlflow.db

  python -m ml.registry tag-stale \\
      --experiment FuelBurn_Baseline \\
      --tracking-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import mlflow
from mlflow import MlflowClient

logger = logging.getLogger("ml.registry")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registered model name — must match what serving resolves.
REGISTRY_NAME: str = "FuelBurn"

#: Aliases used for champion/production tracking.
ALIAS_PRODUCTION: str = "production"
ALIAS_CHAMPION: str = "champion"
ALIAS_CHALLENGER: str = "challenger"

#: The primary official metric used for champion/challenger comparison.
#: Lower is better (RMSE in kg).  cv.py logs this as "cv_mean_rmse_kg" on the
#: parent ChronologicalCV run; train.py logs "best_val_rmse".  We probe both.
PRIMARY_METRIC: str = "cv_mean_rmse_kg"
FALLBACK_METRIC: str = "best_val_rmse"

#: Exact production URI that serving (ml-10 / ML_MODEL_URI) must use.
PRODUCTION_URI: str = f"models:/{REGISTRY_NAME}@{ALIAS_PRODUCTION}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suppress_stage_warnings():
    """Suppress the FutureWarning about deprecated stage APIs.

    We only call stage APIs in _get_existing_versions() for informational
    purposes (to list all registered versions).  The warning is expected and
    does not indicate a bug in our code.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*model registry stages.*",
        category=FutureWarning,
    )


def _get_run_rmse(client: MlflowClient, run_id: str) -> Optional[float]:
    """Return the best available RMSE metric for a run, or None if absent.

    Priority: cv_mean_rmse_kg > best_val_rmse.
    """
    run = client.get_run(run_id)
    metrics = run.data.metrics
    if PRIMARY_METRIC in metrics:
        return float(metrics[PRIMARY_METRIC])
    if FALLBACK_METRIC in metrics:
        return float(metrics[FALLBACK_METRIC])
    return None


def _is_stale(client: MlflowClient, run_id: str) -> bool:
    """Return True if a run should be considered a pre-leakage stale run.

    Stale criteria (OR):
    1. Run tag ``data.leakage_fix`` == ``"pre"``.
    2. Run tag ``registry.stale`` == ``"true"``.
    """
    run = client.get_run(run_id)
    tags = run.data.tags

    if tags.get("registry.stale") == "true":
        return True
    if tags.get("data.leakage_fix") == "pre":
        return True

    return False


def _ensure_model_registered(client: MlflowClient, name: str) -> None:
    """Create the registered model if it does not already exist."""
    try:
        client.create_registered_model(
            name=name,
            description=(
                "Fuel-burn-per-interval prediction model (FuelBurnMLP). "
                f"Production URI: models:/{name}@{ALIAS_PRODUCTION}. "
                "Champion/challenger managed by ml/registry.py."
            ),
        )
        logger.info("Created registered model '%s'.", name)
    except mlflow.exceptions.MlflowException as exc:
        # RESOURCE_ALREADY_EXISTS — model was already registered, not an error.
        if "RESOURCE_ALREADY_EXISTS" in str(exc) or "already exists" in str(exc).lower():
            logger.debug("Registered model '%s' already exists — skipping create.", name)
        else:
            raise


def _get_current_production_version(client: MlflowClient, name: str) -> Optional[str]:
    """Return the version string currently aliased as production, or None."""
    try:
        mv = client.get_model_version_by_alias(name, ALIAS_PRODUCTION)
        return mv.version
    except mlflow.exceptions.MlflowException:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_run(
    run_id: str,
    name: str = REGISTRY_NAME,
    artifact_path: str = "model",
) -> str:
    """Register a logged model artifact into the MLflow Model Registry.

    Parameters
    ----------
    run_id:
        MLflow run_id that contains the logged model artifact.
    name:
        Registered model name (default: REGISTRY_NAME = "FuelBurn").
    artifact_path:
        Path within the run's artifact store where the model was logged
        (default: "model", matching train.py's ``mlflow.pytorch.log_model``
        call).

    Returns
    -------
    version : str
        The newly created model version string (e.g. "1", "2", ...).

    Raises
    ------
    ValueError
        If the run is tagged as stale (pre-leakage).
    mlflow.exceptions.MlflowException
        On any MLflow API error.
    """
    client = MlflowClient()

    if _is_stale(client, run_id):
        raise ValueError(
            f"Run {run_id} is tagged as stale (pre-leakage). "
            "Call tag_stale_pre_leakage_runs() first, then use a post-fix run."
        )

    _ensure_model_registered(client, name)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Registering model from run %s (artifact: %s) as '%s'.", run_id, artifact_path, name)

    mv = mlflow.register_model(model_uri=model_uri, name=name)
    # MLflow 3.x returns version as int; normalise to str so callers can
    # pass it directly to set_registered_model_alias / get_model_version.
    version = str(mv.version)
    logger.info("Registered model '%s' version %s from run %s.", name, version, run_id)
    return version


def promote_to_production(
    version: str,
    name: str = REGISTRY_NAME,
    archive_previous: bool = True,
) -> None:
    """Assign the production alias to a specific model version.

    Sets both ``production`` and ``champion`` aliases on ``version``.
    If ``archive_previous`` is True (default), the prior champion's aliases
    are removed (the version itself is not deleted — it remains accessible by
    version number for audit).

    Parameters
    ----------
    version:
        The model version string to promote.
    name:
        Registered model name.
    archive_previous:
        If True, remove the production/champion aliases from the old champion.
    """
    client = MlflowClient()

    if archive_previous:
        old_version = _get_current_production_version(client, name)
        if old_version is not None and old_version != version:
            try:
                client.delete_registered_model_alias(name, ALIAS_PRODUCTION)
                logger.info(
                    "Removed '%s' alias from version %s of '%s'.",
                    ALIAS_PRODUCTION, old_version, name,
                )
            except mlflow.exceptions.MlflowException:
                pass  # alias may not exist yet
            try:
                client.delete_registered_model_alias(name, ALIAS_CHAMPION)
                logger.info(
                    "Removed '%s' alias from version %s of '%s'.",
                    ALIAS_CHAMPION, old_version, name,
                )
            except mlflow.exceptions.MlflowException:
                pass

    client.set_registered_model_alias(name, ALIAS_PRODUCTION, version)
    client.set_registered_model_alias(name, ALIAS_CHAMPION, version)
    logger.info(
        "Promoted '%s' version %s to production (aliases: %s, %s).",
        name, version, ALIAS_PRODUCTION, ALIAS_CHAMPION,
    )
    logger.info("Production URI: models:/%s@%s", name, ALIAS_PRODUCTION)


def compare_and_promote(
    candidate_run_id: str,
    name: str = REGISTRY_NAME,
    artifact_path: str = "model",
) -> dict:
    """Champion/challenger promotion: register candidate and promote if it wins.

    Workflow
    --------
    1. Validate: reject stale (pre-leakage) run_ids immediately.
    2. Register the candidate run's model as a new version (challenger).
    3. Set the ``challenger`` alias on that version.
    4. Compare candidate RMSE vs current production (champion) RMSE.
       - If no champion exists: promote candidate unconditionally.
       - If candidate RMSE < champion RMSE (lower = better): promote.
       - If candidate RMSE >= champion RMSE: log and leave champion as-is.
    5. Remove the ``challenger`` alias regardless of outcome (keeps aliases
       clean — no perpetual "challenger" alias cluttering the registry).

    Parameters
    ----------
    candidate_run_id:
        MLflow run_id of the challenger run.
    name:
        Registered model name.
    artifact_path:
        Artifact path within the run (default: "model").

    Returns
    -------
    result : dict with keys:
      promoted         : bool
      candidate_version: str   — newly registered version number
      candidate_rmse   : float | None
      champion_version : str | None — previous champion version (if any)
      champion_rmse    : float | None
      reason           : str   — human-readable decision rationale
    """
    client = MlflowClient()

    # --- 1. Stale guard ---
    if _is_stale(client, candidate_run_id):
        raise ValueError(
            f"Candidate run {candidate_run_id} is tagged as stale (pre-leakage). "
            "Only post-fix runs may be promoted to production."
        )

    # --- 2. Register candidate ---
    candidate_version = register_run(candidate_run_id, name=name, artifact_path=artifact_path)
    candidate_rmse = _get_run_rmse(client, candidate_run_id)

    # --- 3. Tag as challenger ---
    try:
        client.set_registered_model_alias(name, ALIAS_CHALLENGER, candidate_version)
    except mlflow.exceptions.MlflowException as exc:
        logger.warning("Could not set challenger alias: %s", exc)

    # --- 4. Get current champion ---
    champion_version = _get_current_production_version(client, name)
    champion_rmse: Optional[float] = None

    if champion_version is not None:
        # Retrieve the run_id of the current champion to get its RMSE.
        _suppress_stage_warnings()
        try:
            champ_mv = client.get_model_version(name, champion_version)
            champ_run_id = champ_mv.run_id
            if champ_run_id:
                champion_rmse = _get_run_rmse(client, champ_run_id)
        except mlflow.exceptions.MlflowException as exc:
            logger.warning("Could not retrieve champion run metrics: %s", exc)

    # --- 4b. Decision ---
    if champion_version is None:
        # No champion yet — promote unconditionally.
        promoted = True
        reason = "No existing champion; promoting candidate unconditionally."
    elif candidate_rmse is None:
        # No metric on candidate — cannot compare; do not promote.
        promoted = False
        reason = (
            f"Candidate run {candidate_run_id} has no RMSE metric "
            f"({PRIMARY_METRIC} or {FALLBACK_METRIC}). Champion retained."
        )
    elif champion_rmse is None:
        # Champion has no metric — promote candidate (it at least has a metric).
        promoted = True
        reason = (
            f"Champion version {champion_version} has no RMSE metric. "
            f"Candidate RMSE {candidate_rmse:.2f} kg wins by default."
        )
    elif candidate_rmse < champion_rmse:
        promoted = True
        reason = (
            f"Candidate RMSE {candidate_rmse:.2f} kg < "
            f"champion RMSE {champion_rmse:.2f} kg — challenger wins."
        )
    else:
        promoted = False
        reason = (
            f"Candidate RMSE {candidate_rmse:.2f} kg >= "
            f"champion RMSE {champion_rmse:.2f} kg — champion retained."
        )

    logger.info("Promotion decision: %s", reason)

    # --- 5. Act on decision ---
    if promoted:
        promote_to_production(candidate_version, name=name, archive_previous=True)

    # Clean up challenger alias (regardless of outcome)
    try:
        client.delete_registered_model_alias(name, ALIAS_CHALLENGER)
    except mlflow.exceptions.MlflowException:
        pass  # alias may not exist if set failed above

    return {
        "promoted": promoted,
        "candidate_version": candidate_version,
        "candidate_rmse": candidate_rmse,
        "champion_version": champion_version,
        "champion_rmse": champion_rmse,
        "reason": reason,
    }


def tag_stale_pre_leakage_runs(
    experiment_name: str = "FuelBurn_Baseline",
    dry_run: bool = False,
) -> list[str]:
    """Tag pre-leakage stale runs in the given experiment.

    Stale identification criteria:
    Run tag ``data.leakage_fix == "pre"`` (explicit pre-fix marker), or a
    pre-existing ``registry.stale == "true"`` tag.

    For each stale run the following tags are set:
      registry.stale          = "true"
      registry.stale_reason   = "pre_leakage_fix_ml01"
      registry.stale_since    = <UTC ISO timestamp>

    Parameters
    ----------
    experiment_name:
        Name of the MLflow experiment to scan.
    dry_run:
        If True, identify stale runs but do NOT write any tags (log only).

    Returns
    -------
    stale_run_ids : list[str]
        Run IDs that were identified (and tagged, unless dry_run=True) as stale.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(
            "Experiment '%s' not found. No runs to tag.", experiment_name
        )
        return []

    experiment_id = experiment.experiment_id

    # Fetch all runs in the experiment (active + deleted).
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=mlflow.entities.ViewType.ALL,
        output_format="list",
    )

    stale_run_ids: list[str] = []
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    for run in runs_df:
        run_id = run.info.run_id
        tags = run.data.tags
        # Already tagged — count it but don't re-tag.
        if tags.get("registry.stale") == "true":
            stale_run_ids.append(run_id)
            logger.debug("Run %s already tagged stale — skipping.", run_id)
            continue

        is_stale_flag = False
        stale_reason = "unknown"

        # Explicit provenance is the only safe stale-run signal.
        if tags.get("data.leakage_fix") == "pre":
            is_stale_flag = True
            stale_reason = "pre_leakage_fix_ml01 (explicit tag)"

        if is_stale_flag:
            stale_run_ids.append(run_id)
            logger.info(
                "Stale run identified: %s  reason: %s  dry_run=%s",
                run_id, stale_reason, dry_run,
            )
            if not dry_run:
                client.set_tag(run_id, "registry.stale", "true")
                client.set_tag(run_id, "registry.stale_reason", "pre_leakage_fix_ml01")
                client.set_tag(run_id, "registry.stale_since", now_iso)
        else:
            logger.debug("Run %s has no explicit stale provenance tag.", run_id)

    logger.info(
        "tag_stale_pre_leakage_runs: %d stale run(s) found in '%s' "
        "(dry_run=%s).",
        len(stale_run_ids), experiment_name, dry_run,
    )
    return stale_run_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.registry",
        description="MLflow Model Registry promotion workflow for FuelBurn.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- promote ---
    p_promote = sub.add_parser(
        "promote",
        help="Register a run's model and promote to production if it beats the champion.",
    )
    p_promote.add_argument(
        "--run-id",
        required=True,
        help="MLflow run_id of the candidate (challenger) run.",
    )
    p_promote.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path within the run (default: 'model').",
    )
    p_promote.add_argument(
        "--name",
        default=REGISTRY_NAME,
        help=f"Registered model name (default: {REGISTRY_NAME}).",
    )
    p_promote.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env or ./mlflow.db).",
    )

    # --- tag-stale ---
    p_stale = sub.add_parser(
        "tag-stale",
        help="Tag pre-leakage stale runs in an experiment.",
    )
    p_stale.add_argument(
        "--experiment",
        default="FuelBurn_Baseline",
        help="Experiment name to scan (default: FuelBurn_Baseline).",
    )
    p_stale.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stale runs without writing any tags.",
    )
    p_stale.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI.",
    )

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    if args.command == "promote":
        result = compare_and_promote(
            candidate_run_id=args.run_id,
            name=args.name,
            artifact_path=args.artifact_path,
        )
        print("\n=== Promotion Result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if result["promoted"]:
            print(f"\nProduction URI: {PRODUCTION_URI}")
        print()

    elif args.command == "tag-stale":
        stale_ids = tag_stale_pre_leakage_runs(
            experiment_name=args.experiment,
            dry_run=args.dry_run,
        )
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Tagged {len(stale_ids)} stale run(s):")
        for rid in stale_ids:
            print(f"  {rid}")
        print()


if __name__ == "__main__":
    main()
