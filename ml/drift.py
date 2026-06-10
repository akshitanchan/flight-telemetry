"""
ml/drift.py — Evidently-based feature-drift monitor with retrain trigger.
=========================================================================

Design overview
---------------
The monitor compares a *reference* DataFrame (the distribution the model was
trained on) against a *current* DataFrame (a recent production batch) over the
11 numeric trajectory features defined in ``ml.features._NUMERIC_FEATURES``.

The 27 aircraft-type one-hot columns (``ac_*``) are collapsed into a single
categorical column ``aircraft_type_cat`` so that Evidently's chi-square test
evaluates the aircraft-type marginal distribution rather than individually
testing 27 near-zero binary columns.  The final monitored column set is
therefore: 11 numeric + 1 categorical = 12 columns.

Evidently report used
---------------------
``evidently.legacy.metric_preset.DataDriftPreset``

This is the standard Evidently v2 batch drift preset.  For each column it
selects an appropriate statistical test automatically:
  - Numeric  : Kolmogorov-Smirnov (p-value threshold 0.05)
  - Categorical : chi-square (p-value threshold 0.05)

``DataDriftPreset(drift_share=…)`` controls the fraction of columns that must
be detected as drifted before the *dataset-level* ``dataset_drift`` flag is
set to ``True``.

Return value
------------
``run_drift_report()`` returns a ``DriftResult`` dataclass:

    dataset_drift     : bool   — True if share_of_drifted_columns ≥ drift_share
    share_drifted     : float  — fraction of monitored columns that drifted
    n_drifted         : int    — absolute count of drifted columns
    n_columns         : int    — total monitored columns
    per_feature       : dict   — {column_name: {"drift_detected": bool,
                                                 "drift_score": float,
                                                 "stattest": str}}

Retrain trigger
---------------
``should_retrain(result, threshold)``
  Returns True when ``result.share_drifted >= threshold``.
  Default threshold is 0.2 (20% of monitored columns drifted).

``trigger_retrain(hook=None, **kwargs)``
  Calls the injected ``hook`` callable (if provided) or falls back to
  invoking ``ml.cv.run_cv`` (the existing production training entrypoint)
  with bounded parameters for the offline demo.

  Making the hook injectable is the key design choice that allows unit tests
  to assert the trigger fired (by passing a ``MagicMock``) without starting
  a real training run.

Leakage prevention
------------------
The drift monitor operates on already-split data passed in from the outside.
No normalization statistics or vocabulary is computed here; the monitor is
purely distributional (shape/rank statistics).

Offline demo
------------
Run as a module::

    python -m ml.drift

This builds reference and drifted batches purely from synthetic data (NumPy /
mock data helpers — no DB, no PRC files), runs the monitor twice, and prints
the results:

  Scenario 1: reference vs reference-like current → no drift → no retrain
  Scenario 2: reference vs heavily drifted current → drift detected → retrain fires
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature schema — imported from the single source of truth
# ---------------------------------------------------------------------------

from ml.features import AIRCRAFT_TYPES  # noqa: E402 (keep import at top of section)

# The 11 numeric trajectory features to monitor directly.
_NUMERIC_FEATURES: list[str] = [
    "duration_s",
    "alt_change",
    "avg_speed",
    "max_vrate",
    "avg_altitude",
    "max_altitude",
    "alt_std",
    "avg_track_change",
    "avg_mach",
    "avg_tas",
    "avg_cas",
]

# Synthetic categorical column name used in the Evidently report.
_AC_CAT_COLUMN = "aircraft_type_cat"

# All columns that the Evidently report monitors (12 total).
MONITORED_COLUMNS: list[str] = _NUMERIC_FEATURES + [_AC_CAT_COLUMN]

# Default drift threshold: if ≥ this share of monitored columns drift, trigger retrain.
DEFAULT_RETRAIN_THRESHOLD: float = 0.2


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _decode_aircraft_type(df: pd.DataFrame) -> str | None:
    """
    Decode the one-hot ``ac_*`` columns back to a single aircraft-type string.

    Returns the label of the first ``ac_*`` column equal to 1.0, or
    ``"__unknown__"`` when no column fires or when ``ac_*`` columns are absent.
    """
    ac_cols = [f"ac_{t}" for t in AIRCRAFT_TYPES]
    present = [c for c in ac_cols if c in df.columns]
    if not present:
        return "__unknown__"
    # Return first column that is 1 (assumes valid one-hot)
    for col in present:
        if col in df.columns and df[col].iloc[0] == 1.0:
            return col[3:]  # strip "ac_" prefix
    return "__unknown__"


def _add_aircraft_type_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``aircraft_type_cat`` column by decoding the one-hot ``ac_*`` columns.

    This is done row-wise and is used to pass a *single* categorical column to
    Evidently rather than 27 sparse binary columns.
    """
    df = df.copy()
    ac_cols = [f"ac_{t}" for t in AIRCRAFT_TYPES]
    present = [c for c in ac_cols if c in df.columns]

    if not present:
        df[_AC_CAT_COLUMN] = "__unknown__"
        return df

    # For each row pick the first column that == 1.0
    def _decode_row(row: pd.Series) -> str:
        for col in present:
            if row[col] == 1.0:
                return col[3:]  # strip "ac_"
        return "__unknown__"

    df[_AC_CAT_COLUMN] = df.apply(_decode_row, axis=1)
    return df


def prepare_monitor_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame containing only the 12 monitored columns.

    Steps:
    1. Decode one-hot ``ac_*`` columns → ``aircraft_type_cat`` (categorical).
    2. Select ``_NUMERIC_FEATURES + [_AC_CAT_COLUMN]``.
    3. Fill any missing numeric columns with 0.0 (graceful degradation).
    """
    df = _add_aircraft_type_cat(df)

    # Ensure all numeric features are present; fill missing with 0.0.
    for col in _NUMERIC_FEATURES:
        if col not in df.columns:
            logger.warning("Missing feature column '%s'; filling with 0.0", col)
            df[col] = 0.0

    return df[MONITORED_COLUMNS].copy()


# ---------------------------------------------------------------------------
# DriftResult
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Structured output of a single drift monitor run."""

    dataset_drift: bool
    """True when share_drifted >= the drift_share threshold used in the report."""

    share_drifted: float
    """Fraction of monitored columns where drift was detected (0.0 – 1.0)."""

    n_drifted: int
    """Absolute count of drifted monitored columns."""

    n_columns: int
    """Total number of monitored columns."""

    per_feature: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-column drift result: {col: {drift_detected, drift_score, stattest}}."""

    def __str__(self) -> str:
        lines = [
            f"DriftResult(dataset_drift={self.dataset_drift}, "
            f"share_drifted={self.share_drifted:.2%}, "
            f"{self.n_drifted}/{self.n_columns} columns drifted)",
        ]
        for col, info in self.per_feature.items():
            flag = "DRIFT" if info["drift_detected"] else "ok   "
            lines.append(
                f"  [{flag}] {col:<25s}  score={info['drift_score']:.4f}"
                f"  ({info['stattest']})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share: float = DEFAULT_RETRAIN_THRESHOLD,
) -> DriftResult:
    """
    Run an Evidently DataDriftPreset report over the 12 monitored columns.

    Parameters
    ----------
    reference:
        The reference distribution (e.g. training batch).  Must contain
        ``FEATURE_COLUMNS`` columns or a subset.
    current:
        The current production batch to compare against the reference.
    drift_share:
        The fraction of monitored columns that must drift before
        ``dataset_drift`` is set to True (passed directly to
        ``DataDriftPreset``).  Defaults to ``DEFAULT_RETRAIN_THRESHOLD``
        (0.20).

    Returns
    -------
    DriftResult
        Structured result with per-feature scores and a dataset-level flag.

    Notes
    -----
    Evidently requires at least 5 rows in both DataFrames to run statistical
    tests reliably; a ``ValueError`` is raised if either has fewer than 5 rows.
    """
    if len(reference) < 5 or len(current) < 5:
        raise ValueError(
            f"Both reference and current DataFrames must have >= 5 rows; "
            f"got reference={len(reference)}, current={len(current)}."
        )

    # Prepare the 12-column monitor DataFrames.
    ref_monitor = prepare_monitor_df(reference)
    cur_monitor = prepare_monitor_df(current)

    # Import Evidently here so the rest of the module can be imported even when
    # evidently is not yet installed (e.g. during dry-run linting).
    from evidently.legacy.report import Report  # type: ignore[import]
    from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore[import]

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=MONITORED_COLUMNS,
                drift_share=drift_share,
                # Numeric: K-S test (default); Categorical: chi-square (default)
            )
        ]
    )
    report.run(reference_data=ref_monitor, current_data=cur_monitor)
    raw = report.as_dict()

    # --- Parse DatasetDriftMetric ---
    dataset_result: dict = {}
    per_column_drift: dict = {}

    for metric in raw["metrics"]:
        metric_type = metric["metric"]
        result_data = metric["result"]

        if metric_type == "DatasetDriftMetric":
            dataset_result = result_data

        elif metric_type == "DataDriftTable":
            for col_name, col_data in result_data.get("drift_by_columns", {}).items():
                per_column_drift[col_name] = {
                    "drift_detected": bool(col_data.get("drift_detected", False)),
                    "drift_score": float(col_data.get("drift_score", float("nan"))),
                    "stattest": str(col_data.get("stattest_name", "unknown")),
                }

    return DriftResult(
        dataset_drift=bool(dataset_result.get("dataset_drift", False)),
        share_drifted=float(dataset_result.get("share_of_drifted_columns", 0.0)),
        n_drifted=int(dataset_result.get("number_of_drifted_columns", 0)),
        n_columns=int(dataset_result.get("number_of_columns", len(MONITORED_COLUMNS))),
        per_feature=per_column_drift,
    )


# ---------------------------------------------------------------------------
# Retrain decision + trigger
# ---------------------------------------------------------------------------

def should_retrain(result: DriftResult, threshold: float = DEFAULT_RETRAIN_THRESHOLD) -> bool:
    """
    Return True when the share of drifted columns exceeds *threshold*.

    The threshold is independent of the ``drift_share`` parameter passed to
    ``DataDriftPreset`` (which controls the dataset-level flag internally).
    Using ``should_retrain`` allows callers to apply a *different* decision
    boundary on top of the raw Evidently output — for example a lower threshold
    for a more aggressive retrain policy.

    Parameters
    ----------
    result:
        A ``DriftResult`` from ``run_drift_report``.
    threshold:
        Fraction of monitored columns that must drift to warrant retraining.
        Defaults to ``DEFAULT_RETRAIN_THRESHOLD`` (0.20).
    """
    return result.share_drifted >= threshold


def trigger_retrain(
    hook: Optional[Callable[..., Any]] = None,
    data_dir: str = "data/ml/prc_2025_mock",
    epochs: int = 1,
    **kwargs: Any,
) -> Any:
    """
    Fire the retrain hook.

    Parameters
    ----------
    hook:
        Optional injectable callable.  When provided it is called with
        ``(data_dir=data_dir, epochs=epochs, **kwargs)`` and its return
        value is returned.  Pass a ``unittest.mock.MagicMock`` in tests to
        verify the trigger fired without running real training.

    data_dir:
        Data directory forwarded to ``ml.cv.run_cv`` (only used when no hook
        is injected).

    epochs:
        Number of training epochs (bounded for demo/tests; default = 1).

    **kwargs:
        Extra keyword arguments forwarded to the hook or ``run_cv``.

    Returns
    -------
    The return value of the hook, or the result dict from ``run_cv``.
    """
    if hook is not None:
        logger.info("trigger_retrain: calling injected hook %r", hook)
        return hook(data_dir=data_dir, epochs=epochs, **kwargs)

    # No hook — call the existing training entrypoint directly.
    logger.info(
        "trigger_retrain: invoking ml.cv.run_cv(data_dir=%r, epochs=%d)",
        data_dir,
        epochs,
    )
    from ml.cv import run_cv  # local import to avoid circular deps

    return run_cv(data_dir=data_dir, epochs=epochs, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic batch builder (offline demo / testing)
# ---------------------------------------------------------------------------

def make_reference_batch(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic reference DataFrame over the 11 numeric features + ac_* one-hot.

    The distributions are calibrated to look like PRC-2025 mock trajectories:
      - duration_s  : LogNormal (median ≈ 3600s, i.e. ~1 h)
      - alt_change  : Normal(0, 2000)  ft
      - avg_speed   : Normal(200, 20)  kts
      - max_vrate   : HalfNormal(500)  ft/min
      - avg_altitude: Normal(30000, 5000) ft
      - max_altitude: avg_altitude + Uniform(0, 5000)
      - alt_std     : HalfNormal(2000)
      - avg_track_change: Uniform(0, 5)
      - avg_mach    : 0.0 (matches real ADS-B NaN → 0 fill)
      - avg_tas     : 0.0
      - avg_cas     : 0.0
      - aircraft_type: sampled from 4 known types (uniform)
    """
    rng = np.random.default_rng(seed)
    ac_types = ["A320", "B738", "A359", "B77W"]
    ac_probs = [0.35, 0.30, 0.20, 0.15]

    avg_alt = rng.normal(30_000, 5_000, n)
    rows = {
        "duration_s": np.exp(rng.normal(np.log(3600), 0.5, n)),
        "alt_change": rng.normal(0, 2_000, n),
        "avg_speed": rng.normal(200, 20, n),
        "max_vrate": np.abs(rng.normal(0, 500, n)),
        "avg_altitude": avg_alt,
        "max_altitude": avg_alt + rng.uniform(0, 5_000, n),
        "alt_std": np.abs(rng.normal(0, 2_000, n)),
        "avg_track_change": rng.uniform(0, 5, n),
        "avg_mach": np.zeros(n),
        "avg_tas": np.zeros(n),
        "avg_cas": np.zeros(n),
        "_ac_type_label": rng.choice(ac_types, n, p=ac_probs),
    }
    df = pd.DataFrame(rows)

    # Encode aircraft type as one-hot ac_* columns
    from ml.features import AIRCRAFT_TYPES as _AT, encode_aircraft_type

    for t in _AT:
        df[f"ac_{t}"] = 0.0
    for i, label in enumerate(df["_ac_type_label"]):
        enc = encode_aircraft_type(label)
        for j, t in enumerate(_AT):
            df.loc[i, f"ac_{t}"] = enc[j]

    return df.drop(columns=["_ac_type_label"])


def make_drifted_batch(n: int = 300, seed: int = 99) -> pd.DataFrame:
    """
    Build a *heavily drifted* batch to demonstrate drift detection.

    Drift injected:
      - duration_s  : shifted to much longer flights (mean ≈ 14 400s, i.e. 4 h)
      - avg_speed   : shifted from ~200 kts to ~450 kts (high-speed jets)
      - avg_altitude: shifted from ~30 000 ft to ~42 000 ft
      - max_vrate   : doubled
      - aircraft_type distribution: B77W and A388 dominate (was balanced mix)

    These are large, clearly detectable distributional shifts, not subtle noise.
    """
    rng = np.random.default_rng(seed)
    # Shift aircraft distribution heavily toward long-haul types
    ac_types = ["B77W", "A388", "B748", "A332"]
    ac_probs = [0.40, 0.30, 0.20, 0.10]

    avg_alt = rng.normal(42_000, 3_000, n)
    rows = {
        "duration_s": np.exp(rng.normal(np.log(14_400), 0.4, n)),
        "alt_change": rng.normal(0, 3_000, n),
        "avg_speed": rng.normal(450, 30, n),
        "max_vrate": np.abs(rng.normal(0, 1_000, n)),
        "avg_altitude": avg_alt,
        "max_altitude": avg_alt + rng.uniform(0, 3_000, n),
        "alt_std": np.abs(rng.normal(0, 3_500, n)),
        "avg_track_change": rng.uniform(0, 3, n),
        "avg_mach": np.zeros(n),
        "avg_tas": np.zeros(n),
        "avg_cas": np.zeros(n),
        "_ac_type_label": rng.choice(ac_types, n, p=ac_probs),
    }
    df = pd.DataFrame(rows)

    from ml.features import AIRCRAFT_TYPES as _AT, encode_aircraft_type

    for t in _AT:
        df[f"ac_{t}"] = 0.0
    for i, label in enumerate(df["_ac_type_label"]):
        enc = encode_aircraft_type(label)
        for j, t in enumerate(_AT):
            df.loc[i, f"ac_{t}"] = enc[j]

    return df.drop(columns=["_ac_type_label"])


# ---------------------------------------------------------------------------
# High-level monitor entry-point
# ---------------------------------------------------------------------------

def monitor(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    retrain_threshold: float = DEFAULT_RETRAIN_THRESHOLD,
    retrain_hook: Optional[Callable[..., Any]] = None,
    retrain_kwargs: Optional[dict] = None,
) -> tuple[DriftResult, bool]:
    """
    Full drift-monitor cycle: run the report, decide, optionally retrain.

    Parameters
    ----------
    reference:
        Reference distribution DataFrame (training batch).
    current:
        Current production batch to evaluate.
    retrain_threshold:
        Fraction of drifted columns needed to trigger retraining.
    retrain_hook:
        Optional injectable callable; if None and retrain is needed, calls
        ``ml.cv.run_cv``.
    retrain_kwargs:
        Extra kwargs forwarded to ``trigger_retrain``.

    Returns
    -------
    (result, retrained)
        ``result`` is the ``DriftResult``; ``retrained`` is True if the
        retrain trigger fired.
    """
    result = run_drift_report(
        reference=reference,
        current=current,
        drift_share=retrain_threshold,
    )

    retrained = False
    if should_retrain(result, threshold=retrain_threshold):
        logger.info(
            "Drift threshold exceeded (%.0f%% columns drifted >= %.0f%% threshold) — "
            "firing retrain trigger.",
            result.share_drifted * 100,
            retrain_threshold * 100,
        )
        trigger_retrain(hook=retrain_hook, **(retrain_kwargs or {}))
        retrained = True
    else:
        logger.info(
            "No retraining needed (%.0f%% columns drifted < %.0f%% threshold).",
            result.share_drifted * 100,
            retrain_threshold * 100,
        )

    return result, retrained


# ---------------------------------------------------------------------------
# Offline demo  (python -m ml.drift)
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Offline demo: reference vs same-distribution current (no drift) then
    reference vs heavily drifted current (drift → retrain fires).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  [%(name)s] %(message)s",
    )
    print()
    print("=" * 70)
    print("  Flight-Telemetry Feature-Drift Monitor — Offline Demo")
    print("=" * 70)

    # Build batches
    reference = make_reference_batch(n=400, seed=42)
    same_dist = make_reference_batch(n=400, seed=7)      # same distribution, different seed
    drifted = make_drifted_batch(n=400, seed=99)

    retrain_threshold = DEFAULT_RETRAIN_THRESHOLD  # 0.20

    # ------------------------------------------------------------------
    # Scenario 1: no drift
    # ------------------------------------------------------------------
    print()
    print("--- Scenario 1: reference vs same-distribution current ---")
    print(f"    threshold = {retrain_threshold:.0%}  (retrain if ≥ this fraction drifted)")
    print()

    mock_retrain = _make_mock_hook()
    result1, retrained1 = monitor(
        reference=reference,
        current=same_dist,
        retrain_threshold=retrain_threshold,
        retrain_hook=mock_retrain,
    )
    print(result1)
    print()
    if retrained1:
        print("  ACTION: Retrain triggered (unexpected for same-distribution batch)")
    else:
        print("  ACTION: No retrain — distribution stable.")

    # ------------------------------------------------------------------
    # Scenario 2: drifted batch
    # ------------------------------------------------------------------
    print()
    print("--- Scenario 2: reference vs drifted current ---")
    print(f"    threshold = {retrain_threshold:.0%}")
    print()

    mock_retrain2 = _make_mock_hook()
    result2, retrained2 = monitor(
        reference=reference,
        current=drifted,
        retrain_threshold=retrain_threshold,
        retrain_hook=mock_retrain2,
    )
    print(result2)
    print()
    if retrained2:
        print("  ACTION: Retrain triggered — drift threshold exceeded.")
    else:
        print("  ACTION: No retrain (unexpected; drift should have been detected).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Demo Summary")
    print("=" * 70)
    print(f"  Scenario 1 (same dist) — dataset_drift={result1.dataset_drift}"
          f"  share_drifted={result1.share_drifted:.1%}"
          f"  retrained={retrained1}")
    print(f"  Scenario 2 (drifted)   — dataset_drift={result2.dataset_drift}"
          f"  share_drifted={result2.share_drifted:.1%}"
          f"  retrained={retrained2}")

    assert not retrained1, "Scenario 1: should NOT have triggered a retrain."
    assert retrained2, "Scenario 2: SHOULD have triggered a retrain."
    print()
    print("  All demo assertions passed.")
    print()


def _make_mock_hook():
    """Return a simple callable that records whether it was called."""

    class _MockHook:
        called: bool = False

        def __call__(self, **kwargs):
            self.called = True
            logger.info("MOCK retrain hook invoked with kwargs=%r", kwargs)
            return {"mock": True}

    return _MockHook()


if __name__ == "__main__":
    _run_demo()
