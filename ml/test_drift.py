"""
ml/test_drift.py — Unit tests for the Evidently-based feature-drift monitor.
=============================================================================

Coverage
--------
1.  Identical reference / current batches → no drift, no retrain.
2.  Clearly drifted current batch → dataset_drift=True, retrain hook invoked
    exactly once.
3.  ``should_retrain`` returns False below threshold and True at/above it.
4.  ``DriftResult.__str__`` renders without error.
5.  ``prepare_monitor_df`` adds the ``aircraft_type_cat`` column and keeps all
    12 monitored columns.
6.  ``prepare_monitor_df`` tolerates missing ``ac_*`` columns gracefully.
7.  ``prepare_monitor_df`` fills missing numeric features with 0.0.
8.  ``run_drift_report`` raises ``ValueError`` when a batch has fewer than 5 rows.
9.  The injected retrain hook in ``monitor()`` is called exactly once on drift.
10. ``monitor()`` does NOT call the retrain hook when there is no drift.

All tests are fully offline: no real PRC data, no DB, no actual training run.
The retrain hook is always a ``unittest.mock.MagicMock`` so no model is trained.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.drift import (
    DEFAULT_RETRAIN_THRESHOLD,
    MONITORED_COLUMNS,
    DriftResult,
    make_drifted_batch,
    make_reference_batch,
    monitor,
    prepare_monitor_df,
    run_drift_report,
    should_retrain,
    trigger_retrain,
    _AC_CAT_COLUMN,
)
from ml.features import AIRCRAFT_TYPES, FEATURE_COLUMNS, encode_aircraft_type


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_batch() -> pd.DataFrame:
    """300-row synthetic reference batch (fixed seed, reused across tests)."""
    return make_reference_batch(n=300, seed=42)


@pytest.fixture(scope="module")
def same_dist_batch() -> pd.DataFrame:
    """300-row batch drawn from the same distribution as the reference."""
    return make_reference_batch(n=300, seed=7)


@pytest.fixture(scope="module")
def drifted_batch() -> pd.DataFrame:
    """300-row batch with injected large distributional drift."""
    return make_drifted_batch(n=300, seed=99)


# ---------------------------------------------------------------------------
# 1.  No-drift scenario: identical distribution → no drift, no retrain
# ---------------------------------------------------------------------------

class TestNoDriftScenario:
    def test_no_dataset_drift_same_distribution(self, reference_batch, same_dist_batch):
        """Same-distribution batches must not trigger dataset-level drift."""
        result = run_drift_report(reference_batch, same_dist_batch)
        assert result.dataset_drift is False, (
            f"Expected no dataset drift but got dataset_drift=True; "
            f"share_drifted={result.share_drifted:.2%}"
        )

    def test_no_retrain_same_distribution(self, reference_batch, same_dist_batch):
        """should_retrain must return False for a stable distribution."""
        result = run_drift_report(reference_batch, same_dist_batch)
        assert not should_retrain(result, threshold=DEFAULT_RETRAIN_THRESHOLD)

    def test_monitor_does_not_call_hook_no_drift(self, reference_batch, same_dist_batch):
        """monitor() must NOT invoke the retrain hook when no drift is detected."""
        hook = MagicMock()
        _result, retrained = monitor(
            reference=reference_batch,
            current=same_dist_batch,
            retrain_threshold=DEFAULT_RETRAIN_THRESHOLD,
            retrain_hook=hook,
        )
        assert not retrained
        hook.assert_not_called()

    def test_drift_result_has_per_feature_entries(self, reference_batch, same_dist_batch):
        """Per-feature dict must contain entries for all monitored columns."""
        result = run_drift_report(reference_batch, same_dist_batch)
        assert len(result.per_feature) == len(MONITORED_COLUMNS), (
            f"Expected {len(MONITORED_COLUMNS)} per-feature entries, "
            f"got {len(result.per_feature)}"
        )


# ---------------------------------------------------------------------------
# 2.  Drift scenario: clearly drifted batch → dataset_drift=True, retrain fires
# ---------------------------------------------------------------------------

class TestDriftScenario:
    def test_dataset_drift_detected(self, reference_batch, drifted_batch):
        """A heavily drifted batch must set dataset_drift=True."""
        result = run_drift_report(reference_batch, drifted_batch)
        assert result.dataset_drift is True, (
            f"Expected dataset_drift=True but got False; "
            f"share_drifted={result.share_drifted:.2%}"
        )

    def test_share_drifted_above_threshold(self, reference_batch, drifted_batch):
        """share_drifted must exceed DEFAULT_RETRAIN_THRESHOLD for the drifted batch."""
        result = run_drift_report(reference_batch, drifted_batch)
        assert result.share_drifted >= DEFAULT_RETRAIN_THRESHOLD, (
            f"share_drifted={result.share_drifted:.2%} < "
            f"threshold={DEFAULT_RETRAIN_THRESHOLD:.2%}"
        )

    def test_should_retrain_returns_true(self, reference_batch, drifted_batch):
        """should_retrain must return True for the drifted batch."""
        result = run_drift_report(reference_batch, drifted_batch)
        assert should_retrain(result, threshold=DEFAULT_RETRAIN_THRESHOLD)

    def test_retrain_hook_invoked_exactly_once(self, reference_batch, drifted_batch):
        """The injected retrain hook must be called exactly once on drift."""
        hook = MagicMock(return_value={"mock": True})
        _result, retrained = monitor(
            reference=reference_batch,
            current=drifted_batch,
            retrain_threshold=DEFAULT_RETRAIN_THRESHOLD,
            retrain_hook=hook,
        )
        assert retrained is True
        hook.assert_called_once()

    def test_n_drifted_positive(self, reference_batch, drifted_batch):
        """At least one column must be flagged as drifted."""
        result = run_drift_report(reference_batch, drifted_batch)
        assert result.n_drifted > 0

    def test_per_feature_has_drift_detected_true_for_shifted_cols(
        self, reference_batch, drifted_batch
    ):
        """duration_s and avg_speed are strongly shifted — they must be drifted."""
        result = run_drift_report(reference_batch, drifted_batch)
        for col in ("duration_s", "avg_speed"):
            assert result.per_feature[col]["drift_detected"] is True, (
                f"Expected column '{col}' to be drifted but drift_detected=False"
            )


# ---------------------------------------------------------------------------
# 3.  should_retrain threshold logic
# ---------------------------------------------------------------------------

class TestShouldRetrain:
    def _make_result(self, share: float) -> DriftResult:
        n = 12
        n_drifted = round(share * n)
        return DriftResult(
            dataset_drift=(share >= DEFAULT_RETRAIN_THRESHOLD),
            share_drifted=share,
            n_drifted=n_drifted,
            n_columns=n,
            per_feature={},
        )

    def test_below_threshold_returns_false(self):
        result = self._make_result(0.10)
        assert not should_retrain(result, threshold=0.20)

    def test_at_threshold_returns_true(self):
        result = self._make_result(0.20)
        assert should_retrain(result, threshold=0.20)

    def test_above_threshold_returns_true(self):
        result = self._make_result(0.50)
        assert should_retrain(result, threshold=0.20)

    def test_zero_drift_always_false(self):
        result = self._make_result(0.0)
        assert not should_retrain(result, threshold=0.01)

    def test_full_drift_always_true(self):
        result = self._make_result(1.0)
        assert should_retrain(result, threshold=0.99)

    def test_custom_tight_threshold(self):
        """A very tight threshold (5%) fires even for minor drift."""
        result = self._make_result(0.08)
        assert should_retrain(result, threshold=0.05)


# ---------------------------------------------------------------------------
# 4.  DriftResult.__str__ renders cleanly
# ---------------------------------------------------------------------------

class TestDriftResultStr:
    def test_str_contains_dataset_drift(self, reference_batch, drifted_batch):
        result = run_drift_report(reference_batch, drifted_batch)
        text = str(result)
        assert "dataset_drift=" in text

    def test_str_contains_feature_names(self, reference_batch, drifted_batch):
        result = run_drift_report(reference_batch, drifted_batch)
        text = str(result)
        assert "duration_s" in text
        assert _AC_CAT_COLUMN in text

    def test_str_no_exception_empty_per_feature(self):
        result = DriftResult(
            dataset_drift=False,
            share_drifted=0.0,
            n_drifted=0,
            n_columns=12,
            per_feature={},
        )
        text = str(result)
        assert "dataset_drift=False" in text


# ---------------------------------------------------------------------------
# 5.  prepare_monitor_df adds aircraft_type_cat and selects 12 columns
# ---------------------------------------------------------------------------

class TestPrepareMonitorDf:
    def _make_full_row(self, ac_type: str = "A320") -> pd.DataFrame:
        """Build a 1-row DataFrame with all FEATURE_COLUMNS."""
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        enc = encode_aircraft_type(ac_type)
        for i, t in enumerate(AIRCRAFT_TYPES):
            row[f"ac_{t}"] = enc[i]
        return pd.DataFrame([row])

    def test_output_has_exactly_12_columns(self):
        df = make_reference_batch(n=10, seed=0)
        out = prepare_monitor_df(df)
        assert list(out.columns) == MONITORED_COLUMNS

    def test_aircraft_type_cat_column_present(self):
        df = make_reference_batch(n=10, seed=0)
        out = prepare_monitor_df(df)
        assert _AC_CAT_COLUMN in out.columns

    def test_aircraft_type_cat_decodes_correctly(self):
        df = self._make_full_row("B738")
        out = prepare_monitor_df(df)
        assert out[_AC_CAT_COLUMN].iloc[0] == "B738"

    def test_aircraft_type_cat_unknown_when_no_ac_cols(self):
        """If no ac_* columns are present, _AC_CAT_COLUMN should be '__unknown__'."""
        df = pd.DataFrame({col: [0.0] for col in MONITORED_COLUMNS[:-1]})
        # No ac_* columns — prepare_monitor_df should fill with __unknown__
        out = prepare_monitor_df(df)
        assert out[_AC_CAT_COLUMN].iloc[0] == "__unknown__"

    def test_no_nan_in_output_numeric_cols(self):
        df = make_reference_batch(n=50, seed=1)
        out = prepare_monitor_df(df)
        numeric_cols = [c for c in MONITORED_COLUMNS if c != _AC_CAT_COLUMN]
        assert not out[numeric_cols].isna().any().any()


# ---------------------------------------------------------------------------
# 6.  Missing ac_* columns handled gracefully
# ---------------------------------------------------------------------------

class TestMissingAcCols:
    def test_missing_ac_cols_does_not_raise(self):
        """prepare_monitor_df must not raise when ac_* columns are absent."""
        df = pd.DataFrame(
            {col: np.random.normal(0, 1, 20) for col in _NUMERIC_FEATURES_ONLY()}
        )
        out = prepare_monitor_df(df)
        assert _AC_CAT_COLUMN in out.columns

    def test_run_drift_report_without_ac_cols(self):
        """run_drift_report must complete even when ac_* columns are absent."""
        df = pd.DataFrame(
            {col: np.random.normal(0, 1, 100) for col in _NUMERIC_FEATURES_ONLY()}
        )
        result = run_drift_report(df, df.copy())
        # Identical data → no drift
        assert result.dataset_drift is False


# ---------------------------------------------------------------------------
# 7.  Missing numeric features filled with 0.0
# ---------------------------------------------------------------------------

class TestMissingNumericFeatures:
    def test_missing_feature_filled_with_zero(self):
        """A DataFrame with one numeric feature missing gets a 0.0-filled column."""
        df = make_reference_batch(n=20, seed=0)
        df = df.drop(columns=["avg_mach"])
        out = prepare_monitor_df(df)
        assert "avg_mach" in out.columns
        assert (out["avg_mach"] == 0.0).all()


# ---------------------------------------------------------------------------
# 8.  run_drift_report raises ValueError for too-small batches
# ---------------------------------------------------------------------------

class TestSmallBatchValidation:
    def test_raises_on_too_few_rows_reference(self):
        tiny = pd.DataFrame(
            {col: [0.0] * 4 for col in _NUMERIC_FEATURES_ONLY()}
        )
        normal = pd.DataFrame(
            {col: np.random.normal(0, 1, 100) for col in _NUMERIC_FEATURES_ONLY()}
        )
        with pytest.raises(ValueError, match=r">= 5 rows"):
            run_drift_report(tiny, normal)

    def test_raises_on_too_few_rows_current(self):
        normal = pd.DataFrame(
            {col: np.random.normal(0, 1, 100) for col in _NUMERIC_FEATURES_ONLY()}
        )
        tiny = pd.DataFrame(
            {col: [0.0] * 3 for col in _NUMERIC_FEATURES_ONLY()}
        )
        with pytest.raises(ValueError, match=r">= 5 rows"):
            run_drift_report(normal, tiny)


# ---------------------------------------------------------------------------
# 9.  trigger_retrain uses the hook when provided
# ---------------------------------------------------------------------------

class TestTriggerRetrain:
    def test_hook_called_with_expected_kwargs(self):
        hook = MagicMock(return_value=42)
        result = trigger_retrain(hook=hook, data_dir="mock_dir", epochs=1)
        hook.assert_called_once_with(data_dir="mock_dir", epochs=1)
        assert result == 42

    def test_hook_not_none_skips_run_cv(self):
        """When a hook is provided, ml.cv.run_cv must never be called."""
        hook = MagicMock()
        # run_cv is lazily imported inside trigger_retrain from ml.cv,
        # so the correct patch target is the source module, not ml.drift.
        with patch("ml.cv.run_cv") as mock_cv:
            trigger_retrain(hook=hook, data_dir="x", epochs=1)
            mock_cv.assert_not_called()


# ---------------------------------------------------------------------------
# 10. monitor() retrain_kwargs forwarded to hook
# ---------------------------------------------------------------------------

class TestMonitorKwargs:
    def test_retrain_kwargs_forwarded(self, reference_batch, drifted_batch):
        """retrain_kwargs must be forwarded to the hook on drift."""
        hook = MagicMock()
        monitor(
            reference=reference_batch,
            current=drifted_batch,
            retrain_threshold=DEFAULT_RETRAIN_THRESHOLD,
            retrain_hook=hook,
            retrain_kwargs={"data_dir": "mock", "epochs": 1},
        )
        # Hook should have been called with at least data_dir and epochs
        _, call_kwargs = hook.call_args
        assert call_kwargs.get("data_dir") == "mock"
        assert call_kwargs.get("epochs") == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _NUMERIC_FEATURES_ONLY() -> list[str]:
    """Return just the 11 numeric feature names (no ac_* columns)."""
    return [
        "duration_s", "alt_change", "avg_speed", "max_vrate",
        "avg_altitude", "max_altitude", "alt_std", "avg_track_change",
        "avg_mach", "avg_tas", "avg_cas",
    ]
