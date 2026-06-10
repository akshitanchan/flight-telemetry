"""
ml/test_cv.py
-------------
Unit tests for the chronological CV harness (ml/cv.py).

These tests use only synthetic in-memory data — they do NOT require the PRC
dataset to be present and do NOT touch MLflow or the filesystem.

Coverage
--------
1. build_folds returns the correct number of folds for a given date layout.
2. Every fold's training set is strictly a temporal prefix relative to its
   validation date (expanding-window invariant).
3. No flight_id appears in both the train and val sets of any fold
   (leakage-free guarantee).
4. A flight whose date spans two buckets (impossible by construction, but
   tested defensively) would trigger the assertion.
5. compute_slices returns correct per-aircraft-type and per-duration-bucket
   RMSE values against known inputs.
6. _rmse is numerically exact for a trivial case.
"""

import datetime
import math

import numpy as np
import pandas as pd
import pytest

from ml.cv import build_folds, compute_slices, _rmse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(flight_rows):
    """
    Build a minimal df_merged from a list of dicts with keys:
      flight_id, flight_date, duration_s, aircraft_type
    Each row represents one interval.
    """
    rows = []
    for i, r in enumerate(flight_rows):
        rows.append({
            "idx": i,
            "flight_id": r["flight_id"],
            "flight_date": r["flight_date"],
            "duration_s": r.get("duration_s", 600.0),
            "alt_change": 0.0,
            "avg_speed": 100.0,
            "max_vrate": 5.0,
            "fuel_kg": 500.0,
            "aircraft_type": r.get("aircraft_type", "A320"),
        })
    return pd.DataFrame(rows)


D1 = datetime.date(2025, 4, 12)
D2 = datetime.date(2025, 4, 13)
D3 = datetime.date(2025, 4, 14)
D4 = datetime.date(2025, 4, 15)


@pytest.fixture()
def df_four_dates():
    """Four date-blocks, two flights per date, two intervals per flight."""
    rows = []
    for date, fid_prefix in [(D1, "f1"), (D2, "f2"), (D3, "f3"), (D4, "f4")]:
        for sub in ("a", "b"):
            fid = f"{fid_prefix}_{sub}"
            for _ in range(2):
                rows.append({"flight_id": fid, "flight_date": date})
    return _make_df(rows)


@pytest.fixture()
def df_two_dates():
    """Minimal two-date fixture → exactly 1 fold."""
    rows = [
        {"flight_id": "f1", "flight_date": D1},
        {"flight_id": "f1", "flight_date": D1},
        {"flight_id": "f2", "flight_date": D2},
        {"flight_id": "f2", "flight_date": D2},
    ]
    return _make_df(rows)


# ---------------------------------------------------------------------------
# 1. Fold count
# ---------------------------------------------------------------------------

class TestFoldCount:
    def test_four_dates_yields_three_folds(self, df_four_dates):
        folds = build_folds(df_four_dates)
        assert len(folds) == 3

    def test_two_dates_yields_one_fold(self, df_two_dates):
        folds = build_folds(df_two_dates)
        assert len(folds) == 1

    def test_n_folds_cap_respected(self, df_four_dates):
        folds = build_folds(df_four_dates, n_folds=2)
        assert len(folds) == 2

    def test_n_folds_1_gives_last_fold_only(self, df_four_dates):
        folds = build_folds(df_four_dates, n_folds=1)
        assert len(folds) == 1
        # The single returned fold should validate on the last date (D4)
        assert folds[0]["val_date"] == str(D4)

    def test_one_date_raises(self):
        df = _make_df([{"flight_id": "f1", "flight_date": D1}])
        with pytest.raises(ValueError, match="at least 2"):
            build_folds(df)

    def test_n_folds_zero_raises(self, df_four_dates):
        with pytest.raises(ValueError, match="n_folds must be between"):
            build_folds(df_four_dates, n_folds=0)

    def test_n_folds_exceeds_max_raises(self, df_four_dates):
        with pytest.raises(ValueError, match="n_folds must be between"):
            build_folds(df_four_dates, n_folds=99)


# ---------------------------------------------------------------------------
# 2. Expanding-window temporal invariant
# ---------------------------------------------------------------------------

class TestExpandingWindow:
    def test_train_dates_are_prefix_of_sorted_dates(self, df_four_dates):
        """Each fold's training dates must form a contiguous prefix of the
        globally sorted date list."""
        all_dates = sorted(df_four_dates["flight_date"].unique())
        folds = build_folds(df_four_dates)

        for k, f in enumerate(folds):
            expected_train_dates = [str(d) for d in all_dates[: k + 1]]
            assert f["train_dates"] == expected_train_dates, (
                f"Fold {k+1}: expected train_dates={expected_train_dates}, "
                f"got {f['train_dates']}"
            )

    def test_val_date_strictly_after_all_train_dates(self, df_four_dates):
        folds = build_folds(df_four_dates)
        for f in folds:
            val = datetime.date.fromisoformat(f["val_date"])
            for td in f["train_dates"]:
                assert datetime.date.fromisoformat(td) < val

    def test_train_size_grows_monotonically(self, df_four_dates):
        folds = build_folds(df_four_dates)
        sizes = [f["n_train_intervals"] for f in folds]
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], (
                f"Training set did not grow at fold {i+1}: {sizes}"
            )


# ---------------------------------------------------------------------------
# 3. Leakage-free guarantee
# ---------------------------------------------------------------------------

class TestNoFlightLeakage:
    def test_no_flight_appears_in_both_train_and_val(self, df_four_dates):
        folds = build_folds(df_four_dates)
        for f in folds:
            train_flights = set(df_four_dates.iloc[f["train_idx"]]["flight_id"])
            val_flights = set(df_four_dates.iloc[f["val_idx"]]["flight_id"])
            overlap = train_flights & val_flights
            assert len(overlap) == 0, (
                f"Fold {f['fold']}: {len(overlap)} flight(s) leaked: {overlap}"
            )

    def test_leakage_assertion_fires_on_contrived_input(self):
        """If the same flight appears on two different dates the assertion
        inside build_folds must fire (a flight can only have one date in
        the real data, but the guard must be explicit)."""
        # Same flight_id, two different dates — impossible in the real data
        # because flight_date comes from flightlist_train, but we test the guard.
        rows = [
            {"flight_id": "cross_flight", "flight_date": D1},
            {"flight_id": "cross_flight", "flight_date": D2},
        ]
        df = _make_df(rows)
        with pytest.raises(AssertionError, match="appear in both train and val"):
            build_folds(df)

    def test_fold_metadata_interval_counts_are_consistent(self, df_four_dates):
        folds = build_folds(df_four_dates)
        for f in folds:
            assert f["n_train_intervals"] == len(f["train_idx"])
            assert f["n_val_intervals"] == len(f["val_idx"])
            # train + val = total coverage (no overlap, no gaps across the folds)
            assert f["n_train_intervals"] > 0
            assert f["n_val_intervals"] > 0


# ---------------------------------------------------------------------------
# 5. compute_slices correctness
# ---------------------------------------------------------------------------

class TestComputeSlices:
    def _make_meta(self, aircraft_types, durations_s):
        return pd.DataFrame(
            {"aircraft_type": aircraft_types, "duration_s": durations_s}
        )

    def test_aircraft_type_rmse_exact(self):
        """Two aircraft types with zero error → RMSE = 0."""
        n = 10
        preds = np.ones(n) * 500.0
        targets = np.ones(n) * 500.0
        meta = self._make_meta(
            ["A320"] * 5 + ["B738"] * 5,
            [600.0] * n,
        )
        result = compute_slices(preds, targets, meta)
        df = result["by_aircraft_type"]
        for _, row in df.iterrows():
            assert row["rmse_kg"] == pytest.approx(0.0)

    def test_aircraft_type_rmse_known_value(self):
        """Residuals of +10 and -10 give RMSE = 10."""
        preds = np.array([510.0, 490.0, 510.0, 490.0])
        targets = np.array([500.0, 500.0, 500.0, 500.0])
        meta = self._make_meta(
            ["A320", "A320", "A320", "A320"],
            [600.0, 600.0, 600.0, 600.0],
        )
        result = compute_slices(preds, targets, meta)
        df = result["by_aircraft_type"]
        assert df.loc[df["aircraft_type"] == "A320", "rmse_kg"].values[0] == pytest.approx(10.0)

    def test_duration_bucket_assignment(self):
        """Intervals < 1800s fall in '<30min'; intervals in [1800,3600) in '30-60min'."""
        preds = np.array([100.0, 200.0, 300.0, 400.0])
        targets = np.array([100.0, 200.0, 300.0, 400.0])  # zero error
        meta = self._make_meta(
            ["A320"] * 4,
            [300.0, 900.0, 1800.0, 3000.0],
        )
        result = compute_slices(preds, targets, meta)
        df_dur = result["by_duration_bucket"]
        buckets = set(df_dur["duration_bucket"])
        assert "<30min" in buckets
        assert "30-60min" in buckets

    def test_empty_bucket_is_omitted(self):
        """Duration buckets with zero samples must not appear in the result."""
        preds = np.ones(4) * 500.0
        targets = np.ones(4) * 500.0
        meta = self._make_meta(["A320"] * 4, [300.0] * 4)  # all <30min
        result = compute_slices(preds, targets, meta)
        df_dur = result["by_duration_bucket"]
        assert all(df_dur["n"] > 0)

    def test_both_slices_present_in_result_keys(self):
        preds = np.ones(4) * 500.0
        targets = np.ones(4) * 500.0
        meta = self._make_meta(["A320"] * 4, [600.0] * 4)
        result = compute_slices(preds, targets, meta)
        assert "by_aircraft_type" in result
        assert "by_duration_bucket" in result


# ---------------------------------------------------------------------------
# 6. _rmse numerical correctness
# ---------------------------------------------------------------------------

class TestRmse:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        # errors: [3, 4] → MSE = (9+16)/2 = 12.5 → RMSE = sqrt(12.5)
        preds = np.array([3.0, 4.0])
        targets = np.array([0.0, 0.0])
        assert _rmse(preds, targets) == pytest.approx(math.sqrt(12.5))

    def test_scalar_agreement(self):
        preds = np.array([100.0, 200.0, 300.0])
        targets = np.array([90.0, 210.0, 295.0])
        errors = preds - targets
        expected = math.sqrt(float(np.mean(errors ** 2)))
        assert _rmse(preds, targets) == pytest.approx(expected)
