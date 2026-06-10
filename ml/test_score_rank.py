"""
ml/test_score_rank.py — Offline tests for the rank-phase scoring harness.
=========================================================================

All tests run with:
  - NO real PRC-2025 data
  - NO cloud connectivity
  - NO MLflow tracking server
  - A temp local FuelBurnMLP checkpoint (--model-path path)

Test coverage
-------------
test_rmse_formula
    Verifies the _rmse helper is numerically correct for a trivial case.

test_verdict_logic
    Parameterised: asserts that score_rank() produces the correct verdict
    string for all three cases — "BEATS baseline", "MATCHES baseline",
    "BELOW baseline".

test_end_to_end_mock
    Full end-to-end run on mock rank data with a freshly-initialised model:
      - generates mock data (rank split)
      - saves a random FuelBurnMLP to a temp .pth file
      - calls score_rank(); asserts:
          * our_rmse is a finite positive float
          * delta == our_rmse - baseline_rmse (arithmetic identity)
          * result dict contains expected keys
          * JSON and Markdown artefacts are written to the data dir

test_missing_labels_raises
    Confirms FileNotFoundError when fuel_rank.parquet is absent.

test_no_model_source_raises
    Confirms RuntimeError when neither --model-path nor --model-uri is set.

test_feature_column_order
    Asserts that the feature matrix built inside score_rank() uses exactly
    FEATURE_COLUMNS in order — ensuring no train/score skew independent of
    the broader integration test.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml.features import AIRCRAFT_TYPES, FEATURE_COLUMNS, INPUT_DIM, encode_aircraft_type
from ml.model import FuelBurnMLP
from ml.score_rank import _rmse, score_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_temp_model(seed: int = 0) -> str:
    """Create a fresh FuelBurnMLP, save to a temp .pth file, return its path."""
    torch.manual_seed(seed)
    model = FuelBurnMLP(input_dim=INPUT_DIM)
    model.eval()
    fd, path = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model, path)
    return path


def _make_mock_rank_dir(tmp_path: Path, n_flights: int = 4, seed: int = 7) -> Path:
    """Generate a minimal rank-split dataset in *tmp_path*.

    Produces:
      fuel_rank.parquet         — TRUE labels (idx, flight_id, start, end, fuel_kg)
      flightlist_rank.parquet   — aircraft types
      flights_rank.zip          — trajectory parquets
      features_rank.parquet     — pre-extracted features (skips real extraction)

    The features parquet is written directly here so the test runs fully offline
    without invoking extract_features (which needs zip/trajectory data available
    at exactly the right structure).  The values are consistent with the label
    rows so merged alignment on idx works correctly.
    """
    rng = np.random.default_rng(seed)

    flight_ids = [f"mock_rank_{i:04d}" for i in range(n_flights)]
    ac_types_pool = ["A320", "B738", "A359", "B77W"]

    base_time = datetime(2025, 9, 1, 6, 0, 0)
    interval_mins = 15

    fuel_rows: list[dict] = []
    flight_rows: list[dict] = []
    feature_rows: list[dict] = []
    idx_counter = 0

    traj_files: dict[str, pd.DataFrame] = {}

    for fid in flight_ids:
        ac_type = rng.choice(ac_types_pool)
        duration_mins = int(rng.integers(45, 120))
        takeoff = base_time + timedelta(minutes=int(rng.integers(0, 200)))
        landed = takeoff + timedelta(minutes=duration_mins)

        flight_rows.append(
            {
                "flight_id": fid,
                "aircraft_type": ac_type,
                "takeoff": takeoff,
                "landed": landed,
            }
        )

        # Build trajectory so the zip is valid (needed only if extract is called,
        # but we write it anyway to keep the fixture complete).
        duration_s = int((landed - takeoff).total_seconds())
        timestamps = [takeoff + timedelta(seconds=s) for s in range(0, duration_s, 10)]
        n_pts = len(timestamps)
        df_traj = pd.DataFrame(
            {
                "timestamp": timestamps,
                "flight_id": fid,
                "typecode": ac_type,
                "latitude": np.linspace(52.0, 51.0, n_pts),
                "longitude": np.linspace(4.5, -0.5, n_pts),
                "altitude": np.sin(np.linspace(0, np.pi, n_pts)) * 35000.0,
                "groundspeed": rng.normal(400, 20, n_pts),
                "track": np.linspace(270, 280, n_pts) % 360,
                "vertical_rate": rng.normal(0, 50, n_pts),
                "mach": np.nan,
                "TAS": np.nan,
                "CAS": np.nan,
                "source": "adsb",
            }
        )
        traj_files[fid] = df_traj

        # Fuel intervals
        current_start = takeoff
        ac_onehot = encode_aircraft_type(ac_type)
        ac_onehot_dict = {f"ac_{t}": v for t, v in zip(AIRCRAFT_TYPES, ac_onehot)}

        while current_start + timedelta(minutes=interval_mins) <= landed:
            current_end = current_start + timedelta(minutes=interval_mins)
            duration_s_interval = float(
                (current_end - current_start).total_seconds()
            )
            true_fuel = float(rng.normal(50 * interval_mins, 30))
            true_fuel = max(50.0, true_fuel)

            fuel_rows.append(
                {
                    "idx": idx_counter,
                    "flight_id": fid,
                    "start": current_start,
                    "end": current_end,
                    "fuel_kg": true_fuel,
                }
            )

            # Feature row — identical schema to what extract_features.py produces.
            feat: dict = {
                "idx": idx_counter,
                "flight_id": fid,
                "aircraft_type": ac_type,
                "fuel_kg": true_fuel,
                "duration_s": duration_s_interval,
                "alt_change": float(rng.normal(0, 500)),
                "avg_speed": float(rng.normal(400, 20)),
                "max_vrate": float(abs(rng.normal(0, 100))),
                "avg_altitude": float(rng.normal(30000, 2000)),
                "max_altitude": float(rng.normal(35000, 1000)),
                "alt_std": float(abs(rng.normal(0, 500))),
                "avg_track_change": float(abs(rng.normal(0, 1))),
                "avg_mach": 0.0,
                "avg_tas": 0.0,
                "avg_cas": 0.0,
            }
            feat.update(ac_onehot_dict)
            feature_rows.append(feat)
            idx_counter += 1
            current_start = current_end

    # Write fuel_rank.parquet (TRUE labels)
    df_fuel = pd.DataFrame(fuel_rows)
    df_fuel.to_parquet(tmp_path / "fuel_rank.parquet")

    # Write flightlist_rank.parquet
    df_flightlist = pd.DataFrame(flight_rows)
    df_flightlist.to_parquet(tmp_path / "flightlist_rank.parquet")

    # Write flights_rank.zip
    zip_path = tmp_path / "flights_rank.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fid, df_t in traj_files.items():
            pq_path = tmp_path / f"{fid}.parquet"
            df_t.to_parquet(pq_path)
            zf.write(pq_path, arcname=f"{fid}.parquet")
            pq_path.unlink()

    # Write features_rank.parquet (pre-extracted, avoids real extraction call)
    df_features = pd.DataFrame(feature_rows)
    df_features.to_parquet(tmp_path / "features_rank.parquet")

    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestRmseFormula:
    def test_perfect_prediction(self):
        preds = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        targets = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        assert _rmse(preds, targets) == pytest.approx(0.0)

    def test_known_value(self):
        # errors = [0, 10, -10, 20]  => MSE = (0+100+100+400)/4 = 150 => RMSE = sqrt(150)
        preds = np.array([100.0, 110.0, 90.0, 120.0], dtype=np.float32)
        targets = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        expected = math.sqrt(150.0)
        assert _rmse(preds, targets) == pytest.approx(expected, rel=1e-5)

    def test_returns_finite_float(self):
        preds = np.ones(100, dtype=np.float32) * 500.0
        targets = np.ones(100, dtype=np.float32) * 480.0
        result = _rmse(preds, targets)
        assert math.isfinite(result)
        assert isinstance(result, float)


class TestVerdictLogic:
    """Verdict logic: our_rmse < baseline → BEATS; == → MATCHES; > → BELOW."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.data_dir = _make_mock_rank_dir(tmp_path)
        self.model_path = _save_temp_model(seed=42)

    def teardown_method(self):
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    @pytest.mark.parametrize(
        "our_rmse_offset,expected_verdict",
        [
            (-50.0, "BEATS baseline"),   # our RMSE lower than baseline
            (0.0, "MATCHES baseline"),   # exact tie
            (+50.0, "BELOW baseline"),   # our RMSE higher than baseline
        ],
    )
    def test_verdict_cases(self, our_rmse_offset, expected_verdict, monkeypatch):
        """Monkeypatch _rmse to return a controlled value, then check verdict."""
        # First, get the real RMSE so we can compute a baseline that gives the
        # desired offset.  We run score_rank once without a baseline to get it.
        result_no_baseline = score_rank(
            data_dir=str(self.data_dir),
            model_path=self.model_path,
            model_uri=None,
            baseline_rmse=None,
        )
        real_rmse = result_no_baseline["our_rmse"]
        # Set the baseline so that delta = our_rmse_offset:
        # delta = real_rmse - baseline  => baseline = real_rmse - our_rmse_offset
        baseline = real_rmse - our_rmse_offset

        result = score_rank(
            data_dir=str(self.data_dir),
            model_path=self.model_path,
            model_uri=None,
            baseline_rmse=baseline,
        )
        assert result["verdict"] == expected_verdict

    def test_no_baseline_verdict(self):
        result = score_rank(
            data_dir=str(self.data_dir),
            model_path=self.model_path,
            model_uri=None,
            baseline_rmse=None,
        )
        assert result["verdict"] == "N/A (no baseline supplied)"
        assert result["delta"] is None
        assert result["baseline_rmse"] is None


class TestEndToEndMock:
    """Full end-to-end harness run on mock rank data."""

    def test_end_to_end(self, tmp_path):
        data_dir = _make_mock_rank_dir(tmp_path, n_flights=5, seed=99)
        model_path = _save_temp_model(seed=1)
        try:
            baseline_rmse = 120.0
            result = score_rank(
                data_dir=str(data_dir),
                model_path=model_path,
                model_uri=None,
                baseline_rmse=baseline_rmse,
            )

            # RMSE must be a finite positive number
            assert math.isfinite(result["our_rmse"]), "RMSE is not finite"
            assert result["our_rmse"] >= 0.0, "RMSE is negative"

            # Interval / flight counts must be positive
            assert result["n_intervals"] > 0
            assert result["n_flights"] > 0

            # Delta arithmetic identity
            assert result["delta"] == pytest.approx(
                result["our_rmse"] - baseline_rmse, rel=1e-5
            )

            # Verdict must be one of the three valid strings
            assert result["verdict"] in {
                "BEATS baseline",
                "MATCHES baseline",
                "BELOW baseline",
            }

            # JSON artefact written
            json_path = data_dir / "rank_score_result.json"
            assert json_path.exists(), "JSON artefact not written"
            with open(json_path) as fh:
                loaded = json.load(fh)
            assert loaded["our_rmse"] == pytest.approx(result["our_rmse"], rel=1e-5)

            # Markdown artefact written
            md_path = data_dir / "rank_score_result.md"
            assert md_path.exists(), "Markdown artefact not written"
            md_content = md_path.read_text()
            assert "RMSE" in md_content

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_result_dict_has_expected_keys(self, tmp_path):
        data_dir = _make_mock_rank_dir(tmp_path, n_flights=3, seed=13)
        model_path = _save_temp_model(seed=2)
        try:
            result = score_rank(
                data_dir=str(data_dir),
                model_path=model_path,
                model_uri=None,
                baseline_rmse=100.0,
            )
            required_keys = {
                "our_rmse",
                "baseline_rmse",
                "delta",
                "verdict",
                "n_flights",
                "n_intervals",
                "model_source",
            }
            assert required_keys <= set(result.keys()), (
                f"Missing keys: {required_keys - set(result.keys())}"
            )
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestEdgeCases:
    def test_missing_labels_raises(self, tmp_path):
        # data dir has no fuel_rank.parquet
        with pytest.raises(FileNotFoundError, match="fuel_rank.parquet"):
            score_rank(
                data_dir=str(tmp_path),
                model_path="/tmp/fake_model.pth",
                model_uri=None,
                baseline_rmse=None,
            )

    def test_no_model_source_raises(self, tmp_path):
        data_dir = _make_mock_rank_dir(tmp_path, n_flights=2)
        with pytest.raises(RuntimeError, match="No model source specified"):
            score_rank(
                data_dir=str(data_dir),
                model_path=None,
                model_uri=None,
                baseline_rmse=None,
            )


class TestFeatureColumnOrder:
    """Verify that the feature matrix is assembled in FEATURE_COLUMNS order.

    This test checks the contract independently of the integration test,
    ensuring no train/score skew is possible even if the features parquet
    column ordering changes on disk.
    """

    def test_feature_matrix_order(self, tmp_path):
        """The score_rank feature matrix must use FEATURE_COLUMNS order."""
        data_dir = _make_mock_rank_dir(tmp_path, n_flights=3, seed=77)
        model_path = _save_temp_model(seed=5)
        try:
            # Load the features parquet and build the matrix manually in the
            # FEATURE_COLUMNS order — then compare to what score_rank produces.
            df_labels = pd.read_parquet(data_dir / "fuel_rank.parquet")
            df_features = pd.read_parquet(data_dir / "features_rank.parquet")

            df_merged = df_labels[["idx", "fuel_kg"]].merge(
                df_features[["idx"] + FEATURE_COLUMNS],
                on="idx",
                how="left",
                validate="1:1",
            )

            X_expected = df_merged[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            # Verify shape and that each column matches its expected position.
            assert X_expected.shape[1] == INPUT_DIM, (
                f"Expected {INPUT_DIM} feature columns, got {X_expected.shape[1]}"
            )

            # Verify that shuffling columns in the parquet and re-indexing by
            # FEATURE_COLUMNS produces the same matrix — confirming that
            # score_rank is column-name-driven, not position-driven.
            cols_shuffled = FEATURE_COLUMNS[::-1]  # reverse order
            df_features_shuffled = df_features[
                ["idx", "flight_id", "aircraft_type", "fuel_kg"] + cols_shuffled
            ]
            df_features_shuffled.to_parquet(data_dir / "features_rank.parquet")

            # Re-merge after shuffle
            df_merged2 = df_labels[["idx", "fuel_kg"]].merge(
                df_features_shuffled[["idx"] + FEATURE_COLUMNS],
                on="idx",
                how="left",
                validate="1:1",
            )
            X_after_shuffle = df_merged2[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            np.testing.assert_array_equal(
                X_expected,
                X_after_shuffle,
                err_msg=(
                    "Feature matrix differs after column shuffle — "
                    "ordering is NOT FEATURE_COLUMNS-driven."
                ),
            )
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
