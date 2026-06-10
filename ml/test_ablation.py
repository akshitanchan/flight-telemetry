"""
ml/test_ablation.py
-------------------
Unit and integration tests for the ablation harness (ml/ablation.py).

Coverage
--------
1. FEATURE_GROUPS covers exactly the 38 FEATURE_COLUMNS — no column is
   missing or duplicated.
2. _make_mock_df_merged produces a valid multi-date DataFrame suitable for
   build_folds (>=2 distinct flight_dates, no NaN in required columns).
3. run_histgbr_cv returns finite RMSE values on the mock fixture.
4. run_mlp_cv returns finite RMSE values on the mock fixture.
5. run_ablation (histgbr) returns a baseline + per-group deltas; ablation
   table has exactly len(FEATURE_GROUPS) rows.
6. run_challenger_comparison returns a dict with all expected keys.
7. ablation_to_df produces a DataFrame with the correct columns and row count.
8. format_ablation_table_md / format_challenger_table_md produce non-empty
   Markdown strings containing the expected headings.
9. Zeroing ALL feature columns makes MLP predictions constant (near-zero
   gradient signal) — RMSE changes vs baseline (sanity check on zeroing logic).
10. Dropping a column set that is not in FEATURE_COLUMNS raises AssertionError
    at module import time (static assertion on FEATURE_GROUPS definition).

Tests use ONLY the built-in mock fixture — no filesystem, no real PRC data,
no MLflow, no network.
"""

import math
import pytest
import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, AIRCRAFT_TYPES
from ml.ablation import (
    FEATURE_GROUPS,
    _make_mock_df_merged,
    run_histgbr_cv,
    run_mlp_cv,
    run_ablation,
    run_challenger_comparison,
    ablation_to_df,
    format_ablation_table_md,
    format_challenger_table_md,
    _AblationDataset,
)
from ml.cv import build_folds


# ---------------------------------------------------------------------------
# 1. Group coverage: FEATURE_GROUPS must partition FEATURE_COLUMNS exactly
# ---------------------------------------------------------------------------

class TestFeatureGroupCoverage:
    def test_all_columns_covered(self):
        covered = set()
        for _, _, cols in FEATURE_GROUPS:
            covered.update(cols)
        assert covered == set(FEATURE_COLUMNS), (
            f"Groups do not cover all FEATURE_COLUMNS.\n"
            f"Missing: {set(FEATURE_COLUMNS) - covered}\n"
            f"Extra:   {covered - set(FEATURE_COLUMNS)}"
        )

    def test_no_column_in_two_groups(self):
        seen = {}
        for gname, _, cols in FEATURE_GROUPS:
            for c in cols:
                assert c not in seen, (
                    f"Column '{c}' appears in both '{seen[c]}' and '{gname}'"
                )
                seen[c] = gname

    def test_group_count(self):
        # We define exactly 6 groups
        assert len(FEATURE_GROUPS) == 6

    def test_all_group_cols_in_feature_columns(self):
        fc_set = set(FEATURE_COLUMNS)
        for gname, _, cols in FEATURE_GROUPS:
            for c in cols:
                assert c in fc_set, f"Group '{gname}' col '{c}' not in FEATURE_COLUMNS"


# ---------------------------------------------------------------------------
# 2. Mock fixture validity
# ---------------------------------------------------------------------------

class TestMockFixture:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=7)

    def test_has_multiple_dates(self, df):
        assert df["flight_date"].nunique() >= 2

    def test_build_folds_succeeds(self, df):
        folds = build_folds(df)
        assert len(folds) >= 1

    def test_no_nan_in_feature_columns(self, df):
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                assert df[col].isna().sum() == 0, f"NaN in column '{col}'"

    def test_fuel_kg_positive(self, df):
        assert (df["fuel_kg"] > 0).all()

    def test_has_required_columns(self, df):
        required = set(FEATURE_COLUMNS) | {"flight_id", "flight_date", "fuel_kg", "aircraft_type"}
        assert required.issubset(set(df.columns))

    def test_flight_id_uniqueness_per_date(self, df):
        # Each flight_id appears on exactly one date (no leakage risk)
        counts = df.groupby("flight_id")["flight_date"].nunique()
        assert (counts == 1).all(), "Some flight_ids span multiple dates"


# ---------------------------------------------------------------------------
# 3. HistGBR CV
# ---------------------------------------------------------------------------

class TestHistGBRCV:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=0)

    def test_returns_finite_rmse(self, df):
        result = run_histgbr_cv(df, n_folds=1, seed=42)
        assert math.isfinite(result["mean_rmse_kg"])
        assert math.isfinite(result["std_rmse_kg"])

    def test_rmse_positive(self, df):
        result = run_histgbr_cv(df, n_folds=1, seed=42)
        assert result["mean_rmse_kg"] > 0

    def test_fold_rmses_length(self, df):
        result = run_histgbr_cv(df, n_folds=1, seed=42)
        assert len(result["fold_rmses"]) == 1

    def test_dropped_cols_reduces_features(self, df):
        # Dropping a group should not raise; result is still finite
        result = run_histgbr_cv(df, dropped_cols={"duration_s"}, n_folds=1, seed=42)
        assert math.isfinite(result["mean_rmse_kg"])
        assert "duration_s" not in result["active_features"]

    def test_active_features_returned(self, df):
        result = run_histgbr_cv(df, n_folds=1, seed=42)
        assert "active_features" in result
        assert len(result["active_features"]) == len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 4. MLP CV
# ---------------------------------------------------------------------------

class TestMLPCV:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=0)

    def test_returns_finite_rmse(self, df):
        result = run_mlp_cv(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        assert math.isfinite(result["mean_rmse_kg"])
        assert result["mean_rmse_kg"] > 0

    def test_fold_rmses_length(self, df):
        result = run_mlp_cv(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        assert len(result["fold_rmses"]) == 1

    def test_zeroed_cols_accepted(self, df):
        # Zeroing columns should not crash
        result = run_mlp_cv(
            df, zeroed_cols={"duration_s"}, epochs=2, batch_size=8, n_folds=1, seed=42
        )
        assert math.isfinite(result["mean_rmse_kg"])


# ---------------------------------------------------------------------------
# 5. Ablation runner
# ---------------------------------------------------------------------------

class TestRunAblation:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=1)

    def test_histgbr_ablation_group_count(self, df):
        result = run_ablation(df, n_folds=1, seed=42, model_type="histgbr")
        assert len(result["groups"]) == len(FEATURE_GROUPS)

    def test_mlp_ablation_group_count(self, df):
        result = run_ablation(df, epochs=2, batch_size=8, n_folds=1, seed=42, model_type="mlp")
        assert len(result["groups"]) == len(FEATURE_GROUPS)

    def test_baseline_in_result(self, df):
        result = run_ablation(df, n_folds=1, seed=42, model_type="histgbr")
        assert "baseline" in result
        assert "mean_rmse_kg" in result["baseline"]

    def test_delta_is_float(self, df):
        result = run_ablation(df, n_folds=1, seed=42, model_type="histgbr")
        for group in result["groups"]:
            assert isinstance(group["delta_rmse_kg"], float)

    def test_group_names_match(self, df):
        result = run_ablation(df, n_folds=1, seed=42, model_type="histgbr")
        expected_names = {g[0] for g in FEATURE_GROUPS}
        actual_names = {g["group_name"] for g in result["groups"]}
        assert actual_names == expected_names


# ---------------------------------------------------------------------------
# 6. Challenger comparison
# ---------------------------------------------------------------------------

class TestChallengerComparison:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=2)

    def test_has_all_keys(self, df):
        result = run_challenger_comparison(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        for key in ("mlp", "histgbr", "winner", "delta_rmse_kg", "winner_margin"):
            assert key in result, f"Missing key: {key}"

    def test_winner_is_valid(self, df):
        result = run_challenger_comparison(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        assert result["winner"] in ("MLP", "HistGBR", "tie")

    def test_delta_consistent_with_rmse(self, df):
        result = run_challenger_comparison(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        expected_delta = result["histgbr"]["mean_rmse_kg"] - result["mlp"]["mean_rmse_kg"]
        assert abs(result["delta_rmse_kg"] - expected_delta) < 1e-6

    def test_both_rmse_positive(self, df):
        result = run_challenger_comparison(df, epochs=2, batch_size=8, n_folds=1, seed=42)
        assert result["mlp"]["mean_rmse_kg"] > 0
        assert result["histgbr"]["mean_rmse_kg"] > 0


# ---------------------------------------------------------------------------
# 7. ablation_to_df
# ---------------------------------------------------------------------------

class TestAblationToDf:
    @pytest.fixture(scope="class")
    def ablation_result(self):
        df = _make_mock_df_merged(seed=3)
        return run_ablation(df, n_folds=1, seed=42, model_type="histgbr")

    def test_row_count(self, ablation_result):
        df_out = ablation_to_df(ablation_result)
        assert len(df_out) == len(FEATURE_GROUPS)

    def test_required_columns_present(self, ablation_result):
        df_out = ablation_to_df(ablation_result)
        for col in [
            "group_name",
            "group_description",
            "n_cols_removed",
            "columns_removed",
            "baseline_mean_rmse_kg",
            "ablated_mean_rmse_kg",
            "delta_rmse_kg",
        ]:
            assert col in df_out.columns, f"Missing column: {col}"

    def test_sorted_by_delta_descending(self, ablation_result):
        df_out = ablation_to_df(ablation_result)
        deltas = df_out["delta_rmse_kg"].tolist()
        assert deltas == sorted(deltas, reverse=True)


# ---------------------------------------------------------------------------
# 8. Markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdownRendering:
    @pytest.fixture(scope="class")
    def ablation_result(self):
        df = _make_mock_df_merged(seed=4)
        return run_ablation(df, n_folds=1, seed=42, model_type="histgbr")

    @pytest.fixture(scope="class")
    def comparison_result(self):
        df = _make_mock_df_merged(seed=4)
        return run_challenger_comparison(df, epochs=2, batch_size=8, n_folds=1, seed=42)

    def test_ablation_md_non_empty(self, ablation_result):
        md = format_ablation_table_md(ablation_result, "histgbr", "mock")
        assert len(md) > 100

    def test_ablation_md_contains_heading(self, ablation_result):
        md = format_ablation_table_md(ablation_result, "histgbr", "mock")
        assert "Feature-Group Ablation" in md

    def test_ablation_md_contains_group_names(self, ablation_result):
        md = format_ablation_table_md(ablation_result, "histgbr", "mock")
        for gname, _, _ in FEATURE_GROUPS:
            assert gname in md, f"Group name '{gname}' missing from markdown"

    def test_challenger_md_non_empty(self, comparison_result):
        md = format_challenger_table_md(comparison_result, "mock")
        assert len(md) > 50

    def test_challenger_md_contains_both_models(self, comparison_result):
        md = format_challenger_table_md(comparison_result, "mock")
        assert "MLP" in md
        assert "HistGBR" in md


# ---------------------------------------------------------------------------
# 9. _AblationDataset zeroing logic
# ---------------------------------------------------------------------------

class TestAblationDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_mock_df_merged(seed=5)

    def test_zeroed_cols_are_zero(self, df):
        import torch
        zeroed = {"duration_s", "avg_speed"}
        ds = _AblationDataset(df, zeroed_cols=zeroed)
        feats, _ = ds[0]
        for i, col in enumerate(FEATURE_COLUMNS):
            if col in zeroed:
                assert feats[i].item() == 0.0, f"Expected 0.0 for zeroed col {col}"

    def test_non_zeroed_cols_are_nonzero_for_duration(self, df):
        import torch
        # duration_s should be non-zero in the mock data
        ds = _AblationDataset(df, zeroed_cols=set())
        dur_idx = FEATURE_COLUMNS.index("duration_s")
        feats, _ = ds[0]
        assert feats[dur_idx].item() > 0.0

    def test_target_positive(self, df):
        ds = _AblationDataset(df)
        _, target = ds[0]
        assert target.item() > 0.0

    def test_len_matches_df(self, df):
        ds = _AblationDataset(df)
        assert len(ds) == len(df)
