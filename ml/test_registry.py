"""
ml/test_registry.py — Offline tests for the MLflow Model Registry workflow.
=============================================================================

All tests run against a TEMPORARY local MLflow tracking/registry store backed
by a SQLite file in a pytest tmp_path directory.  No running MLflow server,
no real PRC data, and no network access required.

Coverage
--------
1. register_run
   - Registers a model and returns a version string.
   - A second call on the same run returns a new (higher) version.
   - Raises ValueError when the run is tagged stale.

2. promote_to_production
   - First promotion: both "production" and "champion" aliases are set.
   - Second promotion: old aliases are removed; new version gets them.
   - Raises no exception when no prior production version exists.

3. compare_and_promote — better challenger WINS
   - A challenger with lower RMSE gets promoted to production.
   - The production alias resolves to the challenger version.

4. compare_and_promote — worse challenger does NOT get promoted
   - A challenger with higher RMSE leaves the champion unchanged.
   - The production alias still resolves to the original champion version.

5. compare_and_promote — no champion yet
   - The first candidate is always promoted unconditionally.

6. compare_and_promote — stale run is rejected before registration
   - ValueError is raised immediately; no new model version is created.

7. tag_stale_pre_leakage_runs
   - Only runs with explicit stale provenance are tagged.
   - Low RMSE alone is never treated as stale.
   - dry_run=True identifies stale runs without writing any tags.

8. URI contract
   - After promotion the production alias resolves to the correct version.
   - The PRODUCTION_URI constant is exactly "models:/FuelBurn@production".
"""

from __future__ import annotations

import os
import tempfile
import warnings

import pytest
import torch
import torch.nn as nn

import mlflow
from mlflow import MlflowClient

from ml.registry import (
    ALIAS_CHAMPION,
    ALIAS_CHALLENGER,
    ALIAS_PRODUCTION,
    PRODUCTION_URI,
    REGISTRY_NAME,
    compare_and_promote,
    promote_to_production,
    register_run,
    tag_stale_pre_leakage_runs,
)


# ---------------------------------------------------------------------------
# Tiny model used throughout (no FuelBurnMLP dependency — keeps test hermetic)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """One-parameter model; enough for mlflow.pytorch.log_model to work."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.w * x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_tracking_uri(tmp_path):
    """Return a file-based SQLite tracking URI in a fresh temp directory.

    Each test gets its own isolated store so there are no cross-test
    interference effects.
    """
    db_path = tmp_path / "test_registry.db"
    return f"sqlite:///{db_path}"


@pytest.fixture()
def isolated_mlflow(tmp_tracking_uri):
    """Configure MLflow to use the isolated temp store for one test.

    Restores the original tracking URI after the test finishes.
    """
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_tracking_uri)
    yield tmp_tracking_uri
    mlflow.set_tracking_uri(original_uri)


def _log_model_run(
    rmse: float,
    experiment_name: str = "FuelBurn_Baseline",
    extra_tags: dict | None = None,
    metric_key: str = "best_val_rmse",
) -> str:
    """Log a tiny model + RMSE metric to MLflow and return the run_id."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model = _TinyModel()
        # Suppress the pickle/CloudPickle advisory for test output cleanliness.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        mlflow.log_metric(metric_key, rmse)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)
        return run.info.run_id


# ---------------------------------------------------------------------------
# 8. URI contract (tested first — no model needed, pure constant check)
# ---------------------------------------------------------------------------

class TestURIContract:
    def test_production_uri_constant(self):
        """PRODUCTION_URI must be exactly 'models:/FuelBurn@production'."""
        assert PRODUCTION_URI == "models:/FuelBurn@production"

    def test_registry_name_constant(self):
        assert REGISTRY_NAME == "FuelBurn"

    def test_alias_production_constant(self):
        assert ALIAS_PRODUCTION == "production"

    def test_alias_champion_constant(self):
        assert ALIAS_CHAMPION == "champion"


# ---------------------------------------------------------------------------
# 1. register_run
# ---------------------------------------------------------------------------

class TestRegisterRun:
    def test_registers_model_returns_version(self, isolated_mlflow):
        """register_run returns a non-empty version string."""
        run_id = _log_model_run(rmse=450.0)
        version = register_run(run_id)
        assert isinstance(version, str)
        assert version != ""

    def test_second_registration_increments_version(self, isolated_mlflow):
        """Two distinct runs produce version 1 then version 2."""
        run_id_1 = _log_model_run(rmse=450.0)
        run_id_2 = _log_model_run(rmse=445.0)
        v1 = register_run(run_id_1)
        v2 = register_run(run_id_2)
        assert int(v2) > int(v1), f"Expected v2 > v1, got {v1} and {v2}"

    def test_stale_run_raises_value_error(self, isolated_mlflow):
        """Registering a stale (pre-leakage) run raises ValueError."""
        run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )
        with pytest.raises(ValueError, match="stale"):
            register_run(run_id)

    def test_stale_explicit_tag_raises(self, isolated_mlflow):
        """A run tagged data.leakage_fix=pre is rejected even with high RMSE."""
        # RMSE is above threshold, but explicit stale tag must still fire.
        run_id = _log_model_run(
            rmse=500.0,
            extra_tags={"data.leakage_fix": "pre"},
        )
        with pytest.raises(ValueError, match="stale"):
            register_run(run_id)

    def test_registered_version_is_accessible(self, isolated_mlflow):
        """The registered version can be retrieved from the registry."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=450.0)
        version = register_run(run_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv = client.get_model_version(REGISTRY_NAME, version)
        # MLflow 3.x stores version as int internally; normalise for comparison.
        assert str(mv.version) == str(version)
        assert mv.name == REGISTRY_NAME


# ---------------------------------------------------------------------------
# 2. promote_to_production
# ---------------------------------------------------------------------------

class TestPromoteToProduction:
    def test_first_promotion_sets_aliases(self, isolated_mlflow):
        """Promoting version 1 creates both production and champion aliases."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=450.0)
        version = register_run(run_id)
        promote_to_production(version)

        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        champ_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_CHAMPION)
        # MLflow 3.x returns version as int internally; normalise for comparison.
        assert str(prod_mv.version) == str(version)
        assert str(champ_mv.version) == str(version)

    def test_second_promotion_reassigns_aliases(self, isolated_mlflow):
        """Promoting version 2 moves aliases from v1 to v2."""
        client = MlflowClient()
        run_id_1 = _log_model_run(rmse=450.0)
        run_id_2 = _log_model_run(rmse=440.0)
        v1 = register_run(run_id_1)
        v2 = register_run(run_id_2)

        promote_to_production(v1)
        promote_to_production(v2)

        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(v2), f"Expected v2={v2}, got {prod_mv.version}"

    def test_old_alias_removed_after_reassignment(self, isolated_mlflow):
        """After promoting v2, v1 no longer has the production alias."""
        client = MlflowClient()
        run_id_1 = _log_model_run(rmse=450.0)
        run_id_2 = _log_model_run(rmse=440.0)
        v1 = register_run(run_id_1)
        v2 = register_run(run_id_2)

        promote_to_production(v1)
        promote_to_production(v2)

        # v1 should no longer be reachable by the production alias.
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert prod_mv.version != v1


# ---------------------------------------------------------------------------
# 3. compare_and_promote — better challenger WINS
# ---------------------------------------------------------------------------

class TestChallengerWins:
    def test_better_challenger_promoted(self, isolated_mlflow):
        """Challenger with lower RMSE replaces the champion."""
        client = MlflowClient()

        # Establish a champion.
        champ_run_id = _log_model_run(rmse=460.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        # Challenger has lower (better) RMSE.
        challenger_run_id = _log_model_run(rmse=440.0)
        result = compare_and_promote(challenger_run_id)

        assert result["promoted"] is True
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(result["candidate_version"])

    def test_result_contains_expected_keys(self, isolated_mlflow):
        run_id = _log_model_run(rmse=450.0)
        result = compare_and_promote(run_id)
        for key in (
            "promoted", "candidate_version", "candidate_rmse",
            "champion_version", "champion_rmse", "reason",
        ):
            assert key in result, f"Missing key: {key}"

    def test_candidate_rmse_correct(self, isolated_mlflow):
        """result.candidate_rmse matches the logged metric."""
        run_id = _log_model_run(rmse=443.5)
        result = compare_and_promote(run_id)
        assert result["candidate_rmse"] == pytest.approx(443.5, abs=1e-4)

    def test_challenger_alias_cleaned_up_after_win(self, isolated_mlflow):
        """The challenger alias is removed after a winning promotion."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=450.0)
        compare_and_promote(run_id)
        # challenger alias must no longer exist
        with pytest.raises(mlflow.exceptions.MlflowException):
            client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_CHALLENGER)


# ---------------------------------------------------------------------------
# 4. compare_and_promote — worse challenger does NOT get promoted
# ---------------------------------------------------------------------------

class TestChallengerLoses:
    def test_worse_challenger_not_promoted(self, isolated_mlflow):
        """Challenger with higher RMSE leaves the champion unchanged."""
        client = MlflowClient()

        # Establish a champion with a good (low) RMSE.
        champ_run_id = _log_model_run(rmse=440.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        # Challenger is worse (higher RMSE).
        challenger_run_id = _log_model_run(rmse=480.0)
        result = compare_and_promote(challenger_run_id)

        assert result["promoted"] is False
        # The production alias must still point to the original champion.
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(champ_version)

    def test_reason_mentions_champion_retained(self, isolated_mlflow):
        """The reason string explicitly mentions the champion was retained."""
        champ_run_id = _log_model_run(rmse=440.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        challenger_run_id = _log_model_run(rmse=480.0)
        result = compare_and_promote(challenger_run_id)
        assert "champion retained" in result["reason"].lower()

    def test_challenger_alias_cleaned_up_after_loss(self, isolated_mlflow):
        """The challenger alias is removed even after a losing attempt."""
        client = MlflowClient()

        champ_run_id = _log_model_run(rmse=440.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        challenger_run_id = _log_model_run(rmse=480.0)
        compare_and_promote(challenger_run_id)

        with pytest.raises(mlflow.exceptions.MlflowException):
            client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_CHALLENGER)

    def test_equal_rmse_does_not_promote(self, isolated_mlflow):
        """Equal RMSE should NOT promote (requires strictly lower to win)."""
        client = MlflowClient()

        champ_run_id = _log_model_run(rmse=450.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        # Identical RMSE.
        challenger_run_id = _log_model_run(rmse=450.0)
        result = compare_and_promote(challenger_run_id)

        assert result["promoted"] is False
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(champ_version)


# ---------------------------------------------------------------------------
# 5. compare_and_promote — no prior champion
# ---------------------------------------------------------------------------

class TestNoPriorChampion:
    def test_first_candidate_always_promoted(self, isolated_mlflow):
        """When no production model exists, the first candidate wins."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=460.0)
        result = compare_and_promote(run_id)

        assert result["promoted"] is True
        assert result["champion_version"] is None  # no prior champion
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(result["candidate_version"])

    def test_reason_mentions_no_existing_champion(self, isolated_mlflow):
        run_id = _log_model_run(rmse=460.0)
        result = compare_and_promote(run_id)
        assert "no existing champion" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 6. compare_and_promote — stale run rejection
# ---------------------------------------------------------------------------

class TestStaleRunRejection:
    def test_explicit_stale_run_raises_before_registration(self, isolated_mlflow):
        """A stale run is rejected before any model version is created."""
        client = MlflowClient()
        run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )

        with pytest.raises(ValueError, match="stale"):
            compare_and_promote(run_id)

        # No version must have been registered.
        try:
            versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
        except mlflow.exceptions.MlflowException:
            versions = []
        assert len(list(versions)) == 0, "No versions should exist after stale rejection."

    def test_stale_explicit_tag_rejected(self, isolated_mlflow):
        """Explicit data.leakage_fix=pre tag rejects even with high RMSE."""
        run_id = _log_model_run(
            rmse=500.0,
            extra_tags={"data.leakage_fix": "pre"},
        )
        with pytest.raises(ValueError, match="stale"):
            compare_and_promote(run_id)

    def test_stale_rejection_leaves_champion_unchanged(self, isolated_mlflow):
        """Stale rejection does not disturb an existing champion."""
        client = MlflowClient()

        champ_run_id = _log_model_run(rmse=450.0)
        champ_version = register_run(champ_run_id)
        promote_to_production(champ_version)

        stale_run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )
        with pytest.raises(ValueError, match="stale"):
            compare_and_promote(stale_run_id)

        # Champion must still be intact.
        prod_mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(prod_mv.version) == str(champ_version)


# ---------------------------------------------------------------------------
# 7. tag_stale_pre_leakage_runs
# ---------------------------------------------------------------------------

class TestTagStaleRuns:
    def test_stale_runs_tagged(self, isolated_mlflow):
        """Explicit pre-fix runs receive the registry.stale=true tag."""
        client = MlflowClient()
        run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        assert run_id in stale_ids

        run = client.get_run(run_id)
        assert run.data.tags.get("registry.stale") == "true"
        assert run.data.tags.get("registry.stale_reason") == "pre_leakage_fix_ml01"
        assert "registry.stale_since" in run.data.tags

    def test_low_rmse_without_provenance_is_not_tagged(self, isolated_mlflow):
        """A strong metric is not evidence that a run is stale."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=350.0)

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        assert run_id not in stale_ids

        run = client.get_run(run_id)
        assert run.data.tags.get("registry.stale") != "true"

    def test_dry_run_does_not_write_tags(self, isolated_mlflow):
        """dry_run=True identifies stale runs without writing tags."""
        client = MlflowClient()
        run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline", dry_run=True)
        assert run_id in stale_ids

        run = client.get_run(run_id)
        assert run.data.tags.get("registry.stale") != "true", (
            "dry_run must not write tags to MLflow"
        )

    def test_explicit_leakage_tag_identified(self, isolated_mlflow):
        """A run tagged data.leakage_fix=pre is always flagged stale."""
        client = MlflowClient()
        run_id = _log_model_run(
            rmse=500.0,
            extra_tags={"data.leakage_fix": "pre"},
        )

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        assert run_id in stale_ids

        run = client.get_run(run_id)
        assert run.data.tags.get("registry.stale") == "true"

    def test_missing_experiment_returns_empty(self, isolated_mlflow):
        """Scanning a non-existent experiment returns an empty list."""
        stale_ids = tag_stale_pre_leakage_runs("NonExistentExperiment_XYZ_999")
        assert stale_ids == []

    def test_mixed_run_set(self, isolated_mlflow):
        """Only the stale run in a mixed batch gets tagged."""
        client = MlflowClient()
        stale_run = _log_model_run(
            rmse=450.0,
            extra_tags={"data.leakage_fix": "pre"},
        )
        valid_run = _log_model_run(rmse=350.0)

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        assert stale_run in stale_ids
        assert valid_run not in stale_ids

        stale_obj = client.get_run(stale_run)
        valid_obj = client.get_run(valid_run)
        assert stale_obj.data.tags.get("registry.stale") == "true"
        assert valid_obj.data.tags.get("registry.stale") != "true"

    def test_idempotent_tagging(self, isolated_mlflow):
        """Calling tag_stale_pre_leakage_runs twice does not error."""
        run_id = _log_model_run(
            rmse=390.0,
            extra_tags={"data.leakage_fix": "pre"},
        )

        ids1 = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        ids2 = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        # Both calls must return the same stale run.
        assert run_id in ids1
        assert run_id in ids2

    def test_low_cv_mean_rmse_is_not_treated_as_stale(self, isolated_mlflow):
        """Metric values never substitute for lineage."""
        client = MlflowClient()
        run_id = _log_model_run(
            rmse=300.0,
            metric_key="cv_mean_rmse_kg",  # primary metric
        )

        stale_ids = tag_stale_pre_leakage_runs("FuelBurn_Baseline")
        assert run_id not in stale_ids

        run = client.get_run(run_id)
        assert run.data.tags.get("registry.stale") != "true"


# ---------------------------------------------------------------------------
# URI resolution integration test
# ---------------------------------------------------------------------------

class TestURIResolution:
    def test_production_alias_resolves_to_promoted_version(self, isolated_mlflow):
        """After promotion, get_model_version_by_alias(name, 'production')
        returns the exact version we promoted."""
        client = MlflowClient()
        run_id = _log_model_run(rmse=450.0)
        result = compare_and_promote(run_id)

        assert result["promoted"] is True
        mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        # MLflow 3.x stores version as int internally; normalise for comparison.
        assert str(mv.version) == str(result["candidate_version"])
        assert mv.name == REGISTRY_NAME

    def test_production_uri_matches_serving_hook(self, isolated_mlflow):
        """The PRODUCTION_URI constant matches serve.py's expected ML_MODEL_URI value.

        serve.py (ml-10) documents:
            ML_MODEL_URI = "models:/FuelBurn/Production"  (stage-based, deprecated)
            ml-09 will promote and document this URI.

        Since we use aliases (MLflow 3.x), the correct URI is:
            models:/FuelBurn@production

        This test asserts that our constant matches the alias-based form.
        """
        assert PRODUCTION_URI == f"models:/{REGISTRY_NAME}@{ALIAS_PRODUCTION}"

    def test_load_model_from_production_alias(self, isolated_mlflow):
        """mlflow.pytorch.load_model can resolve the production alias URI."""
        run_id = _log_model_run(rmse=450.0)
        compare_and_promote(run_id)

        # This exercises the same code path that serve.py uses at startup.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = mlflow.pytorch.load_model(PRODUCTION_URI)

        assert loaded is not None
        # Verify the loaded model can do a forward pass.
        out = loaded(torch.tensor(2.0))
        assert torch.isfinite(out), "Loaded model produced non-finite output"
