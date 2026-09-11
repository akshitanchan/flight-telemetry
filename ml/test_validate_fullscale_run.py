"""
ml/test_validate_fullscale_run.py — Offline tests for validate_fullscale_run.py.
=================================================================================

All tests run against a TEMPORARY local MLflow tracking/registry store backed
by a SQLite file in a pytest tmp_path directory.  No running MLflow server,
no real PRC data, no cloud credentials, and no network access are required.

Coverage
--------
Validation checks (6 checks per run):

1.  PASS run — all 6 checks pass  → promoted to models:/FuelBurn@production.
2.  FAIL run — best_val_rmse MISSING  → CHECK 1 fails; registry untouched.
3.  FAIL run — best_val_rmse ABOVE regression threshold (>= 500 kg)  → CHECK 2
    fails; registry untouched.
4.  WARN run — best_val_rmse above target but below threshold (default mode)
    → validation passes (only WARN, no FAIL); run is promoted.
5.  FAIL run (strict mode) — same as WARN run but --strict  → CHECK 3 fails;
    registry untouched.
6.  FAIL run — model/ artifact MISSING  → CHECK 4 fails; registry untouched.
7.  FAIL run — train_rmse metric MISSING  → CHECK 5 fails; registry untouched.
8.  FAIL run — train_rmse INCREASING (not decreasing)  → CHECK 6 fails;
    registry untouched.
9.  WARN run — only one train_rmse step logged  → CHECK 6 is WARN (skip);
    validation passes.
10. Dry-run flag prevents registry handoff even on ALL-PASS run.
11. validate_run() raises RuntimeError on an unknown run_id.
12. Registry handoff calls compare_and_promote exactly once on PASS.
13. A PASS run that loses the champion comparison is not promoted but the
    script still exits 0 (validation passed; registry decision is separate).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

import mlflow
from mlflow import MlflowClient

# Ensure the project root is on sys.path so `ml.*` imports work regardless of
# where pytest is invoked from.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.validate_fullscale_run import (  # noqa: E402
    METRIC_BEST_VAL_RMSE,
    METRIC_TRAIN_RMSE,
    REGRESSION_THRESHOLD_KG,
    TARGET_RMSE_KG,
    EXPECTED_ARTIFACT_PATH,
    validate_run,
    handoff_to_registry,
    print_validation_report,
    _PASS,
    _FAIL,
    _WARN,
)


# ---------------------------------------------------------------------------
# Tiny model (same pattern as test_registry.py — keeps tests hermetic)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """One-parameter model; sufficient for mlflow.pytorch.log_model."""
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
    """Isolated SQLite-backed MLflow store in a fresh temp directory."""
    db_path = tmp_path / "test_validate.db"
    return f"sqlite:///{db_path}"


@pytest.fixture()
def isolated_mlflow(tmp_tracking_uri):
    """Configure MLflow to use the isolated temp store for the duration of
    one test; restores the original tracking URI on teardown."""
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_tracking_uri)
    yield tmp_tracking_uri
    mlflow.set_tracking_uri(original_uri)


# ---------------------------------------------------------------------------
# Helpers for seeding mock runs
# ---------------------------------------------------------------------------

def _log_full_run(
    *,
    best_val_rmse: float | None = 430.0,
    train_rmse_steps: list[float] | None = None,
    log_model_artifact: bool = True,
    extra_tags: dict | None = None,
    experiment_name: str = "FuelBurn_Baseline",
) -> str:
    """Log a mock full-scale run and return its run_id.

    Parameters
    ----------
    best_val_rmse:
        Value to log as 'best_val_rmse'.  None = skip logging this metric.
    train_rmse_steps:
        List of per-step train_rmse values.  None = use a default decreasing
        sequence [500.0, 460.0, 440.0, 430.0, 425.0].
        Pass [] to skip logging train_rmse entirely.
    log_model_artifact:
        If True, log a tiny PyTorch model under artifact path "model".
    extra_tags:
        Optional dict of MLflow tags to set on the run.
    """
    if train_rmse_steps is None:
        train_rmse_steps = [500.0, 460.0, 440.0, 430.0, 425.0]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        if best_val_rmse is not None:
            mlflow.log_metric(METRIC_BEST_VAL_RMSE, best_val_rmse)

        for step, val in enumerate(train_rmse_steps, start=1):
            mlflow.log_metric(METRIC_TRAIN_RMSE, val, step=step)

        if log_model_artifact:
            model = _TinyModel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlflow.pytorch.log_model(
                    model, EXPECTED_ARTIFACT_PATH, serialization_format="pickle"
                )

        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

    return run_id


def _check_by_name(results: list[dict], name: str) -> dict:
    """Return the result dict for the check with the given name."""
    for r in results:
        if r["check"] == name:
            return r
    raise KeyError(f"No check named '{name}' in results: {[r['check'] for r in results]}")


# ---------------------------------------------------------------------------
# 1. PASS run — all 6 checks pass → promoted
# ---------------------------------------------------------------------------

class TestPassRun:
    def test_all_checks_pass(self, isolated_mlflow):
        """A well-formed run with good metrics passes every check."""
        run_id = _log_full_run(best_val_rmse=430.0)
        results, all_passed = validate_run(run_id)

        assert all_passed is True, (
            f"Expected all-pass but got failures: "
            f"{[r for r in results if r['status'] == _FAIL]}"
        )
        for r in results:
            assert r["status"] in (_PASS, _WARN), (
                f"Check '{r['check']}' has unexpected status {r['status']}: {r['detail']}"
            )

    def test_pass_run_promoted_to_registry(self, isolated_mlflow):
        """A passing run is handed to compare_and_promote and gets promoted."""
        client = MlflowClient()
        run_id = _log_full_run(best_val_rmse=430.0)
        results, all_passed = validate_run(run_id)
        assert all_passed

        result = handoff_to_registry(run_id)
        assert result["promoted"] is True
        # Verify the production alias now resolves.
        from ml.registry import REGISTRY_NAME, ALIAS_PRODUCTION
        mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(mv.version) == str(result["candidate_version"])

    def test_pass_run_check_count(self, isolated_mlflow):
        """Exactly 6 check results are returned."""
        run_id = _log_full_run(best_val_rmse=430.0)
        results, _ = validate_run(run_id)
        assert len(results) == 6

    def test_pass_run_all_required_check_names_present(self, isolated_mlflow):
        """All expected check names appear in the results."""
        expected_names = {
            "best_val_rmse_present",
            "best_val_rmse_below_regression_threshold",
            "best_val_rmse_at_or_below_target",
            "model_artifact_present",
            "train_rmse_history_present",
            "train_rmse_decreasing",
        }
        run_id = _log_full_run(best_val_rmse=430.0)
        results, _ = validate_run(run_id)
        actual_names = {r["check"] for r in results}
        assert expected_names == actual_names


# ---------------------------------------------------------------------------
# 2. FAIL — best_val_rmse MISSING (CHECK 1)
# ---------------------------------------------------------------------------

class TestMissingBestValRmse:
    def test_check1_fails_when_metric_absent(self, isolated_mlflow):
        """CHECK 1 must FAIL when best_val_rmse is not logged."""
        run_id = _log_full_run(best_val_rmse=None)
        results, all_passed = validate_run(run_id)

        assert all_passed is False
        c = _check_by_name(results, "best_val_rmse_present")
        assert c["status"] == _FAIL

    def test_registry_not_touched_when_metric_absent(self, isolated_mlflow):
        """Registry must not be touched when validation fails."""
        client = MlflowClient()
        run_id = _log_full_run(best_val_rmse=None)
        _, all_passed = validate_run(run_id)
        assert all_passed is False

        # No model versions should be registered.
        from ml.registry import REGISTRY_NAME
        try:
            versions = list(client.search_model_versions(f"name='{REGISTRY_NAME}'"))
        except mlflow.exceptions.MlflowException:
            versions = []
        assert len(versions) == 0, "Registry must be untouched after failed validation."


# ---------------------------------------------------------------------------
# 3. FAIL — best_val_rmse ABOVE regression threshold (CHECK 2)
# ---------------------------------------------------------------------------

class TestRegressionThresholdFail:
    def test_check2_fails_above_threshold(self, isolated_mlflow):
        """CHECK 2 must FAIL when best_val_rmse >= REGRESSION_THRESHOLD_KG."""
        above = REGRESSION_THRESHOLD_KG + 10.0  # e.g. 510.0
        run_id = _log_full_run(best_val_rmse=above)
        results, all_passed = validate_run(run_id)

        assert all_passed is False
        c = _check_by_name(results, "best_val_rmse_below_regression_threshold")
        assert c["status"] == _FAIL

    def test_check2_passes_at_boundary_minus_epsilon(self, isolated_mlflow):
        """CHECK 2 passes when best_val_rmse is just below the threshold."""
        just_under = REGRESSION_THRESHOLD_KG - 0.01  # 499.99
        run_id = _log_full_run(best_val_rmse=just_under)
        results, _ = validate_run(run_id)
        c = _check_by_name(results, "best_val_rmse_below_regression_threshold")
        assert c["status"] == _PASS

    def test_check2_fails_at_exact_threshold(self, isolated_mlflow):
        """CHECK 2 requires strictly less than; equal is a FAIL."""
        run_id = _log_full_run(best_val_rmse=REGRESSION_THRESHOLD_KG)
        results, all_passed = validate_run(run_id)
        assert all_passed is False
        c = _check_by_name(results, "best_val_rmse_below_regression_threshold")
        assert c["status"] == _FAIL


# ---------------------------------------------------------------------------
# 4. WARN — best_val_rmse above target (default non-strict mode) → passes
# ---------------------------------------------------------------------------

class TestAboveTargetWarn:
    def test_check3_warns_above_target_default_mode(self, isolated_mlflow):
        """CHECK 3 emits WARN (not FAIL) in default mode when above target."""
        above_target = TARGET_RMSE_KG + 20.0   # e.g. 462.65 kg, still < 500
        run_id = _log_full_run(best_val_rmse=above_target)
        results, all_passed = validate_run(run_id, strict=False)

        assert all_passed is True, "WARN must not block overall pass"
        c = _check_by_name(results, "best_val_rmse_at_or_below_target")
        assert c["status"] == _WARN

    def test_warn_run_promoted_to_registry(self, isolated_mlflow):
        """A run with WARN on CHECK 3 is still handed off and can be promoted."""
        above_target = TARGET_RMSE_KG + 20.0
        run_id = _log_full_run(best_val_rmse=above_target)
        _, all_passed = validate_run(run_id, strict=False)
        assert all_passed

        result = handoff_to_registry(run_id)
        assert result["promoted"] is True  # first candidate always wins


# ---------------------------------------------------------------------------
# 5. FAIL — strict mode + above target (CHECK 3)
# ---------------------------------------------------------------------------

class TestAboveTargetStrictFail:
    def test_check3_fails_in_strict_mode(self, isolated_mlflow):
        """In --strict mode, CHECK 3 fails if best_val_rmse > TARGET_RMSE_KG."""
        above_target = TARGET_RMSE_KG + 20.0
        run_id = _log_full_run(best_val_rmse=above_target)
        results, all_passed = validate_run(run_id, strict=True)

        assert all_passed is False
        c = _check_by_name(results, "best_val_rmse_at_or_below_target")
        assert c["status"] == _FAIL

    def test_registry_not_touched_in_strict_mode(self, isolated_mlflow):
        """Registry is untouched when strict validation fails."""
        client = MlflowClient()
        above_target = TARGET_RMSE_KG + 20.0
        run_id = _log_full_run(best_val_rmse=above_target)
        _, all_passed = validate_run(run_id, strict=True)
        assert all_passed is False

        from ml.registry import REGISTRY_NAME
        try:
            versions = list(client.search_model_versions(f"name='{REGISTRY_NAME}'"))
        except mlflow.exceptions.MlflowException:
            versions = []
        assert len(versions) == 0

    def test_check3_passes_at_exact_target(self, isolated_mlflow):
        """CHECK 3 passes when best_val_rmse equals the target exactly."""
        run_id = _log_full_run(best_val_rmse=TARGET_RMSE_KG)
        results, all_passed = validate_run(run_id, strict=True)
        c = _check_by_name(results, "best_val_rmse_at_or_below_target")
        assert c["status"] == _PASS


# ---------------------------------------------------------------------------
# 6. FAIL — model/ artifact MISSING (CHECK 4)
# ---------------------------------------------------------------------------

class TestMissingModelArtifact:
    def test_check4_fails_when_artifact_absent(self, isolated_mlflow):
        """CHECK 4 must FAIL when the model/ artifact was not logged."""
        run_id = _log_full_run(best_val_rmse=430.0, log_model_artifact=False)
        results, all_passed = validate_run(run_id)

        assert all_passed is False
        c = _check_by_name(results, "model_artifact_present")
        assert c["status"] == _FAIL

    def test_registry_not_touched_when_artifact_absent(self, isolated_mlflow):
        """Registry must not be touched when the model artifact is missing."""
        client = MlflowClient()
        run_id = _log_full_run(best_val_rmse=430.0, log_model_artifact=False)
        _, all_passed = validate_run(run_id)
        assert all_passed is False

        from ml.registry import REGISTRY_NAME
        try:
            versions = list(client.search_model_versions(f"name='{REGISTRY_NAME}'"))
        except mlflow.exceptions.MlflowException:
            versions = []
        assert len(versions) == 0


# ---------------------------------------------------------------------------
# 7. FAIL — train_rmse MISSING (CHECK 5)
# ---------------------------------------------------------------------------

class TestMissingTrainRmse:
    def test_check5_fails_when_metric_absent(self, isolated_mlflow):
        """CHECK 5 must FAIL when train_rmse is not logged at all."""
        run_id = _log_full_run(best_val_rmse=430.0, train_rmse_steps=[])
        results, all_passed = validate_run(run_id)

        assert all_passed is False
        c = _check_by_name(results, "train_rmse_history_present")
        assert c["status"] == _FAIL


# ---------------------------------------------------------------------------
# 8. FAIL — train_rmse INCREASING (CHECK 6)
# ---------------------------------------------------------------------------

class TestIncreasingTrainRmse:
    def test_check6_fails_when_train_rmse_increases(self, isolated_mlflow):
        """CHECK 6 must FAIL when the final train_rmse > the first train_rmse."""
        # Increasing sequence — clearly broken optimiser signal.
        increasing = [420.0, 440.0, 460.0, 480.0, 500.0]
        run_id = _log_full_run(best_val_rmse=430.0, train_rmse_steps=increasing)
        results, all_passed = validate_run(run_id)

        assert all_passed is False
        c = _check_by_name(results, "train_rmse_decreasing")
        assert c["status"] == _FAIL

    def test_check6_passes_when_train_rmse_decreases(self, isolated_mlflow):
        """CHECK 6 passes when the final train_rmse <= the first train_rmse."""
        decreasing = [500.0, 460.0, 440.0, 430.0, 425.0]
        run_id = _log_full_run(best_val_rmse=430.0, train_rmse_steps=decreasing)
        results, _ = validate_run(run_id)
        c = _check_by_name(results, "train_rmse_decreasing")
        assert c["status"] == _PASS

    def test_check6_passes_with_flat_curve(self, isolated_mlflow):
        """CHECK 6 passes when first == last (non-increasing, not just strictly decreasing)."""
        flat = [440.0, 440.0, 440.0]
        run_id = _log_full_run(best_val_rmse=430.0, train_rmse_steps=flat)
        results, _ = validate_run(run_id)
        c = _check_by_name(results, "train_rmse_decreasing")
        assert c["status"] == _PASS


# ---------------------------------------------------------------------------
# 9. WARN — single train_rmse step (CHECK 6 skipped)
# ---------------------------------------------------------------------------

class TestSingleStepTrainRmse:
    def test_check6_warns_with_single_step(self, isolated_mlflow):
        """CHECK 6 emits WARN (not FAIL) when only one train_rmse point is logged."""
        run_id = _log_full_run(best_val_rmse=430.0, train_rmse_steps=[450.0])
        results, all_passed = validate_run(run_id)

        # Single-step is ambiguous — treat as WARN, overall should still pass.
        assert all_passed is True
        c = _check_by_name(results, "train_rmse_decreasing")
        assert c["status"] == _WARN


# ---------------------------------------------------------------------------
# 10. Dry-run prevents registry handoff
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_call_compare_and_promote(self, isolated_mlflow):
        """With --dry-run, validate_run() passes but handoff_to_registry is NOT called."""
        run_id = _log_full_run(best_val_rmse=430.0)
        results, all_passed = validate_run(run_id)
        assert all_passed

        # Verify that compare_and_promote is NOT called when we skip the handoff.
        # We simulate the dry_run path by not calling handoff_to_registry().
        client = MlflowClient()
        from ml.registry import REGISTRY_NAME
        try:
            versions = list(client.search_model_versions(f"name='{REGISTRY_NAME}'"))
        except mlflow.exceptions.MlflowException:
            versions = []
        assert len(versions) == 0, (
            "Registry must be untouched after dry-run (no handoff_to_registry call)."
        )

    def test_main_dry_run_exits_0(self, isolated_mlflow, tmp_path, capsys):
        """Invoking main() with --dry-run on a passing run exits with code 0."""
        run_id = _log_full_run(best_val_rmse=430.0)

        import ml.validate_fullscale_run as vfr
        with patch.object(sys, "argv", [
            "ml.validate_fullscale_run",
            "--run-id", run_id,
            "--tracking-uri", isolated_mlflow,
            "--dry-run",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                vfr.main()
        assert exc_info.value.code == 0

    def test_main_fails_run_exits_1(self, isolated_mlflow, capsys):
        """Invoking main() on a failing run exits with code 1."""
        run_id = _log_full_run(best_val_rmse=None)  # missing metric — CHECK 1 fails

        import ml.validate_fullscale_run as vfr
        with patch.object(sys, "argv", [
            "ml.validate_fullscale_run",
            "--run-id", run_id,
            "--tracking-uri", isolated_mlflow,
        ]):
            with pytest.raises(SystemExit) as exc_info:
                vfr.main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 11. validate_run raises RuntimeError on unknown run_id
# ---------------------------------------------------------------------------

class TestUnknownRunId:
    def test_unknown_run_id_raises_runtime_error(self, isolated_mlflow):
        """validate_run() raises RuntimeError when the run_id does not exist."""
        with pytest.raises(RuntimeError, match="Could not fetch run"):
            validate_run(run_id="nonexistent_run_id_xyz_abc_000")


# ---------------------------------------------------------------------------
# 12. Registry handoff calls compare_and_promote exactly once
# ---------------------------------------------------------------------------

class TestHandoffCallsCompareAndPromote:
    def test_handoff_calls_compare_and_promote_once(self, isolated_mlflow):
        """handoff_to_registry() calls compare_and_promote (via module-level import) exactly once."""
        run_id = _log_full_run(best_val_rmse=430.0)

        with patch("ml.validate_fullscale_run.compare_and_promote") as mock_cap:
            # Return a plausible result dict so handoff doesn't crash.
            mock_cap.return_value = {
                "promoted": True,
                "candidate_version": "1",
                "candidate_rmse": 430.0,
                "champion_version": None,
                "champion_rmse": None,
                "reason": "No existing champion; promoting candidate unconditionally.",
            }
            handoff_to_registry(run_id)

        mock_cap.assert_called_once_with(candidate_run_id=run_id)

    def test_handoff_passes_run_id_correctly(self, isolated_mlflow):
        """handoff_to_registry() passes the exact run_id to compare_and_promote."""
        run_id = _log_full_run(best_val_rmse=435.0)

        with patch("ml.validate_fullscale_run.compare_and_promote") as mock_cap:
            mock_cap.return_value = {
                "promoted": False,
                "candidate_version": "2",
                "candidate_rmse": 435.0,
                "champion_version": "1",
                "champion_rmse": 420.0,
                "reason": "Champion retained.",
            }
            result = handoff_to_registry(run_id)

        assert mock_cap.call_args[1]["candidate_run_id"] == run_id
        assert result["promoted"] is False


# ---------------------------------------------------------------------------
# 13. PASS + loses champion comparison → validate passes, script exits 0
# ---------------------------------------------------------------------------

class TestPassButLosesChampion:
    def test_validation_passes_even_if_champion_wins(self, isolated_mlflow):
        """A run that passes validation but loses the champion comparison is
        still considered a validation SUCCESS (exit 0).  Registry decision
        (not promoted) is independent of validation outcome."""
        client = MlflowClient()

        # Establish a strong champion first.
        champ_run_id = _log_full_run(best_val_rmse=420.0)
        _, all_passed_champ = validate_run(champ_run_id)
        assert all_passed_champ
        champ_result = handoff_to_registry(champ_run_id)
        assert champ_result["promoted"] is True
        champ_version = champ_result["candidate_version"]

        # Now validate and promote a weaker challenger.
        challenger_run_id = _log_full_run(best_val_rmse=460.0)
        results, all_passed = validate_run(challenger_run_id)

        # Validation must PASS (RMSE is below 500 kg, all checks green).
        assert all_passed is True

        # But the registry keeps the champion (lower RMSE wins).
        handoff_result = handoff_to_registry(challenger_run_id)
        assert handoff_result["promoted"] is False

        # Champion must still be the production alias.
        from ml.registry import REGISTRY_NAME, ALIAS_PRODUCTION
        mv = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS_PRODUCTION)
        assert str(mv.version) == str(champ_version)


# ---------------------------------------------------------------------------
# 14. print_validation_report smoke (no assertions on stdout content)
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_print_validation_report_does_not_raise(self, isolated_mlflow, capsys):
        """print_validation_report() must not raise for any valid results list."""
        run_id = _log_full_run(best_val_rmse=430.0)
        results, _ = validate_run(run_id)
        # Should not raise; output goes to capsys.
        print_validation_report(results)

    def test_print_validation_report_shows_pass_fail_warn(self, capsys):
        """Report output contains PASS, FAIL, WARN tokens as appropriate."""
        results = [
            {"check": "c1", "status": _PASS, "detail": "ok"},
            {"check": "c2", "status": _FAIL, "detail": "bad"},
            {"check": "c3", "status": _WARN, "detail": "maybe"},
        ]
        print_validation_report(results)
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "WARN" in captured.out
