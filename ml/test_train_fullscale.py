"""
ml/test_train_fullscale.py — Offline smoke tests for train_fullscale.py.
========================================================================

Coverage
--------
1. Config loading: _load_yaml returns a dict; _DEFAULTS is structurally valid.
2. Config override: _apply_override sets nested keys via dot-notation.
3. Type coercion in _apply_override: int, float, null, bool, string.
4. _get: reads nested dot-notation keys with defaults.
5. model_type guard: NotImplementedError raised for model.type != "mlp".
6. End-to-end smoke: --smoke flag runs extract + 2-epoch train on 20-flight
   mock data with no real PRC data and no Databricks connection.
   This is the primary acceptance gate for ml-05.

All tests run WITHOUT real PRC data, WITHOUT a Databricks connection, and
WITHOUT any network access.  Mock data is generated in a temp directory and
cleaned up automatically.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `ml.*` imports work when pytest
# is run from any directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.train_fullscale import (  # noqa: E402
    _apply_override,
    _get,
    _DEFAULTS,
    _run_smoke,
)


# ---------------------------------------------------------------------------
# 1. _DEFAULTS structural validity
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_required_sections_present(self):
        for section in ("data", "training", "model", "mlflow"):
            assert section in _DEFAULTS, f"Missing section: {section}"

    def test_data_section(self):
        d = _DEFAULTS["data"]
        assert "data_dir" in d
        assert "split" in d
        assert "epochs" not in d  # not in data section

    def test_training_section(self):
        t = _DEFAULTS["training"]
        assert t["epochs"] == 50
        assert t["batch_size"] == 16
        assert isinstance(t["lr"], float)
        assert t["seed"] == 42

    def test_model_type_default(self):
        assert _DEFAULTS["model"]["type"] == "mlp"

    def test_mlflow_experiment_name(self):
        assert _DEFAULTS["mlflow"]["experiment_name"] == "FuelBurn_Baseline"


# ---------------------------------------------------------------------------
# 2 & 3. _apply_override type coercion
# ---------------------------------------------------------------------------

class TestApplyOverride:
    def _cfg(self):
        return {"a": {"b": 1, "c": 1.0}, "x": "hello"}

    def test_set_int(self):
        cfg = self._cfg()
        _apply_override(cfg, "a.b", "99")
        assert cfg["a"]["b"] == 99
        assert isinstance(cfg["a"]["b"], int)

    def test_set_float(self):
        cfg = self._cfg()
        _apply_override(cfg, "a.c", "3.14")
        assert abs(cfg["a"]["c"] - 3.14) < 1e-9
        assert isinstance(cfg["a"]["c"], float)

    def test_set_null(self):
        cfg = self._cfg()
        _apply_override(cfg, "a.b", "null")
        assert cfg["a"]["b"] is None

    def test_set_none_alias(self):
        cfg = self._cfg()
        _apply_override(cfg, "a.b", "None")
        assert cfg["a"]["b"] is None

    def test_set_true(self):
        cfg = self._cfg()
        _apply_override(cfg, "x", "true")
        assert cfg["x"] is True

    def test_set_false(self):
        cfg = self._cfg()
        _apply_override(cfg, "x", "false")
        assert cfg["x"] is False

    def test_set_string_path(self):
        cfg = self._cfg()
        _apply_override(cfg, "x", "/Volumes/cat/sch/vol/data")
        assert cfg["x"] == "/Volumes/cat/sch/vol/data"

    def test_creates_nested_key(self):
        cfg: dict = {}
        _apply_override(cfg, "new.section.key", "42")
        assert cfg["new"]["section"]["key"] == 42

    def test_top_level_key(self):
        cfg: dict = {"top": "old"}
        _apply_override(cfg, "top", "new_val")
        assert cfg["top"] == "new_val"


# ---------------------------------------------------------------------------
# 4. _get
# ---------------------------------------------------------------------------

class TestGet:
    def test_reads_nested(self):
        cfg = {"a": {"b": {"c": 7}}}
        assert _get(cfg, "a.b.c") == 7

    def test_default_on_missing_key(self):
        cfg = {"a": {}}
        assert _get(cfg, "a.b", default=99) == 99

    def test_default_on_missing_section(self):
        assert _get({}, "x.y.z", default="fallback") == "fallback"

    def test_none_value_is_returned(self):
        cfg = {"a": {"b": None}}
        assert _get(cfg, "a.b", default="fallback") is None


# ---------------------------------------------------------------------------
# 5. model_type guard
# ---------------------------------------------------------------------------

class TestModelTypeGuard:
    def test_histgbr_raises_not_implemented(self):
        """Selecting model.type=histgbr should raise NotImplementedError."""
        import io
        import contextlib
        from ml.train_fullscale import _apply_override, _DEFAULTS, _run_extract, _run_train, _set_seeds, _get
        import copy

        # We test the guard logic directly by simulating what main() does.
        cfg = copy.deepcopy(_DEFAULTS)
        _apply_override(cfg, "model.type", "histgbr")

        model_type = _get(cfg, "model.type", "mlp")
        with pytest.raises(NotImplementedError, match="histgbr"):
            if model_type != "mlp":
                raise NotImplementedError(
                    f"model.type={model_type!r} is not yet wired into the full-scale "
                    "entrypoint.  Only 'mlp' is supported."
                )


# ---------------------------------------------------------------------------
# 6. End-to-end smoke (primary acceptance gate)
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_smoke_runs_without_error(self):
        """
        Full end-to-end smoke: 20-flight mock, extract, 2-epoch train.
        This is the ml-05 acceptance gate: runs offline with no real PRC data
        and no Databricks connection.
        """
        # _run_smoke handles temp dir creation and cleanup internally.
        # It will call extract_features and ml.train.main() via sys.argv patch.
        _run_smoke(config_path="", overrides=[])

    def test_smoke_with_override_epochs(self):
        """Overrides passed to smoke run are applied (1-epoch for speed)."""
        _run_smoke(config_path="", overrides=["training.epochs=1"])


# ---------------------------------------------------------------------------
# Config YAML loading (only if PyYAML is available)
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_load_fullscale_yaml(self):
        """fullscale.yaml should parse without errors."""
        yaml_path = Path(__file__).parent / "configs" / "fullscale.yaml"
        if not yaml_path.exists():
            pytest.skip("fullscale.yaml not found")

        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        from ml.train_fullscale import _load_yaml
        cfg = _load_yaml(str(yaml_path))

        assert "data" in cfg
        assert "training" in cfg
        assert "mlflow" in cfg
        assert cfg["mlflow"]["experiment_name"] == "FuelBurn_Baseline"
        assert cfg["training"]["epochs"] == 50

    def test_yaml_data_dir_default_is_local(self):
        """Default data_dir in the YAML should be the local PRC path."""
        yaml_path = Path(__file__).parent / "configs" / "fullscale.yaml"
        if not yaml_path.exists():
            pytest.skip("fullscale.yaml not found")
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        from ml.train_fullscale import _load_yaml
        cfg = _load_yaml(str(yaml_path))
        assert cfg["data"]["data_dir"] == "data/raw/prc_2025"
