#!/usr/bin/env python3
"""
test_lint_job_spec.py — offline unit tests for lint_job_spec.py
================================================================
W4.3 / ds-07

Tests:
  1. The real medallion_job.json passes the linter (zero violations).
  2. A spec pointing at a non-existent notebook is rejected.
  3. A spec with a reversed depends_on order is rejected.
  4. A spec containing a hard-coded /Volumes/ literal is rejected.
  5. The linter rejects a broken-JSON file gracefully.
  6. The linter accepts a minimal well-formed spec.
  7. A source task that incorrectly declares depends_on is rejected.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from this directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.cloud.databricks.jobs.lint_job_spec import lint  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOBS_DIR = _PROJECT_ROOT / "data" / "cloud" / "databricks" / "jobs"
_REAL_SPEC = _JOBS_DIR / "medallion_job.json"

# Notebooks that actually exist in the repo (relative to repo root, no .py).
_NB_BRONZE = "data/cloud/databricks/notebooks/01_bronze_to_silver"
_NB_GOLD = "data/cloud/databricks/notebooks/02_silver_to_gold"

_MINIMAL_VALID_SPEC = {
    "name": "test_job",
    "tasks": [
        {
            "task_key": "bronze_to_silver",
            "notebook_task": {
                "notebook_path": _NB_BRONZE,
                "base_parameters": {},
                "source": "GIT",
            },
        },
        {
            "task_key": "silver_to_gold",
            "depends_on": [{"task_key": "bronze_to_silver"}],
            "notebook_task": {
                "notebook_path": _NB_GOLD,
                "base_parameters": {},
                "source": "GIT",
            },
        },
    ],
    "job_clusters": [
        {
            "job_cluster_key": "test_cluster",
            "new_cluster": {"spark_version": "15.4.x-scala2.12", "num_workers": 0},
        }
    ],
}


def _write_tmp(spec: dict) -> Path:
    """Write *spec* to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(spec, tmp)
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealSpecPassesLint(unittest.TestCase):
    """The committed medallion_job.json must pass all lint checks."""

    def test_real_spec_zero_violations(self):
        violations = lint(_REAL_SPEC)
        self.assertEqual(
            violations,
            [],
            msg=f"Real spec has lint violations:\n"
            + "\n".join(f"  {v}" for v in violations),
        )


class TestMinimalValidSpec(unittest.TestCase):
    """A hand-crafted minimal spec that satisfies all rules."""

    def test_minimal_valid_passes(self):
        tmp = _write_tmp(_MINIMAL_VALID_SPEC)
        try:
            violations = lint(tmp)
            self.assertEqual(violations, [], msg=str(violations))
        finally:
            tmp.unlink(missing_ok=True)


class TestNonExistentNotebook(unittest.TestCase):
    """Pointing a task at a notebook that doesn't exist must be flagged."""

    def test_bad_notebook_path_is_rejected(self):
        import copy

        spec = copy.deepcopy(_MINIMAL_VALID_SPEC)
        spec["tasks"][0]["notebook_task"]["notebook_path"] = (
            "data/cloud/databricks/notebooks/99_does_not_exist"
        )
        tmp = _write_tmp(spec)
        try:
            violations = lint(tmp)
            # Must have at least one violation referencing the bad path.
            self.assertTrue(
                any("99_does_not_exist" in v for v in violations),
                msg=f"Expected a violation for missing notebook. Got: {violations}",
            )
        finally:
            tmp.unlink(missing_ok=True)


class TestWrongDependsOnOrder(unittest.TestCase):
    """silver_to_gold must depend on bronze_to_silver; reversing this is a violation."""

    def test_reversed_order_is_rejected(self):
        import copy

        spec = copy.deepcopy(_MINIMAL_VALID_SPEC)
        # Swap depends_on: bronze now depends on silver (wrong).
        spec["tasks"][0]["depends_on"] = [{"task_key": "silver_to_gold"}]
        spec["tasks"][1]["depends_on"] = []
        tmp = _write_tmp(spec)
        try:
            violations = lint(tmp)
            self.assertTrue(
                any("bronze_to_silver" in v and "depends_on" in v for v in violations),
                msg=f"Expected a depends_on violation. Got: {violations}",
            )
        finally:
            tmp.unlink(missing_ok=True)


class TestMissingSinkDependency(unittest.TestCase):
    """silver_to_gold with empty depends_on is a violation."""

    def test_missing_sink_dep_is_rejected(self):
        import copy

        spec = copy.deepcopy(_MINIMAL_VALID_SPEC)
        spec["tasks"][1]["depends_on"] = []
        tmp = _write_tmp(spec)
        try:
            violations = lint(tmp)
            self.assertTrue(
                any("silver_to_gold" in v for v in violations),
                msg=f"Expected violation for missing sink dependency. Got: {violations}",
            )
        finally:
            tmp.unlink(missing_ok=True)


class TestHardCodedVolumesRejected(unittest.TestCase):
    """Any non-comment field containing '/Volumes/' must be flagged."""

    def test_hard_coded_volume_in_param_default(self):
        import copy

        spec = copy.deepcopy(_MINIMAL_VALID_SPEC)
        spec["parameters"] = [
            {
                "name": "bronze_volume_path",
                "default": "/Volumes/main/default/bronze_vol/bronze/",
            }
        ]
        tmp = _write_tmp(spec)
        try:
            violations = lint(tmp)
            self.assertTrue(
                any("/Volumes/" in v for v in violations),
                msg=f"Expected a /Volumes/ hard-code violation. Got: {violations}",
            )
        finally:
            tmp.unlink(missing_ok=True)

    def test_hard_coded_volume_in_notebook_base_param(self):
        import copy

        spec = copy.deepcopy(_MINIMAL_VALID_SPEC)
        spec["tasks"][0]["notebook_task"]["base_parameters"] = {
            "bronze_volume_path": "/Volumes/main/default/bronze_vol/bronze/"
        }
        tmp = _write_tmp(spec)
        try:
            violations = lint(tmp)
            self.assertTrue(
                any("/Volumes/" in v for v in violations),
                msg=f"Expected a /Volumes/ violation in base_parameters. Got: {violations}",
            )
        finally:
            tmp.unlink(missing_ok=True)


class TestBrokenJsonRejected(unittest.TestCase):
    """Malformed JSON must produce a parse-error violation, not raise."""

    def test_broken_json_gives_violation(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write("{broken json:::}")
        tmp.flush()
        path = Path(tmp.name)
        try:
            violations = lint(path)
            self.assertTrue(
                any("JSON parse error" in v or "parse" in v.lower() for v in violations),
                msg=f"Expected a JSON parse violation. Got: {violations}",
            )
        finally:
            path.unlink(missing_ok=True)


class TestMissingSpecFile(unittest.TestCase):
    """Pointing the linter at a non-existent file gives a clear violation."""

    def test_missing_file_gives_violation(self):
        violations = lint(Path("/tmp/definitely_does_not_exist_xyz.json"))
        self.assertTrue(
            any("not found" in v.lower() for v in violations),
            msg=f"Expected 'not found' violation. Got: {violations}",
        )


if __name__ == "__main__":
    unittest.main()
