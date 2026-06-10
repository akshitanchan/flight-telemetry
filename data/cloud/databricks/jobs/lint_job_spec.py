#!/usr/bin/env python3
"""
lint_job_spec.py — offline validator for medallion_job.json
===========================================================
W4.3 / ds-07

Checks performed (in order):
  1. JSON is well-formed and the spec file exists.
  2. Required top-level keys are present.
  3. Every notebook_task.notebook_path references a notebook file that EXISTS
     in the repository (resolves relative to the repo root).
  4. The bronze_to_silver task has NO depends_on (it is the source).
  5. The silver_to_gold task depends_on bronze_to_silver (correct order).
  6. No literal "/Volumes/" string appears in any parameter default or
     notebook_path value (paths must flow through job parameters only).

Usage
-----
  # Validate the default spec (always resolves relative to this file's repo root):
  python data/cloud/databricks/jobs/lint_job_spec.py

  # Validate an alternative spec file:
  python data/cloud/databricks/jobs/lint_job_spec.py path/to/other_job.json

Exit codes:
  0 — all checks passed
  1 — one or more violations found (details printed to stderr)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
# jobs/ -> databricks/ -> cloud/ -> data/ -> flight-telemetry/ (repo root)
_REPO_ROOT = _THIS_DIR.parent.parent.parent.parent  # flight-telemetry/
_DEFAULT_SPEC = _THIS_DIR / "medallion_job.json"

# Notebooks live at data/cloud/databricks/notebooks/ relative to the repo root.
_NOTEBOOKS_DIR = _REPO_ROOT / "data" / "cloud" / "databricks" / "notebooks"

# Expected task ordering: first task key -> second task key.
_EXPECTED_ORDER = ("bronze_to_silver", "silver_to_gold")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_strings(obj: object) -> list[str]:
    """Recursively collect all string leaf values from a JSON-parsed object."""
    results: list[str] = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_strings(item))
    return results


def _notebook_exists(notebook_path: str) -> bool:
    """
    Return True if the notebook resolves to an existing file in the repo.

    Databricks notebook_path values use the notebook name WITHOUT the .py
    extension (e.g. "data/cloud/databricks/notebooks/01_bronze_to_silver").
    We check for both the bare path and the .py variant.
    """
    candidates = [
        _REPO_ROOT / notebook_path,
        _REPO_ROOT / (notebook_path + ".py"),
    ]
    return any(p.exists() for p in candidates)


# ---------------------------------------------------------------------------
# Main lint function
# ---------------------------------------------------------------------------


def lint(spec_path: Path) -> list[str]:
    """
    Validate the job spec at *spec_path*.

    Returns a list of violation strings.  An empty list means the spec is
    valid.
    """
    violations: list[str] = []

    # ------------------------------------------------------------------
    # Check 1: File exists and is valid JSON.
    # ------------------------------------------------------------------
    if not spec_path.exists():
        violations.append(f"Spec file not found: {spec_path}")
        return violations  # cannot continue

    try:
        with open(spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
    except json.JSONDecodeError as exc:
        violations.append(f"JSON parse error in {spec_path}: {exc}")
        return violations  # cannot continue

    # ------------------------------------------------------------------
    # Check 2: Required top-level keys.
    # ------------------------------------------------------------------
    required_keys = {"name", "tasks", "job_clusters"}
    missing = required_keys - set(spec.keys())
    if missing:
        violations.append(f"Missing required top-level keys: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Check 3: notebook_path values resolve to existing repo files.
    # ------------------------------------------------------------------
    tasks = spec.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) == 0:
        violations.append("'tasks' must be a non-empty list.")
        return violations  # cannot check ordering without tasks

    for task in tasks:
        nb_task = task.get("notebook_task", {})
        nb_path = nb_task.get("notebook_path", "")
        if not nb_path:
            violations.append(
                f"Task '{task.get('task_key', '?')}' has no notebook_path."
            )
            continue
        if not _notebook_exists(nb_path):
            violations.append(
                f"Task '{task.get('task_key', '?')}': notebook_path "
                f"'{nb_path}' does not resolve to an existing file under "
                f"repo root '{_REPO_ROOT}'."
            )

    # ------------------------------------------------------------------
    # Check 4 & 5: depends_on order: bronze_to_silver -> silver_to_gold.
    # ------------------------------------------------------------------
    task_map = {t.get("task_key"): t for t in tasks}

    source_key, sink_key = _EXPECTED_ORDER

    if source_key not in task_map:
        violations.append(
            f"Expected task '{source_key}' not found in task list. "
            f"Found: {list(task_map.keys())}"
        )
    else:
        source_task = task_map[source_key]
        source_deps = [
            d.get("task_key") for d in source_task.get("depends_on", [])
        ]
        if source_deps:
            violations.append(
                f"Task '{source_key}' must have NO depends_on (it is the "
                f"pipeline source), but depends on: {source_deps}"
            )

    if sink_key not in task_map:
        violations.append(
            f"Expected task '{sink_key}' not found in task list. "
            f"Found: {list(task_map.keys())}"
        )
    else:
        sink_task = task_map[sink_key]
        sink_deps = [d.get("task_key") for d in sink_task.get("depends_on", [])]
        if source_key not in sink_deps:
            violations.append(
                f"Task '{sink_key}' must depend on '{source_key}', "
                f"but its depends_on is: {sink_deps}"
            )

    # ------------------------------------------------------------------
    # Check 6: No literal "/Volumes/" in any string value.
    # (Volume paths must be injected via job parameters, not hard-coded.)
    # ------------------------------------------------------------------
    all_strings = _collect_strings(spec)
    # Exclude the _comment key contents and the _comment string itself —
    # comments are documentation and may legitimately reference /Volumes/
    # as examples.  We check only "live" fields by filtering comments out
    # of the scan.  Strategy: re-parse without any key whose name starts
    # with "_comment" and whose value is documentation.
    live_spec = _strip_comment_keys(spec)
    live_strings = _collect_strings(live_spec)

    hard_coded_volumes = [s for s in live_strings if "/Volumes/" in s]
    if hard_coded_volumes:
        violations.append(
            "Hard-coded '/Volumes/' literal found in non-comment spec "
            "fields (use job parameters instead):\n"
            + "\n".join(f"  {v!r}" for v in hard_coded_volumes)
        )

    return violations


def _strip_comment_keys(obj: object) -> object:
    """Return a copy of *obj* with all keys named '_comment' removed (any depth)."""
    if isinstance(obj, dict):
        return {
            k: _strip_comment_keys(v)
            for k, v in obj.items()
            if k != "_comment"
        }
    if isinstance(obj, list):
        return [_strip_comment_keys(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    spec_path = Path(args[0]) if args else _DEFAULT_SPEC

    # Resolve relative paths from the current working directory.
    if not spec_path.is_absolute():
        spec_path = Path.cwd() / spec_path

    print(f"Linting: {spec_path}")
    print(f"Repo root: {_REPO_ROOT}")
    print(f"Notebooks dir: {_NOTEBOOKS_DIR}")
    print()

    violations = lint(spec_path)

    if violations:
        print("LINT FAILED — violations found:", file=sys.stderr)
        for i, v in enumerate(violations, 1):
            print(f"  [{i}] {v}", file=sys.stderr)
        return 1

    print("LINT PASSED — all checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
