#!/usr/bin/env python3
"""
lint_job_spec.py — offline validator for serverless Databricks job specs
========================================================================
W4.3 / ds-07

Checks performed (in order):
  1. JSON is well-formed and the spec file exists.
  2. Required serverless top-level keys are present.
  3. Every task declares exactly one supported workload: notebook_task or
     spark_python_task.
  4. Notebook paths and Python entrypoints exist in the repository.
  5. All dependency references resolve; medallion ordering is correct.
  6. Every task references a declared serverless environment.
  7. Classic-compute fields and unresolved OWNER-FILL placeholders are absent.

Usage
-----
  # Validate the default spec (always resolves relative to this file's repo root):
  python data/cloud/databricks/jobs/lint_job_spec.py

  # Validate an alternative spec file:
  python data/cloud/databricks/jobs/lint_job_spec.py \
    ml/configs/databricks_job.json \
    data/cloud/databricks/jobs/medallion_job.json

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


def _repo_file_exists(repo_path: str, allow_py_suffix: bool = False) -> bool:
    """Return True when a Git-source path resolves inside the repository."""
    candidates = [_REPO_ROOT / repo_path]
    if allow_py_suffix:
        candidates.append(_REPO_ROOT / (repo_path + ".py"))
    return any(path.is_file() for path in candidates)


def _notebook_exists(notebook_path: str) -> bool:
    """
    Return True if the notebook resolves to an existing file in the repo.

    Databricks notebook_path values use the notebook name WITHOUT the .py
    extension (e.g. "data/cloud/databricks/notebooks/01_bronze_to_silver").
    We check for both the bare path and the .py variant.
    """
    return _repo_file_exists(notebook_path, allow_py_suffix=True)


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
    required_keys = {"name", "tasks", "environments", "git_source"}
    missing = required_keys - set(spec.keys())
    if missing:
        violations.append(f"Missing required top-level keys: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Checks 3 & 4: supported workload shape and repository paths.
    # ------------------------------------------------------------------
    tasks = spec.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) == 0:
        violations.append("'tasks' must be a non-empty list.")
        return violations  # cannot check ordering without tasks

    for task in tasks:
        task_key = task.get("task_key", "?")
        workload_keys = [
            key
            for key in ("notebook_task", "spark_python_task")
            if key in task
        ]
        if len(workload_keys) != 1:
            violations.append(
                f"Task '{task_key}' must declare exactly one supported workload "
                "(notebook_task or spark_python_task)."
            )
            continue

        if workload_keys[0] == "notebook_task":
            nb_path = task["notebook_task"].get("notebook_path", "")
            if not nb_path:
                violations.append(f"Task '{task_key}' has no notebook_path.")
            elif not _notebook_exists(nb_path):
                violations.append(
                    f"Task '{task_key}': notebook_path '{nb_path}' does not "
                    f"resolve to an existing file under repo root '{_REPO_ROOT}'."
                )
        else:
            python_file = task["spark_python_task"].get("python_file", "")
            if not python_file:
                violations.append(f"Task '{task_key}' has no python_file.")
            elif not _repo_file_exists(python_file):
                violations.append(
                    f"Task '{task_key}': python_file '{python_file}' does not "
                    f"resolve to an existing file under repo root '{_REPO_ROOT}'."
                )

    # ------------------------------------------------------------------
    # Check 5: dependencies resolve; medallion order is correct.
    # ------------------------------------------------------------------
    task_map = {t.get("task_key"): t for t in tasks}
    task_keys = set(task_map)
    for task in tasks:
        task_key = task.get("task_key", "?")
        for dependency in task.get("depends_on", []):
            dependency_key = dependency.get("task_key")
            if dependency_key not in task_keys:
                violations.append(
                    f"Task '{task_key}' depends on unknown task "
                    f"'{dependency_key}'."
                )

    source_key, sink_key = _EXPECTED_ORDER
    tags = spec.get("tags", {})
    is_medallion = (
        isinstance(tags, dict) and tags.get("layer") == "medallion"
    ) or any(key in task_map for key in _EXPECTED_ORDER)

    if is_medallion:
        if source_key not in task_map:
            violations.append(
                f"Expected task '{source_key}' not found in task list. "
                f"Found: {list(task_map.keys())}"
            )
        else:
            source_deps = [
                d.get("task_key")
                for d in task_map[source_key].get("depends_on", [])
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
            sink_deps = [
                d.get("task_key")
                for d in task_map[sink_key].get("depends_on", [])
            ]
            if source_key not in sink_deps:
                violations.append(
                    f"Task '{sink_key}' must depend on '{source_key}', "
                    f"but its depends_on is: {sink_deps}"
                )

    # ------------------------------------------------------------------
    # Check 6: Every task references a declared serverless environment.
    # ------------------------------------------------------------------
    environments = spec.get("environments", [])
    environment_keys = {
        env.get("environment_key")
        for env in environments
        if isinstance(env, dict)
    }
    if not environment_keys:
        violations.append("'environments' must declare at least one environment_key.")

    for task in tasks:
        task_key = task.get("task_key", "?")
        environment_key = task.get("environment_key")
        if not environment_key:
            violations.append(
                f"Task '{task_key}' has no environment_key for serverless compute."
            )
        elif environment_key not in environment_keys:
            violations.append(
                f"Task '{task_key}' references unknown environment_key "
                f"'{environment_key}'."
            )

    # ------------------------------------------------------------------
    # Check 7: No classic-compute fields or unresolved placeholders.
    # ------------------------------------------------------------------
    if "job_clusters" in spec:
        violations.append(
            "Classic-compute field 'job_clusters' is not allowed; "
            "Free Edition jobs must use serverless environments."
        )

    for task in tasks:
        for field in ("new_cluster", "existing_cluster_id", "job_cluster_key"):
            if field in task:
                violations.append(
                    f"Task '{task.get('task_key', '?')}' uses classic-compute "
                    f"field '{field}'."
                )

    unresolved = [
        value for value in _collect_strings(spec) if "<OWNER-FILL" in value
    ]
    if unresolved:
        violations.append(
            "Unresolved <OWNER-FILL> placeholders remain:\n"
            + "\n".join(f"  {value!r}" for value in unresolved)
        )

    return violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    spec_paths = [Path(arg) for arg in args] if args else [_DEFAULT_SPEC]
    failed = False

    for spec_path in spec_paths:
        if not spec_path.is_absolute():
            spec_path = Path.cwd() / spec_path

        print(f"Linting: {spec_path}")
        print(f"Repo root: {_REPO_ROOT}")
        print(f"Notebooks dir: {_NOTEBOOKS_DIR}")
        print()

        violations = lint(spec_path)
        if violations:
            failed = True
            print("LINT FAILED — violations found:", file=sys.stderr)
            for i, violation in enumerate(violations, 1):
                print(f"  [{i}] {violation}", file=sys.stderr)
        else:
            print("LINT PASSED — all checks OK.")
        print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
