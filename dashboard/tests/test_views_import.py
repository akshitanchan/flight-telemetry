#!/usr/bin/env python3
"""Import-smoke tests for dashboard view modules.

Guards against crash-on-import regressions in the three Streamlit view modules.
Each module is imported via importlib; no ``render()`` is called and no
Streamlit server is launched.

Conventions mirror the existing dashboard/tests/ files:
  - unittest.TestCase subclasses
  - PROJECT_ROOT path insertion for runability without an editable install
  - Fully offline: passes without any running services or network access.

Run with:
    python -m pytest dashboard/tests/test_views_import.py -v
    python -m unittest dashboard.tests.test_views_import -v
"""

import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_VIEW_MODULES = (
    "dashboard.views.operational",
    "dashboard.views.ask_ai",
    "dashboard.views.platform_health",
)


class TestViewModulesImport(unittest.TestCase):
    """Each dashboard view module must be importable without raising."""

    def _import(self, module_name: str) -> None:
        """Helper: import *module_name* and assert no exception is raised."""
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"Importing {module_name!r} raised unexpectedly: {exc}")

    def test_import_operational(self):
        self._import("dashboard.views.operational")

    def test_import_ask_ai(self):
        self._import("dashboard.views.ask_ai")

    def test_import_platform_health(self):
        self._import("dashboard.views.platform_health")

    def test_all_modules_importable_idempotent(self):
        """Importing all three modules a second time must not raise (idempotency)."""
        for mod in _VIEW_MODULES:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"Re-importing {mod!r} raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
