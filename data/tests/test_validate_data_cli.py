#!/usr/bin/env python3
"""Tests for the `--data <path> --schema <name>` CLI mode of validate_schemas.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_SCRIPT = _REPO_ROOT / "shared" / "contracts" / "validate_schemas.py"
FIXTURES_DIR = _REPO_ROOT / "shared" / "contracts" / "fixtures"


def _clean(record: dict) -> dict:
    # fixtures carry a "_description" meta field the contracts reject via
    # additionalProperties: false, so strip "_"-prefixed keys before writing
    return {k: v for k, v in record.items() if not k.startswith("_")}


class TestValidateDataCLI(unittest.TestCase):
    """`validate_schemas.py --data <path> --schema <name>` against a JSONL file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_jsonl(self, records: list[dict], name: str) -> Path:
        path = self.tmp_dir / name
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return path

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )

    def test_valid_jsonl_file_exits_zero(self):
        fixtures = json.loads(
            (FIXTURES_DIR / "gold_sector_load_valid.json").read_text()
        )
        records = [_clean(r) for r in fixtures]
        data_path = self._write_jsonl(records, "sector_load.jsonl")

        result = self._run("--data", str(data_path), "--schema", "gold_sector_load")

        self.assertEqual(
            result.returncode, 0, f"expected exit 0, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_broken_record_exits_nonzero_and_names_the_line(self):
        fixtures = json.loads(
            (FIXTURES_DIR / "gold_sector_load_valid.json").read_text()
        )
        records = [_clean(r) for r in fixtures]
        # break the second record: drop a required field
        broken_line = 2
        del records[broken_line - 1]["aircraft_count"]
        data_path = self._write_jsonl(records, "sector_load_broken.jsonl")

        result = self._run("--data", str(data_path), "--schema", "gold_sector_load")

        self.assertNotEqual(
            result.returncode, 0,
            f"expected non-zero exit for a broken record\nstdout:\n{result.stdout}"
        )
        self.assertIn(
            f"line {broken_line}", result.stdout,
            f"failure output does not name the offending line:\n{result.stdout}"
        )

    def test_no_arguments_still_exits_zero(self):
        result = self._run()

        self.assertEqual(
            result.returncode, 0, f"expected exit 0, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
