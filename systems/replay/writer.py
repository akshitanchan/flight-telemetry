#!/usr/bin/env python3
"""
Landing writer — writes normalized records to durable local storage.

Writes JSONL output to the landing directory. Each record is one
normalized state vector in the landing format.

Production would write to Parquet / Delta Lake / a message queue.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LandingWriter:
    """Write normalized records to a JSONL landing file."""

    def __init__(self, output_path: Path, dry_run: bool = False):
        """
        Args:
            output_path: Path to the output JSONL file.
            dry_run: If True, count records but don't write to disk.
        """
        self._output_path = output_path
        self._dry_run = dry_run
        self._count = 0
        self._file = None

    def open(self) -> None:
        """Open the output file for writing."""
        if self._dry_run:
            logger.info("DRY RUN — would write to %s", self._output_path)
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_path, "a")
        logger.info("Opened landing file: %s", self._output_path)

    def write(self, record: dict) -> None:
        """Write a single normalized record."""
        self._count += 1
        if self._dry_run:
            return

        if self._file is None:
            raise RuntimeError("Writer not opened. Call open() first.")

        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Close the output file and log stats."""
        if self._file is not None:
            self._file.close()
            self._file = None

        mode = "DRY RUN" if self._dry_run else "WRITTEN"
        logger.info("%s: %d records → %s", mode, self._count, self._output_path.name)

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
