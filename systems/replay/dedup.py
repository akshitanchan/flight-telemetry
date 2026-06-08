#!/usr/bin/env python3
"""
Idempotency store — deduplicates landing records on (icao24, event_ts).

Uses an in-memory set backed by an optional persistent journal file.
The key format is "icao24:unix_ts" (precomputed as idem_key by the normalizer).

This is a simple file-based approach suitable for bounded local replay.
Production would use a database-backed store (e.g., Redis SET, or a
UNIQUE constraint on (icao24, event_ts) in the landing table).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IdempotencyStore:
    """Track which (icao24, event_ts) pairs have been ingested."""

    def __init__(self, journal_path: Path | None = None):
        """Initialize the store, optionally loading existing keys from journal.

        Args:
            journal_path: If provided, persist seen keys to this file.
                         If the file exists, load previously seen keys.
        """
        self._seen: set[str] = set()
        self._journal_path = journal_path
        self._new_count = 0
        self._dup_count = 0

        if journal_path and journal_path.exists():
            self._load_journal(journal_path)

    def _load_journal(self, path: Path) -> None:
        """Load previously seen keys from journal file."""
        with open(path) as f:
            for line in f:
                key = line.strip()
                if key:
                    self._seen.add(key)
        logger.info("Loaded %d existing keys from journal", len(self._seen))

    def check_and_mark(self, idem_key: str) -> bool:
        """Check if a key has been seen before. If not, mark it as seen.

        Args:
            idem_key: The idempotency key (format: "icao24:unix_ts").

        Returns:
            True if the record is NEW (not seen before).
            False if it is a DUPLICATE.
        """
        if idem_key in self._seen:
            self._dup_count += 1
            return False

        self._seen.add(idem_key)
        self._new_count += 1
        return True

    def flush_journal(self) -> None:
        """Write all seen keys to the journal file."""
        if self._journal_path is None:
            return

        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._journal_path, "w") as f:
            for key in sorted(self._seen):
                f.write(key + "\n")

        logger.info("Flushed %d keys to journal", len(self._seen))

    @property
    def stats(self) -> dict:
        """Return dedup statistics."""
        return {
            "total_seen": len(self._seen),
            "new_this_run": self._new_count,
            "duplicates_this_run": self._dup_count,
        }
