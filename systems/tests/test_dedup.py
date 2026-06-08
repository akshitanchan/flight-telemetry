#!/usr/bin/env python3
"""Tests for the idempotency store."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import unittest
from systems.replay.dedup import IdempotencyStore


class TestIdempotencyStore(unittest.TestCase):
    """Test idempotency key tracking and journal persistence."""

    def test_new_key_returns_true(self):
        """First time seeing a key returns True (is new)."""
        store = IdempotencyStore()
        self.assertTrue(store.check_and_mark("abc:123"))

    def test_duplicate_key_returns_false(self):
        """Second time seeing the same key returns False (is duplicate)."""
        store = IdempotencyStore()
        store.check_and_mark("abc:123")
        self.assertFalse(store.check_and_mark("abc:123"))

    def test_different_keys_both_new(self):
        """Different keys are both marked as new."""
        store = IdempotencyStore()
        self.assertTrue(store.check_and_mark("abc:123"))
        self.assertTrue(store.check_and_mark("def:456"))

    def test_stats_tracking(self):
        """Stats correctly count new and duplicate records."""
        store = IdempotencyStore()
        store.check_and_mark("a:1")
        store.check_and_mark("b:2")
        store.check_and_mark("a:1")  # duplicate
        store.check_and_mark("c:3")
        store.check_and_mark("b:2")  # duplicate

        stats = store.stats
        self.assertEqual(stats["total_seen"], 3)
        self.assertEqual(stats["new_this_run"], 3)
        self.assertEqual(stats["duplicates_this_run"], 2)

    def test_journal_persistence(self):
        """Keys persist across store instances via journal file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".journal", delete=False) as f:
            journal_path = Path(f.name)

        try:
            # First run: add some keys
            store1 = IdempotencyStore(journal_path=journal_path)
            store1.check_and_mark("abc:123")
            store1.check_and_mark("def:456")
            store1.flush_journal()

            # Second run: load journal, check dedup works
            store2 = IdempotencyStore(journal_path=journal_path)
            self.assertFalse(store2.check_and_mark("abc:123"))  # already seen
            self.assertFalse(store2.check_and_mark("def:456"))  # already seen
            self.assertTrue(store2.check_and_mark("ghi:789"))   # new
        finally:
            journal_path.unlink(missing_ok=True)

    def test_idempotent_replay(self):
        """Running the same batch twice only writes new records once."""
        store = IdempotencyStore()
        batch = ["a:1", "b:2", "c:3"]

        # First pass — all new
        first_pass = [k for k in batch if store.check_and_mark(k)]
        self.assertEqual(len(first_pass), 3)

        # Second pass — all duplicates
        second_pass = [k for k in batch if store.check_and_mark(k)]
        self.assertEqual(len(second_pass), 0)


if __name__ == "__main__":
    unittest.main()
