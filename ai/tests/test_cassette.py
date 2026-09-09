#!/usr/bin/env python3
"""Tests for the eval matrix's record/replay cassette (offline, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.providers import Provider
from ai.eval.cassette import Cassette, CassetteMiss, wrap_provider, wrap_retrieval


def _raising_fn(*args, **kwargs):
    raise AssertionError("underlying call should not have been made")


class TestCassette(unittest.TestCase):
    def test_round_trip_through_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            payload = {"question": "what is the metar for eham"}

            c = Cassette(path, record=True)
            c.store(payload, {"answer": "clear skies"})
            c.save()

            c2 = Cassette(path, record=False)
            self.assertEqual(c2.fetch(payload), {"answer": "clear skies"})

    def test_key_stable_across_payload_dict_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            c = Cassette(path, record=True)
            c.store({"a": 1, "b": 2}, "first")
            c.save()

            c2 = Cassette(path, record=False)
            self.assertEqual(c2.fetch({"b": 2, "a": 1}), "first")

    def test_replay_miss_raises_naming_the_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            c = Cassette(path, record=False)
            with self.assertRaises(CassetteMiss) as ctx:
                c.fetch({"question": "how far is the diversion airport"})
            self.assertIn("how far is the diversion airport", str(ctx.exception))

    def test_record_mode_allows_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            c = Cassette(path, record=True)
            with self.assertRaises(CassetteMiss):
                c.fetch({"question": "unseen"})

    def test_save_writes_sorted_indented_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            c = Cassette(path, record=True)
            c.store({"b": 1}, "x")
            c.save()
            raw = path.read_text()
            self.assertIn("\n", raw)
            data = json.loads(raw)
            self.assertIn("entries", data)
            self.assertIn("recorded_at", data)


class TestWrapProvider(unittest.TestCase):
    def test_record_then_replay_calls_underlying_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            calls = {"n": 0}

            def _fn(messages):
                calls["n"] += 1
                return "the answer", 10, 5

            messages = [{"role": "user", "content": "hello"}]

            provider = Provider("openai", "gpt-4o-mini", _fn)
            cassette = Cassette(path, record=True)
            wrapped = wrap_provider(provider, cassette)

            content, tokens = wrapped(messages)
            self.assertEqual(content, "the answer")
            self.assertEqual(tokens, 15)
            self.assertEqual(calls["n"], 1)

            content2, tokens2 = wrapped(messages)
            self.assertEqual(content2, "the answer")
            self.assertEqual(tokens2, 15)
            self.assertEqual(calls["n"], 1, "second call should hit the cassette")

    def test_replay_never_calls_underlying_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            messages = [{"role": "user", "content": "hello"}]

            recording_provider = Provider(
                "openai", "gpt-4o-mini",
                lambda messages: ("the answer", 10, 5),
            )
            cassette = Cassette(path, record=True)
            wrap_provider(recording_provider, cassette)(messages)
            cassette.save()

            replay_cassette = Cassette(path, record=False)
            broken_provider = Provider("openai", "gpt-4o-mini", _raising_fn)
            wrapped = wrap_provider(broken_provider, replay_cassette)

            content, tokens = wrapped(messages)
            self.assertEqual(content, "the answer")
            self.assertEqual(tokens, 15)

    def test_replay_populates_calls_with_recorded_tokens_and_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            messages = [{"role": "user", "content": "hello"}]

            recording_provider = Provider(
                "openai", "gpt-4o-mini",
                lambda messages: ("the answer", 10, 5),
            )
            cassette = Cassette(path, record=True)
            wrap_provider(recording_provider, cassette)(messages)
            cassette.save()

            replay_cassette = Cassette(path, record=False)
            broken_provider = Provider("openai", "gpt-4o-mini", _raising_fn)
            wrapped = wrap_provider(broken_provider, replay_cassette)
            wrapped(messages)

            self.assertEqual(len(wrapped.calls), 1)
            call = wrapped.calls[0]
            self.assertEqual(call["input_tokens"], 10)
            self.assertEqual(call["output_tokens"], 5)
            self.assertIn("latency_s", call)

    def test_replay_miss_raises_and_never_touches_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            cassette = Cassette(path, record=False)
            broken_provider = Provider("openai", "gpt-4o-mini", _raising_fn)
            wrapped = wrap_provider(broken_provider, cassette)

            with self.assertRaises(CassetteMiss):
                wrapped([{"role": "user", "content": "never recorded"}])

    def test_wrapped_provider_exposes_same_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            provider = Provider("ollama", "llama3", lambda messages: ("x", 1, 1))
            wrapped = wrap_provider(provider, Cassette(path, record=True))
            self.assertEqual(wrapped.name, "ollama")
            self.assertEqual(wrapped.model, "llama3")
            self.assertEqual(wrapped.label, "ollama:llama3")
            self.assertEqual(wrapped.calls, [])


class _StubRetrieval:
    def __init__(self):
        self.searched = False

    def search(self, query, top_k=3):
        self.searched = True
        return {"answer": "snippet", "sources": ["doc-1"], "results": []}


class TestWrapRetrieval(unittest.TestCase):
    def test_record_then_replay_returns_stored_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            tool = _StubRetrieval()
            cassette = Cassette(path, record=True)
            wrapped = wrap_retrieval(tool, cassette)

            result = wrapped.search("metar eham")
            self.assertTrue(tool.searched)
            self.assertEqual(result["sources"], ["doc-1"])

            replay_tool = _StubRetrieval()
            replay_cassette = Cassette(path, record=False)
            replay_wrapped = wrap_retrieval(replay_tool, replay_cassette)

            replayed = replay_wrapped.search("metar eham")
            self.assertEqual(replayed, result)
            self.assertFalse(replay_tool.searched, "replay must not touch the tool")

    def test_replay_miss_raises_naming_the_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            tool = _StubRetrieval()
            cassette = Cassette(path, record=False)
            wrapped = wrap_retrieval(tool, cassette)

            with self.assertRaises(CassetteMiss) as ctx:
                wrapped.search("unrecorded query")
            self.assertIn("unrecorded query", str(ctx.exception))
            self.assertFalse(tool.searched)

    def test_top_k_is_part_of_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cassette.json"
            tool = _StubRetrieval()
            cassette = Cassette(path, record=True)
            wrapped = wrap_retrieval(tool, cassette)
            wrapped.search("metar eham", top_k=5)
            cassette.save()

            replay_cassette = Cassette(path, record=False)
            replay_wrapped = wrap_retrieval(_StubRetrieval(), replay_cassette)
            with self.assertRaises(CassetteMiss):
                replay_wrapped.search("metar eham", top_k=3)
            replay_wrapped.search("metar eham", top_k=5)


if __name__ == "__main__":
    unittest.main()
