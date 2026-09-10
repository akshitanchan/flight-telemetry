#!/usr/bin/env python3
"""Offline unit tests for ai/tools/embeddings.py and the vector path in retrieval.py.

All tests are fully offline:
  - No real OpenAI API call is made.
  - No real Postgres connection is opened.
  - Stubs / mocks replace the network + DB layers.

Coverage targets:
  1. ``embeddings.is_available()`` reflects OPENAI_API_KEY presence.
  2. ``embeddings.embed()`` raises when no API key is set.
  3. ``embeddings.embed()`` correctly parses a stubbed API response.
  4. ``embeddings.index_corpus()`` raises when OPENAI_API_KEY is absent.
  5. ``embeddings.index_corpus()`` executes the expected SQL when services are up
     (fully mocked: no real HTTP, no real DB).
  6. ``RetrievalTool.vector_backend_available()`` returns False when key absent.
  7. ``RetrievalTool.search()`` uses the keyword fallback in offline CI conditions.
  8. ``RetrievalTool._vector_search()`` maps pgvector rows back to correct doc ids
     and returns the ``{answer, sources, results}`` shape.
  9. ``RetrievalTool.search()`` falls back to keyword scorer on vector-path failure.
"""

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ai.tools.embeddings as embeddings_mod
from ai.tools.retrieval import RetrievalTool

CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"

# A fake 1536-dim vector (all zeros except first element) — small but valid.
FAKE_VEC = [0.1] + [0.0] * 1535


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_api_response(n: int) -> bytes:
    """Construct a minimal OpenAI embeddings API JSON response for n inputs."""
    data = [{"index": i, "embedding": FAKE_VEC} for i in range(n)]
    return json.dumps({"data": data}).encode()


# ---------------------------------------------------------------------------
# Tests: embeddings module
# ---------------------------------------------------------------------------

class TestIsAvailable(unittest.TestCase):
    def test_false_when_key_absent(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            # Ensure key is not present.
            env = dict(__import__("os").environ)
            env.pop("OPENAI_API_KEY", None)
            with unittest.mock.patch.dict("os.environ", env, clear=True):
                self.assertFalse(embeddings_mod.is_available())

    def test_true_when_key_present(self):
        with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            self.assertTrue(embeddings_mod.is_available())


class TestEmbed(unittest.TestCase):
    def test_raises_without_api_key(self):
        env = dict(__import__("os").environ)
        env.pop("OPENAI_API_KEY", None)
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaises(EnvironmentError):
                embeddings_mod.embed(["hello"])

    def test_returns_vectors_from_stubbed_response(self):
        fake_response_body = _make_fake_api_response(2)

        class _FakeHTTPResponse:
            def read(self):
                return fake_response_body
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        fake_ctx = _FakeHTTPResponse()

        with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with unittest.mock.patch(
                "urllib.request.urlopen", return_value=fake_ctx
            ) as mock_open:
                # Patch json.load to parse our fake bytes as JSON.
                with unittest.mock.patch(
                    "json.load",
                    side_effect=lambda r: json.loads(fake_response_body),
                ):
                    result = embeddings_mod.embed(["text one", "text two"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1536)
        self.assertAlmostEqual(result[0][0], 0.1)

    def test_raises_on_dimension_mismatch(self):
        wrong_dim_vec = [0.0] * 10  # 10-dim, not 1536
        fake_body = json.dumps({"data": [{"index": 0, "embedding": wrong_dim_vec}]}).encode()

        class _FakeHTTPResponse:
            def read(self):
                return fake_body
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with unittest.mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse()):
                with unittest.mock.patch(
                    "json.load",
                    side_effect=lambda r: json.loads(fake_body),
                ):
                    with self.assertRaises(ValueError):
                        embeddings_mod.embed(["text"])


class TestIndexCorpus(unittest.TestCase):
    def test_raises_without_api_key(self):
        env = dict(__import__("os").environ)
        env.pop("OPENAI_API_KEY", None)
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaises(EnvironmentError):
                embeddings_mod.index_corpus([{"id": "x", "title": "t", "text": "body"}])

    def test_raises_when_db_unavailable(self):
        mock_pg = unittest.mock.MagicMock()
        mock_pg.healthcheck.return_value = False

        with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with unittest.mock.patch.dict(
                "sys.modules", {"shared.store.pg": mock_pg, "shared.store": unittest.mock.MagicMock()}
            ):
                # Re-import embeddings with mocked pg module in place.
                import importlib
                with unittest.mock.patch("shared.store.pg", mock_pg):
                    with self.assertRaises((EnvironmentError, RuntimeError)):
                        # We call is_available (True) then pg.healthcheck (False).
                        # The simplest way: patch the internal module reference.
                        with unittest.mock.patch.object(
                            embeddings_mod, "is_available", return_value=True
                        ):
                            import types
                            fake_pg_mod = types.ModuleType("shared.store.pg")
                            fake_pg_mod.healthcheck = lambda: False
                            fake_pg_mod.get_conn = lambda: None
                            sys.modules["shared.store.pg"] = fake_pg_mod
                            try:
                                embeddings_mod.index_corpus(
                                    [{"id": "x", "title": "t", "text": "body"}]
                                )
                            finally:
                                sys.modules.pop("shared.store.pg", None)

    def test_upserts_with_mocked_services(self):
        """Full path: embed + DELETE + INSERT executes the expected SQL calls."""
        docs = [
            {"id": "metar-eham", "type": "metar", "title": "METAR EHAM", "text": "some metar text"},
        ]
        fake_vec = FAKE_VEC

        # Mock execute and conn context manager.
        mock_conn = unittest.mock.MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = unittest.mock.MagicMock(return_value=False)

        import types
        # Inject a fake shared.store.pg module into sys.modules so that the
        # ``from shared.store import pg`` inside index_corpus resolves to it.
        fake_shared_store = types.ModuleType("shared.store")
        fake_pg_mod = types.ModuleType("shared.store.pg")
        fake_pg_mod.healthcheck = lambda: True
        fake_pg_mod.get_conn = lambda: mock_conn
        fake_shared_store.pg = fake_pg_mod
        sys.modules["shared.store"] = fake_shared_store
        sys.modules["shared.store.pg"] = fake_pg_mod

        try:
            with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                with unittest.mock.patch.object(
                    embeddings_mod, "embed", return_value=[fake_vec]
                ):
                    count = embeddings_mod.index_corpus(docs)

            self.assertEqual(count, 1)
            # Verify a DELETE and INSERT were issued.
            call_args_list = mock_conn.execute.call_args_list
            sql_calls = [str(c[0][0]).strip() for c in call_args_list]
            delete_calls = [s for s in sql_calls if s.startswith("DELETE")]
            insert_calls = [s for s in sql_calls if s.startswith("INSERT")]
            self.assertEqual(len(delete_calls), 1)
            self.assertEqual(len(insert_calls), 1)
            mock_conn.commit.assert_called_once()
        finally:
            sys.modules.pop("shared.store.pg", None)
            sys.modules.pop("shared.store", None)


# ---------------------------------------------------------------------------
# Tests: retrieval vector backend (mocked)
# ---------------------------------------------------------------------------

class TestRetrievalToolVectorBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = RetrievalTool(CORPUS)

    def test_vector_backend_unavailable_without_key(self):
        """vector_backend_available() must return False when API key is absent."""
        env = dict(__import__("os").environ)
        env.pop("OPENAI_API_KEY", None)
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            self.assertFalse(RetrievalTool.vector_backend_available())

    def test_keyword_fallback_used_in_offline_ci(self):
        """search() uses keyword fallback when vector backend is unavailable."""
        with unittest.mock.patch.object(
            RetrievalTool, "vector_backend_available", return_value=False
        ):
            result = self.tool.search("METAR EHAM Amsterdam Schiphol decode")

        self.assertIn("metar-eham", result["sources"])
        self.assertIn("answer", result)
        self.assertIn("results", result)

    def _make_pg_mock(self, pg_rows):
        """Return a (fake_pg_mod, mock_conn) pair that yields pg_rows on execute().fetchall()."""
        import types
        mock_conn = unittest.mock.MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = unittest.mock.MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = pg_rows
        fake_shared_store = types.ModuleType("shared.store")
        fake_pg_mod = types.ModuleType("shared.store.pg")
        fake_pg_mod.healthcheck = lambda: True
        fake_pg_mod.get_conn = lambda: mock_conn
        fake_shared_store.pg = fake_pg_mod
        return fake_shared_store, fake_pg_mod, mock_conn

    def test_vector_search_maps_rows_to_corpus_docs(self):
        """_vector_search() correctly maps pgvector rows to the {answer, sources, results} shape."""
        # Simulate pgvector returning two rows: metar-eham (score 0.92), ref-squawk-codes (0.80)
        pg_rows = [("metar-eham", 0.92), ("ref-squawk-codes", 0.80)]
        fake_shared_store, fake_pg_mod, mock_conn = self._make_pg_mock(pg_rows)

        sys.modules["shared.store"] = fake_shared_store
        sys.modules["shared.store.pg"] = fake_pg_mod

        try:
            # Patch embed on the real already-imported embeddings module, and set env key.
            with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                with unittest.mock.patch.object(
                    embeddings_mod, "embed", return_value=[FAKE_VEC]
                ):
                    result = self.tool._vector_search("METAR EHAM", top_k=2)
        finally:
            sys.modules.pop("shared.store.pg", None)
            sys.modules.pop("shared.store", None)

        # Shape check.
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("results", result)

        # Sources must only contain known doc ids (no fabrication).
        self.assertIn("metar-eham", result["sources"])
        self.assertIn("ref-squawk-codes", result["sources"])

        # Results carry id, score, title, snippet.
        for r in result["results"]:
            self.assertIn("id", r)
            self.assertIn("score", r)
            self.assertIn("title", r)
            self.assertIn("snippet", r)

    def test_vector_search_skips_unknown_doc_ids(self):
        """_vector_search() omits rows whose doc_id is not in the corpus (never fabricates)."""
        pg_rows = [("not-in-corpus-at-all", 0.99), ("metar-eham", 0.80)]
        fake_shared_store, fake_pg_mod, _ = self._make_pg_mock(pg_rows)

        sys.modules["shared.store"] = fake_shared_store
        sys.modules["shared.store.pg"] = fake_pg_mod

        try:
            with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                with unittest.mock.patch.object(
                    embeddings_mod, "embed", return_value=[FAKE_VEC]
                ):
                    result = self.tool._vector_search("anything", top_k=2)
        finally:
            sys.modules.pop("shared.store.pg", None)
            sys.modules.pop("shared.store", None)

        self.assertNotIn("not-in-corpus-at-all", result["sources"])
        self.assertIn("metar-eham", result["sources"])

    def test_search_falls_back_on_vector_exception(self):
        """search() gracefully falls back to keyword scorer when vector path raises."""
        import types

        # Make the backend appear available but raise during _vector_search.
        with unittest.mock.patch.object(
            RetrievalTool, "vector_backend_available", return_value=True
        ):
            with unittest.mock.patch.object(
                self.tool, "_vector_search", side_effect=RuntimeError("DB unavailable")
            ):
                result = self.tool.search("engine failure during climb incident report")

        # Must still return correct shape with correct citation via keyword fallback.
        self.assertIn("ntsb-eng-failure", result["sources"])
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("results", result)


# ---------------------------------------------------------------------------
# Tests: import safety (no socket on import)
# ---------------------------------------------------------------------------

class TestImportSafety(unittest.TestCase):
    def test_embeddings_module_importable_without_openai(self):
        """ai.tools.embeddings must be importable without the openai package."""
        import importlib

        # Simulate openai not installed by blocking it.
        with unittest.mock.patch.dict("sys.modules", {"openai": None}):
            # Should not raise.
            try:
                importlib.reload(embeddings_mod)
            except ImportError as e:
                if "openai" in str(e).lower():
                    self.fail(f"embeddings.py imports openai at module level: {e}")
                # Other ImportErrors (e.g. missing native dep) are acceptable.

    def test_retrieval_module_importable_without_openai(self):
        """ai.tools.retrieval must be importable without the openai package."""
        from ai.tools import retrieval as retrieval_mod
        import importlib

        with unittest.mock.patch.dict("sys.modules", {"openai": None}):
            try:
                importlib.reload(retrieval_mod)
            except ImportError as e:
                if "openai" in str(e).lower():
                    self.fail(f"retrieval.py imports openai at module level: {e}")


if __name__ == "__main__":
    unittest.main()
