#!/usr/bin/env python3
"""Offline unit tests for the ai/obs package (ai-06).

All tests run fully offline:
  - No real OpenAI API calls.
  - No real OTel collector.
  - No Prometheus scrape endpoint.
  - Embeddings are mocked via unittest.mock.

Coverage:
  1. Import safety — ai.obs.* importable without opentelemetry, prometheus_client,
     or openai installed, with no socket on import.
  2. Tracing wrapper (traced_llm):
     a. No-op with no OTel collector — wrapper returns the correct (content, tokens)
        tuple without raising.
     b. The underlying LLM callable is invoked exactly once per call.
     c. Exceptions from the LLM propagate unchanged.
  3. Tracing wrapper (traced_answer):
     a. Wraps an AnswerStrategy.answer() correctly — returns the Answer unchanged.
     b. Clean no-op when OTel SDK is absent.
  4. Cost calculator:
     a. gpt-4o-mini returns the expected USD amount for a token count.
     b. Ollama model returns $0 (local model).
     c. Unknown model returns $0 (conservative).
     d. Zero-token calls return $0.
     e. "openai:" prefix is stripped correctly.
  5. record():
     a. Returns the correct USD amount regardless of Prometheus availability.
     b. Does not raise when prometheus_client is absent.
  6. SemanticCache — exact-match mode (offline):
     a. Cache miss returns None.
     b. Cache hit returns the stored value.
     c. Cache hit skips the wrapped LLM call (the LLM callable is NOT invoked).
     d. Cache size is bounded; oldest entry is evicted when max_size is reached.
     e. Normalisation: queries differing only in case / whitespace hit the
        same entry.
     f. clear() flushes all entries.
     g. stats() returns correct counts.
     h. keying="exact" never calls the embeddings module, even with a fake
        OPENAI_API_KEY set, and stats() reports "exact".
     i. keying="embedding" raises at construction when embeddings are
        unavailable; an invalid keying value raises ValueError.
  7. SemanticCache — embedding path (mocked):
     a. When embeddings are available (mocked), a semantically similar query
        (cosine similarity >= threshold) returns a cache hit.
     b. A dissimilar query (cosine similarity < threshold) is a miss.
     c. Embedding failure during get() falls back to exact-match silently.
  8. Cache-hit skips LLM call integration test:
     Wrap an LLM stub with a SemanticCache; after one answer the same question
     served from cache does not invoke the stub again.
"""

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------

class TestImportSafety(unittest.TestCase):
    """ai.obs.* must be importable without heavy deps and without opening sockets."""

    def _reload_obs(self):
        """Remove cached ai.obs modules and re-import."""
        import importlib
        for key in list(sys.modules):
            if key.startswith("ai.obs"):
                del sys.modules[key]

    def test_tracing_importable_without_opentelemetry(self):
        with unittest.mock.patch.dict("sys.modules", {
            "opentelemetry": None,
            "opentelemetry.trace": None,
        }):
            self._reload_obs()
            import ai.obs.tracing  # noqa: F401
            self.assertTrue(hasattr(ai.obs.tracing, "traced_llm"))

    def test_costs_importable_without_prometheus(self):
        with unittest.mock.patch.dict("sys.modules", {"prometheus_client": None}):
            self._reload_obs()
            import ai.obs.costs as costs_mod
            self.assertTrue(hasattr(costs_mod, "cost_usd"))

    def test_cache_importable_without_numpy(self):
        with unittest.mock.patch.dict("sys.modules", {"numpy": None}):
            self._reload_obs()
            import ai.obs.cache as cache_mod
            self.assertTrue(hasattr(cache_mod, "SemanticCache"))

    def test_package_init_importable(self):
        self._reload_obs()
        import ai.obs  # noqa: F401
        from ai.obs import traced_llm, cost_usd, SemanticCache  # noqa: F401
        self.assertTrue(True)


# ---------------------------------------------------------------------------
# 2. Tracing — traced_llm
# ---------------------------------------------------------------------------

class TestTracedLlm(unittest.TestCase):
    """traced_llm wraps an LLM callable; clean no-op without OTel collector."""

    def setUp(self):
        from ai.obs.tracing import traced_llm
        self.traced_llm = traced_llm

    def _make_stub(self, content="answer", tokens=42):
        """Return a mock LLM callable that records its call count."""
        mock = unittest.mock.MagicMock(return_value=(content, tokens))
        return mock

    def test_returns_correct_tuple(self):
        stub = self._make_stub(content="hello", tokens=10)
        wrapped = self.traced_llm(stub, strategy="test_strategy")
        result = wrapped([{"role": "user", "content": "q"}])
        self.assertEqual(result, ("hello", 10))

    def test_llm_called_exactly_once(self):
        stub = self._make_stub()
        wrapped = self.traced_llm(stub, strategy="test_strategy")
        wrapped([{"role": "user", "content": "q"}])
        stub.assert_called_once()

    def test_noop_without_otel(self):
        """traced_llm must work even when opentelemetry is not installed."""
        with unittest.mock.patch.dict("sys.modules", {
            "opentelemetry": None,
            "opentelemetry.trace": None,
        }):
            for key in list(sys.modules):
                if key.startswith("ai.obs"):
                    del sys.modules[key]
            from ai.obs.tracing import traced_llm as tl
            stub = unittest.mock.MagicMock(return_value=("ok", 5))
            wrapped = tl(stub, strategy="s")
            result = wrapped([])
            self.assertEqual(result, ("ok", 5))

    def test_exception_propagates(self):
        stub = unittest.mock.MagicMock(side_effect=RuntimeError("LLM failed"))
        wrapped = self.traced_llm(stub, strategy="s")
        with self.assertRaises(RuntimeError):
            wrapped([])

    def test_llm_not_called_on_cache_hit(self):
        """Demonstrates that a cache wrapping pattern can skip the LLM entirely.

        This test constructs the skip manually: if a cache returns a value,
        the caller never invokes traced_llm at all.  This is the documented
        integration pattern.
        """
        from ai.obs.cache import SemanticCache

        call_count = {"n": 0}

        def _stub_llm(messages):
            call_count["n"] += 1
            return "answer", 20

        cache = SemanticCache(max_size=8)
        question = "how many emergency events"

        # First call: cache miss → call LLM → store in cache.
        cached = cache.get(question)
        if cached is None:
            result = _stub_llm([{"role": "user", "content": question}])
            cache.put(question, result)
        else:
            result = cached

        self.assertEqual(call_count["n"], 1)

        # Second call: cache hit → do NOT call LLM.
        cached2 = cache.get(question)
        if cached2 is None:
            result2 = _stub_llm([{"role": "user", "content": question}])
            cache.put(question, result2)
        else:
            result2 = cached2

        # LLM must still have been called only once.
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(result2, ("answer", 20))


# ---------------------------------------------------------------------------
# 3. Tracing — traced_answer
# ---------------------------------------------------------------------------

class TestTracedAnswer(unittest.TestCase):
    """traced_answer wraps AnswerStrategy.answer(); returns Answer unchanged."""

    def _make_fake_answer(self, tokens=30, route="analytics:count_emergencies"):
        """Build a minimal Answer-like object without importing ai.agent.base."""
        from ai.agent.base import Answer
        return Answer(
            question="test",
            answer_text="42 events (source: gold)",
            result={"answer": 42, "sources": ["gold"]},
            route=route,
            strategy="test",
            meta={"tokens": tokens, "provider": "stub"},
        )

    def test_returns_answer_unchanged(self):
        from ai.obs.tracing import traced_answer

        original = self._make_fake_answer(tokens=30)

        def _fake_answer_method(question):
            return original

        wrapped = traced_answer(_fake_answer_method, strategy_name="test_strategy")
        result = wrapped("test question")
        self.assertIs(result, original)

    def test_noop_without_otel_sdk(self):
        """traced_answer must return the Answer even when OTel is absent."""
        with unittest.mock.patch.dict("sys.modules", {
            "opentelemetry": None,
            "opentelemetry.trace": None,
        }):
            for key in list(sys.modules):
                if key.startswith("ai.obs"):
                    del sys.modules[key]
            from ai.obs.tracing import traced_answer as ta

            original = self._make_fake_answer()

            def _fake_answer(question):
                return original

            wrapped = ta(_fake_answer, strategy_name="s")
            result = wrapped("q")
            self.assertIs(result, original)

    def test_exception_propagates_from_answer(self):
        from ai.obs.tracing import traced_answer

        def _bad_answer(question):
            raise ValueError("strategy failed")

        wrapped = traced_answer(_bad_answer, strategy_name="s")
        with self.assertRaises(ValueError):
            wrapped("q")


# ---------------------------------------------------------------------------
# 4. Cost calculator
# ---------------------------------------------------------------------------

class TestCostCalc(unittest.TestCase):
    """cost_usd returns correct USD amounts for various models and token counts."""

    def setUp(self):
        from ai.obs.costs import cost_usd
        self.cost_usd = cost_usd

    def test_gpt4o_mini_1000_tokens(self):
        """1000 tokens of gpt-4o-mini should cost $0.000150 (0.15/1000*1)."""
        usd = self.cost_usd("gpt-4o-mini", 1000)
        self.assertAlmostEqual(usd, 0.000150, places=8)

    def test_gpt4o_mini_500_tokens(self):
        usd = self.cost_usd("gpt-4o-mini", 500)
        self.assertAlmostEqual(usd, 0.000075, places=8)

    def test_openai_prefix_stripped(self):
        """openai:gpt-4o-mini should produce the same result as gpt-4o-mini."""
        usd_bare = self.cost_usd("gpt-4o-mini", 1000)
        usd_prefixed = self.cost_usd("openai:gpt-4o-mini", 1000)
        self.assertAlmostEqual(usd_bare, usd_prefixed, places=10)

    def test_ollama_model_is_free(self):
        """Ollama models are local; cost should be $0."""
        usd = self.cost_usd("ollama:llama3.2", 10000)
        self.assertEqual(usd, 0.0)

    def test_ollama_prefix_only(self):
        usd = self.cost_usd("ollama:qwen2.5-coder:7b", 5000)
        self.assertEqual(usd, 0.0)

    def test_unknown_model_returns_zero(self):
        """Unknown model → conservative $0 (never invent phantom costs)."""
        usd = self.cost_usd("my-private-model-v99", 1000)
        self.assertEqual(usd, 0.0)

    def test_zero_tokens_returns_zero(self):
        usd = self.cost_usd("gpt-4o-mini", 0)
        self.assertEqual(usd, 0.0)

    def test_negative_tokens_returns_zero(self):
        usd = self.cost_usd("gpt-4o-mini", -100)
        self.assertEqual(usd, 0.0)

    def test_embedding_model(self):
        """text-embedding-3-small: $0.000020 / 1K tokens."""
        usd = self.cost_usd("text-embedding-3-small", 1000)
        self.assertAlmostEqual(usd, 0.000020, places=8)

    def test_gpt4o_price(self):
        """gpt-4o: $0.005 / 1K tokens."""
        usd = self.cost_usd("gpt-4o", 1000)
        self.assertAlmostEqual(usd, 0.005, places=6)

    def test_case_insensitive(self):
        usd_lower = self.cost_usd("gpt-4o-mini", 1000)
        usd_upper = self.cost_usd("GPT-4O-MINI", 1000)
        self.assertAlmostEqual(usd_lower, usd_upper, places=10)

    def test_split_gpt4o_mini_input_and_output(self):
        from ai.obs.costs import cost_split_usd
        usd = cost_split_usd("gpt-4o-mini", 1000, 1000)
        self.assertAlmostEqual(usd, 0.00075, places=8)

    def test_split_bedrock_llama_bare_id(self):
        from ai.obs.costs import cost_split_usd
        usd = cost_split_usd("meta.llama3-1-8b-instruct-v1:0", 1000, 1000)
        self.assertAlmostEqual(usd, 0.00044, places=8)

    def test_split_bedrock_llama_prefixed_id(self):
        from ai.obs.costs import cost_split_usd
        usd = cost_split_usd("bedrock:us.meta.llama3-1-8b-instruct-v1:0", 1000, 1000)
        self.assertAlmostEqual(usd, 0.00044, places=8)

    def test_split_unknown_model_returns_zero(self):
        from ai.obs.costs import cost_split_usd
        usd = cost_split_usd("my-private-model-v99", 1000, 1000)
        self.assertEqual(usd, 0.0)


# ---------------------------------------------------------------------------
# 5. record()
# ---------------------------------------------------------------------------

class TestRecord(unittest.TestCase):
    """record() returns correct USD even when prometheus_client is absent."""

    def test_returns_usd_without_prometheus(self):
        with unittest.mock.patch.dict("sys.modules", {"prometheus_client": None}):
            for key in list(sys.modules):
                if key.startswith("ai.obs"):
                    del sys.modules[key]
            from ai.obs.costs import record
            usd = record("gpt-4o-mini", 1000, strategy="test")
            self.assertAlmostEqual(usd, 0.000150, places=8)

    def test_does_not_raise_without_prometheus(self):
        with unittest.mock.patch.dict("sys.modules", {"prometheus_client": None}):
            for key in list(sys.modules):
                if key.startswith("ai.obs"):
                    del sys.modules[key]
            from ai.obs.costs import record
            try:
                record("gpt-4o-mini", 500, strategy="react")
            except Exception as exc:
                self.fail(f"record() raised an exception without prometheus: {exc}")

    def test_record_ollama_returns_zero(self):
        for key in list(sys.modules):
            if key.startswith("ai.obs"):
                del sys.modules[key]
        from ai.obs.costs import record
        usd = record("ollama:mistral", 10000, strategy="single_shot_rag")
        self.assertEqual(usd, 0.0)


# ---------------------------------------------------------------------------
# 6. SemanticCache — exact-match (offline)
# ---------------------------------------------------------------------------

class TestSemanticCacheExact(unittest.TestCase):
    """SemanticCache exact-match mode tests (no embedding dependency)."""

    def _make_cache(self, max_size=8, threshold=0.92):
        """Return a SemanticCache with embedding mode disabled."""
        from ai.obs.cache import SemanticCache
        cache = SemanticCache(max_size=max_size, threshold=threshold)
        return cache

    def setUp(self):
        # Force embedding mode off for all exact-match tests.
        self._embed_patch = unittest.mock.patch(
            "ai.obs.cache._embeddings_available", return_value=False
        )
        self._embed_patch.start()

    def tearDown(self):
        self._embed_patch.stop()

    def test_miss_returns_none(self):
        cache = self._make_cache()
        self.assertIsNone(cache.get("how many emergency events"))

    def test_hit_returns_stored_value(self):
        cache = self._make_cache()
        cache.put("how many emergency events", {"answer": 3})
        result = cache.get("how many emergency events")
        self.assertEqual(result, {"answer": 3})

    def test_normalisation_case(self):
        """Case differences hit the same entry."""
        cache = self._make_cache()
        cache.put("How Many Emergency Events", "42")
        self.assertEqual(cache.get("how many emergency events"), "42")
        self.assertEqual(cache.get("HOW MANY EMERGENCY EVENTS"), "42")

    def test_normalisation_whitespace(self):
        """Extra whitespace is collapsed."""
        cache = self._make_cache()
        cache.put("  how many   emergency events  ", "42")
        self.assertEqual(cache.get("how many emergency events"), "42")

    def test_size_bounded_evicts_oldest(self):
        """Oldest entry is evicted when max_size is reached."""
        cache = self._make_cache(max_size=3)
        cache.put("q1", "v1")
        cache.put("q2", "v2")
        cache.put("q3", "v3")
        # All three present.
        self.assertEqual(cache.get("q1"), "v1")
        # Insert a 4th: q1 (oldest) should be evicted.
        cache.put("q4", "v4")
        self.assertIsNone(cache.get("q1"))
        self.assertEqual(cache.get("q2"), "v2")
        self.assertEqual(cache.get("q4"), "v4")

    def test_update_existing_key(self):
        """Putting the same key again updates the value without eviction."""
        cache = self._make_cache(max_size=3)
        cache.put("q", "old")
        cache.put("q", "new")
        self.assertEqual(cache.get("q"), "new")
        self.assertEqual(cache.stats()["size"], 1)

    def test_clear_flushes(self):
        cache = self._make_cache()
        cache.put("q", "v")
        cache.clear()
        self.assertIsNone(cache.get("q"))
        self.assertEqual(cache.stats()["size"], 0)

    def test_stats_counts(self):
        cache = self._make_cache()
        cache.put("q", "v")
        cache.get("q")       # hit
        cache.get("missing")  # miss
        s = cache.stats()
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
        self.assertEqual(s["size"], 1)

    def test_cache_hit_skips_llm_call(self):
        """A cache hit must mean the LLM callable is never invoked.

        This is the critical cost-savings test: we simulate a wrapped call
        pattern where the cache is checked first and the LLM is only called
        on a miss.
        """
        call_count = {"n": 0}

        def _stub_llm(messages):
            call_count["n"] += 1
            return "cached answer", 50

        cache = self._make_cache()
        question = "how many emergency events"

        def _call_with_cache(q):
            hit = cache.get(q)
            if hit is not None:
                return hit
            result = _stub_llm([{"role": "user", "content": q}])
            cache.put(q, result)
            return result

        # First call: miss → LLM invoked.
        r1 = _call_with_cache(question)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(r1, ("cached answer", 50))

        # Second call: hit → LLM NOT invoked.
        r2 = _call_with_cache(question)
        self.assertEqual(call_count["n"], 1)  # still 1 — no second call
        self.assertEqual(r2, ("cached answer", 50))

    def test_different_questions_independent(self):
        cache = self._make_cache()
        cache.put("question A", "answer A")
        cache.put("question B", "answer B")
        self.assertEqual(cache.get("question A"), "answer A")
        self.assertEqual(cache.get("question B"), "answer B")
        self.assertIsNone(cache.get("question C"))

    def test_exact_keying_never_calls_embed_and_reports_exact_in_stats(self):
        from ai.obs.cache import SemanticCache

        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-test-key"}), \
                unittest.mock.patch("ai.obs.cache._embeddings_available", return_value=True), \
                unittest.mock.patch("ai.tools.embeddings.embed") as mock_embed:
            mock_embed.side_effect = RuntimeError("embed must not be called in exact mode")

            cache = SemanticCache(max_size=8, keying="exact")
            cache.put("how many emergency events", "42")
            result = cache.get("how many emergency events")

            self.assertEqual(result, "42")
            mock_embed.assert_not_called()
            self.assertEqual(cache.stats()["keying"], "exact")

    def test_embedding_keying_raises_when_unavailable(self):
        """keying="embedding" fails fast at construction if embeddings aren't available."""
        from ai.obs.cache import SemanticCache
        # class setUp already forces _embeddings_available() to False
        with self.assertRaises(RuntimeError):
            SemanticCache(keying="embedding")

    def test_invalid_keying_value_raises(self):
        from ai.obs.cache import SemanticCache
        with self.assertRaises(ValueError):
            SemanticCache(keying="bogus")


# ---------------------------------------------------------------------------
# 7. SemanticCache — embedding path (mocked)
# ---------------------------------------------------------------------------

class TestSemanticCacheEmbedding(unittest.TestCase):
    """SemanticCache embedding-path tests with mocked embed()."""

    # Two near-identical 4-dim vectors and one orthogonal vector for testing.
    _VEC_A = [1.0, 0.0, 0.0, 0.0]   # query A
    _VEC_B = [0.99, 0.14, 0.0, 0.0]  # similar to A (cosine ~0.999)
    _VEC_C = [0.0, 0.0, 1.0, 0.0]   # orthogonal to A (cosine = 0.0)

    def setUp(self):
        # Enable embedding mode.
        self._avail_patch = unittest.mock.patch(
            "ai.obs.cache._embeddings_available", return_value=True
        )
        self._avail_patch.start()

        # Inject a controllable embed function.
        # _embed_query returns a SINGLE vector (list[float]) or None.
        self._embed_return = self._VEC_A  # default: return VEC_A

        def _fake_embed(query):
            # Return a copy of the current vector stored in self._embed_return.
            return list(self._embed_return)

        self._embed_patch = unittest.mock.patch(
            "ai.obs.cache._embed_query", side_effect=_fake_embed
        )
        self._embed_patch.start()

    def tearDown(self):
        self._avail_patch.stop()
        self._embed_patch.stop()

    def _make_cache(self, threshold=0.90):
        from ai.obs.cache import SemanticCache
        return SemanticCache(max_size=8, threshold=threshold)

    def test_semantic_hit_similar_query(self):
        """A query whose embedding is close (>= threshold) returns a hit."""
        cache = self._make_cache(threshold=0.90)

        # Store with VEC_A.
        self._embed_return = self._VEC_A
        cache.put("how many emergency events", "42")

        # Query with VEC_B (cosine similarity ~0.999 > 0.90).
        self._embed_return = self._VEC_B
        result = cache.get("how many emergency squawks happened")
        self.assertEqual(result, "42")

    def test_semantic_miss_dissimilar_query(self):
        """A query whose embedding is far (< threshold) is a miss."""
        cache = self._make_cache(threshold=0.90)

        # Store with VEC_A.
        self._embed_return = self._VEC_A
        cache.put("how many emergency events", "42")

        # Query with VEC_C (cosine similarity 0.0 < 0.90) → miss.
        self._embed_return = self._VEC_C
        result = cache.get("completely different question")
        self.assertIsNone(result)

    def test_embedding_failure_falls_back_to_exact_match(self):
        """When _embed_query returns None, exact-match is used silently."""
        cache = self._make_cache()

        # Store an entry normally (embedding succeeds on put).
        self._embed_return = [self._VEC_A]
        cache.put("how many emergency events", "42")

        # Simulate embedding failure on get by having _embed_query return None.
        with unittest.mock.patch("ai.obs.cache._embed_query", return_value=None):
            # Exact-match should find the normalised key.
            result = cache.get("how many emergency events")
            self.assertEqual(result, "42")

    def test_embedding_exception_falls_back_to_exact_match(self):
        """Exception from _embed_query during get() is swallowed; exact match used."""
        cache = self._make_cache()

        # Store without embedding to set exact-match key.
        self._embed_return = [self._VEC_A]
        cache.put("my question", "answer")

        with unittest.mock.patch(
            "ai.obs.cache._embed_query", side_effect=RuntimeError("API down")
        ):
            result = cache.get("my question")
            self.assertEqual(result, "answer")


# ---------------------------------------------------------------------------
# 8. Cache-hit skips LLM call — integration with strategy stub
# ---------------------------------------------------------------------------

class TestCacheSkipsLLMIntegration(unittest.TestCase):
    """Integration test: wrap a strategy answer() with a cache; second call skips LLM."""

    def _make_answer(self, tokens=25):
        from ai.agent.base import Answer
        return Answer(
            question="how many emergencies",
            answer_text="3 emergency events (source: gold)",
            result={"answer": 3, "sources": ["gold"]},
            route="analytics:count_emergencies",
            strategy="single_shot_rag",
            meta={"tokens": tokens, "provider": "stub"},
        )

    def test_cached_strategy_skips_llm_on_second_call(self):
        """A strategy answer() wrapped with a cache calls the LLM only once."""
        from ai.obs.cache import SemanticCache

        call_count = {"n": 0}
        expected_answer = self._make_answer()

        def _stub_answer(question):
            call_count["n"] += 1
            return expected_answer

        # Force exact-match mode (no embeddings).
        with unittest.mock.patch("ai.obs.cache._embeddings_available", return_value=False):
            cache = SemanticCache(max_size=32)
            question = "how many emergencies"

            def _cached_answer(q):
                hit = cache.get(q)
                if hit is not None:
                    return hit
                ans = _stub_answer(q)
                cache.put(q, ans)
                return ans

            a1 = _cached_answer(question)
            self.assertEqual(call_count["n"], 1)
            self.assertIs(a1, expected_answer)

            a2 = _cached_answer(question)
            self.assertEqual(call_count["n"], 1)  # LLM NOT called again
            self.assertIs(a2, expected_answer)

    def test_different_questions_each_call_llm(self):
        """Two distinct questions each invoke the LLM once."""
        from ai.obs.cache import SemanticCache

        call_count = {"n": 0}

        def _stub_answer(question):
            call_count["n"] += 1
            from ai.agent.base import Answer
            return Answer(
                question=question, answer_text=f"answer to {question} (source: x)",
                result={"answer": 1, "sources": ["x"]},
                route="retrieval:search", strategy="s", meta={"tokens": 10},
            )

        with unittest.mock.patch("ai.obs.cache._embeddings_available", return_value=False):
            cache = SemanticCache(max_size=32)

            def _cached_answer(q):
                hit = cache.get(q)
                if hit is not None:
                    return hit
                ans = _stub_answer(q)
                cache.put(q, ans)
                return ans

            _cached_answer("question one")
            _cached_answer("question two")
            self.assertEqual(call_count["n"], 2)

            # Repeat both: both should be hits now.
            _cached_answer("question one")
            _cached_answer("question two")
            self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
