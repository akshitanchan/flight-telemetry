#!/usr/bin/env python3
"""ai/obs/cache.py
------------------
Semantic answer cache for the AI agent layer.

Design
------
The cache stores full Answer objects (or arbitrary result dicts) keyed by
query.  A cache hit skips the LLM call entirely, saving tokens and cost.

Two keying modes
~~~~~~~~~~~~~~~~
1. **Embedding mode** (online, preferred)
   - The query is embedded with ``ai.tools.embeddings.embed()`` (text-embedding-3-small).
   - Cosine similarity between the new query embedding and all stored embeddings
     is computed with pure NumPy (no DB required).
   - A hit is declared when similarity >= ``threshold`` (default 0.92).
   - Available only when ``ai.tools.embeddings.is_available()`` returns True
     (i.e. OPENAI_API_KEY is set) AND numpy is importable.

2. **Exact-match mode** (offline, fallback)
   - The normalised query string (lowercased, whitespace-collapsed) is used as
     the cache key directly.
   - Always works offline — no API key, no DB, no numpy required.
   - Used when embeddings are unavailable or when the embedding call fails.

The cache automatically downgrades to exact-match on any embedding failure so
it never crashes the answer path.

Size bounding
~~~~~~~~~~~~~
The cache is bounded to ``max_size`` entries (default 256).  When the bound is
reached the oldest entry (insertion order, via ``collections.OrderedDict``) is
evicted (FIFO).

Prometheus metric
~~~~~~~~~~~~~~~~~
``ai_cache_hits_total`` (Counter, labelled by ``keying``) — incremented on
every cache hit.  Availability-gated: no-op when prometheus_client absent.

Offline discipline
------------------
- Importing this module NEVER opens a network connection.
- All embedding / metric calls are wrapped in try/except.
- The module works correctly with only the Python stdlib.

Public API
----------
``SemanticCache(max_size=256, threshold=0.92, keying="auto")``
    The main cache class.  ``keying`` is ``"auto"`` (embeddings when
    available, else exact), ``"exact"`` (never embed) or ``"embedding"``
    (require embeddings; raises at construction time if unavailable).

    ``get(query) -> Any | None``
        Return the cached value for ``query``, or None on a miss.

    ``put(query, value) -> None``
        Store ``value`` under ``query``.  Evicts the oldest entry if full.

    ``stats() -> dict``
        Return ``{hits, misses, size, keying}`` for diagnostics.

    ``clear() -> None``
        Flush all entries (useful in tests).
"""

from __future__ import annotations

import collections
import logging
import re
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus cache-hit counter
# ---------------------------------------------------------------------------

_cache_metric_lock = threading.Lock()
_cache_metric_ready: bool = False
_cache_hits_counter = None


def _init_cache_metric() -> None:
    """Create the cache-hit Prometheus counter once; no-op if lib absent."""
    global _cache_metric_ready, _cache_hits_counter

    with _cache_metric_lock:
        if _cache_metric_ready:
            return
        try:
            from prometheus_client import Counter, REGISTRY

            existing_names = {m.name for m in REGISTRY._names_to_collectors.values()}  # type: ignore[attr-defined]
            if "ai_cache_hits_total" not in existing_names:
                _cache_hits_counter = Counter(
                    "ai_cache_hits_total",
                    "Total AI answer cache hits.",
                    labelnames=("keying",),
                )
            else:
                _cache_hits_counter = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                    "ai_cache_hits_total"
                )
            _cache_metric_ready = True
        except ImportError:
            _cache_metric_ready = True  # Not installed; don't retry.
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai.obs.cache: failed to init metric: %s", exc)
            _cache_metric_ready = True


def _inc_hit(keying: str) -> None:
    """Increment the cache-hit counter; no-op on any failure."""
    if not _cache_metric_ready:
        _init_cache_metric()
    if _cache_hits_counter is not None:
        try:
            _cache_hits_counter.labels(keying=keying).inc()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ai.obs.cache: hit counter inc failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalise(query: str) -> str:
    """Lowercase and collapse whitespace for exact-match keying."""
    return _WS.sub(" ", query.lower().strip())


def _embeddings_available() -> bool:
    """Return True only when OPENAI_API_KEY is set and numpy is importable."""
    try:
        from ai.tools.embeddings import is_available  # type: ignore[import]
        if not is_available():
            return False
    except ImportError:
        return False
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors.

    Uses numpy when available; falls back to pure Python for portability.
    Returns 0.0 on any error.
    """
    try:
        import numpy as np
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:  # noqa: BLE001
        # Pure-Python fallback (slow but correct for small vectors in tests).
        try:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:  # noqa: BLE001
            return 0.0


def _embed_query(query: str) -> Optional[list[float]]:
    """Embed a single query string; return None on any failure."""
    try:
        from ai.tools.embeddings import embed  # type: ignore[import]
        vectors = embed([query])
        if vectors and len(vectors) == 1:
            return vectors[0]
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("ai.obs.cache: embed failed (%s); falling back to exact match.", exc)
        return None


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Embedding-keyed answer cache with exact-match fallback.

    Parameters
    ----------
    max_size:
        Maximum number of entries to hold.  Oldest entries are evicted when
        the cache is full (FIFO order).
    threshold:
        Cosine-similarity threshold for a semantic cache hit (0.0 – 1.0).
        Queries whose embedding is this similar to a stored entry are
        considered equivalent.  Default 0.92 is intentionally high to avoid
        false positives on aviation queries.
    keying:
        ``"auto"`` (default): embeddings when available, else exact-match.
        ``"exact"``: always exact-match, never calls the embeddings module.
        ``"embedding"``: always embeds; raises ``RuntimeError`` at
        construction time if embeddings are unavailable.
    """

    def __init__(
        self, max_size: int = 256, threshold: float = 0.92, keying: str = "auto"
    ) -> None:
        if keying not in ("auto", "exact", "embedding"):
            raise ValueError(f"keying must be 'auto', 'exact', or 'embedding'; got {keying!r}")
        if keying == "embedding" and not _embeddings_available():
            raise RuntimeError(
                "SemanticCache(keying='embedding') requires OPENAI_API_KEY and numpy, "
                "neither of which is available in this environment"
            )

        self.max_size = max_size
        self.threshold = threshold
        self.keying = keying
        self._lock = threading.Lock()

        # Ordered dict for FIFO eviction.  Each value is a dict:
        #   {"value": Any, "embedding": list[float] | None, "key_norm": str}
        self._store: collections.OrderedDict[str, dict] = collections.OrderedDict()

        # Diagnostics.
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, query: str) -> Any:
        """Return cached value for ``query``, or None on a miss.

        Tries semantic (embedding) lookup first; falls back to exact-match
        when embeddings are unavailable.
        """
        norm = _normalise(query)

        with self._lock:
            if not self._store:
                self._misses += 1
                return None

            # --- Embedding path (preferred when available) ---
            if self._use_embeddings():
                try:
                    query_vec = _embed_query(query)
                    if query_vec is not None:
                        best_key, best_score = self._best_embedding_match(query_vec)
                        if best_key is not None and best_score >= self.threshold:
                            self._hits += 1
                            _inc_hit("embedding")
                            return self._store[best_key]["value"]
                        # No embedding hit — fall through to exact match.
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "ai.obs.cache: embedding lookup failed (%s); trying exact match.", exc
                    )

            # --- Exact-match path (offline fallback) ---
            if norm in self._store:
                self._hits += 1
                _inc_hit("exact")
                return self._store[norm]["value"]

        self._misses += 1
        return None

    def put(self, query: str, value: Any) -> None:
        """Store ``value`` under ``query``; evicts oldest entry if at capacity.

        Attempts to also embed the query for future semantic lookups.  If the
        embedding call fails the entry is stored with a None embedding and
        future lookups fall back to exact-match for this entry.
        """
        norm = _normalise(query)
        embedding: Optional[list[float]] = None

        if self._use_embeddings():
            try:
                embedding = _embed_query(query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ai.obs.cache: embed on put failed (%s); storing without vector.", exc)

        with self._lock:
            # If the same normalised key already exists, update in place and
            # move to end (most-recent) without counting as an eviction.
            if norm in self._store:
                self._store.move_to_end(norm)
                self._store[norm] = {"value": value, "embedding": embedding, "key_norm": norm}
                return

            # Evict oldest entry if at max capacity.
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)

            self._store[norm] = {"value": value, "embedding": embedding, "key_norm": norm}

    def stats(self) -> dict:
        """Return ``{hits, misses, size, keying}`` diagnostics."""
        with self._lock:
            keying = "embedding" if self._use_embeddings() else "exact"
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "keying": keying,
            }

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _use_embeddings(self) -> bool:
        """Return True when this instance should attempt the embedding path."""
        if self.keying == "exact":
            # exact exists so an eval that must replay offline takes the same cache path live and in replay
            return False
        if self.keying == "embedding":
            return True
        return _embeddings_available()

    def _best_embedding_match(
        self, query_vec: list[float]
    ) -> tuple[Optional[str], float]:
        """Find the stored entry with the highest cosine similarity.

        Returns (key, score); key is None when the store is empty or all
        stored entries have no embedding.

        Must be called with ``self._lock`` held.
        """
        best_key: Optional[str] = None
        best_score: float = -1.0

        for key, entry in self._store.items():
            stored_vec = entry.get("embedding")
            if stored_vec is None:
                continue
            score = _cosine_similarity(query_vec, stored_vec)
            if score > best_score:
                best_score = score
                best_key = key

        return best_key, best_score
