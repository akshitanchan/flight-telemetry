#!/usr/bin/env python3
"""Retrieval tool over a small unstructured corpus.

The corpus holds METAR strings, aviation reference notes, and (synthetic)
incident-report snippets.

Backend selection (chosen at search-time, not import-time):
  - VECTOR backend: used when BOTH ``embeddings.is_available()`` (i.e.
    OPENAI_API_KEY is set) AND ``pg.healthcheck()`` succeed. Embeds the query
    with text-embedding-3-small, runs a pgvector cosine nearest-neighbour
    (``<=>`` operator) query against ``ai_embeddings``, maps result rows back
    to corpus doc ids, and fills title/snippet from the in-memory corpus.
  - KEYWORD fallback: the original token-overlap scorer. Used by default and
    in CI (no OPENAI_API_KEY / no DB). Fully offline and deterministic.

``search`` returns a dict with:
  - ``answer``  : concatenated snippets of the top hits (text for downstream use)
  - ``sources`` : the retrieved document ids (for citation checks)
  - ``results`` : per-hit id / score / title / snippet
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class RetrievalTool:
    """Retrieval over a JSON corpus of documents.

    Uses a pgvector cosine-similarity backend when OpenAI + Postgres are
    available; falls back to keyword token-overlap scoring otherwise.
    """

    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path)
        with open(self.corpus_path) as f:
            self.docs = json.load(f)
        self._by_id = {d["id"]: d for d in self.docs}
        # Keyword index: pre-tokenised doc term sets.
        self._index = {
            d["id"]: set(_tokens(f"{d.get('title', '')} {d.get('text', '')}"))
            for d in self.docs
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> dict:
        """Return top-k matching documents for *query*.

        Attempts the vector backend first (when available); falls back to
        the keyword scorer on any failure or when prerequisites are absent.
        The returned dict always has the shape::

            {
                "answer":  "<concatenated snippets>",
                "sources": ["doc-id-1", "doc-id-2", ...],
                "results": [
                    {"id": "doc-id", "score": 0.92, "title": "...", "snippet": "..."},
                    ...
                ],
            }
        """
        if self._vector_backend_available():
            try:
                return self._vector_search(query, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Vector search failed (%s); falling back to keyword scorer.", exc
                )
        return self._keyword_search(query, top_k)

    def get(self, doc_id: str):
        return self._by_id.get(doc_id)

    # ------------------------------------------------------------------
    # Backend selector
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_backend_available() -> bool:
        """Return True only when BOTH embeddings and Postgres are ready.

        Evaluated lazily at search-time so that:
        - Import is always clean (no network, no DB).
        - CI (offline, no OPENAI_API_KEY) always uses the keyword fallback.
        - Live environments with both services get the semantic backend.
        """
        try:
            from ai.tools import embeddings  # type: ignore[import]
            from shared.store import pg  # type: ignore[import]
        except ImportError:
            return False

        if not embeddings.is_available():
            return False
        try:
            return pg.healthcheck()
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Vector backend
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int) -> dict:
        """Embed the query and run a pgvector cosine NN query.

        SQL used::

            SELECT doc_id, 1 - (embedding <=> %s::vector) AS score
            FROM   ai_embeddings
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s;

        ``<=>`` is pgvector's cosine-distance operator; smaller = more similar.
        We convert to a similarity score via ``1 - distance`` so higher scores
        rank better (consistent with the keyword scorer's 0–1 range).

        Results are mapped back to the in-memory corpus for title and snippet
        so the caller always receives the same ``{answer, sources, results}``
        shape regardless of which backend ran.
        """
        from ai.tools import embeddings  # type: ignore[import]
        from shared.store import pg  # type: ignore[import]

        # Embed the query (returns a list with one vector).
        query_vec = embeddings.embed([query])[0]
        vec_literal = "[" + ",".join(str(v) for v in query_vec) + "]"

        sql = """
            SELECT doc_id,
                   1 - (embedding <=> %s::vector) AS score
            FROM   ai_embeddings
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """

        results = []
        with pg.get_conn() as conn:
            rows = conn.execute(sql, (vec_literal, vec_literal, top_k)).fetchall()

        for doc_id, score in rows:
            doc = self._by_id.get(doc_id)
            if doc is None:
                # Row in DB references a doc_id not in the current corpus;
                # skip rather than fabricate a citation.
                logger.warning("ai_embeddings references unknown doc_id %r; skipping.", doc_id)
                continue
            results.append(
                {
                    "id": doc_id,
                    "score": round(float(score), 4),
                    "title": doc.get("title"),
                    "snippet": doc.get("text", "")[:200],
                }
            )

        return {
            "answer": " ".join(r["snippet"] for r in results),
            "sources": [r["id"] for r in results],
            "results": results,
        }

    # ------------------------------------------------------------------
    # Keyword fallback backend (original implementation)
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, top_k: int) -> dict:
        """Token-overlap keyword scorer. Deterministic, fully offline."""
        q = set(_tokens(query))
        scored = []
        for d in self.docs:
            overlap = len(q & self._index[d["id"]])
            if overlap:
                scored.append((overlap / max(1, len(q)), d))
        # Highest overlap first; tie-break on id for determinism.
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        results = [
            {
                "id": d["id"],
                "score": round(s, 3),
                "title": d.get("title"),
                "snippet": d.get("text", "")[:200],
            }
            for s, d in scored[:top_k]
        ]
        return {
            "answer": " ".join(r["snippet"] for r in results),
            "sources": [r["id"] for r in results],
            "results": results,
        }
