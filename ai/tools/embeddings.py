#!/usr/bin/env python3
"""Embeddings client for OpenAI text-embedding-3-small (1536-dim).

Design rules (mirrors ai/agent/ollama_llm.py):
- Availability-gated: importing this module NEVER opens a socket and NEVER
  requires the ``openai`` package to be installed.
- Uses stdlib ``urllib`` only for the HTTP call — no heavy SDK dependency.
- ``is_available()`` returns True only when OPENAI_API_KEY is set in the env.
- ``embed(texts)`` calls the OpenAI embeddings API and returns a list of 1536-
  dimensional float vectors, one per input text.
- ``index_corpus(docs, corpus_path)`` embeds each corpus document and upserts
  the result into the ``ai_embeddings`` pgvector table, gated on pg.healthcheck().

The embedding model is fixed at ``text-embedding-3-small`` (1536 dims) to match
the ``ai_embeddings`` DDL in infra/db/001_init.sql.
"""

import json
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 1536
_API_URL = "https://api.openai.com/v1/embeddings"
_BATCH_SIZE = 64  # OpenAI allows up to 2048 inputs per request; 64 is safe


def is_available() -> bool:
    """Return True only when OPENAI_API_KEY is present in the environment.

    Never raises. Checks only the env var — no network call.
    """
    return bool(os.environ.get("OPENAI_API_KEY"))


def embed(texts: list[str], timeout: int = 30) -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API.

    Uses stdlib urllib — no openai package required.

    Args:
        texts: Non-empty list of strings to embed.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of 1536-dimensional float vectors, one per input text.
        Preserves the input order (OpenAI guarantees index ordering).

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        urllib.error.URLError: On network / HTTP failure.
        ValueError: If the API response is malformed or dimensions mismatch.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it before calling embed(), e.g.:\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    all_vectors: list[list[float]] = []

    # Process in batches to stay within API limits.
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]

        payload = {
            "model": _MODEL,
            "input": batch,
            "dimensions": _DIMENSIONS,
            "encoding_format": "float",
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _API_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        logger.debug("Calling OpenAI embeddings API for %d texts", len(batch))
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)

        # API returns: {"data": [{"index": N, "embedding": [...], ...}, ...], ...}
        embedding_items = data.get("data", [])
        if len(embedding_items) != len(batch):
            raise ValueError(
                f"OpenAI returned {len(embedding_items)} embeddings "
                f"for {len(batch)} inputs"
            )

        # Sort by index to guarantee correct order regardless of API response order.
        embedding_items.sort(key=lambda x: x["index"])
        for item in embedding_items:
            vec = item["embedding"]
            if len(vec) != _DIMENSIONS:
                raise ValueError(
                    f"Expected {_DIMENSIONS}-dim embedding, got {len(vec)}-dim"
                )
            all_vectors.append(vec)

    return all_vectors


def index_corpus(docs: list[dict], timeout: int = 60) -> int:
    """Embed all corpus documents and upsert them into ai_embeddings.

    Availability-gated on both OPENAI_API_KEY and pg.healthcheck().
    Each document is treated as a single chunk (chunk_id=0) — chunking
    can be layered on top in a future iteration without schema changes.

    Args:
        docs: List of corpus document dicts, each with keys
              ``id``, ``title``, and ``text``.
        timeout: Per-batch embed HTTP timeout in seconds.

    Returns:
        Number of documents indexed.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        RuntimeError: If the Postgres database is unreachable.
    """
    # Late import — shared.store.pg is systems-owned and psycopg may not be
    # installed in all environments. We gate on healthcheck before get_conn.
    from shared.store import pg  # type: ignore[import]

    if not is_available():
        raise EnvironmentError("OPENAI_API_KEY is not set; cannot index corpus.")
    if not pg.healthcheck():
        raise RuntimeError(
            "Postgres healthcheck failed; cannot upsert into ai_embeddings."
        )

    texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in docs]
    doc_ids = [d["id"] for d in docs]
    sources = [d.get("type", "unknown") for d in docs]
    contents = [d.get("text", "") for d in docs]

    logger.info("Embedding %d corpus documents with %s ...", len(docs), _MODEL)
    vectors = embed(texts, timeout=timeout)
    logger.info("Received %d embeddings. Upserting into ai_embeddings ...", len(vectors))

    upserted = 0
    with pg.get_conn() as conn:
        for doc_id, source, content, vec in zip(doc_ids, sources, contents, vectors):
            # Delete any existing rows for this doc_id (idempotent full re-index).
            conn.execute(
                "DELETE FROM ai_embeddings WHERE doc_id = %s AND chunk_id = 0",
                (doc_id,),
            )
            # pgvector accepts Python lists for the vector column when psycopg3 +
            # pgvector adapter registers the type; pass as a stringified literal
            # as a safe fallback that works without the adapter installed.
            vec_literal = "[" + ",".join(str(v) for v in vec) + "]"
            conn.execute(
                """
                INSERT INTO ai_embeddings
                    (doc_id, chunk_id, source, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (doc_id, 0, source, content, vec_literal),
            )
            upserted += 1
        conn.commit()

    logger.info("Indexed %d documents into ai_embeddings.", upserted)
    return upserted
