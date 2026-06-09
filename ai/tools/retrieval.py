#!/usr/bin/env python3
"""Retrieval tool over a small unstructured corpus.

The corpus holds METAR strings, aviation reference notes, and (synthetic)
incident-report snippets. W3.1 uses offline keyword retrieval (token overlap) —
no embeddings, no network — so the eval is deterministic. A semantic/embedding
backend can later replace the scorer behind this same interface.

``search`` returns a dict with:
  - ``answer``  : concatenated snippets of the top hits (text for downstream use)
  - ``sources`` : the retrieved document ids (for citation checks)
  - ``results`` : per-hit id / score / title / snippet
"""

import json
import re
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class RetrievalTool:
    """Keyword retrieval over a JSON corpus of documents."""

    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path)
        with open(self.corpus_path) as f:
            self.docs = json.load(f)
        self._by_id = {d["id"]: d for d in self.docs}
        self._index = {
            d["id"]: set(_tokens(f"{d.get('title', '')} {d.get('text', '')}"))
            for d in self.docs
        }

    def search(self, query: str, top_k: int = 3) -> dict:
        q = set(_tokens(query))
        scored = []
        for d in self.docs:
            overlap = len(q & self._index[d["id"]])
            if overlap:
                scored.append((overlap / max(1, len(q)), d))
        # Highest overlap first; tie-break on id for determinism.
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        results = [
            {"id": d["id"], "score": round(s, 3),
             "title": d.get("title"), "snippet": d.get("text", "")[:200]}
            for s, d in scored[:top_k]
        ]
        return {
            "answer": " ".join(r["snippet"] for r in results),
            "sources": [r["id"] for r in results],
            "results": results,
        }

    def get(self, doc_id: str):
        return self._by_id.get(doc_id)
