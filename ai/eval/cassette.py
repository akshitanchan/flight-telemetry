#!/usr/bin/env python3
"""Record-and-replay for the eval matrix.

A :class:`Cassette` maps a hash of a call's payload to its recorded result.
In record mode a miss calls through and stores the result; in replay mode a
miss raises :class:`CassetteMiss` so CI never falls back to a live call.
:func:`wrap_provider` and :func:`wrap_retrieval` give the strategies and the
retrieval tool a same-shaped stand-in backed by the cassette.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class CassetteMiss(KeyError):
    """Raised in replay mode when a payload has no recorded entry."""


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _key(payload):
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _describe(payload):
    described = payload.get("question") or payload.get("query")
    if described:
        return described
    messages = payload.get("messages")
    if messages:
        # last user turn only, so a provider-payload miss never dumps the system prompt
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content")
    return repr(payload)


class Cassette:
    """A JSON-backed store of payload-hash -> recorded result.

    ``record=True`` allows misses (the caller records the live result);
    ``record=False`` (replay) turns a miss into a :class:`CassetteMiss`.
    """

    def __init__(self, path, record=False):
        self.path = Path(path)
        self.record = record
        self._dirty = False
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.recorded_at = data.get("recorded_at")
            self.meta = data.get("meta", {})
            self.entries = data.get("entries", {})
        else:
            self.recorded_at = datetime.now(timezone.utc).date().isoformat()
            self.meta = {}
            self.entries = {}

    def fetch(self, payload):
        key = _key(payload)
        try:
            return self.entries[key]
        except KeyError:
            raise CassetteMiss(f"no cassette entry for {_describe(payload)!r}") from None

    def store(self, payload, value):
        self.entries[_key(payload)] = value
        self._dirty = True

    def save(self):
        if not self._dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "recorded_at": self.recorded_at,
                "meta": self.meta,
                "entries": self.entries,
            }, f, sort_keys=True, indent=2)
        self._dirty = False


class _WrappedProvider:
    def __init__(self, provider, cassette):
        self._provider = provider
        self._cassette = cassette
        self.name = provider.name
        self.model = provider.model
        self.label = provider.label
        self.calls = []

    def __call__(self, messages):
        payload = {"provider": self.name, "model": self.model, "messages": messages}
        try:
            entry = self._cassette.fetch(payload)
        except CassetteMiss:
            if not self._cassette.record:
                raise
            content, _ = self._provider(messages)
            entry = dict(self._provider.calls[-1], content=content)
            self._cassette.store(payload, entry)
            self._cassette.save()
        self.calls.append({k: entry[k] for k in ("input_tokens", "output_tokens", "latency_s")})
        return entry["content"], entry["input_tokens"] + entry["output_tokens"]


def wrap_provider(provider, cassette):
    """Wrap *provider* so calls are served from *cassette*.

    Returns a Provider-like callable (same ``name``/``model``/``label``/
    ``calls`` attributes) for injection as a strategy's ``llm=``.
    """
    return _WrappedProvider(provider, cassette)


class _WrappedRetrieval:
    def __init__(self, retrieval, cassette):
        self._retrieval = retrieval
        self._cassette = cassette

    # default mirrors RetrievalTool.search so keys match calls that omit top_k
    def search(self, query, top_k=3):
        payload = {"kind": "retrieval", "query": query, "top_k": top_k}
        try:
            return self._cassette.fetch(payload)
        except CassetteMiss:
            if not self._cassette.record:
                raise
            result = self._retrieval.search(query, top_k)
            self._cassette.store(payload, result)
            self._cassette.save()
            return result


def wrap_retrieval(retrieval, cassette):
    """Wrap *retrieval* so ``search`` calls are served from *cassette*."""
    return _WrappedRetrieval(retrieval, cassette)
