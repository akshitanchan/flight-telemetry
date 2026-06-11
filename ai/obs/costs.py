#!/usr/bin/env python3
"""ai/obs/costs.py
------------------
Token-to-cost calculator and Prometheus counters for the AI agent layer.

Naming convention follows C6 (shared/obs/telemetry.py): ``<service>_<noun>_<unit>``
  - ``ai_llm_tokens_total``   — Counter, labelled by model and strategy
  - ``ai_llm_cost_usd_total`` — Counter (float64), labelled by model and strategy

Price table (USD per 1 000 tokens, input+output blended rate):
  - openai:gpt-4o-mini   $0.000150/1K tokens  (input $0.15/1M, output $0.60/1M,
                          blended ~$0.15/1M for routing-heavy workloads)
  - openai:gpt-4o        $0.005000/1K tokens  (input $5/1M, output $15/1M, blended)
  - text-embedding-3-small $0.000020/1K tokens ($0.02/1M)
  - ollama:*             $0.000000/1K tokens  (local, no cost)
  - unknown/other        $0.000000/1K tokens  (conservative: do not invent costs)

Availability-gated
------------------
prometheus_client may not be installed in all environments (e.g. offline CI).
The module degrades gracefully:
  - When prometheus_client IS available the real Counter objects are created
    once (double-registration guard) and ``record`` updates them.
  - When prometheus_client is NOT available ``record`` is a no-op.

No socket is opened at import time in either case.

Public API
----------
``cost_usd(model: str, tokens: int) -> float``
    Return the estimated USD cost for ``tokens`` tokens on ``model``.

``record(model: str, tokens: int, strategy: str = "") -> float``
    Record tokens + estimated cost to Prometheus counters (no-op if
    prometheus_client absent) and return the USD cost.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price table  (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

# Keys are normalised model strings (lowercase, leading "openai:" stripped for
# matching).  The lookup normalises the caller-supplied model string before
# matching so "openai:gpt-4o-mini", "gpt-4o-mini", "GPT-4O-MINI" all match.

_PRICE_PER_1K: dict[str, float] = {
    # OpenAI chat
    "gpt-4o-mini":             0.000150,   # blended ~$0.15/1M
    "gpt-4o":                  0.005000,   # blended ~$5/1M
    "gpt-4-turbo":             0.010000,
    "gpt-3.5-turbo":           0.000500,
    # OpenAI embeddings
    "text-embedding-3-small":  0.000020,   # $0.02/1M
    "text-embedding-3-large":  0.000130,   # $0.13/1M
    "text-embedding-ada-002":  0.000100,
    # Ollama (local — always $0)
    "ollama":                  0.000000,
}

_DEFAULT_PRICE: float = 0.000000  # unknown models — conservative zero


def _normalise_model(model: str) -> str:
    """Lower-case and strip common provider prefixes for table lookup."""
    m = model.lower().strip()
    for prefix in ("openai:", "ollama:", "anthropic:", "cohere:"):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    return m


def cost_usd(model: str, tokens: int) -> float:
    """Return the estimated USD cost for ``tokens`` tokens on ``model``.

    Args:
        model:  Model identifier string (provider-prefixed or bare).
                Examples: "gpt-4o-mini", "openai:gpt-4o-mini",
                          "ollama:llama3.2", "text-embedding-3-small".
        tokens: Total token count (prompt + completion).

    Returns:
        Estimated cost in USD (float).  Returns 0.0 for unknown models so
        callers never get inflated phantom costs.
    """
    if tokens <= 0:
        return 0.0

    norm = _normalise_model(model)

    # Exact match first.
    price = _PRICE_PER_1K.get(norm)

    # Prefix match for Ollama variants (e.g. "llama3.2:8b" → "ollama" → $0).
    if price is None:
        # Check original model for "ollama:" prefix before normalisation.
        if model.lower().startswith("ollama:"):
            price = _PRICE_PER_1K["ollama"]

    if price is None:
        # Partial-key scan (handles "gpt-4o-mini-2024-07-18" → "gpt-4o-mini").
        for key in _PRICE_PER_1K:
            if norm.startswith(key) or key.startswith(norm):
                price = _PRICE_PER_1K[key]
                break

    if price is None:
        price = _DEFAULT_PRICE

    return (tokens / 1000.0) * price


# ---------------------------------------------------------------------------
# Prometheus metric singletons (created once; guarded against double-register)
# ---------------------------------------------------------------------------

_metrics_lock = threading.Lock()
_metrics_ready: bool = False
_tokens_counter = None
_cost_counter = None


def _init_metrics() -> None:
    """Create Prometheus counters once; no-op if already created or lib absent."""
    global _metrics_ready, _tokens_counter, _cost_counter

    with _metrics_lock:
        if _metrics_ready:
            return

        try:
            from prometheus_client import Counter, REGISTRY

            # Guard against double-registration (e.g. test re-imports).
            existing_names = {m.name for m in REGISTRY._names_to_collectors.values()}  # type: ignore[attr-defined]

            if "ai_llm_tokens_total" not in existing_names:
                _tokens_counter = Counter(
                    "ai_llm_tokens_total",
                    "Total LLM tokens consumed by the AI agent layer.",
                    labelnames=("model", "strategy"),
                )
            else:
                # Retrieve the already-registered collector.
                _tokens_counter = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                    "ai_llm_tokens_total"
                )

            if "ai_llm_cost_usd_total" not in existing_names:
                _cost_counter = Counter(
                    "ai_llm_cost_usd_total",
                    "Total estimated USD cost of LLM calls by the AI agent layer.",
                    labelnames=("model", "strategy"),
                )
            else:
                _cost_counter = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                    "ai_llm_cost_usd_total"
                )

            _metrics_ready = True
            logger.debug("ai.obs.costs: Prometheus counters initialised.")

        except ImportError:
            # prometheus_client not installed — metrics silently disabled.
            logger.debug(
                "ai.obs.costs: prometheus_client not available; metrics disabled."
            )
            _metrics_ready = True  # Don't retry on every call.
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai.obs.costs: failed to initialise metrics: %s", exc)
            _metrics_ready = True


# ---------------------------------------------------------------------------
# Public API: record
# ---------------------------------------------------------------------------

def record(model: str, tokens: int, strategy: str = "") -> float:
    """Record token usage + cost to Prometheus and return the USD cost.

    Safe to call without prometheus_client installed: the cost is still
    calculated and returned; only the Prometheus increment is skipped.

    Args:
        model:    Model identifier (e.g. "gpt-4o-mini", "ollama:llama3.2").
        tokens:   Total token count (prompt + completion).
        strategy: Strategy name for the Prometheus label (e.g. "react").

    Returns:
        Estimated cost in USD.
    """
    usd = cost_usd(model, tokens)

    # Lazy metric init (first call only).
    if not _metrics_ready:
        _init_metrics()

    if _tokens_counter is not None and tokens > 0:
        try:
            _tokens_counter.labels(model=model, strategy=strategy).inc(tokens)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ai.obs.costs: tokens counter inc failed: %s", exc)

    if _cost_counter is not None and usd > 0.0:
        try:
            _cost_counter.labels(model=model, strategy=strategy).inc(usd)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ai.obs.costs: cost counter inc failed: %s", exc)

    return usd
