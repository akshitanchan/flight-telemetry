#!/usr/bin/env python3
"""ai/obs/tracing.py
--------------------
Per-call OTel tracing wrapper for the AI agent layer.

Reuses the C6 observability conventions from shared/obs/telemetry.py:
  - ``get_tracer(name)`` for instrumentation scope
  - ``service.name`` resource attribute (value "ai-agent")
  - Availability-gated: no OTel SDK → clean no-op; no collector → spans
    silently dropped (same pattern as ml/serve.py / shared/obs/telemetry.py)

Design
------
``trace_llm_call(fn, *, span_name, strategy, route, **extra_attrs)``
    Functional wrapper: calls ``fn()`` inside an OTel span.  The span records:
      - ``ai.strategy``    — strategy name (single_shot_rag / react / plan_execute)
      - ``ai.route``       — tool route (e.g. analytics:count_emergencies)
      - ``ai.tokens``      — token count returned by the LLM callable
      - ``ai.provider``    — model/provider string when available
      - any extra keyword attrs passed by the caller

``traced_llm``
    A decorator factory that wraps an LLM callable ``(messages) -> (str, int)``
    in a span per call.  Designed to wrap the injectable ``llm`` argument of
    any AnswerStrategy without modifying the strategy files:

        from ai.obs.tracing import traced_llm
        strategy = SingleShotRAGStrategy(analytics, retrieval,
                                         llm=traced_llm(my_llm, strategy="single_shot_rag"))

``traced_answer``
    A decorator that wraps an ``AnswerStrategy.answer`` method in a single span
    covering the full strategy invocation, recording tokens from ``Answer.meta``.

Offline discipline
------------------
- Importing this module NEVER opens a network connection.
- All OTel calls are wrapped in try/except; failures are logged at DEBUG and
  ignored so the underlying LLM call always propagates.
- When opentelemetry is not installed the module falls back to no-ops without
  raising ImportError at import time.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTel availability check (lazy, never raises at import time)
# ---------------------------------------------------------------------------

def _otel_available() -> bool:
    """Return True only when the opentelemetry SDK is installed.

    Never raises.  No network is opened.
    """
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


def _get_tracer():
    """Return an OTel tracer from shared/obs/telemetry.py if available.

    Falls back to a no-op stub when the SDK is absent or setup_tracing has
    not been called.  The no-op context manager never raises.
    """
    if not _otel_available():
        return _NoOpTracer()
    try:
        from shared.obs.telemetry import get_tracer as _c6_get_tracer
        return _c6_get_tracer("ai.obs.tracing")
    except Exception:  # noqa: BLE001
        return _NoOpTracer()


# ---------------------------------------------------------------------------
# No-op tracer / span stubs (used when OTel SDK is absent)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Minimal context-manager span that accepts set_attribute silently."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def record_exception(self, exc: Exception) -> None:  # noqa: ARG002
        pass

    def set_status(self, *args, **kwargs) -> None:
        pass


class _NoOpTracer:
    """Minimal tracer stub whose start_as_current_span returns a no-op span."""

    def start_as_current_span(self, name: str, **kwargs):  # noqa: ARG002
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Core tracing helper
# ---------------------------------------------------------------------------

def _build_span_context(tracer, span_name: str, attrs: dict):
    """Return (context_manager, span) pair; always returns _NoOpSpan on error."""
    try:
        ctx = tracer.start_as_current_span(span_name)
        return ctx
    except Exception as exc:  # noqa: BLE001
        logger.debug("ai.obs.tracing: failed to start span %r: %s", span_name, exc)
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Public API: trace_llm_call
# ---------------------------------------------------------------------------

def trace_llm_call(
    fn: Callable,
    *,
    span_name: str = "ai.llm.call",
    strategy: str = "",
    route: str = "",
    **extra_attrs: Any,
) -> Any:
    """Call ``fn()`` inside an OTel span; return the result unchanged.

    ``fn`` must be a zero-argument callable (use ``functools.partial`` or a
    lambda to capture arguments before calling this helper).

    The span records:
      - ``ai.strategy``: strategy name (e.g. "single_shot_rag")
      - ``ai.route``:    resolved tool route (e.g. "analytics:count_emergencies")
      - ``ai.tokens``:   token count from the LLM response (int)
      - any additional kwargs as span attributes

    All OTel errors are silently swallowed; ``fn()`` always executes.
    """
    tracer = _get_tracer()
    span_ctx = _build_span_context(tracer, span_name, {})

    with span_ctx as span:
        try:
            span.set_attribute("ai.strategy", strategy)
            span.set_attribute("ai.route", route)
            for k, v in extra_attrs.items():
                span.set_attribute(k, str(v))
        except Exception as exc:  # noqa: BLE001
            logger.debug("ai.obs.tracing: set_attribute failed: %s", exc)

        try:
            result = fn()
        except Exception as exc:
            try:
                span.record_exception(exc)
            except Exception:  # noqa: BLE001
                pass
            raise

        # If the result is a (content, tokens) 2-tuple (raw LLM callable return)
        # surface the token count on the span.
        if isinstance(result, tuple) and len(result) == 2:
            try:
                _, tokens = result
                if isinstance(tokens, int):
                    span.set_attribute("ai.tokens", tokens)
            except Exception:  # noqa: BLE001
                pass

        return result


# ---------------------------------------------------------------------------
# Public API: traced_llm decorator factory
# ---------------------------------------------------------------------------

def traced_llm(
    llm: Callable,
    *,
    strategy: str = "",
    provider: str = "",
) -> Callable:
    """Wrap an LLM callable in a per-call OTel span.

    The wrapped callable has the same signature:
        ``(messages: list[dict]) -> (content: str, tokens: int)``

    Span name: ``ai.llm.call``
    Span attributes: ``ai.strategy``, ``ai.provider``, ``ai.tokens``

    Usage (inject at strategy construction time, not inside the strategy):

        from ai.obs.tracing import traced_llm
        from ai.agent.single_shot_rag import SingleShotRAGStrategy

        instrumented_llm = traced_llm(raw_llm, strategy="single_shot_rag",
                                      provider="openai:gpt-4o-mini")
        strat = SingleShotRAGStrategy(analytics, retrieval, llm=instrumented_llm)

    Args:
        llm:       the LLM callable to wrap.
        strategy:  strategy name for span attributes.
        provider:  provider string (e.g. "openai:gpt-4o-mini") for span attrs.

    Returns:
        A new callable with the same interface as ``llm``.
    """

    @functools.wraps(llm)
    def _wrapper(messages):
        return trace_llm_call(
            lambda: llm(messages),
            span_name="ai.llm.call",
            strategy=strategy,
            provider=provider,
        )

    return _wrapper


# ---------------------------------------------------------------------------
# Public API: traced_answer decorator
# ---------------------------------------------------------------------------

def traced_answer(method: Callable | None = None, *, strategy_name: str = "") -> Callable:
    """Decorator that wraps an AnswerStrategy.answer() method in an OTel span.

    The span covers the entire strategy invocation.  After the call completes,
    tokens and route are read from the returned Answer object and set on the
    span.

    Usage as a decorator (apply at callsite, not inside agent files):

        from ai.obs.tracing import traced_answer
        from ai.agent.react import ReActStrategy

        class InstrumentedReAct(ReActStrategy):
            answer = traced_answer(ReActStrategy.answer)

    Or wrap an instance's bound method:

        strat = ReActStrategy(analytics, retrieval, llm=my_llm)
        strat.answer = traced_answer(strat.answer, strategy_name="react")

    Args:
        method:        the ``answer(self, question)`` bound/unbound method.
        strategy_name: optional override for the span's ``ai.strategy`` attr.

    Returns:
        Wrapped callable.
    """
    def _decorator(fn: Callable) -> Callable:
        sname = strategy_name or getattr(fn, "__qualname__", "") or "unknown"

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            tracer = _get_tracer()
            span_ctx = _build_span_context(tracer, "ai.strategy.answer", {})

            with span_ctx as span:
                try:
                    span.set_attribute("ai.strategy", sname)
                except Exception:  # noqa: BLE001
                    pass

                try:
                    answer = fn(*args, **kwargs)
                except Exception as exc:
                    try:
                        span.record_exception(exc)
                    except Exception:  # noqa: BLE001
                        pass
                    raise

                # Enrich span from the returned Answer object.
                try:
                    tokens = answer.meta.get("tokens", 0)
                    route = getattr(answer, "route", "")
                    provider = answer.meta.get("provider", "")
                    span.set_attribute("ai.tokens", int(tokens))
                    span.set_attribute("ai.route", str(route))
                    span.set_attribute("ai.provider", str(provider))
                except Exception:  # noqa: BLE001
                    pass

                return answer

        return _wrapper

    # Support both @traced_answer and @traced_answer(strategy_name="x")
    if method is not None:
        return _decorator(method)
    return _decorator
