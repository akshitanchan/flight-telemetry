"""ai/obs — Observability layer for the AI agent.

Provides per-call OTel tracing, token/cost surfacing, and a semantic cache.
All components are availability-gated: they degrade to no-ops when the OTel
collector, OpenAI API key, or Prometheus client are absent.

Re-exports:
    tracing:  traced_llm, traced_answer, trace_llm_call
    costs:    cost_usd, record
    cache:    SemanticCache
"""

from ai.obs.tracing import trace_llm_call, traced_llm, traced_answer
from ai.obs.costs import cost_usd, record as record_cost
from ai.obs.cache import SemanticCache

__all__ = [
    "trace_llm_call",
    "traced_llm",
    "traced_answer",
    "cost_usd",
    "record_cost",
    "SemanticCache",
]
