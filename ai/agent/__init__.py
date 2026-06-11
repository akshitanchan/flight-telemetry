"""Answer strategies for the AI layer.

An ``AnswerStrategy`` turns a natural-language question into a grounded answer by
routing it to the analytics/retrieval tools and formatting the result with source
citations. W3.2 ships the deterministic rule-based baseline; W3.3 adds further
strategies behind the same interface for comparison. W4.4 (ai-04) adds three
LLM-backed architectures: single-shot RAG, ReAct, and plan-execute.
"""

from ai.agent.single_shot_rag import SingleShotRAGStrategy
from ai.agent.react import ReActStrategy
from ai.agent.plan_execute import PlanExecuteStrategy

__all__ = [
    "SingleShotRAGStrategy",
    "ReActStrategy",
    "PlanExecuteStrategy",
]
