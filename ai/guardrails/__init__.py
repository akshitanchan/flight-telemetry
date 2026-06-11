"""ai.guardrails — input + output guardrail layer for the flight-telemetry AI agent.

Public surface
--------------
  InputGuardrail   — detect/block prompt-injection patterns before routing.
  OutputGuardrail  — reject answers whose sources are empty or fabricated.
  GuardedStrategy  — thin AnswerStrategy wrapper that applies both guardrails.

Usage (composing with any ai-04 architecture)::

    from ai.guardrails import GuardedStrategy
    from ai.agent.single_shot_rag import SingleShotRAGStrategy

    inner = SingleShotRAGStrategy(analytics, retrieval, llm=stub)
    guarded = GuardedStrategy(inner, corpus_path="ai/fixtures/corpus/corpus.json")
    answer = guarded.answer(question)

If the input guardrail fires, a safe refusal Answer is returned immediately
(the inner strategy is never called).  If the output guardrail fires, the
inner Answer is replaced with a safe refusal Answer.
"""

from ai.guardrails.input_guard import InputGuardrail, InputGuardrailResult
from ai.guardrails.output_guard import OutputGuardrail, OutputGuardrailResult
from ai.guardrails.guarded_strategy import GuardedStrategy

__all__ = [
    "InputGuardrail",
    "InputGuardrailResult",
    "OutputGuardrail",
    "OutputGuardrailResult",
    "GuardedStrategy",
]
