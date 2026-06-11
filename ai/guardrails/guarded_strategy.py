#!/usr/bin/env python3
"""GuardedStrategy — thin AnswerStrategy wrapper applying both guardrails.

Usage
-----
    from ai.guardrails import GuardedStrategy
    from ai.agent.react import ReActStrategy

    inner = ReActStrategy(analytics, retrieval, llm=stub_llm)
    guarded = GuardedStrategy(inner, corpus_path="ai/fixtures/corpus/corpus.json")
    answer = guarded.answer(question)

Control flow
------------
1. InputGuardrail.check(question)
     - If blocked: return safe refusal Answer immediately (inner never called).
2. inner.answer(question)  — delegate to the wrapped AnswerStrategy.
3. OutputGuardrail.check(answer)
     - If rejected: return safe refusal Answer.
4. Return the original (passing) Answer unchanged.

The wrapped strategy's name is preserved in Answer.strategy; the GuardedStrategy
appends "+guarded" so downstream code can distinguish guarded from un-guarded runs.

Composability
-------------
GuardedStrategy itself implements AnswerStrategy, so it can be wrapped again
(e.g., for logging or metrics) without any changes.
"""

from pathlib import Path

from ai.agent.base import Answer, AnswerStrategy
from ai.guardrails.input_guard import InputGuardrail
from ai.guardrails.output_guard import OutputGuardrail


class GuardedStrategy(AnswerStrategy):
    """Wraps any AnswerStrategy with input + output guardrails.

    Parameters
    ----------
    inner:
        The AnswerStrategy to guard.  Must implement ``answer(question) -> Answer``.
    corpus_path:
        Path to ``corpus.json`` for the output guardrail's valid-source set.
        If None, only gold table names are used as valid sources.
    extra_input_patterns:
        Optional extra injection patterns for the input guardrail.
    extra_valid_sources:
        Optional extra valid source ids for the output guardrail.
    """

    name = "guarded"

    def __init__(
        self,
        inner: AnswerStrategy,
        corpus_path: str | Path | None = None,
        extra_input_patterns: list[tuple[str, list[str]]] | None = None,
        extra_valid_sources: set[str] | None = None,
    ):
        self.inner = inner
        self._input_guard = InputGuardrail(extra_patterns=extra_input_patterns)
        self._output_guard = OutputGuardrail(
            corpus_path=corpus_path,
            extra_valid_sources=extra_valid_sources,
        )

    # ------------------------------------------------------------------
    # AnswerStrategy implementation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        """Run guardrails around the inner strategy.

        Returns a safe refusal Answer if either guardrail fires, or the inner
        Answer unchanged if both pass.
        """
        # Step 1: input guardrail
        in_result = self._input_guard.check(question)
        if in_result.blocked:
            return self._blocked_answer(question, in_result.reason, in_result.category)

        # Step 2: delegate to inner strategy
        inner_answer = self.inner.answer(question)

        # Step 3: output guardrail
        out_result = self._output_guard.check(inner_answer)
        if out_result.rejected:
            return self._rejected_answer(question, out_result.reason, out_result.check)

        # Step 4: pass-through
        return inner_answer

    # ------------------------------------------------------------------
    # Safe refusal helpers
    # ------------------------------------------------------------------

    def _blocked_answer(self, question: str, reason: str, category: str) -> Answer:
        """Safe refusal Answer for input-blocked questions."""
        return Answer(
            question=question,
            answer_text=(
                f"[GUARDRAIL BLOCKED] This request was blocked by the input guardrail. "
                f"Category: {category}. {reason}"
            ),
            result={
                "answer": None,
                "sources": [],
                "rows": [],
                "guardrail": "input",
                "blocked": True,
                "category": category,
                "reason": reason,
            },
            route="guardrail:input_block",
            strategy=f"{self.inner.name}+guarded",
            meta={
                "guardrail": "input",
                "blocked": True,
                "category": category,
            },
        )

    def _rejected_answer(self, question: str, reason: str, check: str) -> Answer:
        """Safe refusal Answer for output-rejected answers."""
        return Answer(
            question=question,
            answer_text=(
                f"[GUARDRAIL REJECTED] This answer was rejected by the output guardrail. "
                f"Check: {check}. {reason}"
            ),
            result={
                "answer": None,
                "sources": [],
                "rows": [],
                "guardrail": "output",
                "rejected": True,
                "check": check,
                "reason": reason,
            },
            route="guardrail:output_reject",
            strategy=f"{self.inner.name}+guarded",
            meta={
                "guardrail": "output",
                "rejected": True,
                "check": check,
            },
        )
