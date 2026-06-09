#!/usr/bin/env python3
"""Answer-strategy interface shared by all AI answer paths."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Answer:
    """A grounded answer produced by a strategy.

    Attributes:
        question:    the original natural-language question.
        answer_text: human-facing answer string, including a source citation.
        result:      the raw tool result dict ({"answer": ..., "sources": [...]}),
                     used by the deterministic eval checks.
        route:       which tool/operation handled it, e.g. "analytics:count_emergencies".
        strategy:    name of the strategy that produced the answer.
        meta:        strategy-specific extras (e.g. LLM token count, model) used as
                     a cost proxy in the strategy comparison.
    """
    question: str
    answer_text: str
    result: dict
    route: str
    strategy: str
    meta: dict = field(default_factory=dict)

    @property
    def sources(self) -> list:
        return self.result.get("sources", [])


class AnswerStrategy(ABC):
    """Base class for answer strategies. Subclasses implement ``answer``."""

    name: str = "base"

    @abstractmethod
    def answer(self, question: str) -> Answer:
        """Route ``question`` to a tool and return a grounded :class:`Answer`."""
        ...
