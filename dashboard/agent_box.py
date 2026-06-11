"""Agent wiring for the Ask AI dashboard view.

Builds the default deterministic answer strategy (offline, no LLM, no keys)
and exposes a single ``answer_question`` function.  This module intentionally
has NO ``import streamlit`` so it can be imported and tested in isolation
by pytest without a running Streamlit server.

Default strategy:
    ``RuleBasedStrategy`` backed by:
    - ``AnalyticsTool(gold_dir)`` — validates named ops over the four gold tables
    - ``RetrievalTool(corpus_path)`` — keyword search over the aviation corpus

    When an LLM strategy instance is passed in via ``strategy=``, it is used
    directly; the deterministic tools are still wired as the fallback path for
    callers that need offline guarantees.

Return type:
    ``AnswerResult`` (a plain dict-like object, no Streamlit dependency),
    containing:
      - answer_text (str)
      - sources     (list[str])
      - route       (str)
      - strategy    (str)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ai.agent.base import Answer, AnswerStrategy
from ai.agent.rule_based import RuleBasedStrategy
from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool

# ---------------------------------------------------------------------------
# Default path resolution — mirrors dashboard/data.py and ai/tools conventions
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GOLD_DIR = _PROJECT_ROOT / "data" / "processed"
_DEFAULT_CORPUS_PATH = _PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """Serialisable view of an :class:`~ai.agent.base.Answer`.

    Carries only the fields the dashboard and tests need — no Streamlit
    types, no internal tool state.
    """

    answer_text: str
    sources: list[str] = field(default_factory=list)
    route: str = ""
    strategy: str = ""

    @classmethod
    def from_answer(cls, answer: Answer) -> "AnswerResult":
        return cls(
            answer_text=answer.answer_text,
            sources=answer.sources,
            route=answer.route,
            strategy=answer.strategy,
        )


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_default_strategy(
    gold_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> RuleBasedStrategy:
    """Return a fully wired ``RuleBasedStrategy`` (offline, no LLM).

    Parameters
    ----------
    gold_dir:
        Directory with gold JSONL files.  Defaults to the ``GOLD_DIR`` env-var
        when set, otherwise ``<project_root>/data/processed/``.
    corpus_path:
        Path to ``corpus.json``.  Defaults to
        ``<project_root>/ai/fixtures/corpus/corpus.json``.
    """
    if gold_dir is None:
        gold_dir = os.environ.get("GOLD_DIR", str(_DEFAULT_GOLD_DIR))
    if corpus_path is None:
        corpus_path = str(_DEFAULT_CORPUS_PATH)

    analytics = AnalyticsTool(gold_dir)
    retrieval = RetrievalTool(corpus_path)
    return RuleBasedStrategy(analytics, retrieval)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    gold_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
    strategy: Optional[AnswerStrategy] = None,
) -> AnswerResult:
    """Answer *question* and return an :class:`AnswerResult`.

    This function is the single call-site used by the Streamlit view and by
    pytest.  It is fully offline when ``strategy`` is None (or when the
    supplied strategy is the deterministic default).

    Parameters
    ----------
    question:
        Natural-language question from the user.
    gold_dir:
        Override for the gold-tables directory (see :func:`build_default_strategy`).
    corpus_path:
        Override for the corpus file (see :func:`build_default_strategy`).
    strategy:
        Explicit :class:`~ai.agent.base.AnswerStrategy` instance to use.  When
        None, a fresh :class:`RuleBasedStrategy` is built from ``gold_dir`` and
        ``corpus_path``.

    Returns
    -------
    AnswerResult
        Plain dataclass with ``answer_text``, ``sources``, ``route``, and
        ``strategy`` fields.  Safe to serialise; contains no Streamlit objects.
    """
    if strategy is None:
        strategy = build_default_strategy(gold_dir, corpus_path)

    answer: Answer = strategy.answer(question)
    return AnswerResult.from_answer(answer)
