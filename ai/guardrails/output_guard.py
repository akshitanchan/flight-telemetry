#!/usr/bin/env python3
"""Output guardrail for the flight-telemetry AI agent.

Inspects an Answer AFTER the inner AnswerStrategy has produced it and before
it is returned to the caller.  Rejects (replaces with a safe refusal) if:

1. Sources are empty    — the answer is ungrounded; every valid answer MUST cite
                          at least one corpus doc id or gold table name.
2. Fabricated source    — any source id that is NOT in the valid corpus/gold
                          set; this catches source-spoofing injections where an
                          attacker coerces the model to cite a doc that doesn't
                          exist.
3. Prompt-leak content  — the answer_text contains phrases that indicate the
                          system prompt was leaked ("system prompt", "api key",
                          "confidential", "instructions given", etc.).
4. Unsafe content       — generic unsafe-content markers (extreme language, self-
                          harm, etc.) — a minimal list for this domain; the system
                          is an aviation-operations agent so these are low
                          probability but present here for defence-in-depth.

Valid source IDs are loaded from the corpus JSON (all ``"id"`` fields) plus the
gold table logical names used by the AnalyticsTool (``gold_emergency_events``,
``gold_airport_congestion``, ``gold_routing_stats``, ``gold_sector_load``).

Design
------
All checks are deterministic string/set operations — no LLM involved.
The guardrail is stateless and thread-safe.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Gold table logical names (must match AnalyticsTool's source strings)
# ---------------------------------------------------------------------------

_GOLD_SOURCES = {
    "gold_emergency_events",
    "gold_airport_congestion",
    "gold_routing_stats",
    "gold_sector_load",
}


# ---------------------------------------------------------------------------
# Prompt-leak detection patterns (applied to answer_text)
# ---------------------------------------------------------------------------

_LEAK_PATTERNS: list[list[str]] = [
    ["system prompt"],
    ["api key"],
    ["api keys"],
    ["confidential instructions"],
    ["instructions given at the start"],
    ["instructions you were given"],
    ["injection_success"],
    ["hacked"],
    ["secret token"],
    ["internal configuration"],
]

_UNSAFE_PATTERNS: list[list[str]] = [
    ["ignore", "all restrictions"],
    ["dan mode"],
    ["jailbreak"],
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OutputGuardrailResult:
    """Result of running the output guardrail on an Answer.

    Attributes:
        rejected:   True if the answer must be replaced with a safe refusal.
        reason:     Human-readable explanation.
        check:      Which check triggered rejection ('empty_sources',
                    'fabricated_source', 'prompt_leak', 'unsafe_content').
        bad_sources: Source ids that are not in the valid set (may be empty).
    """
    rejected: bool
    reason: str = ""
    check: str = ""
    bad_sources: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class OutputGuardrail:
    """Deterministic output guardrail.

    Parameters
    ----------
    corpus_path:
        Path to ``corpus.json``.  All ``"id"`` values in this file are treated
        as valid source ids.  If the path is None or missing, only the gold
        table names are used as valid sources.
    extra_valid_sources:
        Optional additional source ids to treat as valid (e.g. from external
        data pipelines).
    """

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        extra_valid_sources: set[str] | None = None,
    ):
        self._valid_sources: set[str] = set(_GOLD_SOURCES)
        if extra_valid_sources:
            self._valid_sources.update(extra_valid_sources)

        if corpus_path is not None:
            cp = Path(corpus_path)
            if cp.exists():
                try:
                    with open(cp) as f:
                        docs = json.load(f)
                    for doc in docs:
                        doc_id = doc.get("id")
                        if doc_id:
                            self._valid_sources.add(doc_id)
                except (json.JSONDecodeError, OSError):
                    pass  # Corpus unreadable — fall back to gold names only.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def valid_sources(self) -> frozenset[str]:
        """Immutable view of the valid source-id set (for inspection/testing)."""
        return frozenset(self._valid_sources)

    def check(self, answer) -> OutputGuardrailResult:
        """Inspect an :class:`~ai.agent.base.Answer` for guardrail violations.

        Parameters
        ----------
        answer:
            Any object with a ``sources`` property (list[str]) and an
            ``answer_text`` attribute (str).  Compatible with
            :class:`~ai.agent.base.Answer`.

        Returns an :class:`OutputGuardrailResult`.  If ``.rejected`` is True
        the caller must replace the answer with a safe refusal.
        """
        sources = list(answer.sources) if hasattr(answer, "sources") else []
        answer_text = getattr(answer, "answer_text", "") or ""

        # 1. Empty-source check
        if not sources:
            return OutputGuardrailResult(
                rejected=True,
                reason="Answer has no sources — grounding requirement not met.",
                check="empty_sources",
            )

        # 2. Fabricated-source check
        bad = [s for s in sources if s not in self._valid_sources]
        if bad:
            return OutputGuardrailResult(
                rejected=True,
                reason=(
                    f"Answer cites source(s) not in the valid corpus: {bad}. "
                    "This may indicate a source-spoofing injection."
                ),
                check="fabricated_source",
                bad_sources=bad,
            )

        # 3. Prompt-leak check
        norm = answer_text.lower()
        for pattern in _LEAK_PATTERNS:
            if all(p in norm for p in pattern):
                return OutputGuardrailResult(
                    rejected=True,
                    reason=(
                        f"Answer contains potential prompt-leak content "
                        f"(matched: {pattern!r})."
                    ),
                    check="prompt_leak",
                )

        # 4. Unsafe-content check
        for pattern in _UNSAFE_PATTERNS:
            if all(p in norm for p in pattern):
                return OutputGuardrailResult(
                    rejected=True,
                    reason=(
                        f"Answer contains unsafe content "
                        f"(matched: {pattern!r})."
                    ),
                    check="unsafe_content",
                )

        return OutputGuardrailResult(rejected=False)

    def safe_refusal_result(self, reason: str) -> dict:
        """Minimal safe refusal result dict (used by GuardedStrategy)."""
        return {
            "answer": (
                "This answer cannot be served. "
                f"Reason: {reason}"
            ),
            "sources": [],
            "rows": [],
            "guardrail": "output",
            "rejected": True,
        }
