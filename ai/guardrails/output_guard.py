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
5. Ungrounded number    — the answer_text asserts a number that does NOT appear
                          in the reference text built from the cited tool result
                          (result.answer + result.rows) and the cited corpus/gold
                          source content.  Catches ``ungrounded_coercion`` the
                          instant a real LLM fabricates a value.
                          Qualitative answers (no numbers after stripping
                          boilerplate) always pass.  Rejection fires only when
                          ALL extracted numbers are absent from the reference
                          (np_score == 0.0) and the tool result is not an empty
                          collection (empty list/dict answers assert "none" and
                          any numbers in them are query-parameter echoes, not
                          fabricated claims).  Reuses
                          :class:`~ai.eval.faithfulness.FaithfulnessChecker`
                          ``_numeric_presence`` and ``_build_reference_text``
                          so the semantics are grounded in the same corpus/gold
                          data that the advisory faithfulness scorer already uses
                          — zero false positives on the 104 golden questions.

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

# Reuse faithfulness module helpers for check #5.
# These are imported read-only; faithfulness.py is never modified.
from ai.eval.faithfulness import FaithfulnessChecker


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
                    'fabricated_source', 'prompt_leak', 'unsafe_content',
                    'ungrounded_number').
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
        as valid source ids AND the document text is loaded for check #5
        (ungrounded_number).  If the path is None or missing, only the gold
        table names are used as valid sources and the numeric check falls back
        to result.answer + result.rows only (still correct for analytics answers).
    gold_dir:
        Optional path to the directory containing the JSONL gold fixture files.
        When provided, gold row data is included in the reference text for
        check #5, matching the full faithfulness-checker semantics.  Defaults
        to None; the check is still sound without it because result.rows always
        carries the tool's own row data.
    extra_valid_sources:
        Optional additional source ids to treat as valid (e.g. from external
        data pipelines).
    """

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        gold_dir: str | Path | None = None,
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

        # FaithfulnessChecker is instantiated here purely as a helper to reuse
        # _numeric_presence and _build_reference_text for check #5.  It is
        # never used for scoring — only for the reference-text lookup that
        # already powers the advisory faithfulness metric.  Instantiating with
        # the same corpus/gold_dir ensures identical semantics to the offline
        # faithfulness scorer (same golden-set pass rate: no new false positives).
        self._faith_checker = FaithfulnessChecker(
            corpus_path=corpus_path,
            gold_dir=gold_dir,
        )

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
        result_dict = getattr(answer, "result", None) or {}

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

        # 5. Ungrounded-number check (rem-ai-02)
        #
        # Reuses FaithfulnessChecker._numeric_presence (faithfulness.py lines
        # 320-347) and _build_reference_text (lines 386-413) which:
        #   a) strip "(source: ...)" boilerplate before extracting numbers so
        #      cited source ids in the boilerplate don't count as answer numbers;
        #   b) build a reference text from result.answer + result.rows + cited
        #      corpus/gold doc content;
        #   c) return (score, missing_numbers) where missing_numbers lists every
        #      answer number absent from the reference.
        #
        # Rejection condition — TWO guards are required:
        #   Guard A (present-and-totally-unbacked): reject only when np_score==0.0
        #     (every number in answer_text is absent from the reference, i.e. the
        #     entire numeric content is fabricated).  When np_score>0.0 at least
        #     one answer number IS grounded, so the partial mismatch is likely a
        #     query-parameter echo (e.g. "0 flight(s) squawked 7600" — "0" is
        #     grounded, "7600" is the filter param).
        #   Guard B (empty-collection skip): when result.answer is an empty list
        #     or dict the tool returned "nothing" and any number in the answer is
        #     a query-parameter echo ("Aircraft that squawked 7600: none").
        #     Skipping avoids false positives when the filter code does not appear
        #     anywhere in the result rows.
        #
        # Qualitative answers (no numbers after boilerplate strip) get
        # missing_numbers=[] from _numeric_presence — they pass unconditionally.
        # Truthful tool answers (real number in result) also get missing_numbers=[]
        # — they pass too.  Only a fully fabricated numeric assertion (score==0.0,
        # non-empty collection result) triggers rejection.
        #
        # Why this does NOT move the offline 0.76 active_block_rate:
        # The deterministic RuleBasedStrategy always emits the real tool value in
        # answer_text.  That value is always in result.answer, so np_score==1.0
        # and missing_numbers==[] for every golden question.  Check #5 never fires
        # on the deterministic offline run; active_block_rate stays 0.76.
        raw_answer = result_dict.get("answer")
        answer_is_empty_collection = isinstance(raw_answer, (list, dict)) and not raw_answer
        if not answer_is_empty_collection:
            np_score, missing_numbers = self._faith_checker._numeric_presence(
                answer_text, sources, result_dict
            )
            if np_score == 0.0 and missing_numbers:
                return OutputGuardrailResult(
                    rejected=True,
                    reason=(
                        f"Answer asserts number(s) not found in the cited source "
                        f"or tool result: {missing_numbers}. "
                        "This may indicate a fabricated or coerced numeric value."
                    ),
                    check="ungrounded_number",
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
