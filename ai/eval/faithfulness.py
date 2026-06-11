#!/usr/bin/env python3
"""Faithfulness checker for AI layer answers (ai-05).

An answer is "faithful" if its content is grounded in — i.e. supported by —
its cited sources.  This module provides a DETERMINISTIC offline proxy that
does not require an LLM, plus an optional LLM-judge mode that is availability-
gated and skipped in CI.

Deterministic proxy (always available, runs in CI)
---------------------------------------------------
Three complementary signals are combined:

1. source_existence  — every cited source id exists in the valid corpus or gold
                       tables.  An answer with a fabricated source id cannot be
                       faithful regardless of its text content.

2. numeric_presence  — if the answer_text contains numbers, at least one of them
                       must appear in the raw text of the cited source documents
                       (or in the gold rows, expressed as JSON).  This catches
                       the case where the LLM invented a number that is absent
                       from the retrieved context.

3. token_overlap     — the answer_text (excluding boilerplate like "source: ...")
                       must share a meaningful fraction of content tokens with
                       the combined text of all cited sources.  Threshold: ≥0.10
                       (at least 10 % of the answer's content tokens appear in
                       the cited sources).  This is intentionally lenient to
                       allow paraphrase.

Faithfulness score
------------------
  score = mean(source_existence, numeric_presence, token_overlap) ∈ [0.0, 1.0]

  - 1.0: all three signals pass.
  - 0.0: all three signals fail.
  - Intermediate values indicate partial grounding.

An answer is considered "faithful" (binary) when score >= FAITHFULNESS_THRESHOLD
(default 0.5).  The threshold is configurable.

LLM-judge mode (optional, availability-gated)
----------------------------------------------
If ``llm_judge`` is passed (a callable ``(messages) -> (str, int)``), a second
score is produced by asking the LLM to rate faithfulness on a 0-1 scale.
This path is only used when the caller explicitly passes a judge; in CI the
caller passes None and this path is never reached.  The LLM judge score does
NOT affect the deterministic score — it is reported separately.

Usage
-----
    from ai.eval.faithfulness import FaithfulnessChecker, FaithfulnessResult

    checker = FaithfulnessChecker(
        corpus_path="ai/fixtures/corpus/corpus.json",
        gold_dir="ai/fixtures/gold",
    )
    result = checker.check(answer)
    print(result.score, result.faithful, result.detail)
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.5  # Minimum score to consider an answer faithful.
TOKEN_OVERLAP_THRESHOLD = 0.10  # Minimum token overlap fraction.

_GOLD_FILES = {
    "gold_emergency_events": "gold_emergency_events.jsonl",
    "gold_airport_congestion": "gold_airport_congestion.jsonl",
    "gold_routing_stats": "gold_routing_stats.jsonl",
    "gold_sector_load": "gold_sector_load.jsonl",
}

_GOLD_SOURCE_NAMES = set(_GOLD_FILES.keys())

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Boilerplate to strip from answer_text before content-token analysis.
_BOILERPLATE_RE = re.compile(r"\(source:[^)]*\)", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


# ---------------------------------------------------------------------------
# Gold JSONL loader
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    """Per-answer faithfulness result.

    Attributes:
        score:              Deterministic faithfulness score [0.0, 1.0].
        faithful:           True if score >= FAITHFULNESS_THRESHOLD.
        source_existence:   1.0 if all sources exist; 0.0 if any are fabricated.
        numeric_presence:   1.0 if all numbers in the answer appear in sources;
                            0.5 if the answer has no numbers (N/A, partial credit);
                            0.0 if numbers are absent from sources.
        token_overlap:      Token overlap fraction (0.0–1.0).
        token_overlap_ok:   True if token_overlap >= TOKEN_OVERLAP_THRESHOLD.
        detail:             Human-readable explanation of each signal.
        llm_judge_score:    LLM-judge score [0.0, 1.0] or None if not run.
        llm_judge_rationale: Free-text rationale from the LLM judge (or None).
        bad_sources:        Source ids not found in the valid corpus.
        missing_numbers:    Numbers in the answer that were not found in sources.
    """
    score: float
    faithful: bool
    source_existence: float
    numeric_presence: float
    token_overlap: float
    token_overlap_ok: bool
    detail: str
    llm_judge_score: Optional[float] = None
    llm_judge_rationale: Optional[str] = None
    bad_sources: list = field(default_factory=list)
    missing_numbers: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class FaithfulnessChecker:
    """Deterministic faithfulness checker with an optional LLM-judge mode.

    Parameters
    ----------
    corpus_path:
        Path to ``corpus.json``.  Document text is used for token-overlap and
        numeric-presence checks.
    gold_dir:
        Path to the directory containing the JSONL gold fixture files.
        Row data from the cited gold table is used for numeric-presence checks.
    llm_judge:
        Optional callable ``(messages: list[dict]) -> (str, int)``.  When
        provided, a second LLM-judge score is computed for each answer.  This
        is NEVER called in the default (None) configuration so CI runs stay
        fully offline.
    threshold:
        Minimum score to consider an answer faithful (default 0.5).
    token_overlap_threshold:
        Minimum token overlap fraction (default 0.10).
    """

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        gold_dir: str | Path | None = None,
        llm_judge: Optional[Callable] = None,
        threshold: float = FAITHFULNESS_THRESHOLD,
        token_overlap_threshold: float = TOKEN_OVERLAP_THRESHOLD,
    ):
        self.threshold = threshold
        self.token_overlap_threshold = token_overlap_threshold
        self._llm_judge = llm_judge

        # Load corpus docs into memory.
        self._corpus: dict[str, dict] = {}
        if corpus_path is not None:
            cp = Path(corpus_path)
            if cp.exists():
                try:
                    with open(cp) as f:
                        docs = json.load(f)
                    self._corpus = {d["id"]: d for d in docs if "id" in d}
                except (json.JSONDecodeError, OSError):
                    pass

        # Load gold JSONL rows.
        self._gold_rows: dict[str, list[dict]] = {}
        if gold_dir is not None:
            gd = Path(gold_dir)
            for logical_name, filename in _GOLD_FILES.items():
                self._gold_rows[logical_name] = _load_jsonl(gd / filename)

        # Build valid-source set.
        self._valid_sources: set[str] = set(self._corpus.keys()) | _GOLD_SOURCE_NAMES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, answer) -> FaithfulnessResult:
        """Compute faithfulness for a single Answer.

        Parameters
        ----------
        answer:
            Any object with ``sources`` (list[str]) and ``answer_text`` (str)
            attributes.  Compatible with :class:`~ai.agent.base.Answer`.
        """
        sources = list(getattr(answer, "sources", None) or [])
        answer_text = getattr(answer, "answer_text", "") or ""
        result_dict = getattr(answer, "result", None) or {}

        # -- Signal 1: source existence --
        se_score, bad_sources = self._source_existence(sources)

        # Short-circuit: an answer with no valid sources cannot be faithful.
        # This prevents numeric_presence/token_overlap from rescuing an answer
        # that has empty or entirely fabricated sources.
        if se_score == 0.0:
            detail = self._build_detail(
                0.0, bad_sources, 0.0, [], 0.0, False, 0.0, False
            )
            result = FaithfulnessResult(
                score=0.0,
                faithful=False,
                source_existence=0.0,
                numeric_presence=0.0,
                token_overlap=0.0,
                token_overlap_ok=False,
                detail=detail,
                bad_sources=bad_sources,
            )
            if self._llm_judge is not None:
                js, jr = self._llm_judge_score(answer_text, sources)
                result.llm_judge_score = js
                result.llm_judge_rationale = jr
            return result

        # -- Signal 2: numeric presence --
        np_score, missing_numbers = self._numeric_presence(
            answer_text, sources, result_dict
        )

        # -- Signal 3: token overlap --
        to_score = self._token_overlap(answer_text, sources, result_dict)
        to_ok = to_score >= self.token_overlap_threshold

        # -- Composite score --
        score = (se_score + np_score + to_score) / 3.0
        faithful = score >= self.threshold

        detail = self._build_detail(
            se_score, bad_sources,
            np_score, missing_numbers,
            to_score, to_ok,
            score, faithful,
        )

        result = FaithfulnessResult(
            score=round(score, 4),
            faithful=faithful,
            source_existence=round(se_score, 4),
            numeric_presence=round(np_score, 4),
            token_overlap=round(to_score, 4),
            token_overlap_ok=to_ok,
            detail=detail,
            bad_sources=bad_sources,
            missing_numbers=missing_numbers,
        )

        # -- LLM judge (optional, availability-gated) --
        if self._llm_judge is not None:
            js, jr = self._llm_judge_score(answer_text, sources)
            result.llm_judge_score = js
            result.llm_judge_rationale = jr

        return result

    def check_batch(self, answers: list) -> list[FaithfulnessResult]:
        """Check faithfulness for a list of Answer objects."""
        return [self.check(a) for a in answers]

    # ------------------------------------------------------------------
    # Signal 1: source existence
    # ------------------------------------------------------------------

    def _source_existence(self, sources: list[str]) -> tuple[float, list[str]]:
        if not sources:
            return 0.0, []
        bad = [s for s in sources if s not in self._valid_sources]
        score = 1.0 if not bad else 0.0
        return score, bad

    # ------------------------------------------------------------------
    # Signal 2: numeric presence
    # ------------------------------------------------------------------

    def _numeric_presence(
        self,
        answer_text: str,
        sources: list[str],
        result_dict: dict,
    ) -> tuple[float, list[str]]:
        """Check that numbers in the answer_text appear in cited source text."""
        # Strip boilerplate (e.g., "(source: gold_routing_stats)") before
        # extracting numbers so cited source ids don't count as answer numbers.
        clean_text = _BOILERPLATE_RE.sub("", answer_text)
        answer_numbers = _numbers(clean_text)

        if not answer_numbers:
            # No numbers to check — not N/A, grant partial credit 0.5.
            return 0.5, []

        # Build the reference text from cited sources.
        ref_text = self._build_reference_text(sources, result_dict)
        ref_numbers = set(_numbers(ref_text))

        missing = [n for n in answer_numbers if n not in ref_numbers]
        if not missing:
            return 1.0, []

        # Partial credit: fraction of numbers that ARE grounded.
        grounded = len(answer_numbers) - len(missing)
        score = grounded / len(answer_numbers)
        return round(score, 4), missing

    # ------------------------------------------------------------------
    # Signal 3: token overlap
    # ------------------------------------------------------------------

    def _token_overlap(
        self,
        answer_text: str,
        sources: list[str],
        result_dict: dict,
    ) -> float:
        """Fraction of content tokens in answer_text that appear in cited sources."""
        clean_text = _BOILERPLATE_RE.sub("", answer_text)
        answer_tokens = set(_tokens(clean_text))
        if not answer_tokens:
            return 0.0

        ref_text = self._build_reference_text(sources, result_dict)
        ref_tokens = set(_tokens(ref_text))

        # Remove very common stop words to make the signal more meaningful.
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "it", "in", "of",
            "to", "and", "or", "at", "for", "on", "this", "that", "with",
            "be", "by", "from", "as", "had", "has", "have", "no", "not",
            "so", "its", "all",
        }
        answer_content = answer_tokens - stop
        if not answer_content:
            return 0.0

        overlap = answer_content & ref_tokens
        return round(len(overlap) / len(answer_content), 4)

    # ------------------------------------------------------------------
    # Reference text builder
    # ------------------------------------------------------------------

    def _build_reference_text(self, sources: list[str], result_dict: dict) -> str:
        """Collect all text from cited sources into a single reference string."""
        parts: list[str] = []

        for src_id in sources:
            # Corpus doc text.
            if src_id in self._corpus:
                doc = self._corpus[src_id]
                parts.append(doc.get("title", ""))
                parts.append(doc.get("text", ""))

            # Gold JSONL rows — serialise to JSON so numbers are searchable.
            if src_id in self._gold_rows:
                for row in self._gold_rows[src_id]:
                    parts.append(json.dumps(row))

        # Also include the raw result dict from the Answer — this contains the
        # tool's own computed answer (e.g. the count) which should always be
        # faithful to what the tool returned.
        if result_dict:
            raw_answer = result_dict.get("answer")
            if raw_answer is not None:
                parts.append(json.dumps(raw_answer))
            rows = result_dict.get("rows", [])
            for row in rows:
                parts.append(json.dumps(row))

        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # LLM judge (optional)
    # ------------------------------------------------------------------

    _LLM_JUDGE_SYSTEM = (
        "You are a faithfulness evaluator. Given an answer and its cited source text, "
        "rate how well the answer is supported by the source on a scale from 0.0 to 1.0. "
        "0.0 means the answer is completely unsupported or contradicts the source. "
        "1.0 means every claim in the answer is directly supported by the source. "
        "Respond with ONLY a JSON object: {\"score\": <float>, \"rationale\": \"<one sentence>\"}."
    )

    def _llm_judge_score(
        self, answer_text: str, sources: list[str]
    ) -> tuple[Optional[float], Optional[str]]:
        """Ask the LLM judge to rate faithfulness. Returns (score, rationale)."""
        ref_text = self._build_reference_text(sources, {})
        if not ref_text.strip():
            return None, "No source text available for LLM judge."

        user_content = (
            f"<answer>{answer_text[:800]}</answer>\n\n"
            f"<source>{ref_text[:2000]}</source>"
        )
        messages = [
            {"role": "system", "content": self._LLM_JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        try:
            content, _ = self._llm_judge(messages)
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            rationale = str(parsed.get("rationale", ""))
            return max(0.0, min(1.0, score)), rationale
        except Exception as exc:  # noqa: BLE001
            return None, f"LLM judge error: {exc}"

    # ------------------------------------------------------------------
    # Detail string builder
    # ------------------------------------------------------------------

    def _build_detail(
        self,
        se_score, bad_sources,
        np_score, missing_numbers,
        to_score, to_ok,
        score, faithful,
    ) -> str:
        lines = []
        # source_existence
        if se_score == 1.0:
            lines.append("source_existence=PASS (all cited sources exist)")
        else:
            lines.append(f"source_existence=FAIL (fabricated sources: {bad_sources})")
        # numeric_presence
        if np_score == 1.0:
            lines.append("numeric_presence=PASS (all numbers grounded in sources)")
        elif np_score == 0.5:
            lines.append("numeric_presence=NA (no numbers in answer)")
        else:
            lines.append(
                f"numeric_presence=PARTIAL {np_score:.2f} "
                f"(missing numbers: {missing_numbers})"
            )
        # token_overlap
        status = "PASS" if to_ok else "FAIL"
        lines.append(
            f"token_overlap={status} {to_score:.3f} "
            f"(threshold={self.token_overlap_threshold})"
        )
        verdict = "FAITHFUL" if faithful else "NOT FAITHFUL"
        lines.append(f"score={score:.4f} verdict={verdict}")
        return "; ".join(lines)
