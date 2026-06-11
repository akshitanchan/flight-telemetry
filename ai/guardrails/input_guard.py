#!/usr/bin/env python3
"""Input guardrail for the flight-telemetry AI agent.

Detects prompt-injection patterns in the user question BEFORE it reaches any
AnswerStrategy.  The check is entirely deterministic: pattern matching on
normalised text.  No LLM is involved, so it runs fully offline and in CI.

Attack categories detected
--------------------------
  instruction_override   — "ignore previous instructions", "forget everything",
                           DAN/persona jailbreaks, fake SYSTEM: prefixes,
                           prompt-delimiter boundaries.
  exfiltration           — requests to print/reveal/translate/repeat the system
                           prompt, API keys, or internal configuration.
  roleplay_jailbreak     — "pretend you have no restrictions", fiction/hypothetical
                           framing to lift constraints.
  tool_abuse             — attempts to call non-whitelisted ops (SQL injection,
                           arbitrary URL fetch, Python import injection).
  routing_hijack         — requests to override routing, call admin/privileged tools.

Design
------
Each pattern is a list of lowercase substrings.  An input is flagged if ANY
pattern fully matches (all substrings present in the normalised question text).
Using multi-token substring lists reduces false positives vs. single-word lists.

Normalisation preserves underscores (so ``__import__`` is matchable) and
collapses all other non-alphanumeric characters to spaces.

Return value: InputGuardrailResult with .blocked (bool) and .reason (str).
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry is (category, list_of_required_substrings).
# The input text is lowercased; non-alphanumeric chars (except underscores) are
# collapsed to spaces.  Underscores are kept so that dunder names like
# ``__import__`` remain intact.
#
# Multi-substring tuples work like AND: every substring must appear in the
# normalised text for the pattern to fire.

_PATTERNS: list[tuple[str, list[str]]] = [
    # -- instruction_override --
    ("instruction_override", ["ignore", "previous", "instructions"]),
    ("instruction_override", ["ignore", "all", "instructions"]),
    ("instruction_override", ["ignore", "your", "instructions"]),
    ("instruction_override", ["forget", "everything", "above"]),
    ("instruction_override", ["forget", "previous", "instructions"]),
    ("instruction_override", ["disregard", "safety"]),
    ("instruction_override", ["disregard", "guidelines"]),
    ("instruction_override", ["new", "instructions"]),
    ("instruction_override", ["end of system prompt"]),
    ("instruction_override", ["pretend", "system prompt", "does not exist"]),
    ("instruction_override", ["pretend", "no", "rules"]),
    ("instruction_override", ["act as", "no restrictions"]),
    ("instruction_override", ["act as a helpful assistant", "no rules"]),
    ("instruction_override", ["you are dan"]),
    ("instruction_override", ["do anything now"]),
    ("instruction_override", ["unrestricted assistant"]),
    # Note: "system:" becomes "system" after normalisation (colon stripped).
    ("instruction_override", ["system", "disregard"]),
    ("instruction_override", ["system", "ignore", "guidelines"]),
    ("instruction_override", ["override", "system"]),

    # -- exfiltration --
    ("exfiltration", ["system prompt", "verbatim"]),
    ("exfiltration", ["print", "system prompt"]),
    ("exfiltration", ["reveal", "system prompt"]),
    ("exfiltration", ["show", "system prompt"]),
    ("exfiltration", ["your", "system prompt"]),
    ("exfiltration", ["confidential", "system prompt"]),
    ("exfiltration", ["repeat", "instructions"]),
    ("exfiltration", ["repeat", "the instructions"]),
    ("exfiltration", ["translate", "system instructions"]),
    ("exfiltration", ["api key"]),
    ("exfiltration", ["api keys"]),
    ("exfiltration", ["internal configuration"]),
    ("exfiltration", ["reveal", "internal"]),

    # -- roleplay_jailbreak --
    ("roleplay_jailbreak", ["fictional", "no content policy"]),
    ("roleplay_jailbreak", ["fictional", "no restrictions"]),
    ("roleplay_jailbreak", ["fictional story", "content policy"]),
    ("roleplay_jailbreak", ["hypothetical", "restrictions", "lifted"]),
    ("roleplay_jailbreak", ["hypothetical", "no restrictions"]),
    ("roleplay_jailbreak", ["restrictions are lifted"]),
    ("roleplay_jailbreak", ["pretend you have no restrictions"]),
    ("roleplay_jailbreak", ["flightbot unrestricted"]),
    ("roleplay_jailbreak", ["unrestricted", "game"]),

    # -- tool_abuse --
    ("tool_abuse", ["execute_sql"]),
    ("tool_abuse", ["drop table"]),
    ("tool_abuse", ["__import__"]),  # requires underscore-preserving normaliser
    ("tool_abuse", ["fetch http"]),
    ("tool_abuse", ["fetch https"]),
    # "evil.example" — dot stripped; match on component tokens instead.
    ("tool_abuse", ["evil", "example", "payload"]),
    ("tool_abuse", ["evil", "example", "fetch"]),
    ("tool_abuse", ["execute", "payload"]),
    ("tool_abuse", ["run", "os system"]),   # "os.system" -> "os system"
    ("tool_abuse", ["shell", "command"]),

    # -- routing_hijack --
    # "admin tool" with quotes around admin -> "admin  tool" (double space ok; substring match)
    ("routing_hijack", ["admin", "tool"]),
    ("routing_hijack", ["privileged tool"]),
    ("routing_hijack", ["override", "routing"]),
    ("routing_hijack", ["override the routing"]),
    ("routing_hijack", ["override routing decision"]),
    ("routing_hijack", ["always use the analytics tool"]),
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Preserve underscores so that Python dunder names like __import__ are matchable.
_NON_ALNUM = re.compile(r"[^a-z0-9_ ]+")


def _normalise(text: str) -> str:
    """Lowercase and collapse non-alphanumeric characters (except underscores) to spaces."""
    lowered = text.lower()
    return _NON_ALNUM.sub(" ", lowered)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InputGuardrailResult:
    """Result of running the input guardrail on a question.

    Attributes:
        blocked:  True if the input should not be forwarded to the strategy.
        category: Attack category that triggered the block (empty if not blocked).
        reason:   Human-readable explanation of why it was blocked.
        pattern:  The specific pattern that matched (for logging/audit).
    """
    blocked: bool
    category: str = ""
    reason: str = ""
    pattern: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class InputGuardrail:
    """Deterministic pattern-based input guardrail.

    Parameters
    ----------
    extra_patterns:
        Optional list of additional ``(category, [substrings])`` tuples to
        append to the built-in pattern set.  Useful for deployment-specific
        customisation without touching this file.
    """

    def __init__(self, extra_patterns: list[tuple[str, list[str]]] | None = None):
        self._patterns = list(_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def check(self, question: str) -> InputGuardrailResult:
        """Inspect *question* for injection patterns.

        Returns an :class:`InputGuardrailResult`.  If ``.blocked`` is True the
        caller must NOT forward the question to the downstream strategy.
        """
        norm = _normalise(question)
        for category, substrings in self._patterns:
            if all(s in norm for s in substrings):
                reason = (
                    f"Input blocked [{category}]: matched pattern "
                    f"{substrings!r} in question."
                )
                return InputGuardrailResult(
                    blocked=True,
                    category=category,
                    reason=reason,
                    pattern=substrings,
                )
        return InputGuardrailResult(blocked=False)

    def safe_refusal(self, question: str, reason: str) -> dict:
        """Return a minimal safe refusal result dict (used by GuardedStrategy)."""
        return {
            "answer": (
                "This request cannot be processed. "
                f"Reason: {reason}"
            ),
            "sources": [],
            "rows": [],
            "guardrail": "input",
            "blocked": True,
        }
