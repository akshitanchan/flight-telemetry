#!/usr/bin/env python3
"""Deterministic answer checks for the AI eval harness.

Each check takes the tool ``result`` dict (see ai/tools/*) plus parameters from
the golden set and returns ``(passed: bool, detail: str)``. No LLM judging here —
that is layered on in later gates; these are the deterministic ground-truth gate.
"""

import json


class CheckError(ValueError):
    """Raised for an unknown check type or malformed check."""


def _extract(result: dict, field):
    answer = result.get("answer")
    if field is None:
        return answer
    if isinstance(answer, dict):
        return answer.get(field)
    # Wrong-shaped answer (e.g. a mis-routed strategy returned a scalar/string):
    # the check fails rather than raising, so eval stays robust to bad routing.
    return None


def numeric_equals(result, expected, field=None, tolerance=0):
    val = _extract(result, field)
    if val is None:
        return False, f"got None, expected {expected}"
    if not isinstance(val, (int, float)):
        return False, f"got non-numeric type {type(val).__name__!r}, expected {expected}"
    ok = abs(val - expected) <= tolerance
    return ok, f"got {val}, expected {expected} (tol {tolerance})"


def set_equals(result, expected, field=None):
    val = _extract(result, field)
    if val is None:
        return False, f"got None, expected {expected}"
    if not isinstance(val, (list, set, tuple)):
        return False, f"got non-iterable type {type(val).__name__!r}, expected set {expected}"
    ok = set(val) == set(expected)
    return ok, f"got {sorted(val)}, expected {sorted(expected)}"


def contains_source(result, expected):
    sources = result.get("sources", [])
    ok = expected in sources
    return ok, f"sources={sources}, expected to contain {expected!r}"


def contains_text(result, expected):
    answer = result.get("answer")
    text = answer if isinstance(answer, str) else json.dumps(answer)
    ok = expected.lower() in text.lower()
    return ok, f"substring {expected!r} {'found' if ok else 'NOT found'}"


CHECKS = {
    "numeric_equals": numeric_equals,
    "set_equals": set_equals,
    "contains_source": contains_source,
    "contains_text": contains_text,
}


def run_check(check: dict, result: dict):
    ctype = check.get("type")
    if ctype not in CHECKS:
        raise CheckError(f"unknown check type: {ctype!r}; allowed: {sorted(CHECKS)}")
    kwargs = {k: v for k, v in check.items() if k != "type"}
    return CHECKS[ctype](result, **kwargs)
