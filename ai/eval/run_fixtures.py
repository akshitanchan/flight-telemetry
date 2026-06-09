#!/usr/bin/env python3
"""W3.1 fixtures harness.

Validates that:
  1. the golden question set loads and is well-formed;
  2. every analytics operation it references exists on the tool (no free-form SQL);
  3. each question's deterministic checks pass against the *frozen* gold + corpus
     fixtures.

No LLM is involved — this is the deterministic ground-truth gate the later
answer-path gates (W3.2/W3.3) build on. Exits non-zero if any check fails.

Usage:
    python -m ai.eval.run_fixtures
    python -m ai.eval.run_fixtures --gold ai/fixtures/gold \\
        --corpus ai/fixtures/corpus/corpus.json --golden ai/eval/golden_set.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool, AnalyticsError
from ai.tools.retrieval import RetrievalTool
from ai.eval.checks import run_check

DEFAULT_GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
DEFAULT_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"

REQUIRED_KEYS = {"id", "question", "tool", "checks"}


def _validate_golden(questions, analytics: AnalyticsTool) -> list[str]:
    """Structural validation. Returns a list of error strings (empty == ok)."""
    errors = []
    seen = set()
    for i, q in enumerate(questions):
        missing = REQUIRED_KEYS - q.keys()
        if missing:
            errors.append(f"question[{i}] missing keys: {sorted(missing)}")
            continue
        if q["id"] in seen:
            errors.append(f"duplicate question id: {q['id']!r}")
        seen.add(q["id"])
        if q["tool"] == "analytics":
            op = q.get("operation")
            if op not in analytics.operations:
                errors.append(f"{q['id']}: unknown analytics operation {op!r}")
        elif q["tool"] != "retrieval":
            errors.append(f"{q['id']}: unknown tool {q['tool']!r}")
        if not q.get("checks"):
            errors.append(f"{q['id']}: no checks defined")
    return errors


def evaluate(gold_dir=DEFAULT_GOLD, corpus_path=DEFAULT_CORPUS, golden_path=DEFAULT_GOLDEN):
    """Run the golden set against the fixtures.

    Returns (results, all_passed). Raises ValueError on structural problems so
    a malformed golden set fails loudly rather than silently passing.
    """
    with open(golden_path) as f:
        golden = json.load(f)
    questions = golden.get("questions", [])

    analytics = AnalyticsTool(gold_dir)
    retrieval = RetrievalTool(corpus_path)

    struct_errors = _validate_golden(questions, analytics)
    if struct_errors:
        raise ValueError("golden set invalid:\n  - " + "\n  - ".join(struct_errors))

    results = []
    for q in questions:
        try:
            if q["tool"] == "analytics":
                result = analytics.call(q["operation"], **q.get("params", {}))
            else:
                result = retrieval.search(**q.get("params", {}))
        except (AnalyticsError, TypeError) as e:
            results.append({"id": q["id"], "passed": False,
                            "details": [("call", False, str(e))]})
            continue

        details = []
        passed = True
        for chk in q["checks"]:
            ok, msg = run_check(chk, result)
            details.append((chk["type"], ok, msg))
            passed = passed and ok
        results.append({"id": q["id"], "question": q["question"],
                        "passed": passed, "details": details})

    all_passed = all(r["passed"] for r in results)
    return results, all_passed


def _print_report(results, all_passed):
    print("\nAI eval fixtures — golden set results")
    print("=" * 60)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}: {r.get('question', '')}")
        if not r["passed"]:
            for ctype, ok, msg in r["details"]:
                if not ok:
                    print(f"        ✗ {ctype}: {msg}")
    n_pass = sum(1 for r in results if r["passed"])
    print("-" * 60)
    print(f"{n_pass}/{len(results)} questions passed")
    print("OVERALL:", "PASS" if all_passed else "FAIL")


def main():
    parser = argparse.ArgumentParser(description="Validate AI golden set against fixtures")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    args = parser.parse_args()

    try:
        results, all_passed = evaluate(args.gold, args.corpus, args.golden)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    _print_report(results, all_passed)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
