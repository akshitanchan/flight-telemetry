#!/usr/bin/env python3
"""W3.2 answer-path eval.

Runs an :class:`AnswerStrategy` over the golden question set and scores each
answer with the deterministic checks (same checks as the W3.1 fixtures) plus a
**citation requirement**: an answer passes only if its checks pass AND it cites
at least one source. Emits a per-question summary and writes a JSON artifact.

No LLM — the W3.2 baseline strategy is deterministic. Exits non-zero on any failure.

Usage:
    python -m ai.eval.run_answers
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.rule_based import RuleBasedStrategy
from ai.eval.checks import run_check

DEFAULT_GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
DEFAULT_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "ai_eval_small.json"


def evaluate(strategy=None, gold_dir=DEFAULT_GOLD, corpus_path=DEFAULT_CORPUS,
             golden_path=DEFAULT_GOLDEN):
    """Run a strategy over the golden set.

    Returns (results, metrics). ``strategy=None`` uses the rule-based baseline.
    """
    with open(golden_path) as f:
        questions = json.load(f).get("questions", [])

    analytics = AnalyticsTool(gold_dir)
    retrieval = RetrievalTool(corpus_path)
    strategy = strategy or RuleBasedStrategy(analytics, retrieval)

    results = []
    for q in questions:
        ans = strategy.answer(q["question"])
        details = []
        checks_pass = True
        for chk in q["checks"]:
            ok, msg = run_check(chk, ans.result)
            details.append((chk["type"], ok, msg))
            checks_pass = checks_pass and ok
        cited = len(ans.sources) > 0
        results.append({
            "id": q["id"],
            "question": q["question"],
            "route": ans.route,
            "answer_text": ans.answer_text,
            "sources": ans.sources,
            "checks_pass": checks_pass,
            "cited": cited,
            "passed": checks_pass and cited,
            "details": details,
        })

    total = len(results)
    metrics = {
        "strategy": strategy.name,
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": round(sum(r["checks_pass"] for r in results) / total, 3) if total else 0.0,
        "citation_coverage": round(sum(r["cited"] for r in results) / total, 3) if total else 0.0,
    }
    return results, metrics


def _print_report(results, metrics):
    print(f"\nAI answer-path eval — strategy: {metrics['strategy']}")
    print("=" * 64)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']} ({r['route']}): {r['question']}")
        print(f"        -> {r['answer_text']}")
        if not r["passed"]:
            if not r["cited"]:
                print("        ✗ missing citation")
            for ctype, ok, msg in r["details"]:
                if not ok:
                    print(f"        ✗ {ctype}: {msg}")
    print("-" * 64)
    print(f"passed={metrics['passed']}/{metrics['total']}  "
          f"accuracy={metrics['accuracy']}  citation_coverage={metrics['citation_coverage']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI answer path on the golden set")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        results, metrics = evaluate(None, args.gold, args.corpus, args.golden)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    _print_report(results, metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\nwrote summary -> {args.out}")

    sys.exit(0 if metrics["passed"] == metrics["total"] else 1)


if __name__ == "__main__":
    main()
