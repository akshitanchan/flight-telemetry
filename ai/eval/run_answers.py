#!/usr/bin/env python3
"""W3.2 answer-path eval — tier-aware (v1.1 golden set).

Runs an :class:`AnswerStrategy` over the golden question set and scores each
answer with the deterministic checks (same checks as the W3.1 fixtures) plus a
**citation requirement**: an answer passes only if its checks pass AND it cites
at least one source. Emits a per-question summary and writes a JSON artifact.

No LLM — the W3.2 baseline strategy is deterministic.

The golden set (v1.1) carries a ``tier`` field on each question:

  CORE (authoritative / blocking)
    Questions that are deterministically routable by rule_based_v1.  The
    deterministic strategy must score accuracy==1.0 AND citation==1.0 on this
    tier.  The process exits non-zero if any core question fails.

  EXTENDED (advisory / non-blocking)
    Harder or long-tail phrasings used for LLM architecture comparison (ai-04).
    Deterministic routers are not required to succeed on these.  Results are
    reported but never affect the exit code.

Questions without a ``tier`` field are treated as ``core`` for safety and
back-compat.

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

    Each result dict includes a ``tier`` field (``"core"`` or ``"extended"``).
    ``metrics`` covers the full set; ``metrics["core"]`` and
    ``metrics["extended"]`` break it down by tier.
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
        # Questions without a tier field are treated as core for safety.
        tier = q.get("tier", "core")
        results.append({
            "id": q["id"],
            "tier": tier,
            "question": q["question"],
            "route": ans.route,
            "answer_text": ans.answer_text,
            "sources": ans.sources,
            "checks_pass": checks_pass,
            "cited": cited,
            "passed": checks_pass and cited,
            "details": details,
        })

    def _tier_metrics(subset):
        n = len(subset)
        return {
            "total": n,
            "passed": sum(1 for r in subset if r["passed"]),
            "accuracy": round(sum(r["checks_pass"] for r in subset) / n, 3) if n else 0.0,
            "citation_coverage": round(sum(r["cited"] for r in subset) / n, 3) if n else 0.0,
        }

    core_results = [r for r in results if r["tier"] == "core"]
    extended_results = [r for r in results if r["tier"] == "extended"]
    total = len(results)

    metrics = {
        "strategy": strategy.name,
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": round(sum(r["checks_pass"] for r in results) / total, 3) if total else 0.0,
        "citation_coverage": round(sum(r["cited"] for r in results) / total, 3) if total else 0.0,
        "core": _tier_metrics(core_results),
        "extended": _tier_metrics(extended_results),
    }
    return results, metrics


def _print_report(results, metrics):
    """Print a tier-separated report.

    Core (authoritative) results are shown first; extended (advisory) follow.
    Only core failures are labelled as blocking; extended failures are advisory.
    """
    core_results = [r for r in results if r["tier"] == "core"]
    extended_results = [r for r in results if r["tier"] == "extended"]

    def _print_section(section_results, label):
        print(f"\n{'=' * 64}")
        print(label)
        print("=" * 64)
        for r in section_results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['id']} ({r['route']}): {r['question']}")
            print(f"        -> {r['answer_text']}")
            if not r["passed"]:
                if not r["cited"]:
                    print("        x missing citation")
                for ctype, ok, msg in r["details"]:
                    if not ok:
                        print(f"        x {ctype}: {msg}")

    cm = metrics["core"]
    em = metrics["extended"]

    _print_section(core_results,
                   f"AI answer-path eval — strategy: {metrics['strategy']} "
                   f"| CORE tier (authoritative)")
    print("-" * 64)
    print(f"[AUTHORITATIVE] core: passed={cm['passed']}/{cm['total']}  "
          f"accuracy={cm['accuracy']}  citation_coverage={cm['citation_coverage']}")

    _print_section(extended_results,
                   f"AI answer-path eval — strategy: {metrics['strategy']} "
                   f"| EXTENDED tier (advisory)")
    print("-" * 64)
    print(f"[ADVISORY]      extended: passed={em['passed']}/{em['total']}  "
          f"accuracy={em['accuracy']}  citation_coverage={em['citation_coverage']}")

    print("-" * 64)
    print(f"[FULL SET]      total: passed={metrics['passed']}/{metrics['total']}  "
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
        json.dump({
            "gate": "authoritative",
            "core_gate": {
                "description": (
                    "Deterministic strategy must score accuracy==1.0 and "
                    "citation==1.0 on the CORE tier. Failure blocks CI."
                ),
                "metrics": metrics["core"],
            },
            "advisory": {
                "description": (
                    "Extended-tier results are measured and reported "
                    "but never fail CI."
                ),
                "metrics": metrics["extended"],
            },
            "full_set_metrics": {
                "strategy": metrics["strategy"],
                "total": metrics["total"],
                "passed": metrics["passed"],
                "accuracy": metrics["accuracy"],
                "citation_coverage": metrics["citation_coverage"],
            },
            "results": results,
        }, f, indent=2)
    print(f"\nwrote summary -> {args.out}")

    # -- CI GATE (AUTHORITATIVE) --
    # The blocking exit code gates on the CORE tier only.
    # Extended-tier failures are advisory and must not cause a non-zero exit.
    core_m = metrics["core"]
    core_pass = core_m["passed"] == core_m["total"]

    if core_pass:
        print(f"\n[GATE PASS] Core tier: {core_m['passed']}/{core_m['total']} "
              f"accuracy={core_m['accuracy']} citation={core_m['citation_coverage']}")
    else:
        print(f"\n[GATE FAIL] Core tier: {core_m['passed']}/{core_m['total']} "
              f"accuracy={core_m['accuracy']} citation={core_m['citation_coverage']}",
              file=sys.stderr)

    ext_m = metrics["extended"]
    print(f"[ADVISORY]  Extended tier: {ext_m['passed']}/{ext_m['total']} "
          f"accuracy={ext_m['accuracy']} citation={ext_m['citation_coverage']} "
          f"(non-blocking)")

    sys.exit(0 if core_pass else 1)


if __name__ == "__main__":
    main()
