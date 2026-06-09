#!/usr/bin/env python3
"""W3.3 strategy comparison.

Runs every available answer strategy over the same golden set and compares them
on accuracy, citation coverage, latency (p50/p95), and a cost proxy (LLM calls /
tokens). Deterministic strategies always run; the Ollama LLM strategy is included
only if the server + a model are reachable, otherwise it is reported as skipped.

Research question (plan §5.4): which agent architecture gives the best
faithfulness/cost tradeoff for operational analytical queries?

Usage:
    python -m ai.eval.compare
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.rule_based import RuleBasedStrategy
from ai.agent.keyword_router import KeywordScoreStrategy
from ai.agent.ollama_llm import OllamaStrategy
from ai.eval.checks import run_check

DEFAULT_GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
DEFAULT_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "ai_compare_small.json"


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * pct / 100)))
    return sorted_vals[k]


def _run_strategy(strategy, questions):
    latencies, results = [], []
    tokens = calls = 0
    for q in questions:
        t0 = time.perf_counter()
        ans = strategy.answer(q["question"])
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        checks_pass = all(run_check(c, ans.result)[0] for c in q["checks"])
        cited = len(ans.sources) > 0
        results.append({"id": q["id"], "passed": checks_pass and cited, "cited": cited,
                        "route": ans.route, "latency_ms": round(dt_ms, 2)})
        tokens += ans.meta.get("tokens", 0)
        calls += ans.meta.get("llm_calls", 0)

    n = len(results)
    lat_sorted = sorted(latencies)
    return {
        "strategy": strategy.name,
        "total": n,
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": round(sum(r["passed"] for r in results) / n, 3) if n else 0.0,
        "citation_coverage": round(sum(1 for r in results if r["cited"]) / n, 3) if n else 0.0,
        "p50_latency_ms": round(_percentile(lat_sorted, 50), 2),
        "p95_latency_ms": round(_percentile(lat_sorted, 95), 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "llm_calls": calls,
        "tokens": tokens,
        "per_question": results,
    }


def compare(gold_dir=DEFAULT_GOLD, corpus_path=DEFAULT_CORPUS, golden_path=DEFAULT_GOLDEN,
            include_llm=True):
    with open(golden_path) as f:
        questions = json.load(f).get("questions", [])
    analytics = AnalyticsTool(gold_dir)
    retrieval = RetrievalTool(corpus_path)

    strategies = [RuleBasedStrategy(analytics, retrieval),
                  KeywordScoreStrategy(analytics, retrieval)]
    notes = []
    if include_llm:
        ok, model, _ = OllamaStrategy.available()
        if ok:
            strategies.append(OllamaStrategy(analytics, retrieval, model=model))
        else:
            notes.append("ollama strategy skipped — server/model unavailable")

    summaries = [_run_strategy(s, questions) for s in strategies]
    return summaries, notes


def _print_table(summaries, notes):
    cols = ["strategy", "accuracy", "citation_coverage", "p50_latency_ms",
            "p95_latency_ms", "llm_calls", "tokens"]
    print("\nAI strategy comparison (same golden set)")
    print("=" * 92)
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for s in summaries:
        print("| " + " | ".join(str(s[c]) for c in cols) + " |")
    for n in notes:
        print(f"\nNote: {n}")


def main():
    parser = argparse.ArgumentParser(description="Compare AI answer strategies on the golden set")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-llm", action="store_true", help="Skip the Ollama LLM strategy")
    args = parser.parse_args()

    summaries, notes = compare(args.gold, args.corpus, args.golden, include_llm=not args.no_llm)
    _print_table(summaries, notes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summaries": summaries, "notes": notes}, f, indent=2)
    print(f"\nwrote comparison -> {args.out}")

    # Gate: every deterministic strategy must still pass all questions.
    det = [s for s in summaries if not s["strategy"].startswith("ollama")]
    sys.exit(0 if all(s["passed"] == s["total"] for s in det) else 1)


if __name__ == "__main__":
    main()
