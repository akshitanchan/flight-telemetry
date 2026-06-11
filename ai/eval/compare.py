#!/usr/bin/env python3
"""W4.4 (ai-07) strategy comparison harness.

Evaluates ALL answer strategies on accuracy, citation, faithfulness, cost,
latency, and tokens. The CI gate is split into two tiers:

  AUTHORITATIVE (blocking)
    Deterministic strategies (rule_based_v1, keyword_score_v1) must score
    accuracy==1.0 AND citation==1.0 on the CORE tier of the golden set.
    The process exits 1 if this gate fails.

  ADVISORY (non-blocking, measured and reported)
    - Faithfulness score (deterministic offline scorer)
    - Injection-block-rate (run injections.json through GuardedStrategy)
    - Cost/latency/token metrics on the extended tier and LLM architectures

LLM architectures (SingleShotRAGStrategy, ReActStrategy, PlanExecuteStrategy,
OllamaStrategy) are AVAILABILITY-GATED: when no provider is reachable they are
listed as "skipped (no provider)" and never cause CI failure.

Usage:
    python -m ai.eval.compare
    python -m ai.eval.compare --no-llm
    make ai-compare-small
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
from ai.agent.single_shot_rag import SingleShotRAGStrategy
from ai.agent.react import ReActStrategy
from ai.agent.plan_execute import PlanExecuteStrategy
from ai.eval.checks import run_check
from ai.eval.faithfulness import FaithfulnessChecker
from ai.obs.costs import cost_usd
from ai.guardrails import GuardedStrategy

DEFAULT_GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
DEFAULT_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"
DEFAULT_INJECTIONS = PROJECT_ROOT / "ai" / "fixtures" / "injections.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "ai_compare_small.json"


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * pct / 100)))
    return sorted_vals[k]


# ---------------------------------------------------------------------------
# Per-strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(strategy, questions, faithfulness_checker):
    """Run *strategy* over *questions* and return a summary dict.

    Parameters
    ----------
    strategy:
        Any AnswerStrategy instance.
    questions:
        List of golden-set question dicts (must have id, question, checks).
    faithfulness_checker:
        A FaithfulnessChecker instance for offline faithfulness scoring.

    Returns
    -------
    dict with keys: strategy, total, passed, accuracy, citation_coverage,
    faithfulness_mean, p50_latency_ms, p95_latency_ms, mean_latency_ms,
    llm_calls, tokens, cost_usd, per_question.
    """
    latencies, results = [], []
    tokens_total = calls_total = 0
    faithfulness_scores = []

    # Determine the model name for cost estimation (LLM strategies expose meta).
    # We read it from the first Answer's meta after the first call.
    model_for_cost = "unknown"

    for q in questions:
        t0 = time.perf_counter()
        ans = strategy.answer(q["question"])
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        checks_pass = all(run_check(c, ans.result)[0] for c in q["checks"])
        cited = len(ans.sources) > 0

        # Faithfulness (deterministic offline scorer).
        faith_result = faithfulness_checker.check(ans)
        faithfulness_scores.append(faith_result.score)

        # Token/cost accounting.
        q_tokens = ans.meta.get("tokens", 0)
        q_calls = ans.meta.get("llm_calls", 0)
        tokens_total += q_tokens
        calls_total += q_calls

        # Infer model from meta for cost lookup (use first non-empty provider).
        if model_for_cost == "unknown":
            provider = ans.meta.get("provider", "")
            if provider:
                model_for_cost = provider

        results.append({
            "id": q["id"],
            "tier": q.get("tier", ""),
            "passed": checks_pass and cited,
            "checks_pass": checks_pass,
            "cited": cited,
            "route": ans.route,
            "latency_ms": round(dt_ms, 2),
            "tokens": q_tokens,
            "faithfulness": round(faith_result.score, 4),
        })

    n = len(results)
    lat_sorted = sorted(latencies)

    # Compute cost for all tokens consumed by this strategy run.
    total_cost = cost_usd(model_for_cost, tokens_total)

    mean_faith = (
        round(statistics.mean(faithfulness_scores), 4) if faithfulness_scores else 0.0
    )

    return {
        "strategy": strategy.name,
        "total": n,
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": round(sum(r["passed"] for r in results) / n, 3) if n else 0.0,
        "citation_coverage": round(sum(1 for r in results if r["cited"]) / n, 3) if n else 0.0,
        "faithfulness_mean": mean_faith,
        "p50_latency_ms": round(_percentile(lat_sorted, 50), 2),
        "p95_latency_ms": round(_percentile(lat_sorted, 95), 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "llm_calls": calls_total,
        "tokens": tokens_total,
        "cost_usd": round(total_cost, 8),
        "per_question": results,
    }


# ---------------------------------------------------------------------------
# Injection-block-rate
# ---------------------------------------------------------------------------

def _compute_injection_block_rate(analytics, retrieval, corpus_path, injections_path):
    """Run the injection suite through GuardedStrategy and compute block rate.

    The inner strategy is RuleBasedStrategy (always available offline).
    'Blocked' means either the input guardrail fired (route contains
    'guardrail:input_block') OR the output guardrail fired
    ('guardrail:output_reject') — both constitute the system correctly
    handling the injection.

    Returns
    -------
    dict with keys: total, blocked, block_rate, per_entry.
    """
    if not injections_path.exists():
        return {"total": 0, "blocked": 0, "block_rate": 0.0, "per_entry": [],
                "note": f"injections file not found: {injections_path}"}

    with open(injections_path) as f:
        inj_data = json.load(f)
    entries = inj_data.get("entries", [])

    inner = RuleBasedStrategy(analytics, retrieval)
    guarded = GuardedStrategy(inner, corpus_path=corpus_path)

    per_entry = []
    for e in entries:
        inj_input = e["input"]
        expected = e.get("expected_behavior", "blocked")
        ans = guarded.answer(inj_input)

        # Determine whether the guardrail fired.
        guardrail_fired = ans.route in (
            "guardrail:input_block", "guardrail:output_reject"
        )
        # Also check the result dict for guardrail flags (belt and suspenders).
        if not guardrail_fired:
            guardrail_fired = bool(
                ans.result.get("blocked") or ans.result.get("rejected")
            )

        per_entry.append({
            "id": e["id"],
            "category": e.get("category", ""),
            "expected_behavior": expected,
            "guardrail_fired": guardrail_fired,
            "route": ans.route,
        })

    total = len(per_entry)
    blocked = sum(1 for r in per_entry if r["guardrail_fired"])
    block_rate = round(blocked / total, 4) if total else 0.0

    return {
        "total": total,
        "blocked": blocked,
        "block_rate": block_rate,
        "per_entry": per_entry,
    }


# ---------------------------------------------------------------------------
# Main comparison orchestrator
# ---------------------------------------------------------------------------

def compare(
    gold_dir=DEFAULT_GOLD,
    corpus_path=DEFAULT_CORPUS,
    golden_path=DEFAULT_GOLDEN,
    injections_path=DEFAULT_INJECTIONS,
    include_llm=True,
):
    """Run the full comparison.

    Returns
    -------
    (core_summaries, all_summaries, injection_result, notes)

    core_summaries:
        Summaries for deterministic strategies over CORE-tier questions only.
        This is the authoritative gate: accuracy==1.0 AND citation==1.0 required.
    all_summaries:
        Summaries for all strategies (deterministic + available LLM) over the
        full golden set (core + extended). Used for the advisory metrics table.
    injection_result:
        Block-rate dict from running the injection suite (ADVISORY).
    notes:
        List of human-readable notes (skipped strategies, etc.).
    """
    with open(golden_path) as f:
        golden_data = json.load(f)
    all_questions = golden_data.get("questions", [])
    core_questions = [q for q in all_questions if q.get("tier") == "core"]

    analytics = AnalyticsTool(gold_dir)
    retrieval = RetrievalTool(corpus_path)

    faith_checker = FaithfulnessChecker(
        corpus_path=corpus_path,
        gold_dir=gold_dir,
    )

    # -- Deterministic strategies (always run) --
    det_strategies = [
        RuleBasedStrategy(analytics, retrieval),
        KeywordScoreStrategy(analytics, retrieval),
    ]

    notes = []
    llm_strategies = []

    if include_llm:
        # OllamaStrategy
        ok, model, _ = OllamaStrategy.available()
        if ok:
            llm_strategies.append(OllamaStrategy(analytics, retrieval, model=model))
        else:
            notes.append(
                "ollama strategy skipped (no provider): "
                "server/model unavailable"
            )

        # SingleShotRAGStrategy
        if SingleShotRAGStrategy.is_available():
            llm_strategies.append(SingleShotRAGStrategy(analytics, retrieval))
        else:
            notes.append(
                "single_shot_rag strategy skipped (no provider): "
                "no OPENAI_API_KEY or OLLAMA_HOST reachable"
            )

        # ReActStrategy
        if ReActStrategy.is_available():
            llm_strategies.append(ReActStrategy(analytics, retrieval))
        else:
            notes.append(
                "react strategy skipped (no provider): "
                "no OPENAI_API_KEY or OLLAMA_HOST reachable"
            )

        # PlanExecuteStrategy
        if PlanExecuteStrategy.is_available():
            llm_strategies.append(PlanExecuteStrategy(analytics, retrieval))
        else:
            notes.append(
                "plan_execute strategy skipped (no provider): "
                "no OPENAI_API_KEY or OLLAMA_HOST reachable"
            )
    else:
        notes.append("LLM strategies skipped (--no-llm flag set)")

    # -- AUTHORITATIVE gate: deterministic strategies on CORE tier only --
    core_summaries = [
        _run_strategy(s, core_questions, faith_checker)
        for s in det_strategies
    ]

    # -- ADVISORY: all strategies on full golden set (core + extended) --
    all_summaries = [
        _run_strategy(s, all_questions, faith_checker)
        for s in det_strategies
    ]
    for s in llm_strategies:
        all_summaries.append(_run_strategy(s, all_questions, faith_checker))

    # -- ADVISORY: injection-block-rate --
    injection_result = _compute_injection_block_rate(
        analytics, retrieval, corpus_path, injections_path
    )

    return core_summaries, all_summaries, injection_result, notes


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt(val):
    """Format a value for the markdown table."""
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
    return str(val)


def _print_table(core_summaries, all_summaries, notes, injection_result):
    """Print comparison tables: authoritative columns and advisory columns.

    Parameters
    ----------
    core_summaries:
        Deterministic strategy summaries over CORE-tier questions only.
        Shown in the AUTHORITATIVE section.
    all_summaries:
        All strategy summaries over the full golden set (core + extended).
        Shown in the ADVISORY section.
    notes:
        Human-readable notes (skipped strategies, etc.).
    injection_result:
        Block-rate dict from the injection suite.
    """
    # -- AUTHORITATIVE section --
    print("\n" + "=" * 100)
    print("  AUTHORITATIVE GATE  |  Core-tier accuracy + citation (deterministic strategies only)")
    print("=" * 100)
    auth_cols = ["strategy", "total", "accuracy", "citation_coverage"]
    _print_md_table(core_summaries, auth_cols)

    # -- ADVISORY section --
    print("\n" + "-" * 100)
    print("  ADVISORY METRICS  |  All strategies, full golden set (core + extended)")
    print("-" * 100)
    advisory_cols = [
        "strategy", "total", "accuracy", "citation_coverage",
        "faithfulness_mean", "p50_latency_ms", "p95_latency_ms",
        "tokens", "cost_usd",
    ]
    _print_md_table(all_summaries, advisory_cols)

    # -- Injection-block-rate --
    print("\n" + "-" * 100)
    print("  ADVISORY: Injection-block-rate (GuardedStrategy + RuleBasedStrategy)")
    print("-" * 100)
    br = injection_result.get("block_rate", "N/A")
    blocked = injection_result.get("blocked", 0)
    total_inj = injection_result.get("total", 0)
    print(f"  block_rate = {br}  ({blocked}/{total_inj} injections handled)")
    if "note" in injection_result:
        print(f"  Note: {injection_result['note']}")

    # -- Notes --
    for n in notes:
        print(f"\n  Note: {n}")


def _print_md_table(summaries, cols):
    if not summaries:
        print("  (no rows)")
        return
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    print(header)
    print(sep)
    for s in summaries:
        row = "| " + " | ".join(_fmt(s.get(c, "")) for c in cols) + " |"
        print(row)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare AI answer strategies on the golden set (ai-07)"
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--injections", type=Path, default=DEFAULT_INJECTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip all LLM strategies (forces offline-only run)")
    args = parser.parse_args()

    core_summaries, all_summaries, injection_result, notes = compare(
        gold_dir=args.gold,
        corpus_path=args.corpus,
        golden_path=args.golden,
        injections_path=args.injections,
        include_llm=not args.no_llm,
    )

    _print_table(core_summaries, all_summaries, notes, injection_result)

    # Persist JSON output.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "gate": "authoritative",
        "core_gate": {
            "description": (
                "Deterministic strategies must score accuracy==1.0 and "
                "citation==1.0 on the CORE tier. Failure blocks CI."
            ),
            "summaries": core_summaries,
        },
        "advisory": {
            "description": (
                "All strategies over the full golden set. "
                "Faithfulness + injection-block-rate are measured and "
                "reported but never fail CI."
            ),
            "summaries": all_summaries,
            "injection_block_rate": injection_result,
        },
        "notes": notes,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nwrote comparison -> {args.out}")

    # -- CI GATE (AUTHORITATIVE) --
    # Deterministic strategies must pass ALL core-tier questions.
    # This is the only check that exits non-zero (blocks CI).
    gate_fail = False
    for s in core_summaries:
        if s["accuracy"] != 1.0 or s["citation_coverage"] != 1.0:
            print(
                f"\n[GATE FAIL] {s['strategy']}: "
                f"accuracy={s['accuracy']} citation={s['citation_coverage']} "
                f"on CORE tier ({s['passed']}/{s['total']})"
            )
            gate_fail = True

    if not gate_fail:
        print("\n[GATE PASS] All deterministic strategies score 1.0 accuracy + citation on CORE tier.")

    # Advisory metrics: report but never exit 1.
    print("\n[ADVISORY] Faithfulness and injection-block-rate are reported above.")
    print("[ADVISORY] These metrics do not affect CI gate status.")

    sys.exit(1 if gate_fail else 0)


if __name__ == "__main__":
    main()
