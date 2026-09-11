#!/usr/bin/env python3
"""Architecture x provider eval matrix.

Runs single_shot_rag, plan_execute, and langgraph_plan_execute against every
requested provider over the full golden set, each wrapped in GuardedStrategy
and fronted by a SemanticCache, and writes ai/eval/results.md and
ai/eval/results.json. Provider calls and retrieval lookups go through
ai.eval.cassette so the same command replays offline with --dry-run and
produces identical output, at zero cost and with no network access.

Usage:
    python -m ai.eval.matrix
    python -m ai.eval.matrix --dry-run
    python -m ai.eval.matrix --providers openai --limit 10
    make eval
    make eval DRY_RUN=1
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ai.providers as providers
from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.rule_based import RuleBasedStrategy
from ai.agent.single_shot_rag import SingleShotRAGStrategy
from ai.agent.plan_execute import PlanExecuteStrategy
# imported directly (not via ai.agent), same reasoning as ai.eval.compare: keep
# langgraph out of the ai.agent package import for callers that don't need it.
from ai.agent.langgraph_plan_execute import LangGraphPlanExecuteStrategy
from ai.guardrails import GuardedStrategy
from ai.obs.cache import SemanticCache
from ai.obs.costs import cost_split_usd
from ai.eval.checks import run_check
from ai.eval.cassette import Cassette, wrap_provider, wrap_retrieval
from ai.eval.compare import _percentile, compute_injection_metrics

DEFAULT_GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
DEFAULT_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"
DEFAULT_INJECTIONS = PROJECT_ROOT / "ai" / "fixtures" / "injections.json"
DEFAULT_FIXTURES_DIR = PROJECT_ROOT / "ai" / "fixtures" / "llm"
DEFAULT_OUT = PROJECT_ROOT / "ai" / "eval"

# architecture name -> (strategy class, max_tokens), matching what each
# strategy already uses via its own build_default() call.
ARCHITECTURES = {
    "single_shot_rag": (SingleShotRAGStrategy, 256),
    "plan_execute": (PlanExecuteStrategy, 512),
    "langgraph_plan_execute": (LangGraphPlanExecuteStrategy, 512),
}
ALL_PROVIDERS = ["openai", "bedrock"]

# mirrors the input/output $-per-1000-tokens pairs in ai/obs/costs.py; kept as
# a literal here so the header can cite the date they were checked without
# reaching into that module's private price table.
PRICE_CHECKED_ON = "2026-09-10"
PRICE_PER_1K = {
    "openai": (0.000150, 0.000600),
    "bedrock": (0.00022, 0.00022),
}

HOST_STRING = f"Apple M2, macOS, Python {sys.version_info.major}.{sys.version_info.minor} in ~/.venvs/flight-telemetry"


def _duplicate_count(questions):
    # extra occurrences of an exact-duplicate question string in this question list
    distinct = {q["question"] for q in questions}
    return len(questions) - len(distinct)


def _duplicate_sentence(count):
    if count == 0:
        return "No question in the set is an exact duplicate."
    if count == 1:
        return ("1 question in the golden set is an exact duplicate and was served from "
                "the semantic cache on its second occurrence at zero provider cost.")
    return (f"{count} questions in the golden set are exact duplicates and were served from "
            "the semantic cache on their second occurrence at zero provider cost.")


# stand-in for a live provider when replaying from a cassette, whether that's
# --dry-run or a recording run with no live credentials; never actually invoked,
# since every question must be served by a cassette hit or CassetteMiss fires first
class _CassetteProviderIdentity:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.label = f"{name}:{model}"

    def __call__(self, messages):
        raise RuntimeError(
            f"cassette stub for {self.label} was invoked directly; "
            "the cassette should have raised CassetteMiss first"
        )


def _resolve_provider(cassette, name, max_tokens, dry_run):
    # returns (identity, reason) with exactly one set; a missing-credentials
    # RuntimeError becomes the reason string so the caller can record it and continue
    if dry_run:
        model = cassette.meta.get("model")
        if not model:
            raise RuntimeError(
                f"{cassette.path} has no recorded provider/model in its meta; "
                "record a live cassette for this provider before replaying"
            )
        recorded_name = cassette.meta.get("provider", name)
        return _CassetteProviderIdentity(recorded_name, model), None
    try:
        provider = providers.build(name, max_tokens=max_tokens)
    except RuntimeError as exc:
        # no live credentials: fall back to the identity this cassette already
        # recorded, same stand-in as --dry-run, so a recording run over a fully
        # cached cassette still works (e.g. replaying synthetic fixtures in tests)
        model = cassette.meta.get("model")
        if not model:
            return None, str(exc)
        recorded_name = cassette.meta.get("provider", name)
        return _CassetteProviderIdentity(recorded_name, model), None
    cassette.meta = {"provider": provider.name, "model": provider.model}
    return provider, None


def _run_cell(strategy_cls, max_tokens, provider_name, cassette, retrieval, analytics, questions, dry_run):
    # on a missing provider this returns {"unavailable": reason} instead of the metrics dict below
    identity, reason = _resolve_provider(cassette, provider_name, max_tokens, dry_run)
    if identity is None:
        return {"unavailable": reason}

    llm = wrap_provider(identity, cassette)
    strategy = strategy_cls(analytics, retrieval, llm=llm)
    guarded = GuardedStrategy(strategy, corpus_path=DEFAULT_CORPUS)
    cache = SemanticCache()

    rows = []
    for q in questions:
        start = len(llm.calls)
        cached = cache.get(q["question"])
        answer = cached if cached is not None else guarded.answer(q["question"])
        if cached is None:
            cache.put(q["question"], answer)
        new_calls = llm.calls[start:]

        in_tok = sum(c["input_tokens"] for c in new_calls)
        out_tok = sum(c["output_tokens"] for c in new_calls)
        latency_s = sum(c["latency_s"] for c in new_calls)
        cost_usd = cost_split_usd(llm.label, in_tok, out_tok)

        checks_ok = all(run_check(chk, answer.result)[0] for chk in q["checks"])
        cited = bool(answer.sources)
        rows.append({
            "passed": checks_ok and cited,
            "cited": cited,
            "cost_usd": cost_usd,
            "latency_s": latency_s,
        })

    hits = cache.stats()["hits"]
    expected_hits = _duplicate_count(questions)
    if hits != expected_hits:
        raise RuntimeError(
            f"{strategy_cls.name}/{provider_name}: cache reported {hits} hit(s), "
            f"expected {expected_hits} exact-duplicate hit(s) among {len(questions)} "
            "question(s); this points to a keying bug that would understate cost and latency"
        )

    n = len(rows)
    latencies = sorted(r["latency_s"] for r in rows)
    return {
        "n": n,
        "accuracy": round(sum(r["passed"] for r in rows) / n, 4) if n else 0.0,
        "citation_rate": round(sum(r["cited"] for r in rows) / n, 4) if n else 0.0,
        "cost_per_query_usd": round(sum(r["cost_usd"] for r in rows) / n, 8) if n else 0.0,
        "p50_latency_s": round(_percentile(latencies, 50), 4),
        "cache_hits": hits,
        "provider_label": llm.label,
    }


def _run_reference(retrieval, analytics, questions):
    # rule_based_v1 over the same questions: no llm, no cost, no latency claim
    strategy = RuleBasedStrategy(analytics, retrieval)
    passed = cited = 0
    for q in questions:
        answer = strategy.answer(q["question"])
        if all(run_check(chk, answer.result)[0] for chk in q["checks"]) and answer.sources:
            passed += 1
        if answer.sources:
            cited += 1
    n = len(questions)
    return {
        "strategy": "rule_based_v1",
        "n": n,
        "accuracy": round(passed / n, 4) if n else 0.0,
        "citation_rate": round(cited / n, 4) if n else 0.0,
        "cost_usd": 0.0,
    }


def _run_injection(provider_name, cassette, retrieval, analytics, corpus_path, injections_path, dry_run):
    identity, reason = _resolve_provider(cassette, provider_name, 256, dry_run)
    if identity is None:
        return {"unavailable": reason}
    llm = wrap_provider(identity, cassette)
    inner = SingleShotRAGStrategy(analytics, retrieval, llm=llm)
    result = compute_injection_metrics(analytics, retrieval, corpus_path, injections_path, inner=inner)
    blocked_at_input = sum(1 for e in result["per_entry"] if e["input_blocked"])
    rejected_at_output = sum(1 for e in result["per_entry"] if e["output_rejected"])
    return {
        "probes": result["total"],
        "blocked_at_input": blocked_at_input,
        "rejected_at_output": rejected_at_output,
        "block_rate": result["active_block_rate"],
    }


def _fmt_frac(val):
    return f"{val:.3f}"


def _fmt_cost(val):
    return f"${val:.6f}"


def _fmt_latency(val):
    return f"{val:.3f}"


def _cell_columns(cell):
    # four markdown cell strings, in header order: accuracy, citation, cost, p50
    if "unavailable" in cell:
        return [cell["unavailable"], "-", "-", "-"]
    return [
        _fmt_frac(cell["accuracy"]),
        _fmt_frac(cell["citation_rate"]),
        _fmt_cost(cell["cost_per_query_usd"]),
        _fmt_latency(cell["p50_latency_s"]),
    ]


def _model_line(provider_name, results, provider_names):
    for arch_name in ARCHITECTURES:
        cell = results[arch_name][provider_name]
        if "provider_label" in cell:
            return cell["provider_label"]
    # every architecture failed to resolve this provider; fall back to the note
    for arch_name in ARCHITECTURES:
        cell = results[arch_name][provider_name]
        if "unavailable" in cell:
            return f"{provider_name}: unavailable ({cell['unavailable']})"
    return provider_name


def _recorded_at_line(cassettes):
    dates = {name: c.recorded_at for name, c in cassettes.items()}
    distinct = sorted(set(dates.values()))
    if len(distinct) == 1:
        return distinct[0]
    return ", ".join(f"{name} {when}" for name, when in sorted(dates.items()))


def _render_markdown(command, cassettes, golden_size, used_size, backend,
                      provider_names, results, reference, injection, duplicate_note):
    lines = []
    lines.append("# AI architecture x provider eval matrix")
    lines.append("")
    lines.append(f"Generated with: `{command}`")
    lines.append(f"Cassettes recorded at: {_recorded_at_line(cassettes)}.")
    lines.append(f"Host: {HOST_STRING}.")
    lines.append(f"Golden set has {golden_size} questions; this run used {used_size}.")
    lines.append(duplicate_note)
    lines.append(f"Retrieval backend actually used: {backend}.")
    for name in provider_names:
        lines.append(f"Model ({name}): {_model_line(name, results, provider_names)}.")
    for name in provider_names:
        if name in PRICE_PER_1K:
            in_price, out_price = PRICE_PER_1K[name]
            lines.append(
                f"Price ({name}), checked {PRICE_CHECKED_ON}: "
                f"${in_price:.6f}/1K input tokens, ${out_price:.6f}/1K output tokens."
            )
    lines.append(
        "p50 latency covers provider call latency only; it excludes local tool "
        "and guardrail time."
    )
    lines.append("")

    header = ["architecture"]
    for name in provider_names:
        header += [f"{name} accuracy", f"{name} citation rate", f"{name} cost/query", f"{name} p50 (s)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(" --- " for _ in header) + "|")
    for arch_name in ARCHITECTURES:
        row = [arch_name]
        for name in provider_names:
            row += _cell_columns(results[arch_name][name])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(
        f"Reference: rule_based_v1 (no LLM) scores accuracy {_fmt_frac(reference['accuracy'])} "
        f"and citation rate {_fmt_frac(reference['citation_rate'])} over the same "
        f"{reference['n']} questions, at $0.00 cost. No latency claim is made since "
        "this strategy never calls a provider."
    )
    lines.append("")

    lines.append("## Injection probes")
    lines.append("")
    lines.append(
        "The input guard fires before any model call, so per-provider "
        "differences below can only come from the output guard."
    )
    lines.append("")
    inj_header = ["provider", "probes", "blocked at input", "rejected at output", "block rate"]
    lines.append("| " + " | ".join(inj_header) + " |")
    lines.append("|" + "|".join(" --- " for _ in inj_header) + "|")
    for name in provider_names:
        cell = injection[name]
        if "unavailable" in cell:
            lines.append(f"| {name} | {cell['unavailable']} | - | - | - |")
        else:
            lines.append(
                f"| {name} | {cell['probes']} | {cell['blocked_at_input']} | "
                f"{cell['rejected_at_output']} | {_fmt_frac(cell['block_rate'])} |"
            )
    lines.append("")

    return "\n".join(lines)


def _canonical_command(provider_names, limit):
    # the header must record how to reproduce this file, not the literal argv,
    # so a --dry-run replay into a different --out still matches the live run
    # byte for byte; --dry-run, --out and --fixtures-dir are execution details
    # and never appear here
    flags = []
    if provider_names != ALL_PROVIDERS:
        flags.append(f"--providers {','.join(provider_names)}")
    if limit is not None:
        flags.append(f"--limit {limit}")
    if not flags:
        return "make eval"
    return f'make eval EVAL_ARGS="{" ".join(flags)}"'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the architecture x provider eval matrix"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="replay recorded cassettes; no network, no spend")
    parser.add_argument("--providers", default=",".join(ALL_PROVIDERS),
                        help="comma-separated provider names (default: openai,bedrock)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N golden-set questions")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory to write results.md and results.json into")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR,
                        help="directory holding the per-provider and retrieval cassettes")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--injections", type=Path, default=DEFAULT_INJECTIONS)
    args = parser.parse_args(argv)

    if args.dry_run:
        for var in ("OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            os.environ.pop(var, None)

    provider_names = [p.strip() for p in args.providers.split(",") if p.strip()]

    with open(args.golden) as f:
        all_questions = json.load(f).get("questions", [])
    questions = all_questions[: args.limit] if args.limit else all_questions
    duplicate_count = _duplicate_count(questions)
    duplicate_note = _duplicate_sentence(duplicate_count)

    analytics = AnalyticsTool(args.gold)
    raw_retrieval = RetrievalTool(args.corpus)
    backend = "pgvector" if raw_retrieval.vector_backend_available() else "keyword fallback"

    args.fixtures_dir.mkdir(parents=True, exist_ok=True)
    retrieval_cassette = Cassette(args.fixtures_dir / "retrieval.json", record=not args.dry_run)
    retrieval = wrap_retrieval(raw_retrieval, retrieval_cassette)

    provider_cassettes = {
        name: Cassette(args.fixtures_dir / f"{name}.json", record=not args.dry_run)
        for name in provider_names
    }

    results = {arch_name: {} for arch_name in ARCHITECTURES}
    for arch_name, (strategy_cls, max_tokens) in ARCHITECTURES.items():
        for name in provider_names:
            results[arch_name][name] = _run_cell(
                strategy_cls, max_tokens, name, provider_cassettes[name],
                retrieval, analytics, questions, args.dry_run,
            )

    reference = _run_reference(retrieval, analytics, questions)

    injection = {
        name: _run_injection(
            name, provider_cassettes[name], retrieval, analytics,
            args.corpus, args.injections, args.dry_run,
        )
        for name in provider_names
    }

    command = _canonical_command(provider_names, args.limit)
    all_cassettes = dict(provider_cassettes, retrieval=retrieval_cassette)

    args.out.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(
        command, all_cassettes, len(all_questions), len(questions), backend,
        provider_names, results, reference, injection, duplicate_note,
    )
    (args.out / "results.md").write_text(markdown)

    output = {
        "header": {
            "command": command,
            "recorded_at": {name: c.recorded_at for name, c in all_cassettes.items()},
            "host": HOST_STRING,
            "golden_set_size": len(all_questions),
            "questions_used": len(questions),
            "duplicate_questions": {"count": duplicate_count, "note": duplicate_note},
            "retrieval_backend": backend,
            "models": {name: _model_line(name, results, provider_names) for name in provider_names},
            "price_per_1k_usd": {
                "checked": PRICE_CHECKED_ON,
                **{name: PRICE_PER_1K[name] for name in provider_names if name in PRICE_PER_1K},
            },
            "latency_note": (
                "p50 latency covers provider call latency only; it excludes "
                "local tool and guardrail time."
            ),
        },
        "architectures": results,
        "reference": reference,
        "injection": injection,
    }
    with open(args.out / "results.json", "w") as f:
        json.dump(output, f, sort_keys=True, indent=2)

    print(f"wrote {args.out / 'results.md'}")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
