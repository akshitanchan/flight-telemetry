#!/usr/bin/env python3
"""
Benchmark runner for spatiotemporal index strategies.

Runs a generated query workload against an index and measures:
  - Query latency: p50, p95, p99
  - Ingest/build throughput
  - Index build time
  - Storage footprint
  - Result counts per query

Plan §5.1: "One command runs the benchmark end-to-end and emits a results table."
"""

import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from systems.index.base import SpatiotemporalIndex, SpatiotemporalQuery


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one index strategy."""
    strategy: str
    record_count: int
    num_queries: int
    query_profile: str
    # Build metrics
    build_time_s: float
    ingest_throughput_rps: float
    index_size_bytes: int
    # Query latency (seconds)
    query_p50_s: float
    query_p95_s: float
    query_p99_s: float
    query_mean_s: float
    query_min_s: float
    query_max_s: float
    # Result metrics
    result_count_mean: float
    result_count_max: int
    result_count_total: int
    # Index-specific stats
    index_stats: dict[str, Any]


def run_benchmark(
    index: SpatiotemporalIndex,
    records: list[dict],
    queries: list[SpatiotemporalQuery],
    query_profile: str = "unknown",
    warmup_queries: int = 5,
) -> BenchmarkResult:
    """Run a complete benchmark: build index, then run query workload.

    Args:
        index: The index strategy to benchmark.
        records: Silver-format records to index.
        queries: Query workload to execute.
        query_profile: Label for the query profile used.
        warmup_queries: Number of warmup queries before measurement.

    Returns:
        BenchmarkResult with all metrics.
    """
    # --- Build phase ---
    build_start = time.monotonic()
    index.build(records)
    build_elapsed = time.monotonic() - build_start

    idx_stats = index.stats()
    ingest_throughput = len(records) / build_elapsed if build_elapsed > 0 else 0

    # --- Warmup phase ---
    for q in queries[:warmup_queries]:
        index.query(q)

    # --- Measurement phase ---
    latencies = []
    result_counts = []

    for q in queries:
        q_start = time.perf_counter()
        results = index.query(q)
        q_elapsed = time.perf_counter() - q_start

        latencies.append(q_elapsed)
        result_counts.append(len(results))

    # --- Compute percentiles ---
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data: list[float], pct: float) -> float:
        """Compute percentile using nearest-rank method."""
        if not data:
            return 0.0
        k = max(0, min(len(data) - 1, int(len(data) * pct / 100)))
        return data[k]

    return BenchmarkResult(
        strategy=index.name,
        record_count=len(records),
        num_queries=len(queries),
        query_profile=query_profile,
        build_time_s=round(build_elapsed, 6),
        ingest_throughput_rps=round(ingest_throughput, 1),
        index_size_bytes=idx_stats.get("index_size_bytes", 0),
        query_p50_s=round(percentile(latencies_sorted, 50), 9),
        query_p95_s=round(percentile(latencies_sorted, 95), 9),
        query_p99_s=round(percentile(latencies_sorted, 99), 9),
        query_mean_s=round(statistics.mean(latencies), 9) if latencies else 0,
        query_min_s=round(min(latencies), 9) if latencies else 0,
        query_max_s=round(max(latencies), 9) if latencies else 0,
        result_count_mean=round(statistics.mean(result_counts), 1) if result_counts else 0,
        result_count_max=max(result_counts) if result_counts else 0,
        result_count_total=sum(result_counts),
        index_stats=idx_stats,
    )


def format_results_markdown(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a Markdown comparison table."""
    lines = [
        "## Spatiotemporal Index Benchmark Results",
        "",
        "| Metric | " + " | ".join(r.strategy for r in results) + " |",
        "|---|" + "|".join("---" for _ in results) + "|",
    ]

    def _row(label: str, accessor):
        vals = [str(accessor(r)) for r in results]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    _row("Records indexed", lambda r: f"{r.record_count:,}")
    _row("Build time (s)", lambda r: f"{r.build_time_s:.6f}")
    _row("Ingest throughput (rec/s)", lambda r: f"{r.ingest_throughput_rps:,.0f}")
    _row("Index size (bytes)", lambda r: f"{r.index_size_bytes:,}")
    _row("Queries executed", lambda r: f"{r.num_queries:,}")
    _row("Query profile", lambda r: r.query_profile)
    _row("**p50 latency (µs)**", lambda r: f"**{r.query_p50_s * 1e6:,.1f}**")
    _row("**p95 latency (µs)**", lambda r: f"**{r.query_p95_s * 1e6:,.1f}**")
    _row("**p99 latency (µs)**", lambda r: f"**{r.query_p99_s * 1e6:,.1f}**")
    _row("Mean latency (µs)", lambda r: f"{r.query_mean_s * 1e6:,.1f}")
    _row("Min latency (µs)", lambda r: f"{r.query_min_s * 1e6:,.1f}")
    _row("Max latency (µs)", lambda r: f"{r.query_max_s * 1e6:,.1f}")
    _row("Result count (mean)", lambda r: f"{r.result_count_mean:.1f}")
    _row("Result count (max)", lambda r: f"{r.result_count_max:,}")
    _row("Result count (total)", lambda r: f"{r.result_count_total:,}")

    return "\n".join(lines)


def format_results_json(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as JSON."""
    return json.dumps([asdict(r) for r in results], indent=2)


def format_results_csv(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as CSV."""
    if not results:
        return ""

    fields = [
        "strategy", "record_count", "num_queries", "query_profile",
        "build_time_s", "ingest_throughput_rps", "index_size_bytes",
        "query_p50_s", "query_p95_s", "query_p99_s",
        "query_mean_s", "query_min_s", "query_max_s",
        "result_count_mean", "result_count_max", "result_count_total",
    ]

    lines = [",".join(fields)]
    for r in results:
        d = asdict(r)
        lines.append(",".join(str(d.get(f, "")) for f in fields))

    return "\n".join(lines)
