# Systems Layer — ingestion + spatiotemporal index benchmark

**What this proves:** high-throughput replay ingestion with idempotent landing, and a
rigorous, reproducible benchmark comparing spatiotemporal index strategies with proper
tail-latency reporting.

## Metrics (measured, 50 records / 200 regional queries)

| Strategy | p50 | p95 | p99 | Ingest throughput |
|---|---|---|---|---|
| geohash-p3 | **35.1 µs** | 40.2 µs | 48.4 µs | **3.33M rec/s** |
| geohash-p5 | 19,062 µs | 62,308 µs | 102,207 µs | 2.59M rec/s |
| H3-r4 | 161.9 µs | 320.7 µs | 469.6 µs | 733k rec/s |
| H3-r5 | 849.8 µs | 1,674 µs | 2,380 µs | 1.60M rec/s |

**Finding:** geohash-p3 is the fastest baseline for coarse/continental queries but degrades
sharply with precision; H3 holds sub-ms tail latency at granular resolutions. **H3-r4** is the
balanced default. Full analysis in [docs/research-findings.md](../docs/research-findings.md).

## Run

```bash
make systems-replay-sample      # replay sample telemetry → landing JSONL (idempotent)
make systems-benchmark-small    # geohash vs H3 benchmark, p50/p95/p99
make test-systems               # unit tests
```

## Key files

- `replay/` — reader, normalizer, dedup (journal-backed), writer, CLI.
- `index/` — `base.py` (interface), `geohash_index.py`, `h3_index.py`, `workload.py`, `benchmark.py`, `cli.py`.

## Notes

Implemented in Python for the three-week build; the plan prefers Go/Rust for this layer (a Go/Rust
rewrite of the ingestion path is a tracked stretch item). See ADR-0002, ADR-0003.
