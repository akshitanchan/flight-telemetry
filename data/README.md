# Data Layer — medallion pipeline + refresh experiment

**What this proves:** a contract-driven bronze→silver→gold pipeline with data-quality checks,
and a measured cost/latency frontier for incremental vs full-recompute aggregation.

## Metrics (measured)

| Refresh strategy (DuckDB) | 10k rows | 100k rows | 1M rows |
|---|---|---|---|
| Full recompute | 0.0039 s | 0.0070 s | 0.0297 s |
| Incremental `MERGE` | 0.0041 s | 0.0031 s | 0.0118 s |
| **Speedup** | 0.95× | 2.25× | **2.51×** |

**Finding:** `MERGE` overtakes full recompute past ~100k rows (2.5× at 1M). Gold outputs on the
sample: **4 tables** — airport congestion (11 real airports, 0 `UNKNOWN`), sector load (13),
emergency events (3), routing stats (10). Full analysis in
[docs/research-findings.md](../docs/research-findings.md).

## Run

```bash
make data-local-silver              # bronze landing → silver_flight_state (clean, dedup, enrich)
make data-local-gold                # silver → gold aggregates
make data-refresh-experiment-small  # incremental MERGE vs full recompute (DuckDB)
make test-data                      # unit tests (transforms, dedup, aggregates)
```

## Key files

- `transforms/` — `bronze_to_silver.py` (type/range checks, geohash7 + H3-r7, nearest-airport
  enrichment within 50 km), `silver_to_gold.py` (4 gold aggregates), CLIs.
- `experiments/refresh_strategy.py` — the DuckDB `MERGE`-vs-recompute harness.
- `reference/airports.json` — bounded 1,189-airport European subset (offline fixture; canonical
  source is OurAirports). See ADR-0006.

## Notes

Local Python medallion + DuckDB stands in for cloud Databricks/Delta + BigQuery (ADR-0004);
numbers are local proxies, not cloud cost figures.
