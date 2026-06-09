# Flight Telemetry Intelligence Platform

A four-layer platform over aircraft telemetry — **Systems**, **Data**, **ML**, and **AI** —
sharing one data spine (`silver_flight_state`). Each layer is a standalone flagship with a
stated research question and a measured answer; every layer runs locally on bounded sample
data from a single `make` target.

## Metrics at a glance

*Measured locally on bounded sample data (2026-06-10). Reproduce with the `make` target in each row.*

| Layer | Headline measured result | Research question → answer | Reproduce |
|---|---|---|---|
| **Systems** | geohash-p3 p50 **35 µs** vs H3-r4 p50 **162 µs**; ingest up to **3.3M rec/s** | Which spatiotemporal index minimizes tail latency? → geohash-p3 for coarse/continental, **H3-r4 best balanced** tail latency | `make systems-benchmark-small` |
| **Data** | incremental `MERGE` **2.5×** faster than full recompute at 1M rows | Where is the incremental-vs-recompute frontier? → **`MERGE` wins past ~100k rows** | `make data-refresh-experiment-small` |
| **ML** | **real PRC-2025 baseline: val RMSE ~395 kg** (real fuel labels, 500-flight subset); serve smoke **2/2** | Can the model match the published PRC-2025 baseline? → **real-data baseline trained; full-scale + leaderboard scoring pending** | `make ml-baseline-real` |
| **AI** | deterministic routers **100%** vs local 7B LLM **92.3%**; citation coverage **100%** | Which agent architecture wins on faithfulness/cost? → **deterministic tool-routing dominates** here | `make ai-compare-small` |
| **Cross-layer** | `make smoke` **7/7** steps in **~7.5 s**, fully offline | Does the bounded platform run end-to-end from one command? → **yes** | `make smoke` |

## Architecture

Four layers over a shared `silver_flight_state` contract; see **[docs/architecture.md](docs/architecture.md)**
for the full as-built diagram and a plan-vs-reality table.

```
ingestion (systems) → landing → bronze→silver→gold (data) → ┬→ ML (PRC fuel-burn)
                                                            └→ AI (eval-driven agent over gold + corpus)
```

## What each layer proves

- **[systems/](systems/README.md)** — high-throughput replay ingestion (idempotent on `icao24+event_ts`) and a rigorous spatiotemporal index benchmark (geohash vs H3) with p50/p95/p99 reporting.
- **[data/](data/README.md)** — a local medallion pipeline (bronze→silver→gold) with DQ checks, plus a DuckDB incremental-`MERGE`-vs-recompute experiment.
- **[ml/](ml/README.md)** — a PRC-2025 fuel-burn baseline wrapped in production scaffolding (feature extraction, MLflow tracking with best-model logging, FastAPI serving).
- **[ai/](ai/README.md)** — an eval-first operational agent: validated tools over the gold tables + retrieval, a golden set with deterministic checks, and three answer strategies compared head-to-head.

## Quick start

```bash
cp .env.example .env     # optional; only needed for live API / MLflow server
make setup               # create local output directories
make smoke               # run every layer end-to-end on bounded data (~7.5s, offline)
make help                # list all targets
```

## One-command entrypoints

| Target | What it does |
|---|---|
| `make smoke` | cross-layer smoke test (all layers, bounded, offline) |
| `make systems-benchmark-small` | geohash vs H3 spatiotemporal index benchmark |
| `make data-local-gold` | bronze→silver→gold transforms on the sample |
| `make data-refresh-experiment-small` | incremental `MERGE` vs full recompute (DuckDB) |
| `make ml-baseline-small` / `make ml-serve-smoke` | train the fuel-burn baseline / serving smoke |
| `make ai-eval-small` / `make ai-compare-small` | evaluate / compare AI answer strategies |
| `make test-data` / `test-systems` / `test-ai` | per-layer test suites |

## Limitations

This is a three-week bounded build; the honest edges:

- **Systems** is implemented in Python (the plan prefers Go/Rust for a stronger backend signal); a rewrite is a stretch item.
- **Data** runs a local medallion + DuckDB experiment rather than cloud Databricks/Delta + BigQuery; numbers are local proxies, not cloud cost figures.
- **ML** now trains on the **real PRC-2025 data** (`make ml-baseline-real`, bounded to a 500-flight subset; val RMSE ~395 kg). Full-scale training over all 11,037 flights and the official leaderboard scoring are still pending — the model is a baseline MLP, not yet competitive with the published leaderboard — and there is no drift/retrain loop yet.
- **AI** numbers are on a small golden set; the LLM strategy needs a local Ollama server (availability-gated, skipped in CI). The LLM comparison figures are a single local `temperature=0` run.
- ML serving still uses a deprecated FastAPI startup hook and a hardcoded fake-model mode (tracked for cleanup).

## Documentation

| Document | Purpose |
|---|---|
| [docs/architecture.md](docs/architecture.md) | As-built architecture + plan-vs-reality |
| [docs/decisions.md](docs/decisions.md) | ADR-style log of every non-trivial tradeoff |
| [docs/research-findings.md](docs/research-findings.md) | The four research questions + measured answers |
| [docs/plan.md](docs/plan.md) | Original implementation plan (intent) |
| [docs/execution.md](docs/execution.md) · [docs/checklist.md](docs/checklist.md) | Execution discipline + progress ledger |

## License

For personal and academic use only. See data-source licenses in [docs/plan.md](docs/plan.md).
