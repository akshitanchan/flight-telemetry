# Flight Telemetry Intelligence Platform

A four-layer platform over aircraft telemetry — **Systems**, **Data**, **ML**, and **AI** —
sharing one data spine (`silver_flight_state`). Each layer is a standalone flagship with a
stated research question and a measured answer; every layer runs locally on bounded sample
data from a single `make` target.

## Metrics at a glance

*Measured locally on 2026-06-11. Reproduce with the command in each row.*

| Layer | Headline measured result | Research question → answer | Reproduce |
|---|---|---|---|
| **Systems** | At 1M rows: geohash-p4 p50 **35.5 ms**, H3-r4 **30.4 ms**, PostGIS GIST **257.9 ms** | Which index wins this regional workload? -> **H3-r4 has the lowest measured latency; PostGIS provides persistence, not a latency win here** | `python -m systems.index.cli --mode postgres --synthetic-rows 1000000 --queries 500 --profile regional` |
| **Data** | DuckDB proxy: incremental `MERGE` **1.5x-2.3x** faster at 10k-200k rows | Where is the incremental-vs-recompute frontier? -> **directional proxy only; real Delta timing pending** | `make data-refresh-experiment-small` |
| **ML** | full-data four-fold RMSE: HistGBR **142.20 +/- 22.68 kg**, MLP **355.04 +/- 89.12 kg** | Which model wins on the extracted 11,037-flight set? -> **HistGBR by 212.83 kg RMSE; cloud training and rank scoring remain** | `python -m ml.ablation --data-dir data/raw/prc_2025 --epochs 10 --n-folds 4 --model histgbr` |
| **AI** | Core deterministic **100%/100%**; local `llama3.2` extended-tier accuracy: single-shot **36%**, ReAct **20%**, plan-execute **52%** | Which live architecture wins here? -> **plan-execute on accuracy; single-shot on latency** | `python -m ai.eval.compare` |
| **Cross-layer** | `make smoke` **7/7** steps in **13.21 s**, fully offline | Does the bounded platform run end-to-end from one command? → **yes** | `make smoke` |

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
make smoke               # run every layer end-to-end on bounded data (~13s, offline)
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
- **Data** has Databricks serverless and BigQuery/dbt deployment artifacts, but the real cloud row counts, Lakeflow metrics, Delta timings, and bytes-scanned captures are still pending.
- **ML** has an honest bounded CV baseline and a measured 11,037-flight
  ablation/challenger result. Databricks full-scale training and held-out rank
  scoring remain pending. The JOAS paper's 201 kg winning score is from the
  separate final phase, not a same-split rank baseline.
- **AI** live architecture numbers are local `llama3.2:latest` measurements, not a provider-independent ranking. CI still skips unavailable providers.
- **Serving** defaults to configurable fake mode for offline smoke tests. Real serving requires `ML_FAKE_MODE=false` plus a model path or registry URI.

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
