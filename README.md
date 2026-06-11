# Flight Telemetry

A four-layer data platform built on a single aircraft-tracking dataset: systems, data engineering, machine learning, and an LLM agent, all sharing one schema so the layers feed each other instead of standing alone. The data is aircraft position reports from OpenSky plus the Eurocontrol PRC-2025 fuel challenge.

The whole platform runs offline from a single `make smoke` in about twelve seconds on bounded sample data. The cloud paths (BigQuery, PostGIS, live LLM providers) activate when credentials are present; everything else works from a clean clone.

Every number below is measured. Work that is built but not yet executed at full scale is labelled as such, and proxies are called out where they stand in for the real thing.

## Architecture

```
OpenSky / PRC-2025
        │
        ▼
   Systems   ingest + spatiotemporal index
        │
        ▼
   Data      clean, then aggregate into gold tables
        │
        ├────────►  ML        fuel-burn model + serving API
        │
        └────────►  AI agent  grounded question answering
                        │
                        ▼
                   Dashboard  Streamlit, reads across every layer
```

Ingestion lands raw state vectors. The data layer cleans and aggregates them through a layered pipeline. ML trains on the aggregated features; the AI agent answers questions against the same tables. A Streamlit dashboard reads across all of it. Six schema contracts hold the seams together and are validated on every run.

## Systems: ingestion and a spatiotemporal index benchmark

A replay ingestion path lands aircraft state vectors idempotently (dedup on `icao24 + event_ts`, so re-running a feed never double-writes). On top of it sits a benchmark that pits three ways of answering "which aircraft were in this box during this time window" against each other at one million rows.

| Index | p50 | p95 | build | size |
|---|---|---|---|---|
| geohash prefix (p4) | 33.6 ms | 45.1 ms | 0.21 s | 9.8 MB |
| H3 (r4) | 30.2 ms | 32.5 ms | 0.38 s | 8.9 MB |
| PostGIS GIST | 260.0 ms | 308.5 ms | 49.8 s | 609 MB |

The in-process indexes are far faster than the database. H3 answers in about 30 ms; the same query through a PostGIS GIST index takes roughly 260 ms once the round trip and the on-disk geography index are paid for. PostGIS buys persistence and real SQL, not latency, for this access pattern. All three return the same ~3,800 rows per query, so the comparison is fair.

## Data: a layered pipeline, local and on the cloud

Raw landings are cleaned into deduplicated, geo-enriched state vectors (the silver stage), then rolled up into four aggregates (the gold stage): airport congestion, sector load, emergency events, and routing stats. The schema contract is checked at every hop. The research question is where an incremental `MERGE` beats a full recompute; on the local DuckDB harness the crossover sits around 100k rows, reaching roughly 2.5x at one million.

Two implementations exist. The local Python and DuckDB version runs offline. The cloud version is split:

- **BigQuery (deployed):** dbt builds four incremental, partitioned, clustered marts; 58 data tests pass against live BigQuery.
- **Databricks Delta (built, runbooked):** the same pipeline on Delta tables, with declarative data-quality checks and a MERGE-vs-recompute experiment hardened for overlapping batches. Written and offline-verified, but left as a runbook rather than spending Free-Edition compute to re-prove what the BigQuery side already shows.

One honest wrinkle: the BigQuery partition-pruning demo came out degenerate. The sample gold is dated 2024 and the marts carry a 60-day partition expiry, so the partitions expired the moment they loaded. The partition and clustering design is correct; the demo data is simply too old to show it off.

## ML: fuel-burn estimation on PRC-2025

The task is to predict kilograms of fuel burned per flight interval from the trajectory and the aircraft type. Thirty-eight features, the single strongest being aircraft type.

The evaluation is built to be trusted. An early version scored a suspiciously good ~395 kg; the cause was leakage, with intervals from the same flight landing on both sides of the train/validation split. Grouping every flight onto one side and switching to a chronological, expanding-window CV moved the honest number to 442 kg. From there a gradient-boosted model clearly beats the MLP:

| Model | full-data CV | held-out rank phase |
|---|---|---|
| HistGBR | 142.2 ± 22.7 kg | 248.6 kg |
| MLP | 355.0 ± 89.1 kg | 411.8 kg |

CV runs over all 11,037 training flights; the rank-phase column is the official held-out set (24,289 intervals across 1,888 separate flights). The held-out number sitting above CV is exactly what a real generalization test should do.

The model is wrapped in production scaffolding: MLflow tracking, a model registry with champion/challenger promotion, an Evidently drift monitor that triggers a retrain when feature distributions shift, and a FastAPI serving endpoint comfortably under its latency budget. The full-scale Databricks training run is set up and offline-validated but, like the Delta pipeline, left as a runbook, since the full-data ablation already settles which model wins.

## AI: an agent that can be graded

The focus of this layer is evaluation, not a chat box. One pattern holds throughout: the LLM only decides which tool to call, a validated tool computes the answer, and every answer cites its source. That makes answers checkable. A golden set of 104 questions splits into a deterministic core (the rule-based router gets it 100% right, and that is the gate that blocks CI) and a harder extended tier where the language models are meant to earn their place.

They mostly do not. Three agent architectures run live against gpt-4o-mini:

| Architecture | accuracy | citation | cost |
|---|---|---|---|
| single-shot RAG | 0.76 | 1.00 | $0.0015 |
| plan-execute | 0.60 | 1.00 | $0.0023 |
| ReAct | 0.20 | 0.40 | $0.0074 |

The elaborate ReAct loop finishes last by a wide margin: 20% accuracy, only 40% of answers grounded in a real source, at five times the token cost. A single retrieval-augmented call beats all of it. The deterministic router still outscores every LLM on the full set; the models only pull ahead on genuinely open-ended phrasings.

The rest of the layer is supporting cast: pgvector retrieval with a keyword fallback for offline use, input and output guardrails, a 25-probe prompt-injection suite, token-cost accounting, and a semantic answer cache.

## Running it

```bash
cp .env.example .env   # only needed for live providers or cloud
make setup
make smoke             # every layer, end to end, offline, ~12s
make help              # everything else
```

Specific entrypoints:

- `make systems-benchmark-small` — the index benchmark
- `make data-local-gold` — the data transforms
- `make ml-serve-smoke` — the serving endpoint
- `make ai-compare-small` — the agent comparison (LLM legs skip themselves if no provider is configured)
- `make test-systems` / `test-data` / `test-ml` / `test-ai` / `test-cloud` / `test-dashboard` — the per-layer suites

A Streamlit dashboard (`streamlit run dashboard/app.py`) provides an operational view over the gold tables, an ask-the-agent box, and a platform-health panel.

## Limitations

- Systems is Python. A production hot path would be a Go or Rust rewrite; that is out of scope here.
- The Databricks Delta pipeline and the full-scale ML training are built and runbooked but not cloud-run, by design: the BigQuery work already demonstrates the cloud data layer, and the local full-data ablation already settles the model question.
- The PRC challenge's published winning score comes from a different phase, so it is not a like-for-like baseline. The rank number here is reported on its own rather than framed as a win it cannot support.

## Stack

Python throughout. PyTorch, scikit-learn, Evidently, and MLflow for ML; FastAPI for serving; DuckDB, dbt, and Delta for data; PostGIS, pgvector, and H3 for spatial indexing and retrieval; Streamlit for the dashboard; OpenTelemetry and Prometheus for observability; Docker Compose for the local stack.

## License

Personal and academic use. The data sources retain their own licenses (OpenSky, Eurocontrol PRC-2025, OurAirports).
