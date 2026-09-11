# Flight Telemetry

A four-layer data platform built on a single aircraft-tracking dataset: systems, data engineering, machine learning, and an LLM agent, all sharing one schema so the layers feed each other. The data is aircraft position reports from OpenSky plus the Eurocontrol PRC-2025 fuel challenge. The whole platform runs offline from `make smoke` in about twelve seconds. The cloud paths (BigQuery, Snowflake, live LLM providers) activate when credentials from .env.example are present; everything else works from a clean clone.

Six schema contracts hold the seams together and are validated on every run. Every number below is measured.

## Running it

```bash
cp .env.example .env
make setup
make smoke              # every layer, end to end, offline, ~12s
make help
```

Specific paths: `make airflow-test` and `cd airflow && astro dev start` for the DAG (see airflow/README.md). Cloud data: `dbt build --target bigquery` and `dbt build --target snowflake` from data/cloud/dbt/flight_telemetry.

## Measured results

Spatiotemporal index benchmark on one million rows: geohash prefix (p4) answers in 33.6 ms at p50 and 45.1 ms at p95, builds in 0.21 s and takes 9.8 MB; H3 (r4) answers in 30.2 ms at p50 and 32.5 ms at p95, builds in 0.38 s and takes 8.9 MB; PostGIS GIST answers in 260.0 ms at p50 and 308.5 ms at p95, builds in 49.8 s and takes 609 MB. All three return roughly 3,800 rows per query, so the comparison is fair.

Fuel burn: an early model scored a suspiciously good 395 kg, which turned out to be leakage between train and validation flights; the honest baseline is 442 kg. HistGBR scores 142.2 ± 22.7 kg on chronological CV over 11,037 training flights and 248.6 kg on the held-out rank phase (24,289 intervals, 1,888 flights); the MLP scores 355.0 ± 89.1 kg and 411.8 kg on the same splits.

Refresh benchmark, full recompute against an incremental MERGE of a 10 percent batch at one million rows: 3.0x on the local DuckDB harness and 0.98x on a Snowflake X-Small warehouse, both measured on 2026-09-11. Both targets pass `dbt build` with 63 nodes (4 models, 54 data tests, 5 unit tests); see bench/results.md for commands and hardware.

AI agent, full 104-question golden set, recorded on 2026-09-11 and replayed offline by `make eval DRY_RUN=1`: single-shot RAG scores 0.942 accuracy and 0.971 citation on gpt-4o-mini against 0.712 and 0.990 on Llama 3.1 8B through Bedrock; plan-execute scores 0.827 and 0.913 against 0.721 and 0.933, and the LangGraph version of plan-execute reproduces those numbers exactly because it sends the same prompts to the same tools; the deterministic router scores 0.904 with full citation at no cost, and the 25-probe injection suite blocks 0.84 on OpenAI and 0.80 on Bedrock. Cost per query, p50 latency, and the per-provider injection table are in ai/eval/results.md, and CI fails if any cell drops below ai/eval/baseline.json. An earlier extended-tier run put ReAct at 20% accuracy and 40% citation at five times the cost, which is why it is not in the matrix.

## Limitations

Systems is Python; production would need Go or Rust, out of scope here. Databricks Delta pipeline and full-scale ML training are built but not cloud-run, by design: BigQuery work demonstrates the cloud layer, and local ablation settles the model question.

## Stack

Python throughout: PyTorch, scikit-learn, MLflow for ML; FastAPI for serving; DuckDB, dbt, Delta for data; PostGIS, pgvector, H3 for spatial indexing; Streamlit for dashboard; Docker Compose for local orchestration.

Personal and academic use. The data sources retain their own licenses (OpenSky, Eurocontrol PRC-2025, OurAirports).
