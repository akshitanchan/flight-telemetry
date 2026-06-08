# Flight Telemetry Intelligence Platform

> A four-layer platform over live and historical aircraft telemetry — Systems, Data, ML, and AI — sharing one data spine.

## Quick Start

```bash
cp .env.example .env   # fill in credentials
make setup             # create local output directories
make help              # list all available targets
```

## Documentation

| Document | Purpose |
|---|---|
| [docs/plan.md](docs/plan.md) | Implementation plan — architecture, data sources, acceptance criteria |
| [docs/execution.md](docs/execution.md) | 3-week agent execution plan — timeline, gates, commit discipline |
| [docs/checklist.md](docs/checklist.md) | Shared progress ledger — current gate, task status, verification log |

## Repository Layout

```
flight-telemetry/
  .env.example           # environment template (never commit .env)
  Makefile               # one-command entrypoints per layer
  docs/                  # project docs — plan, execution, checklist
  shared/contracts/      # silver_flight_state schema + gold schemas
  systems/               # Go/Rust ingestion + spatiotemporal index benchmark
  data/                  # Databricks notebooks/Lakeflow + dbt/BigQuery project
  ml/                    # training pipeline, serving, drift monitor
  ai/                    # agent + eval harness + golden question set
  infra/                 # env setup, IaC, CI config
```

## License

For personal and academic use only. See data source licenses in [docs/plan.md](docs/plan.md).