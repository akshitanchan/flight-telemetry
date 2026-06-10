# Cloud DDL — flight-telemetry data layer

Contract version: **1.0.0**
Wave: **W4.3 (Data cloud)** — task ds-01 (shared upstream)

## Overview

This directory contains byte-faithful DDL for all five tables in the
`flight_telemetry` cloud dataset, generated from the frozen C3/C1 JSON Schema
contracts in `shared/contracts/`. It covers two dialects:

| Dialect | Directory | Target |
|---------|-----------|--------|
| Delta Lake | `delta/` | Databricks / Unity Catalog |
| BigQuery | `bigquery/` | GCP BigQuery dataset `flight_telemetry` |

The five tables are:

| Table | Layer |
|-------|-------|
| `silver_flight_state` | Silver |
| `gold_airport_congestion` | Gold |
| `gold_sector_load` | Gold |
| `gold_emergency_events` | Gold |
| `gold_routing_stats` | Gold |

## Type mapping

| JSON Schema type | Delta Lake | BigQuery | Notes |
|-----------------|------------|----------|-------|
| `string` | `STRING` | `STRING` | |
| `number` / `["number","null"]` | `DOUBLE` | `FLOAT64` | |
| `integer` | `BIGINT` | `INT64` | |
| `boolean` | `BOOLEAN` | `BOOL` | |
| `string` + `format: date-time` | `TIMESTAMP` | `TIMESTAMP` | Overrides base type |

A column is `NOT NULL` iff its name appears in the schema's `required` array AND
its type is not a `[..., "null"]` union. Otherwise it is nullable (no `NOT NULL`
modifier).

## Partition and cluster (C3-locked)

| Table | Partition | Cluster | Rationale |
|-------|-----------|---------|-----------|
| `silver_flight_state` | `DATE(event_ts)` | none | Time-range scans on raw events |
| `gold_airport_congestion` | `DATE(window_start)` | `airport_icao` | Per-airport daily queries |
| `gold_sector_load` | `DATE(window_start)` | `h3_r4` | Per-sector daily queries |
| `gold_emergency_events` | **none** | **none** | Event grain — low volume, ad-hoc |
| `gold_routing_stats` | **none** | **none** | Route grain — wide ML feature scans |

### Delta Lake partition strategy

Delta Lake requires a concrete column in the table schema for `PARTITIONED BY`.
A derived `DATE` column (`event_date` for silver, `window_date` for gold
time-windowed tables) is declared in each Delta DDL and must be written by the
ETL job as `CAST(event_ts AS DATE)` or `CAST(window_start AS DATE)`. This
column is excluded from the schema parity check (it is a Delta-layer concern,
not part of the JSON Schema contract).

`ZORDER BY` is advisory for Delta tables: it is applied post-load via
`OPTIMIZE ... ZORDER BY (airport_icao)` / `ZORDER BY (h3_r4)` on the two
congestion/sector tables. It is NOT embedded in the `CREATE TABLE` statement
because Delta does not support inline `CLUSTER BY` in the same syntax as
BigQuery.

### BigQuery

`PARTITION BY DATE(window_start)` and `CLUSTER BY <col>` are specified inline in
the `CREATE OR REPLACE TABLE` statement for the two partitioned gold tables.
The BigQuery dataset `flight_telemetry` is already provisioned by Terraform
(`infra/terraform/`); these DDL files target tables inside it and must not
redefine the dataset.

**Note:** The BQ gold tables are normally created and managed by dbt (task
ds-06). This BigQuery DDL serves as:
1. The contract reference shape — every column, type, nullability, partition, and
   cluster clause is frozen here and checked by `check_ddl_parity.py`.
2. The `bq load` landing-table shape for the batch-load path.

Keep this DDL in sync with dbt models; the parity checker (`check_ddl_parity.py`)
is the enforcement mechanism.

## Parity checker

```
python data/cloud/ddl/check_ddl_parity.py [--verbose]
```

Parses all 10 DDL files and diffs them against the corresponding schema
contracts. Exits non-zero on any violation: missing column, extra column, wrong
type, wrong nullability, missing/incorrect partition clause, or
missing/incorrect cluster clause.

The checker uses stdlib only (no third-party deps). It is also wired into
`make test-data` via `data/tests/test_ddl_parity.py`.

## Do NOT touch

- `infra/db/001_init.sql` — C2 Postgres store (local, separate)
- `data/transforms/` — local silver/gold transforms
- `infra/` — Terraform, docker-compose
- `shared/contracts/` — frozen authoritative schemas
