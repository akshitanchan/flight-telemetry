# flight_telemetry dbt Project

dbt project targeting BigQuery for the flight-telemetry data cloud (ds-06, W4.3).

## Project layout

```
data/cloud/dbt/flight_telemetry/
├── dbt_project.yml                  # Project config; profile: flight_telemetry
├── profiles.example.yml             # Profile template (copy + set env vars)
├── README.md                        # This file
├── models/
│   ├── staging/
│   │   └── sources.yml              # Source declarations (bq-load landing tables)
│   └── marts/
│       ├── gold_airport_congestion.sql   # incremental, partition+cluster
│       ├── gold_sector_load.sql          # incremental, partition+cluster
│       ├── gold_emergency_events.sql     # incremental, no partition/cluster
│       ├── gold_routing_stats.sql        # incremental, no partition/cluster
│       ├── schema.yml               # Column not_null / accepted_values tests
│       └── _unit_tests.yml          # dbt >= 1.8 native unit tests (offline)
└── analyses/
    └── bytes_scanned_writeup.md     # Before/after partition-pruning template
```

## Mart configurations summary

| Model | materialized | partition_by | cluster_by | unique_key |
|---|---|---|---|---|
| `gold_airport_congestion` | incremental | `window_start` (timestamp, day) | `airport_icao` | `(airport_icao, window_start)` |
| `gold_sector_load` | incremental | `window_start` (timestamp, day) | `h3_r4` | `(h3_r4, window_start)` |
| `gold_emergency_events` | incremental | none (C3 locked) | none | `(icao24, first_seen_ts)` |
| `gold_routing_stats` | incremental | none (C3 locked) | none | `(icao24, window_start)` |

Partition and cluster keys are **locked by contract C3** and must match
`data/cloud/ddl/bigquery/*.sql` exactly.

## Prerequisites

- Python 3.10–3.13 (dbt-core is incompatible with Python 3.14 due to the
  `mashumaro` dependency; use a dedicated venv separate from the project's main
  venv which may run Python 3.14).
- `dbt-core >= 1.8`, `dbt-bigquery >= 1.8`, and `dbt-duckdb >= 1.8` installed
  (all three pinned in `requirements.txt` at repo root under `# dbt (ds-06)`).

Install in a fresh venv:
```bash
python3.13 -m venv .venv-dbt
source .venv-dbt/bin/activate
pip install "dbt-core>=1.8,<2.0" "dbt-bigquery>=1.8,<2.0" "dbt-duckdb>=1.8,<2.0"
```

**Note on offline `dbt compile`:** the `dbt-bigquery` adapter always calls the
BigQuery API to populate a relation cache before compiling — even with
`--no-introspect`.  This is a known adapter limitation.  Use `dbt parse` for
offline static analysis; `dbt ls` works offline too.  Full `dbt compile` and
`dbt test` (schema tests) require a live BigQuery connection.  Native unit tests
(`test_type:unit`) run against DuckDB locally using the `duckdb_unit` target.

## Offline commands (no warehouse required)

All commands run from `data/cloud/dbt/flight_telemetry/`.

### 1. Set required environment variables (dummy values are fine offline)

```bash
export BIGQUERY_PROJECT=dummy-project
export GOOGLE_APPLICATION_CREDENTIALS=/dev/null
export DBT_PROFILES_DIR=$(pwd)   # points dbt at profiles.example.yml in this dir
```

Rename `profiles.example.yml` to `profiles.yml` temporarily for local runs, or
set `--profiles-dir` explicitly:

```bash
cp profiles.example.yml profiles.yml
```

### 2. dbt deps (fetch packages — currently none, but required by dbt)

```bash
dbt deps \
  --profiles-dir . \
  --profile flight_telemetry \
  --target offline
```

Expected output: `Up to date!` (no packages declared in packages.yml).

### 3. dbt parse (validates YAML and SQL syntax, no warehouse connection)

```bash
dbt parse \
  --profiles-dir . \
  --profile flight_telemetry \
  --target offline \
  --no-partial-parse
```

Expected output: `Done.` with no errors.

### 4. dbt ls (list all nodes — fully offline)

```bash
BIGQUERY_PROJECT=dummy-project \
GOOGLE_APPLICATION_CREDENTIALS=/dev/null \
dbt ls \
  --profiles-dir . \
  --profile flight_telemetry \
  --target offline
```

Expected output: 4 mart models, 4 sources, 54 data tests, 5 unit tests listed.

Note: `dbt compile` with the bigquery adapter requires a live BigQuery connection
(the adapter populates a relation cache via the API before compiling).  Run
`dbt compile` only with real credentials using `--target prod`.

### 5. Native unit tests (offline, no BigQuery connection)

dbt >= 1.8 unit tests execute in DuckDB locally.  Use the `duckdb_unit` target
to bypass the BigQuery adapter's mandatory relation-cache API call.

```bash
BIGQUERY_PROJECT=dummy-project \
dbt test \
  --select "test_type:unit" \
  --profiles-dir . \
  --profile flight_telemetry \
  --target duckdb_unit \
  --no-partial-parse
```

Expected: all 5 unit tests pass:
- `ut_airport_congestion_incremental_snapshot` (PASS)
- `ut_airport_congestion_full_refresh` (PASS)
- `ut_sector_load_incremental_snapshot` (PASS)
- `ut_emergency_events_incremental_snapshot` (PASS)
- `ut_routing_stats_incremental_snapshot` (PASS)

### 6. Run all schema tests (online — requires BigQuery)

```bash
dbt test \
  --exclude "test_type:unit" \
  --profiles-dir . \
  --profile flight_telemetry \
  --target prod
```

---

## Owner-cloud commands (requires real BigQuery credentials)

### Step 0: Configure environment

```bash
export BIGQUERY_PROJECT=<your-gcp-project-id>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/dbt-runner-sa-key.json
export DBT_PROFILES_DIR=/path/to/data/cloud/dbt/flight_telemetry
```

Terraform must have already provisioned the dataset and service account
(`infra/terraform/bigquery/`).  dbt owns table creation inside the dataset.

### Step 1: Export and load gold Parquet files into landing tables

```bash
make data-export-gold-parquet

for table in \
  gold_airport_congestion \
  gold_sector_load \
  gold_emergency_events \
  gold_routing_stats
do
  bq load \
    --project_id="${BIGQUERY_PROJECT}" \
    --source_format=PARQUET \
    --replace \
    "${BIGQUERY_PROJECT}:flight_telemetry.${table}_landing" \
    "outputs/bigquery_landing/${table}.parquet"
done
```

### Step 2: dbt deps + build

```bash
cd data/cloud/dbt/flight_telemetry

dbt deps --profiles-dir . --profile flight_telemetry --target prod

dbt build \
  --profiles-dir . \
  --profile flight_telemetry \
  --target prod
```

`dbt build` runs: compile → create/insert incremental tables → run schema tests.

For a full historical reload:
```bash
dbt build \
  --full-refresh \
  --profiles-dir . \
  --profile flight_telemetry \
  --target prod
```

### Step 3: Capture bytes-scanned (for bytes_scanned_writeup.md)

```bash
# Full-table scan baseline
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`'

# Partition-filtered query
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03"'

# Partition + cluster filtered query
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03" AND airport_icao = "EBAR"'
```

Record results in `analyses/bytes_scanned_writeup.md`.

### Step 4: Verify data quality

```bash
dbt test \
  --profiles-dir . \
  --profile flight_telemetry \
  --target prod \
  --select "models/marts/*"
```

---

## Incremental snapshot strategy

`bq load --replace` makes each landing table a complete current snapshot. The
dbt models read the full landing snapshot and use their `unique_key` settings to
MERGE into the destination tables. This is intentionally late-data safe:
corrected rows at an existing timestamp remain eligible for update.

The query benchmark in `analyses/bytes_scanned_writeup.md` measures partition
and clustering behavior of downstream reads. It does not claim the dbt build
itself is partition-pruned.

---

## Data lineage

```
bq load (Parquet)
  └── flight_telemetry.gold_*_landing   [source tables — owner managed]
        └── dbt incremental models
              └── flight_telemetry.gold_*   [mart tables — dbt managed]
```

The gold Parquet files are produced by `data/transforms/silver_to_gold.py`
(local) or equivalent Spark/Dataproc jobs in the cloud pipeline.  dbt reads
only from the landing tables; it does not re-derive aggregation logic.
