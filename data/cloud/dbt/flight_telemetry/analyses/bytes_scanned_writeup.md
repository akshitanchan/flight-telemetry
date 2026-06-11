# BigQuery Partition-Pruning + Clustering: Bytes-Scanned Writeup

**Status:** TEMPLATE — owner fills in measured values after running `bq query --dry-run`  
**Date captured:** ____-__-__  
**GCP project:** ______________________  
**BigQuery dataset:** `flight_telemetry`  
**dbt version:** ______________________

---

## Background

The two partitioned+clustered mart tables (`gold_airport_congestion`,
`gold_sector_load`) were designed to exploit BigQuery's two cost-reduction
mechanisms:

| Mechanism | How it works | Benefit |
|-----------|-------------|---------|
| **Partition pruning** | DATE(window_start) partitioned; query `WHERE DATE(window_start) = '...'` eliminates whole day-partitions | Scans only matching day(s) |
| **Clustering pruning** | `cluster_by airport_icao` / `cluster_by h3_r4`; query `WHERE airport_icao = '...'` skips non-matching blocks within a partition | Reduces bytes scanned within the partition |

The two non-partitioned tables (`gold_emergency_events`, `gold_routing_stats`)
have no partition/cluster per C3 contract — full-table scans are expected and
their small cardinality keeps absolute cost negligible.

---

## Measurement Procedure

### Prerequisites

```bash
export BIGQUERY_PROJECT=<your-project-id>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/dbt-runner-key.json

# Confirm table sizes after dbt build
bq show --format=prettyjson "${BIGQUERY_PROJECT}:flight_telemetry.gold_airport_congestion" \
  | python3 -c "import sys,json; t=json.load(sys.stdin); print(t['numBytes'], 'bytes')"
```

### Benchmark queries

Use `--dry-run` so no query bytes are billed. The fixture data is dated
`2024-06-03`; replace the date and keys only if the loaded data differs.

```bash
# ---------------------------------------------------------------------------
# gold_airport_congestion — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`'

# gold_airport_congestion — AFTER (partition filter on one day)
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03"'

# gold_airport_congestion — AFTER (partition + cluster filter)
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03"
     AND airport_icao = "EBAR"'

# ---------------------------------------------------------------------------
# gold_sector_load — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`'

# gold_sector_load — AFTER (partition + cluster filter)
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`
   WHERE DATE(window_start) = "2024-06-03"
     AND h3_r4 = "841e033ffffffff"'

# ---------------------------------------------------------------------------
# Unpartitioned reference tables
# ---------------------------------------------------------------------------
bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_emergency_events`'

bq query --dry-run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_routing_stats`'
```

---

## Results Table

Fill in from `bq query --dry-run` output line: `Query will process X bytes`.

### gold_airport_congestion

| Query type | Filter applied | Bytes processed | Reduction |
|---|---|---|---|
| Full scan (BEFORE) | none | _______ B | - |
| Partition only (AFTER) | `DATE(window_start) = '2024-06-03'` | _______ B | ____% |
| Partition + cluster (AFTER) | above + `airport_icao = 'EBAR'` | _______ B | ____% |

### gold_sector_load

| Query type | Filter applied | Bytes processed | Reduction |
|---|---|---|---|
| Full scan (BEFORE) | none | _______ B | - |
| Partition + cluster (AFTER) | `DATE(window_start)` + `h3_r4 = '841e033ffffffff'` | _______ B | ____% |

### gold_emergency_events (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | _______ B | Expected; event-grain, small table |

### gold_routing_stats (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | _______ B | Expected; route-grain, small table |

---

## Interpretation

Record what BigQuery reports. Small fixture tables may show no measurable
clustering reduction because clustering block pruning becomes useful only once
the table spans enough storage blocks.

The dbt marts use full-snapshot keyed `MERGE` semantics. This intentionally
allows late and corrected rows at existing timestamps to update; it does not
claim that the dbt build itself is partition-pruned.

To inspect build cost separately, capture the dbt job statistics:

```bash
# List recent dbt jobs (look for CREATE OR INSERT statements)
bq ls --jobs --max_results=20 --project_id="${BIGQUERY_PROJECT}"

# Inspect bytes billed for a specific job
bq show --job --format=prettyjson "${BIGQUERY_PROJECT}:US.<job-id>" \
  | python3 -c "import sys,json; s=json.load(sys.stdin)['statistics']; \
      print('bytes billed:', s.get('query',{}).get('totalBytesBilled','n/a'))"
```

---

## Notes

- Pricing changes over time and depends on the billing model. This template
  records bytes, not an estimated dollar value.
- The `require_partition_filter` option is set to `false` in the dbt configs to
  allow unrestricted queries from dbt itself; the owner may set it to `true` on
  the tables directly after `dbt build` if they want to enforce partition filter
  on ad-hoc queries.
