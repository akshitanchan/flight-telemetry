# BigQuery Partition-Pruning + Clustering: Bytes-Scanned Writeup

**Status:** TEMPLATE — owner fills in measured values after running `bq query --dry-run`  
**Date captured:** ____-__-__  
**GCP project:** ______________________  
**BigQuery dataset:** `flight_telemetry`  
**dbt version:** `dbt-core 1.11.x` / `dbt-bigquery 1.11.x`

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
export GCP_PROJECT=<your-project-id>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/dbt-runner-key.json

# Confirm table sizes after dbt build
bq show --format=prettyjson "${GCP_PROJECT}:flight_telemetry.gold_airport_congestion" \
  | python3 -c "import sys,json; t=json.load(sys.stdin); print(t['numBytes'], 'bytes')"
```

### Benchmark queries

Run each query **twice** — once without a partition/cluster filter (BEFORE)
and once with (AFTER).  Use `--dry-run` so no bytes are actually billed.

```bash
# ---------------------------------------------------------------------------
# gold_airport_congestion — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry-run --use_legacy_sql=false --project_id="${GCP_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`'

# gold_airport_congestion — AFTER (partition filter on one day)
bq query --dry-run --use_legacy_sql=false --project_id="${GCP_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-01-15"'

# gold_airport_congestion — AFTER (partition + cluster filter)
bq query --dry-run --use_legacy_sql=false --project_id="${GCP_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-01-15"
     AND airport_icao = "EGLL"'

# ---------------------------------------------------------------------------
# gold_sector_load — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry-run --use_legacy_sql=false --project_id="${GCP_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`'

# gold_sector_load — AFTER (partition + cluster filter)
bq query --dry-run --use_legacy_sql=false --project_id="${GCP_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`
   WHERE DATE(window_start) = "2024-01-15"
     AND h3_r4 = "8426b47ffffffff"'
```

---

## Results Table

Fill in from `bq query --dry-run` output line: `Query will process X bytes`.

### gold_airport_congestion

| Query type | Filter applied | Bytes scanned | Estimated cost (USD at $6.25/TB) | Reduction |
|---|---|---|---|---|
| Full scan (BEFORE) | none | _______ B | $_______ | — |
| Partition only (AFTER) | `DATE(window_start) = '2024-01-15'` | _______ B | $_______ | ____% |
| Partition + cluster (AFTER) | above + `airport_icao = 'EGLL'` | _______ B | $_______ | ____% |

### gold_sector_load

| Query type | Filter applied | Bytes scanned | Estimated cost (USD) | Reduction |
|---|---|---|---|---|
| Full scan (BEFORE) | none | _______ B | $_______ | — |
| Partition + cluster (AFTER) | `DATE(window_start)` + `h3_r4 = '...'` | _______ B | $_______ | ____% |

### gold_emergency_events (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | _______ B | Expected; event-grain, small table |

### gold_routing_stats (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | _______ B | Expected; route-grain, small table |

---

## Expected Outcomes

Based on the incremental load pattern (daily partitions, one new day per run):

- **Partition pruning alone** should reduce scanned bytes by approximately
  `(total_days - 1) / total_days` — e.g. for 30 days of data, ~97% reduction.
- **Clustering** provides an additional reduction proportional to the
  cardinality of the cluster key within a partition.  For `airport_icao` with
  ~1,000 unique ICAO codes, expect an additional 10–50x reduction per filtered
  query (BigQuery cluster blocks typically hold ~1–4 GB of sorted data).

---

## dbt Incremental Load Bytes

The incremental predicate in each mart model uses a `max(window_start)` subquery
on the destination table.  BigQuery executes this as a partition-pruned read of
the _last_ partition rather than a full-table scan, so the overhead of the
watermark lookup is minimal (typically a few MB on a sorted partition).

To verify, capture the `dbt build` slot-time and bytes-billed from the BigQuery
job history after a run:

```bash
# List recent dbt jobs (look for CREATE OR INSERT statements)
bq ls --jobs --max_results=20 --project_id="${GCP_PROJECT}"

# Inspect bytes billed for a specific job
bq show --job --format=prettyjson "${GCP_PROJECT}:US.<job-id>" \
  | python3 -c "import sys,json; s=json.load(sys.stdin)['statistics']; \
      print('bytes billed:', s.get('query',{}).get('totalBytesBilled','n/a'))"
```

---

## Notes

- Cost formula: `bytes_scanned_TB * $6.25` (on-demand pricing; confirm current
  rate at https://cloud.google.com/bigquery/pricing).
- Slot-based (flat-rate) pricing makes bytes-scanned less meaningful for cost
  but still relevant for performance and quota management.
- The `require_partition_filter` option is set to `false` in the dbt configs to
  allow unrestricted queries from dbt itself; the owner may set it to `true` on
  the tables directly after `dbt build` if they want to enforce partition filter
  on ad-hoc queries.
