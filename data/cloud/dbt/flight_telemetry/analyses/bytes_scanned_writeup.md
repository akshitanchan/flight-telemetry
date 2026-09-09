# BigQuery Partition-Pruning + Clustering: Bytes-Scanned Writeup

**Status:** MEASURED — captured 2026-06-11 on real BigQuery  
**Date captured:** 2026-06-11  
**BigQuery dataset:** `flight_telemetry` (US region); marts materialized into `flight_telemetry_` (see caveat below)  
**dbt result:** 4 incremental marts built; 58 data tests PASS, 0 errors  
**Fixture rows loaded:** airport 11, sector 13, routing 10, emergency 3

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

## Measured Results

### gold_emergency_events (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | **244 bytes** | Expected; event-grain, small fixture (3 rows) |

### gold_routing_stats (no partition/cluster — reference)

| Query type | Bytes scanned | Note |
|---|---|---|
| Full scan | **875 bytes** | Expected; route-grain, small fixture (10 rows) |

### gold_airport_congestion — DEGENERATE (see caveat)

| Query type | Filter applied | Bytes scanned | Reduction |
|---|---|---|---|
| Full scan (BEFORE) | none | 0 bytes | — |
| Partition only (AFTER) | `DATE(window_start) = '2024-06-03'` | 0 bytes | n/a |
| Partition + cluster (AFTER) | above + `airport_icao = 'EBAR'` | 0 bytes | n/a |

### gold_sector_load — DEGENERATE (see caveat)

| Query type | Filter applied | Bytes scanned | Reduction |
|---|---|---|---|
| Full scan (BEFORE) | none | 0 bytes | — |
| Partition + cluster (AFTER) | `DATE(window_start)` + `h3_r4 = '841e033ffffffff'` | 0 bytes | n/a |

---

## Honest Caveat: Degenerate Partition Demo

The partition-pruning demo for the two partitioned marts is **degenerate** and
must not be cited as a pruning result.

**Root cause:** the dbt models carry a **60-day partition expiration**
(`partition_expiration_days = 60` in the dbt config). The fixture data is dated
**2024-06-03** — approximately two years before the capture date of 2026-06-11.
BigQuery auto-expired those partitions shortly after load. By the time the
dry-run queries ran, the partitioned tables contained no live data, so every
query returned 0 bytes regardless of whether a filter was applied.

**What this means:**
- The partition/cluster **design** is correct. The DDL materializes as intended:
  `gold_airport_congestion` is DAY-partitioned on `window_start` and clustered
  on `airport_icao`; `gold_sector_load` is clustered on `h3_r4`. This was
  verified by inspecting table metadata after `dbt build`.
- The **pruning demo** cannot be measured with this fixture data.
- The small fixture (11 rows for airport, 13 for sector) would produce near-zero
  bytes even without expiration; meaningful pruning numbers require a multi-day,
  multi-airport dataset spanning enough storage blocks.

**Recommended fix (follow-up):** either drop the `partition_expiration_days`
setting from the dbt model configs, or re-date the fixture data to a date within
the last 60 days, then re-run `bq load` + `dbt build` + the dry-run queries.

---

## Additional Notes

### Dataset name quirk

The marts materialized into a dataset named **`flight_telemetry_`** (with a
trailing underscore) rather than `flight_telemetry`. This is a dbt
custom-schema config quirk: when dbt's `generate_schema_name` macro receives a
custom schema, it appends it to the target schema with an underscore separator.
The dataset `flight_telemetry` was created as the target, but the custom schema
name produced `flight_telemetry_`. This does not affect correctness; adjust the
`generate_schema_name` macro or set the custom schema to an empty string to
land the marts directly in `flight_telemetry`.

Fixed by removing the empty `+schema: ""` from `dbt_project.yml`; marts now
land directly in the target dataset/schema with no trailing underscore.

### CLI flag correction

The runbook uses `--dry-run` in the `bq query` examples. The correct BigQuery
CLI flag is `--dry_run` (underscore). The `--dry-run` form may be silently
accepted as a positional no-op on some versions; use `--dry_run` to ensure the
estimate is printed and no data is billed.

---

## Measurement Procedure (for re-run with corrected fixture)

### Prerequisites

```bash
export BIGQUERY_PROJECT=<your-project-id>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/dbt-runner-key.json

# Confirm table sizes after dbt build
bq show --format=prettyjson "${BIGQUERY_PROJECT}:flight_telemetry.gold_airport_congestion" \
  | python3 -c "import sys,json; t=json.load(sys.stdin); print(t['numBytes'], 'bytes')"
```

### Benchmark queries

Use `--dry_run` (underscore) so no query bytes are billed. Re-date the fixture
data to a recent date before running, or the partitions will have expired.

```bash
# ---------------------------------------------------------------------------
# gold_airport_congestion — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`'

# gold_airport_congestion — AFTER (partition filter on one day)
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03"'

# gold_airport_congestion — AFTER (partition + cluster filter)
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_airport_congestion`
   WHERE DATE(window_start) = "2024-06-03"
     AND airport_icao = "EBAR"'

# ---------------------------------------------------------------------------
# gold_sector_load — BEFORE (no filter)
# ---------------------------------------------------------------------------
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`'

# gold_sector_load — AFTER (partition + cluster filter)
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_sector_load`
   WHERE DATE(window_start) = "2024-06-03"
     AND h3_r4 = "841e033ffffffff"'

# ---------------------------------------------------------------------------
# Unpartitioned reference tables (measured 2026-06-11)
# ---------------------------------------------------------------------------
bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_emergency_events`'
# Result: 244 bytes

bq query --dry_run --use_legacy_sql=false --project_id="${BIGQUERY_PROJECT}" \
  'SELECT * FROM `flight_telemetry.gold_routing_stats`'
# Result: 875 bytes
```

---

## Interpretation

The two reference tables (`gold_emergency_events` at 244 bytes,
`gold_routing_stats` at 875 bytes) confirm the tables are correctly loaded and
queryable. Their absolute sizes are negligible; they are included as a
correctness check, not a performance claim.

The partitioned mart bytes will only be meaningful once the partition-expiration
or fixture-age issue is resolved. Until then the 0-byte results for those two
tables should not be cited.

The dbt marts use full-snapshot keyed `MERGE` semantics. This intentionally
allows late and corrected rows at existing timestamps to update; it does not
claim that the dbt build itself is partition-pruned.
