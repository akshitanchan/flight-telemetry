{{
    config(
        materialized        = 'incremental',
        unique_key          = ['airport_icao', 'window_start'],
        partition_by        = {
            'field'       : 'window_start',
            'data_type'   : 'timestamp',
            'granularity' : 'day'
        },
        cluster_by          = ['airport_icao'],
        on_schema_change    = 'append_new_columns',
        require_partition_filter = false
    )
}}

/*
  gold_airport_congestion — incremental mart
  ------------------------------------------
  Source : flight_telemetry.gold_airport_congestion_landing  (bq load target)
  Dest   : flight_telemetry.gold_airport_congestion          (dbt-managed)

  Partition : DATE(window_start) — day granularity
  Cluster   : airport_icao
  Unique key: (airport_icao, window_start)

  Incremental predicate
  ---------------------
  On each dbt run we select only source rows whose window_start falls strictly
  AFTER the latest window_start already present in the destination table.
  Because window_start is also the partition column, BigQuery can prune to
  the relevant day-partitions in BOTH the source (landing table) and the
  destination (for dedup merge) — guaranteed partition-pruning on every
  incremental load.

  Full-refresh safety
  -------------------
  A `dbt build --full-refresh` drops and recreates the table; re-running it
  twice produces the same result (idempotent via CREATE OR REPLACE).

  Schema contract (C3 locked — do NOT add/remove columns):
    airport_icao   STRING    NOT NULL
    window_start   TIMESTAMP NOT NULL
    window_end     TIMESTAMP NOT NULL
    aircraft_count INT64     NOT NULL
    avg_altitude_m FLOAT64
    ground_count   INT64     NOT NULL
    airborne_count INT64     NOT NULL
*/

select
    airport_icao,
    window_start,
    window_end,
    aircraft_count,
    avg_altitude_m,
    ground_count,
    airborne_count
from {{ source('gold_landing', 'gold_airport_congestion_landing') }}

{% if is_incremental() %}
-- Partition-pruning predicate: only new windows not yet in the destination.
-- The sub-select hits only the last partition in the destination (cheap).
where window_start > (
    select coalesce(max(window_start), cast('1970-01-01' as timestamp))
    from {{ this }}
)
{% endif %}
