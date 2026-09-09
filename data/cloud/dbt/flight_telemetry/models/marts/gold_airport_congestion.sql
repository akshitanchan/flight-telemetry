{{
    config(
        materialized        = 'incremental',
        unique_key          = ['airport_icao', 'window_start'],
        partition_by        = {
            'field'       : 'window_start',
            'data_type'   : 'timestamp',
            'granularity' : 'day'
        } if target.type == 'bigquery' else none,
        cluster_by          = ['airport_icao'],
        on_schema_change    = 'append_new_columns',
        require_partition_filter = false if target.type == 'bigquery' else none
    )
}}
{# partitioning is bigquery-only (snowflake micro-partitions automatically) #}
{# cluster_by is also used as a clustering key on snowflake #}
{# require_partition_filter has no snowflake equivalent #}

/*
  gold_airport_congestion — incremental mart
  ------------------------------------------
  Source : flight_telemetry.gold_airport_congestion_landing  (bq load target)
  Dest   : flight_telemetry.gold_airport_congestion          (dbt-managed)

  Partition : DATE(window_start) — day granularity (bigquery only)
  Cluster   : airport_icao (also a snowflake clustering key)
  Unique key: (airport_icao, window_start)

  Incremental behavior
  --------------------
  The landing table is replaced as a complete snapshot by `bq load --replace`.
  Each incremental dbt run therefore reads that snapshot and MERGEs it by the
  declared unique key. This intentionally favors correctness: corrected or
  late rows at an existing window_start are updated instead of being skipped by
  a fragile `window_start > max(window_start)` watermark.

  Full-refresh safety
  -------------------
  A `dbt build --full-refresh` drops and recreates the table; re-running it
  twice produces the same result (idempotent via CREATE OR REPLACE).

  Schema contract (C3 locked — do NOT add/remove columns):
    note: timestamp columns land as timestamp_tz on snowflake to match bigquery timestamp
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
