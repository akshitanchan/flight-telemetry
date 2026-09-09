{{
    config(
        materialized        = 'incremental',
        unique_key          = ['h3_r4', 'window_start'],
        partition_by        = {
            'field'       : 'window_start',
            'data_type'   : 'timestamp',
            'granularity' : 'day'
        } if target.type == 'bigquery' else none,
        cluster_by          = ['h3_r4'],
        on_schema_change    = 'append_new_columns',
        require_partition_filter = false if target.type == 'bigquery' else none
    )
}}
{# partitioning is bigquery-only (snowflake micro-partitions automatically) #}
{# cluster_by is also used as a clustering key on snowflake #}
{# require_partition_filter has no snowflake equivalent #}

/*
  gold_sector_load — incremental mart
  ------------------------------------
  Source : flight_telemetry.gold_sector_load_landing  (bq load target)
  Dest   : flight_telemetry.gold_sector_load          (dbt-managed)

  Partition : DATE(window_start) — day granularity (bigquery only)
  Cluster   : h3_r4 (also a snowflake clustering key)
  Unique key: (h3_r4, window_start)

  Incremental behavior
  --------------------
  The landing table is a complete replacement snapshot. The incremental model
  MERGEs every source key so late or corrected rows at an existing timestamp
  are updated rather than silently skipped by a strict max-watermark filter.

  Schema contract (C3 locked):
    note: timestamp columns land as timestamp_tz on snowflake to match bigquery timestamp
    h3_r4          STRING    NOT NULL
    window_start   TIMESTAMP NOT NULL
    window_end     TIMESTAMP NOT NULL
    aircraft_count INT64     NOT NULL
*/

select
    h3_r4,
    window_start,
    window_end,
    aircraft_count
from {{ source('gold_landing', 'gold_sector_load_landing') }}
