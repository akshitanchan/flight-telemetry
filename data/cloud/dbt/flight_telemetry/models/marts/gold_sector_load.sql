{{
    config(
        materialized        = 'incremental',
        unique_key          = ['h3_r4', 'window_start'],
        partition_by        = {
            'field'       : 'window_start',
            'data_type'   : 'timestamp',
            'granularity' : 'day'
        },
        cluster_by          = ['h3_r4'],
        on_schema_change    = 'append_new_columns',
        require_partition_filter = false
    )
}}

/*
  gold_sector_load — incremental mart
  ------------------------------------
  Source : flight_telemetry.gold_sector_load_landing  (bq load target)
  Dest   : flight_telemetry.gold_sector_load          (dbt-managed)

  Partition : DATE(window_start) — day granularity
  Cluster   : h3_r4
  Unique key: (h3_r4, window_start)

  Incremental predicate
  ---------------------
  Selects only landing rows whose window_start exceeds the max already stored.
  Because the partition field is window_start, both the read from the landing
  table and the merge check on the destination table are partition-pruned.

  Schema contract (C3 locked):
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

{% if is_incremental() %}
where window_start > (
    select coalesce(max(window_start), cast('1970-01-01' as timestamp))
    from {{ this }}
)
{% endif %}
