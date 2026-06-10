{{
    config(
        materialized     = 'incremental',
        unique_key       = ['icao24', 'window_start'],
        on_schema_change = 'append_new_columns'
    )
}}
{# No partition_by / cluster_by per C3 locked contract (route/aircraft grain).
   DDL reference: data/cloud/ddl/bigquery/gold_routing_stats.sql #}

/*
  gold_routing_stats — incremental mart (no partition / no cluster)
  -----------------------------------------------------------------
  Source : flight_telemetry.gold_routing_stats_landing  (bq load target)
  Dest   : flight_telemetry.gold_routing_stats          (dbt-managed)

  No partition or cluster — C3 locked contract for route-grain table.

  Incremental predicate
  ---------------------
  Uses window_start as the watermark.  On each run we select only rows
  whose window_start exceeds the max already loaded.

  Schema contract (C3 locked):
    icao24           STRING    NOT NULL
    callsign         STRING
    window_start     TIMESTAMP NOT NULL
    window_end       TIMESTAMP NOT NULL
    origin_lat       FLOAT64   NOT NULL
    origin_lon       FLOAT64   NOT NULL
    destination_lat  FLOAT64   NOT NULL
    destination_lon  FLOAT64   NOT NULL
    max_altitude_m   FLOAT64
    avg_velocity_mps FLOAT64
    ping_count       INT64     NOT NULL
*/

select
    icao24,
    callsign,
    window_start,
    window_end,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    max_altitude_m,
    avg_velocity_mps,
    ping_count
from {{ source('gold_landing', 'gold_routing_stats_landing') }}

{% if is_incremental() %}
where window_start > (
    select coalesce(max(window_start), cast('1970-01-01' as timestamp))
    from {{ this }}
)
{% endif %}
