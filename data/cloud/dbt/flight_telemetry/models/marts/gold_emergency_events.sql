{{
    config(
        materialized     = 'incremental',
        unique_key       = ['icao24', 'first_seen_ts'],
        on_schema_change = 'append_new_columns'
    )
}}
{# No partition_by / cluster_by per C3 locked contract (event grain).
   DDL reference: data/cloud/ddl/bigquery/gold_emergency_events.sql #}

/*
  gold_emergency_events — incremental mart (no partition / no cluster)
  ---------------------------------------------------------------------
  Source : flight_telemetry.gold_emergency_events_landing  (bq load target)
  Dest   : flight_telemetry.gold_emergency_events          (dbt-managed)

  No partition or cluster — C3 locked contract for event-grain table.

  Incremental predicate
  ---------------------
  Uses first_seen_ts as the watermark.  On each run we select only events
  whose first_seen_ts is later than the latest event already loaded.  Because
  there is no partition column on this table, BigQuery will full-scan the
  destination for the max() check — acceptable given the typically small
  cardinality of emergency events.  The landing table is similarly unpartitioned.

  Squawk domain: {"7500", "7600", "7700"} — enforced by schema.yml accepted_values test.

  Schema contract (C3 locked):
    icao24          STRING    NOT NULL
    callsign        STRING
    squawk          STRING    NOT NULL  -- "7500" | "7600" | "7700"
    first_seen_ts   TIMESTAMP NOT NULL
    last_seen_ts    TIMESTAMP NOT NULL
    lat             FLOAT64   NOT NULL
    lon             FLOAT64   NOT NULL
    origin_country  STRING    NOT NULL
    nearest_airport STRING
    duration_s      INT64     NOT NULL
*/

select
    icao24,
    callsign,
    squawk,
    first_seen_ts,
    last_seen_ts,
    lat,
    lon,
    origin_country,
    nearest_airport,
    duration_s
from {{ source('gold_landing', 'gold_emergency_events_landing') }}

{% if is_incremental() %}
where first_seen_ts > (
    select coalesce(max(first_seen_ts), cast('1970-01-01' as timestamp))
    from {{ this }}
)
{% endif %}
