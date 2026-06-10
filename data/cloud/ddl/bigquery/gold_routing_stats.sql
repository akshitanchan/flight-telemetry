-- BigQuery DDL: gold_routing_stats
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_routing_stats.schema.json
-- Dataset: flight_telemetry (provisioned via Terraform; do NOT redefine dataset here)
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Note: BQ gold tables are normally created by dbt (ds-06). This DDL is the
-- contract reference + bq load landing-table shape, kept in sync with dbt.
--
-- Partition/cluster: NONE per C3 locked contract (route/aircraft grain).

CREATE OR REPLACE TABLE `flight_telemetry.gold_routing_stats`
(
    icao24           STRING    NOT NULL,
    callsign         STRING,
    window_start     TIMESTAMP NOT NULL,
    window_end       TIMESTAMP NOT NULL,
    origin_lat       FLOAT64   NOT NULL,
    origin_lon       FLOAT64   NOT NULL,
    destination_lat  FLOAT64   NOT NULL,
    destination_lon  FLOAT64   NOT NULL,
    max_altitude_m   FLOAT64,
    avg_velocity_mps FLOAT64,
    ping_count       INT64     NOT NULL
)
OPTIONS (
    description = 'Route-level aggregates per aircraft per window. Contract v1.0.0.'
);
