-- BigQuery DDL: gold_emergency_events
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_emergency_events.schema.json
-- Dataset: flight_telemetry (provisioned via Terraform; do NOT redefine dataset here)
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Note: BQ gold tables are normally created by dbt (ds-06). This DDL is the
-- contract reference + bq load landing-table shape, kept in sync with dbt.
--
-- Partition/cluster: NONE per C3 locked contract (event grain).

CREATE OR REPLACE TABLE `flight_telemetry.gold_emergency_events`
(
    icao24          STRING    NOT NULL,
    callsign        STRING,
    squawk          STRING    NOT NULL,
    first_seen_ts   TIMESTAMP NOT NULL,
    last_seen_ts    TIMESTAMP NOT NULL,
    lat             FLOAT64   NOT NULL,
    lon             FLOAT64   NOT NULL,
    origin_country  STRING    NOT NULL,
    nearest_airport STRING,
    duration_s      INT64     NOT NULL
)
OPTIONS (
    description = 'Emergency squawk event log. Contract v1.0.0.'
);
