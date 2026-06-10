-- BigQuery DDL: silver_flight_state
-- Contract version: 1.0.0
-- Source schema: shared/contracts/silver_flight_state.schema.json
-- Dataset: flight_telemetry (provisioned via Terraform; do NOT redefine dataset here)
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Partition: by event_ts date (ingestion-time or column-based).
-- No cluster applied at silver layer; query-level clustering is on gold tables.

CREATE OR REPLACE TABLE `flight_telemetry.silver_flight_state`
(
    icao24           STRING    NOT NULL,
    callsign         STRING,
    event_ts         TIMESTAMP NOT NULL,
    lon              FLOAT64   NOT NULL,
    lat              FLOAT64   NOT NULL,
    baro_altitude_m  FLOAT64,
    velocity_ms      FLOAT64,
    true_track_deg   FLOAT64,
    vertical_rate_ms FLOAT64,
    on_ground        BOOL      NOT NULL,
    squawk           STRING,
    origin_country   STRING    NOT NULL,
    nearest_airport  STRING,
    geohash7         STRING    NOT NULL,
    h3_r7            STRING    NOT NULL,
    metar_wind_kt    FLOAT64,
    metar_vis_m      FLOAT64,
    metar_ceiling_ft FLOAT64
)
PARTITION BY DATE(event_ts)
OPTIONS (
    description = 'Canonical silver record per state vector. Contract v1.0.0.'
);
