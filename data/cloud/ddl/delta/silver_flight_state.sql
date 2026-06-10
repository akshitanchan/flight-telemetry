-- Delta Lake DDL: silver_flight_state
-- Contract version: 1.0.0
-- Source schema: shared/contracts/silver_flight_state.schema.json
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Partition strategy: PARTITIONED BY (event_date DATE) where event_date is a
-- generated column derived from event_ts. Delta Lake requires the partition
-- column to be present in the schema; we use a generated/virtual date column
-- so that DATE(event_ts) can be used as the partition predicate without
-- duplicating the timestamp. In practice the column is written by the ETL job
-- via CAST(event_ts AS DATE) and declared here as a regular DATE column so the
-- DDL is self-contained and can be executed with `spark.sql()`.
-- No ZORDER/CLUSTER BY is specified for silver; clustering is applied at the
-- gold layer where query patterns are fixed.

CREATE TABLE IF NOT EXISTS flight_telemetry.silver_flight_state (
    icao24           STRING    NOT NULL,
    callsign         STRING,
    event_ts         TIMESTAMP NOT NULL,
    lon              DOUBLE    NOT NULL,
    lat              DOUBLE    NOT NULL,
    baro_altitude_m  DOUBLE,
    velocity_ms      DOUBLE,
    true_track_deg   DOUBLE,
    vertical_rate_ms DOUBLE,
    on_ground        BOOLEAN   NOT NULL,
    squawk           STRING,
    origin_country   STRING    NOT NULL,
    nearest_airport  STRING,
    geohash7         STRING    NOT NULL,
    h3_r7            STRING    NOT NULL,
    metar_wind_kt    DOUBLE,
    metar_vis_m      DOUBLE,
    metar_ceiling_ft DOUBLE,
    -- Derived partition column (written by ETL as CAST(event_ts AS DATE))
    event_date       DATE      NOT NULL
)
USING DELTA
PARTITIONED BY (event_date)
TBLPROPERTIES (
    'delta.minReaderVersion' = '1',
    'delta.minWriterVersion' = '2',
    '_contract_version'      = '1.0.0'
);
