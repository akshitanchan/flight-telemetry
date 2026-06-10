-- Delta Lake DDL: gold_emergency_events
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_emergency_events.schema.json
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Partition/cluster: NONE per C3 locked contract (event grain — low volume,
-- ad-hoc query pattern; no partition or ZORDER applied).

CREATE TABLE IF NOT EXISTS flight_telemetry.gold_emergency_events (
    icao24          STRING    NOT NULL,
    callsign        STRING,
    squawk          STRING    NOT NULL,
    first_seen_ts   TIMESTAMP NOT NULL,
    last_seen_ts    TIMESTAMP NOT NULL,
    lat             DOUBLE    NOT NULL,
    lon             DOUBLE    NOT NULL,
    origin_country  STRING    NOT NULL,
    nearest_airport STRING,
    duration_s      BIGINT    NOT NULL
)
USING DELTA
TBLPROPERTIES (
    'delta.minReaderVersion' = '1',
    'delta.minWriterVersion' = '2',
    '_contract_version'      = '1.0.0'
);
