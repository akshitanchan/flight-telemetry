-- Delta Lake DDL: gold_routing_stats
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_routing_stats.schema.json
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Partition/cluster: NONE per C3 locked contract (route/aircraft grain —
-- wide scan expected for ML feature extraction; no partition or ZORDER applied).

CREATE TABLE IF NOT EXISTS flight_telemetry.gold_routing_stats (
    icao24          STRING    NOT NULL,
    callsign        STRING,
    window_start    TIMESTAMP NOT NULL,
    window_end      TIMESTAMP NOT NULL,
    origin_lat      DOUBLE    NOT NULL,
    origin_lon      DOUBLE    NOT NULL,
    destination_lat DOUBLE    NOT NULL,
    destination_lon DOUBLE    NOT NULL,
    max_altitude_m  DOUBLE,
    avg_velocity_mps DOUBLE,
    ping_count      BIGINT    NOT NULL
)
USING DELTA
TBLPROPERTIES (
    'delta.minReaderVersion' = '1',
    'delta.minWriterVersion' = '2',
    '_contract_version'      = '1.0.0'
);
