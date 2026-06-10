-- Delta Lake DDL: gold_sector_load
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_sector_load.schema.json
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Partition strategy: PARTITIONED BY (window_date DATE) where window_date is a
-- derived column written by the ETL job as CAST(window_start AS DATE).
-- ZORDER BY h3_r4 co-locates files for per-sector range scans,
-- matching the C3 locked cluster key.

CREATE TABLE IF NOT EXISTS flight_telemetry.gold_sector_load (
    h3_r4          STRING    NOT NULL,
    window_start   TIMESTAMP NOT NULL,
    window_end     TIMESTAMP NOT NULL,
    aircraft_count BIGINT    NOT NULL,
    -- Derived partition column (written by ETL as CAST(window_start AS DATE))
    window_date    DATE      NOT NULL
)
USING DELTA
PARTITIONED BY (window_date)
TBLPROPERTIES (
    'delta.minReaderVersion' = '1',
    'delta.minWriterVersion' = '2',
    '_contract_version'      = '1.0.0'
);
-- ZORDER BY (h3_r4) -- apply via OPTIMIZE after load; see README.
