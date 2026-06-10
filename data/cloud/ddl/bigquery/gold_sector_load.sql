-- BigQuery DDL: gold_sector_load
-- Contract version: 1.0.0
-- Source schema: shared/contracts/gold_sector_load.schema.json
-- Dataset: flight_telemetry (provisioned via Terraform; do NOT redefine dataset here)
-- This is a CLOUD DEMO table; do NOT modify infra/db/001_init.sql (C2 Postgres).
--
-- Note: BQ gold tables are normally created by dbt (ds-06). This DDL is the
-- contract reference + bq load landing-table shape, kept in sync with dbt.
--
-- Partition: DATE(window_start) per C3 locked contract.
-- Cluster: h3_r4 per C3 locked contract.

CREATE OR REPLACE TABLE `flight_telemetry.gold_sector_load`
(
    h3_r4          STRING    NOT NULL,
    window_start   TIMESTAMP NOT NULL,
    window_end     TIMESTAMP NOT NULL,
    aircraft_count INT64     NOT NULL
)
PARTITION BY DATE(window_start)
CLUSTER BY h3_r4
OPTIONS (
    description = 'Sector load density per H3 r4 cell and time window. Contract v1.0.0.'
);
