-- =============================================================================
-- Migration 001 — Initial schema for the Flight Telemetry Intelligence Platform
-- Safe to run multiple times (fully idempotent via IF NOT EXISTS).
-- Target: PostgreSQL 16 + PostGIS + pgvector
-- (Custom image: FROM postgis/postgis:16 with postgresql-16-pgvector installed)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Table: silver_flight_state
--
-- Mirrors shared/contracts/silver_flight_state.schema.json v1.0.0 exactly.
-- Columns are in the same order as the JSON Schema properties object.
-- Primary key: (icao24, event_ts) — one row per transponder per 5-second slot.
-- Generated column `geom` is derived from lon/lat at write time (no app math).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver_flight_state (
    -- identity
    icao24              TEXT                NOT NULL,
    callsign            TEXT                NULL,
    event_ts            TIMESTAMPTZ         NOT NULL,

    -- position
    lon                 DOUBLE PRECISION    NOT NULL,
    lat                 DOUBLE PRECISION    NOT NULL,

    -- flight dynamics (nullable per schema)
    baro_altitude_m     DOUBLE PRECISION    NULL,
    velocity_ms         DOUBLE PRECISION    NULL,
    true_track_deg      DOUBLE PRECISION    NULL,
    vertical_rate_ms    DOUBLE PRECISION    NULL,

    -- state flags
    on_ground           BOOLEAN             NOT NULL,
    squawk              TEXT                NULL,
    origin_country      TEXT                NOT NULL,

    -- enrichment
    nearest_airport     TEXT                NULL,
    geohash7            TEXT                NOT NULL,
    h3_r7               TEXT                NOT NULL,

    -- METAR weather join (nullable)
    metar_wind_kt       DOUBLE PRECISION    NULL,
    metar_vis_m         DOUBLE PRECISION    NULL,
    metar_ceiling_ft    DOUBLE PRECISION    NULL,

    -- Generated geography column — computed from lon/lat at INSERT, never stored
    -- by the application layer. STORED means it is persisted on disk so spatial
    -- operators do not recompute it on every read.
    geom geography(Point, 4326)
        GENERATED ALWAYS AS (ST_MakePoint(lon, lat)::geography) STORED,

    PRIMARY KEY (icao24, event_ts)
);

-- Spatial index — supports ST_DWithin, ST_Distance, and bounding-box queries.
CREATE INDEX IF NOT EXISTS idx_silver_flight_state_geom
    ON silver_flight_state USING GIST (geom);

-- Temporal index — supports time-range queries and time-series aggregations.
CREATE INDEX IF NOT EXISTS idx_silver_flight_state_event_ts
    ON silver_flight_state (event_ts);

-- ---------------------------------------------------------------------------
-- Table: ml_predictions
--
-- Stores per-inference outputs from every MLflow model run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_predictions (
    id                  BIGSERIAL           PRIMARY KEY,
    run_id              TEXT                NOT NULL,
    model_uri           TEXT                NOT NULL,
    icao24              TEXT                NOT NULL,
    event_ts            TIMESTAMPTZ         NULL,
    predicted_fuel_kg   DOUBLE PRECISION    NOT NULL,
    actual_fuel_kg      DOUBLE PRECISION    NULL,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Table: ai_embeddings
--
-- Stores document chunk embeddings for vector similarity search.
--
-- Embedding dimension: 1536 — matches OpenAI text-embedding-3-small output.
-- To change the dimension (e.g. to 3072 for text-embedding-3-large):
--   1. Update the vector(N) declaration below.
--   2. Drop and recreate the HNSW index (it is dimension-specific).
--   3. Re-embed all existing rows with the new model.
--   Do NOT change the dimension in-place on a populated table; ALTER COLUMN
--   is not supported for vector columns — you must migrate the data.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_embeddings (
    id          BIGSERIAL       PRIMARY KEY,
    doc_id      TEXT            NOT NULL,
    chunk_id    INT             NOT NULL,
    source      TEXT            NOT NULL,
    content     TEXT            NOT NULL,
    embedding   vector(1536)    NOT NULL,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- HNSW index for approximate nearest-neighbour search using cosine distance.
-- HNSW is preferred over IVFFlat for this workload: no training step required,
-- better recall at low ef_search, and it handles incremental inserts safely.
-- m=16, ef_construction=64 are the pgvector defaults; tune after load testing.
CREATE INDEX IF NOT EXISTS idx_ai_embeddings_embedding_hnsw
    ON ai_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
