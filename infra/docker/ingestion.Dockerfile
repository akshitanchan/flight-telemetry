# =============================================================================
# infra/docker/ingestion.Dockerfile
# Multi-stage build for the real-time data ingestion service.
#
# DEPENDENCY NOTE (task ws1-02):
#   The entrypoint `python -m systems.ingest.cli` targets a module that does
#   not yet exist in the repository. It will be created in task ws1-02
#   (OpenSky ingestion pipeline). This image will BUILD successfully now
#   because the entrypoint is only evaluated at container start, not at build
#   time. Once ws1-02 lands, `docker compose up ingestion` will be fully
#   functional.
#
# Build context: repo root  (set in docker-compose.yml via `context: .`)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY . .

RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin app \
    && chown -R app:app /app

USER app

# Secrets are never baked in — inject OPENSKY_CLIENT_ID, OPENSKY_CLIENT_SECRET,
# DATABASE_URL, and OTEL_EXPORTER_OTLP_ENDPOINT at runtime via env or env_file.

# DEPENDENCY: systems.ingest.cli is provided by task ws1-02.
# This container will exit with ModuleNotFoundError until that task lands.
ENTRYPOINT ["python", "-m", "systems.ingest.cli"]
