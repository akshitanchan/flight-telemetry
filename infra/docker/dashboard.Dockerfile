# =============================================================================
# infra/docker/dashboard.Dockerfile
# Multi-stage build for the Streamlit dashboard service.
#
# DEPENDENCY NOTE (task ws5-01):
#   The runtime CMD references `dashboard/app.py` and requires the `streamlit`
#   package. Both the dashboard application and the streamlit dependency are
#   added in task ws5-01. This image will BUILD successfully now because:
#     1. `streamlit` is not imported at build time.
#     2. `dashboard/app.py` is referenced only in CMD, which is evaluated at
#        container start, not at image build time.
#   Once ws5-01 lands and `streamlit` is added to requirements.txt (or a
#   dedicated dashboard requirements file), `docker compose up dashboard` will
#   be fully functional.
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

# Expose the Streamlit port
EXPOSE 8501

# Secrets are never baked in — inject DATABASE_URL and OTEL_* at runtime.

# DEPENDENCY: dashboard/app.py and the `streamlit` package are provided by
# task ws5-01. Until then this container will fail at runtime, not build time.
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
