# =============================================================================
# infra/docker/ml-serve.Dockerfile
# Multi-stage build for the ML inference service (ml/serve.py).
#
# Stage 1 (builder): install all Python dependencies into a prefix so the
#   final image receives only compiled wheels — no pip, no build toolchain.
# Stage 2 (runtime): copy the installed site-packages and the source tree;
#   run as non-root user `app`.
#
# Build context: repo root  (set in docker-compose.yml via `context: .`)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS builder

WORKDIR /build

# Install build tools needed for wheels with C extensions (psycopg, torch, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can COPY it cleanly to the runtime stage.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS runtime

# Runtime-only system libraries (libpq for psycopg[binary])
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy the full repo source (dockerignore keeps this lean — see /.dockerignore)
COPY . .

# Create a non-root user and hand over ownership
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin app \
    && chown -R app:app /app

USER app

# Expose the API port
EXPOSE 8000

# Secrets are never baked in — inject DATABASE_URL, OTEL_*, ML_* at runtime
# via environment variables or the env_file directive in docker-compose.yml.
CMD ["uvicorn", "ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
