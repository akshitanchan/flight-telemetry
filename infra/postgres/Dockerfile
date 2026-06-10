# =============================================================================
# infra/postgres/Dockerfile
# Custom Postgres image: PostGIS 16-3.4 + pgvector
#
# Base: postgis/postgis:16-3.4  (Postgres 16, PostGIS 3.4, Debian bookworm)
# Adds:  postgresql-16-pgvector (apt package — same Debian suite as base image)
#
# The init SQL (infra/db/001_init.sql) issues:
#   CREATE EXTENSION IF NOT EXISTS postgis;
#   CREATE EXTENSION IF NOT EXISTS vector;
# Both are available after this build — no manual steps required.
# =============================================================================

FROM postgis/postgis:16-3.4

# Switch to root for package installation; Postgres process still runs as the
# unprivileged `postgres` user (Debian default for the official images).
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Return to the default user expected by the entrypoint script.
USER postgres
