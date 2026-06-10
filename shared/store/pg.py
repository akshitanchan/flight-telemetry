"""
shared/store/pg.py
------------------
Lightweight psycopg (v3) connection helper for the Flight Telemetry platform.

Design rules:
- Lazy connect: importing this module does NOT open a database connection.
  A live DATABASE_URL is only required when connect() / get_conn() is called.
- Single source of truth: all code reads DATABASE_URL from the environment;
  no DSN is hard-coded here.
- Thread-safety: each call to get_conn() returns an independent connection.
  Callers that need a pool should layer one on top (e.g. psycopg_pool).

Usage:
    from shared.store.pg import get_conn, healthcheck

    with get_conn() as conn:
        conn.execute("INSERT INTO silver_flight_state ...")

    ok = healthcheck()   # True on success, False on error (never raises)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _dsn() -> str:
    """Return DATABASE_URL from env, raising a clear error if absent."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise EnvironmentError(
            "DATABASE_URL is not set. "
            "Export it before calling connect(), e.g.:\n"
            "  export DATABASE_URL=postgresql://flight:flight@localhost:5432/flight"
        )
    return dsn


def connect():
    """
    Open and return a new psycopg v3 Connection.

    The connection is NOT autocommit by default (psycopg3 default).
    Callers should use it as a context manager so the connection is
    closed on exit:

        with connect() as conn:
            conn.execute("SELECT 1")

    Or manage the lifecycle explicitly:

        conn = connect()
        try:
            ...
        finally:
            conn.close()

    Raises:
        EnvironmentError: if DATABASE_URL is not set.
        psycopg.OperationalError: if the database is unreachable.
    """
    import psycopg  # type: ignore[import]

    dsn = _dsn()
    logger.debug("Opening new psycopg connection to %s", _redact_dsn(dsn))
    return psycopg.connect(dsn)


# Alias — callers can use whichever name reads more naturally.
get_conn = connect


def healthcheck() -> bool:
    """
    Execute ``SELECT 1`` and return True on success, False on any error.

    Never raises. Safe to call from readiness probes and startup routines.
    """
    try:
        with connect() as conn:
            row = conn.execute("SELECT 1").fetchone()
            return row is not None and row[0] == 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB healthcheck failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _redact_dsn(dsn: str) -> str:
    """
    Strip the password from a DSN string before logging.

    Handles the common ``postgresql://user:password@host/db`` format.
    Falls back to returning the full DSN if parsing fails (shouldn't happen).
    """
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(dsn)
        if parsed.password:
            # Replace password with *** while keeping the rest intact
            netloc = parsed.netloc.replace(
                f":{parsed.password}@", ":***@", 1
            )
            redacted = parsed._replace(netloc=netloc)
            return urlunparse(redacted)
    except Exception:  # noqa: BLE001
        pass
    return dsn
