#!/usr/bin/env python3
"""
PostGIS GIST index — third spatiotemporal index strategy.

Uses PostgreSQL's PostGIS extension and a GIST index on the
``geom geography(Point,4326)`` generated column of ``silver_flight_state``
for spatiotemporal range queries.

Approach:
  - Build (choice a — UPSERT): bulk-inserts/upserts the supplied records
    into ``silver_flight_state`` using ``ON CONFLICT (icao24, event_ts) DO UPDATE``
    so that the benchmark is fully self-contained; a caller can re-run
    without manually pre-populating the table and results remain idempotent.
    The generated column ``geom`` is computed by Postgres at write time, so
    the application never sets it explicitly.
  - Query: a single SQL statement that exploits the GIST index via
    ``ST_Intersects(geom, ST_MakeEnvelope(lon_min, lat_min, lon_max, lat_max, 4326)::geography)``
    combined with an ``event_ts BETWEEN`` clause that drives the temporal index.
    Returning all columns gives the same dict shape as geohash/h3 backends,
    allowing apples-to-apples result comparison.
  - index_size_bytes: derived from ``pg_total_relation_size('silver_flight_state')``
    (includes the table heap + all its indexes), never in-memory sys.getsizeof.

Availability-gating:
  - ``PostGISIndex.is_available()`` is a classmethod that calls
    ``shared.store.pg.healthcheck()``.  It is the single place where DB
    reachability is tested.
  - The CLI consults ``is_available()`` before constructing the instance and
    logs a WARNING + skips the strategy when the DB is unreachable, so the
    offline CI suite stays green.
  - ``build()`` and ``query()`` also guard themselves — they raise
    ``RuntimeError`` if called when unavailable, giving a clear failure
    message instead of a cryptic DB error.

Dependencies: psycopg[binary]>=3.1 (already in requirements.txt), PostGIS
installed in the target Postgres instance (see infra/db/001_init.sql).
"""

import logging
import time as time_mod
from typing import Any

from systems.index.base import SpatiotemporalIndex, SpatiotemporalQuery

logger = logging.getLogger(__name__)

# Columns written during UPSERT (geom is generated — never listed in INSERT).
_INSERT_COLS = (
    "icao24",
    "callsign",
    "event_ts",
    "lon",
    "lat",
    "baro_altitude_m",
    "velocity_ms",
    "true_track_deg",
    "vertical_rate_ms",
    "on_ground",
    "squawk",
    "origin_country",
    "nearest_airport",
    "geohash7",
    "h3_r7",
    "metar_wind_kt",
    "metar_vis_m",
    "metar_ceiling_ft",
)

# Columns to SELECT in query — mirrors _INSERT_COLS so result dicts
# carry the same fields as geohash/h3 in-memory backends.
_SELECT_COLS = ", ".join(_INSERT_COLS)

# The SQL placeholders tuple: (%s, %s, ...) — one per column.
_PLACEHOLDERS = "(" + ", ".join(["%s"] * len(_INSERT_COLS)) + ")"

# DO UPDATE SET clause — update every non-key column so the UPSERT is
# idempotent but always reflects the latest ingested values.
_UPDATE_COLS = ", ".join(
    f"{col} = EXCLUDED.{col}"
    for col in _INSERT_COLS
    if col not in ("icao24", "event_ts")
)

_UPSERT_SQL = f"""
INSERT INTO silver_flight_state ({", ".join(_INSERT_COLS)})
VALUES {_PLACEHOLDERS}
ON CONFLICT (icao24, event_ts) DO UPDATE
    SET {_UPDATE_COLS}
"""

_QUERY_SQL = f"""
SELECT {_SELECT_COLS}
FROM   silver_flight_state
WHERE  geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography
  AND  ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography)
  AND  event_ts BETWEEN %s AND %s
"""

_SIZE_SQL = "SELECT pg_total_relation_size('silver_flight_state')"

_COUNT_SQL = "SELECT COUNT(*) FROM silver_flight_state"


def _record_to_row(rec: dict) -> tuple:
    """Extract ordered column values from a silver record dict.

    Missing nullable fields default to None so the UPSERT never fails
    on incomplete records from the benchmark workload.
    """
    return (
        rec.get("icao24"),
        rec.get("callsign"),
        rec.get("event_ts"),
        rec.get("lon"),
        rec.get("lat"),
        rec.get("baro_altitude_m"),
        rec.get("velocity_ms"),
        rec.get("true_track_deg"),
        rec.get("vertical_rate_ms"),
        rec.get("on_ground", False),
        rec.get("squawk"),
        rec.get("origin_country", ""),
        rec.get("nearest_airport"),
        rec.get("geohash7", ""),
        rec.get("h3_r7", ""),
        rec.get("metar_wind_kt"),
        rec.get("metar_vis_m"),
        rec.get("metar_ceiling_ft"),
    )


class PostGISIndex(SpatiotemporalIndex):
    """Spatiotemporal index backed by PostGIS GIST on silver_flight_state.geom.

    Queries delegate entirely to Postgres: the GIST index on the generated
    ``geom geography`` column accelerates the bounding-box filter, and the
    B-tree index on ``event_ts`` handles the temporal range filter.

    The benchmark harness calls build() once, then query() many times; all
    timing/stats are tracked in-memory, but index_size_bytes is fetched
    from Postgres so it reflects actual on-disk storage.
    """

    def __init__(self, batch_size: int = 1000):
        """
        Args:
            batch_size: Number of rows to UPSERT per database round-trip.
                Larger values reduce round-trips at the cost of memory per
                batch.  1000 is a safe default for the benchmark dataset size.
        """
        self._batch_size = batch_size
        self._record_count = 0
        self._build_time = 0.0
        self._index_size_bytes = 0

    # ------------------------------------------------------------------
    # Availability gate — single source of truth for DB reachability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if a database connection can be established.

        Delegates to shared.store.pg.healthcheck(), which executes
        ``SELECT 1`` and catches all exceptions, so this method never raises.
        Cached once per CLI invocation; for unit tests, call it fresh each
        time so the guard reflects the current environment.
        """
        from shared.store.pg import healthcheck  # lazy import — no DB at import time
        return healthcheck()

    # ------------------------------------------------------------------
    # SpatiotemporalIndex ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "postgis_gist"

    def build(self, records: list[dict]) -> None:
        """UPSERT records into silver_flight_state (choice a).

        Each record is inserted or updated idempotently on the
        (icao24, event_ts) primary key.  Postgres computes the ``geom``
        generated column at write time from lon/lat.

        Raises:
            RuntimeError: if the database is not reachable.
        """
        if not self.is_available():
            raise RuntimeError(
                "PostGISIndex.build() called but database is not reachable. "
                "Set DATABASE_URL and ensure Postgres is running."
            )

        from shared.store.pg import get_conn  # lazy import

        start = time_mod.monotonic()

        # Filter records missing the required NOT NULL fields.
        valid_records = []
        for rec in records:
            if rec.get("icao24") and rec.get("event_ts") and rec.get("lon") is not None and rec.get("lat") is not None:
                valid_records.append(rec)
            else:
                logger.debug("Skipping record with missing required fields: %s", rec.get("icao24"))

        self._record_count = 0

        with get_conn() as conn:
            # Use executemany with batching for throughput.
            for batch_start in range(0, len(valid_records), self._batch_size):
                batch = valid_records[batch_start: batch_start + self._batch_size]
                rows = [_record_to_row(r) for r in batch]
                conn.executemany(_UPSERT_SQL, rows)
                self._record_count += len(batch)
                logger.debug(
                    "Upserted batch %d-%d (%d rows)",
                    batch_start,
                    batch_start + len(batch),
                    len(batch),
                )
            conn.commit()

            # Cache index size right after build for stats().
            row = conn.execute(_SIZE_SQL).fetchone()
            self._index_size_bytes = int(row[0]) if row else 0

        self._build_time = time_mod.monotonic() - start
        logger.info(
            "PostGISIndex.build(): upserted %d records in %.3fs",
            self._record_count,
            self._build_time,
        )

    def query(self, q: SpatiotemporalQuery) -> list[dict]:
        """Execute a spatiotemporal range query via PostGIS.

        Uses ST_Intersects against the GIST-indexed ``geom geography`` column
        combined with event_ts BETWEEN for temporal filtering.  The bbox
        double-filter pattern (``&&`` for fast GIST index scan, then
        ST_Intersects for exact check) is the standard PostGIS practice for
        geography types.

        Returns:
            List of dicts with the same field set as geohash/h3 backends.

        Raises:
            RuntimeError: if the database is not reachable.
        """
        if not self.is_available():
            raise RuntimeError(
                "PostGISIndex.query() called but database is not reachable. "
                "Set DATABASE_URL and ensure Postgres is running."
            )

        from shared.store.pg import get_conn  # lazy import

        bbox = q.bbox
        tw = q.time_window

        # Parameter order matches the SQL placeholders above:
        # 8 positional args for two ST_MakeEnvelope calls (bbox reused),
        # then 2 for the BETWEEN clause.
        # ST_MakeEnvelope signature: (xmin/lon_min, ymin/lat_min, xmax/lon_max, ymax/lat_max, srid)
        params = (
            # first ST_MakeEnvelope (for && bbox operator)
            bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max,
            # second ST_MakeEnvelope (for ST_Intersects)
            bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max,
            # temporal filter
            tw.start, tw.end,
        )

        with get_conn() as conn:
            cursor = conn.execute(_QUERY_SQL, params)
            col_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(col_names, row)) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Return index statistics including on-disk size from Postgres.

        If the DB is unavailable (e.g. between build and stats calls),
        falls back to the cached value from the last successful build.
        """
        index_size = self._index_size_bytes

        # Refresh size if we can reach the DB (best-effort, non-fatal).
        if self.is_available():
            try:
                from shared.store.pg import get_conn
                with get_conn() as conn:
                    size_row = conn.execute(_SIZE_SQL).fetchone()
                    count_row = conn.execute(_COUNT_SQL).fetchone()
                    if size_row:
                        index_size = int(size_row[0])
                    if count_row:
                        self._record_count = int(count_row[0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("PostGISIndex.stats(): could not refresh DB metrics: %s", exc)

        return {
            "strategy": self.name,
            "record_count": self._record_count,
            "build_time_s": round(self._build_time, 6),
            "index_size_bytes": index_size,
            "batch_size": self._batch_size,
            "backend": "postgis_gist",
        }
