"""Gold-data loader for the dashboard.

Reads the four gold JSONL files from a configurable directory and returns
pandas DataFrames with the canonical C3 columns.  Design choices:

* NO ``import streamlit`` — this module is import-testable without a browser.
* ``GOLD_DIR`` defaults to ``data/processed/`` relative to the project root, or
  can be overridden via the ``GOLD_DIR`` environment variable or by passing
  ``gold_dir`` directly to ``load_gold()``.
* Missing files → empty DataFrame (no crash).  The dashboard degrades
  gracefully.
* Optional C2-Postgres source: if ``DATABASE_URL`` is set in the environment
  AND psycopg is available, ``load_gold()`` will try the DB first, falling
  back silently to local JSONL on any error.  A live database is never
  required.
"""

import json
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GOLD_DIR = _PROJECT_ROOT / "data" / "processed"

GOLD_FILES: dict[str, str] = {
    "congestion": "gold_airport_congestion.jsonl",
    "sector": "gold_sector_load.jsonl",
    "emergency": "gold_emergency_events.jsonl",
    "routing": "gold_routing_stats.jsonl",
}

# Canonical columns per table — derived from shared/contracts/gold_*.schema.json
GOLD_COLUMNS: dict[str, list[str]] = {
    "congestion": [
        "airport_icao",
        "window_start",
        "window_end",
        "aircraft_count",
        "avg_altitude_m",
        "ground_count",
        "airborne_count",
    ],
    "sector": [
        "h3_r4",
        "window_start",
        "window_end",
        "aircraft_count",
    ],
    "emergency": [
        "icao24",
        "callsign",
        "squawk",
        "first_seen_ts",
        "last_seen_ts",
        "lat",
        "lon",
        "origin_country",
        "nearest_airport",
        "duration_s",
    ],
    "routing": [
        "icao24",
        "callsign",
        "window_start",
        "window_end",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "max_altitude_m",
        "avg_velocity_mps",
        "ping_count",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning an empty list when the file is absent."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _records_to_df(records: list[dict], table: str) -> pd.DataFrame:
    """Convert a list of dicts to a DataFrame with only the C3 columns.

    Columns absent from the source data are added as ``pd.NA`` so the schema
    is always consistent.
    """
    columns = GOLD_COLUMNS[table]
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    # Keep only known C3 columns; add missing ones as NA
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


# ---------------------------------------------------------------------------
# Optional Postgres (C2) source — availability-gated
# ---------------------------------------------------------------------------

_SQL_QUERIES: dict[str, str] = {
    "congestion": (
        "SELECT airport_icao, window_start, window_end, aircraft_count,"
        " avg_altitude_m, ground_count, airborne_count"
        " FROM gold_airport_congestion"
    ),
    "sector": (
        "SELECT h3_r4, window_start, window_end, aircraft_count"
        " FROM gold_sector_load"
    ),
    "emergency": (
        "SELECT icao24, callsign, squawk, first_seen_ts, last_seen_ts,"
        " lat, lon, origin_country, nearest_airport, duration_s"
        " FROM gold_emergency_events"
    ),
    "routing": (
        "SELECT icao24, callsign, window_start, window_end,"
        " origin_lat, origin_lon, destination_lat, destination_lon,"
        " max_altitude_m, avg_velocity_mps, ping_count"
        " FROM gold_routing_stats"
    ),
}


def _try_load_from_postgres() -> dict[str, pd.DataFrame] | None:
    """Attempt to load all four tables from Postgres.

    Returns a dict of DataFrames if successful, or ``None`` on any error
    (missing DATABASE_URL, psycopg not installed, connection refused, etc.).
    """
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return None
    try:
        import psycopg  # type: ignore[import-untyped]

        with psycopg.connect(database_url) as conn:
            result: dict[str, pd.DataFrame] = {}
            for table, sql in _SQL_QUERIES.items():
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    col_names = [desc[0] for desc in cur.description]
                records = [dict(zip(col_names, row)) for row in rows]
                result[table] = _records_to_df(records, table)
            return result
    except Exception:  # noqa: BLE001 — intentional broad catch; DB is optional
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GoldTables:
    """Container for the four gold DataFrames.

    Attributes are the table names: ``congestion``, ``sector``,
    ``emergency``, ``routing``.
    """

    def __init__(
        self,
        congestion: pd.DataFrame,
        sector: pd.DataFrame,
        emergency: pd.DataFrame,
        routing: pd.DataFrame,
    ) -> None:
        self.congestion = congestion
        self.sector = sector
        self.emergency = emergency
        self.routing = routing

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GoldTables("
            f"congestion={len(self.congestion)} rows, "
            f"sector={len(self.sector)} rows, "
            f"emergency={len(self.emergency)} rows, "
            f"routing={len(self.routing)} rows)"
        )


def load_gold(gold_dir: str | Path | None = None) -> GoldTables:
    """Load the four gold tables, returning a :class:`GoldTables` instance.

    Resolution order:

    1. If ``DATABASE_URL`` is set in the environment, try Postgres (C2).
       On any error, fall through silently.
    2. Read JSONL files from ``gold_dir``.

    Parameters
    ----------
    gold_dir:
        Directory containing the gold ``*.jsonl`` files.  Defaults to the
        ``GOLD_DIR`` environment variable if set, otherwise
        ``<project_root>/data/processed/``.

    Returns
    -------
    GoldTables
        Container with one DataFrame per gold table.  An empty DataFrame (with
        the correct columns) is returned for any missing file.
    """
    # Try Postgres first if DATABASE_URL is configured
    pg_result = _try_load_from_postgres()
    if pg_result is not None:
        return GoldTables(**pg_result)

    # Resolve local JSONL directory
    if gold_dir is None:
        gold_dir = os.environ.get("GOLD_DIR", str(_DEFAULT_GOLD_DIR))
    gold_path = Path(gold_dir)

    tables: dict[str, pd.DataFrame] = {}
    for table, filename in GOLD_FILES.items():
        records = _load_jsonl(gold_path / filename)
        tables[table] = _records_to_df(records, table)

    return GoldTables(**tables)
