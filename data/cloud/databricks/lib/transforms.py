"""
Pure-Python row-transform logic for bronze → silver_flight_state.

This module is intentionally free of Spark / PySpark so it can be imported
and unit-tested offline without a cluster.  The Databricks notebook imports
this module and applies ``transform_record`` inside a Spark UDF or mapInPandas
call.

All validation rules mirror ``data/transforms/bronze_to_silver.py`` exactly:
  - icao24: 6 hex-char string
  - lon/lat: presence + range (-180..180 / -90..90)
  - velocity_ms: nullable, must be >= 0 when present
  - true_track_deg: nullable, must be 0..360 when present
  - baro_altitude_m: nullable, must be -1000..30000 when present (audit M16)
  - squawk: null OR exactly 4 octal digits
  - callsign: strip whitespace; empty string → None (audit M15)
  - geohash7: pygeohash precision 7
  - h3_r7: h3 resolution 7
  - nearest_airport: haversine within 50 km against airports reference list
  - METAR stubs: all null (enrichment not yet wired in cloud path)
  - Dedup key: idem_key field if present, else "{icao24}:{event_ts}"
  - event_date: date portion of event_ts (written as CAST(event_ts AS DATE) in Spark)
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

try:
    import h3 as _h3
    HAS_H3 = True
except ImportError:  # pragma: no cover — never missing in production
    HAS_H3 = False

try:
    import pygeohash as _pygeohash
    HAS_PYGEOHASH = True
except ImportError:  # pragma: no cover
    HAS_PYGEOHASH = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OCTAL_DIGITS = frozenset("01234567")
_ICAO24_RE = re.compile(r"^[0-9a-f]{6}$")
_MAX_AIRPORT_DIST_KM = 50.0


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def haversine_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres.

    Mirrors the identical function in ``data/transforms/bronze_to_silver.py``.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_airport(
    lat: float,
    lon: float,
    airports: list,
    max_km: float = _MAX_AIRPORT_DIST_KM,
) -> Optional[str]:
    """Return the ICAO code of the nearest airport within *max_km*, or None.

    ``airports`` is a list of dicts with keys ``icao``, ``lat``, ``lon``.
    Mirrors the identical function in ``data/transforms/bronze_to_silver.py``.
    """
    closest_dist = float("inf")
    closest_icao = None
    for ap in airports:
        dist = haversine_dist_km(lat, lon, ap["lat"], ap["lon"])
        if dist < closest_dist:
            closest_dist = dist
            closest_icao = ap["icao"]
    return closest_icao if closest_dist <= max_km else None


def load_airports_reference(ref_path) -> list:
    """Load airports JSON reference from a local or DBFS/UC path string/Path.

    Returns an empty list if the file does not exist.
    """
    p = Path(ref_path)
    if not p.exists():
        logger.warning("Airports reference not found at %s — nearest_airport will be null", ref_path)
        return []
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Range validation helper
# ---------------------------------------------------------------------------

def _validate_range(
    value,
    field: str,
    minimum=None,
    maximum=None,
) -> bool:
    """Return False if *value* is outside [minimum, maximum]; None passes."""
    if value is None:
        return True
    if minimum is not None and value < minimum:
        logger.debug("Range check failed: %s=%s < %s", field, value, minimum)
        return False
    if maximum is not None and value > maximum:
        logger.debug("Range check failed: %s=%s > %s", field, value, maximum)
        return False
    return True


# ---------------------------------------------------------------------------
# Core per-row transform
# ---------------------------------------------------------------------------

def transform_record(record: dict) -> Optional[dict]:
    """Transform a single bronze landing record into a silver-compliant dict.

    Returns the transformed dict on success, or ``None`` if the record should
    be dropped.  The caller is responsible for deduplication — this function
    is stateless and operates on a single record.

    Mirrored rules (authoritative source: data/transforms/bronze_to_silver.py):

    Required checks
    ---------------
    * icao24 must be a 6-character lowercase hex string
    * event_ts must be present
    * lon AND lat must be present

    Range validation
    ----------------
    * lon: -180 .. 180
    * lat: -90 .. 90
    * velocity_ms: >= 0 (nullable)
    * true_track_deg: 0 .. 360 (nullable)
    * baro_altitude_m: -1000 .. 30000 (nullable, audit M16)

    Field normalisation
    -------------------
    * squawk: set to None unless it is a 4-character string of octal digits
    * callsign: strip whitespace; empty string becomes None (audit M15)

    Derived fields
    --------------
    * geohash7: pygeohash.encode(lat, lon, precision=7)
    * h3_r7: h3.latlng_to_cell(lat, lon, 7)
    * nearest_airport: passed in by caller after calling nearest_airport()
    * METAR stubs: all None (enrichment not yet wired)

    event_date is NOT added here; it is computed by Spark as
    ``CAST(event_ts AS DATE)`` so it stays in the Spark layer.
    """
    # ------------------------------------------------------------------
    # 1. Required field checks
    # ------------------------------------------------------------------
    icao24 = record.get("icao24")
    if not icao24 or not isinstance(icao24, str) or len(icao24) != 6:
        logger.debug("Dropping record: invalid icao24 %r", icao24)
        return None

    event_ts = record.get("event_ts")
    if not event_ts:
        logger.debug("Dropping record: missing event_ts")
        return None

    lon = record.get("lon")
    lat = record.get("lat")
    if lon is None or lat is None:
        logger.debug("Dropping record: missing lon/lat")
        return None

    # ------------------------------------------------------------------
    # 2. Range validation
    # ------------------------------------------------------------------
    if not _validate_range(lon, "lon", -180, 180):
        return None
    if not _validate_range(lat, "lat", -90, 90):
        return None
    if not _validate_range(record.get("velocity_ms"), "velocity_ms", 0):
        return None
    if not _validate_range(record.get("true_track_deg"), "true_track_deg", 0, 360):
        return None
    if not _validate_range(record.get("baro_altitude_m"), "baro_altitude_m", -1000, 30000):
        return None

    # ------------------------------------------------------------------
    # 3. Squawk validation — must be 4 octal digits or null
    # ------------------------------------------------------------------
    squawk = record.get("squawk")
    if squawk is not None:
        if not isinstance(squawk, str) or len(squawk) != 4:
            squawk = None
        elif not all(c in _OCTAL_DIGITS for c in squawk):
            squawk = None

    # ------------------------------------------------------------------
    # 4. Callsign normalisation — strip whitespace; empty → None (M15)
    # ------------------------------------------------------------------
    callsign = record.get("callsign")
    if isinstance(callsign, str):
        callsign = callsign.strip() or None

    # ------------------------------------------------------------------
    # 5. Derived spatial indices
    # ------------------------------------------------------------------
    geohash7 = _pygeohash.encode(lat, lon, precision=7)
    h3_r7 = _h3.latlng_to_cell(lat, lon, 7)

    # ------------------------------------------------------------------
    # 6. Build silver record
    # ------------------------------------------------------------------
    silver = {
        "icao24": icao24,
        "callsign": callsign,
        "event_ts": event_ts,
        "lon": round(lon, 6),
        "lat": round(lat, 6),
        "baro_altitude_m": record.get("baro_altitude_m"),
        "velocity_ms": record.get("velocity_ms"),
        "true_track_deg": record.get("true_track_deg"),
        "vertical_rate_ms": record.get("vertical_rate_ms"),
        "on_ground": bool(record.get("on_ground", False)),
        "squawk": squawk,
        "origin_country": record.get("origin_country", "Unknown"),
        # Derived spatial
        "geohash7": geohash7,
        "h3_r7": h3_r7,
        # Enrichment stubs — nearest_airport is filled by caller
        "nearest_airport": None,
        "metar_wind_kt": None,
        "metar_vis_m": None,
        "metar_ceiling_ft": None,
    }
    return silver


# ---------------------------------------------------------------------------
# Dedup key helper (used by both notebook and tests)
# ---------------------------------------------------------------------------

def idem_key_for(record: dict) -> str:
    """Return the deduplication key for a bronze record.

    Prefers the ``idem_key`` field when present; falls back to
    ``"{icao24}:{event_ts}"``, mirroring ``run_transform`` in
    ``data/transforms/bronze_to_silver.py``.
    """
    return record.get("idem_key") or f"{record.get('icao24')}:{record.get('event_ts')}"
