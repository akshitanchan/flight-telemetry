#!/usr/bin/env python3
"""
Normalizer — converts raw OpenSky state-vector arrays into structured
landing-format dicts.

OpenSky /states/all returns each state vector as a 17- or 18-element array:
    [0]  icao24            str
    [1]  callsign          str|null
    [2]  origin_country    str
    [3]  time_position     int|null
    [4]  last_contact      int
    [5]  longitude         float|null
    [6]  latitude          float|null
    [7]  baro_altitude     float|null   (metres)
    [8]  on_ground         bool
    [9]  velocity           float|null   (m/s)
    [10] true_track         float|null   (degrees)
    [11] vertical_rate      float|null   (m/s)
    [12] sensors            list|null
    [13] geo_altitude       float|null   (metres)
    [14] squawk             str|null
    [15] spi                bool
    [16] position_source    int
    [17] category           int|null     (aircraft category; present in 18-field responses)

This module converts each array into a flat dict in the "landing" format,
which is the raw-but-structured representation before silver transforms.
Vectors of length >= 17 are accepted; index 17 (category) is read only
when present, otherwise None is used (contract C4).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum number of fields in a valid OpenSky state vector
MIN_FIELDS = 17


def normalize_state_vector(snapshot_time: int, sv: list) -> dict | None:
    """Convert a raw OpenSky state-vector array into a landing-format dict.

    Accepts vectors of length >= 17 (contract C4).  When the vector contains
    18 or more fields, index 17 is read as ``category``; otherwise ``category``
    is ``None``.

    Args:
        snapshot_time: Unix timestamp of the API snapshot.
        sv: List of >= 17 elements from the OpenSky states array.

    Returns:
        A normalized dict, or None if the record is malformed.
    """
    if not sv or len(sv) < MIN_FIELDS:
        logger.warning(
            "Skipping state vector with %d fields (expected %d)",
            len(sv) if sv else 0,
            MIN_FIELDS,
        )
        return None

    icao24 = sv[0]
    if not icao24 or not isinstance(icao24, str):
        logger.warning("Skipping state vector with invalid icao24: %r", icao24)
        return None

    # Normalize callsign: strip whitespace, convert empty to None
    callsign = sv[1]
    if isinstance(callsign, str):
        callsign = callsign.strip() or None

    # Use time_position if available, else snapshot_time
    event_time = sv[3] if sv[3] is not None else snapshot_time

    # Index 17 (category) is only present in 18-field responses (contract C4)
    category = sv[17] if len(sv) > 17 else None

    record = {
        # --- Identity ---
        "icao24": icao24.lower().strip(),
        "callsign": callsign,
        "origin_country": sv[2],
        # --- Time ---
        "event_ts": datetime.fromtimestamp(event_time, tz=timezone.utc).isoformat(),
        "snapshot_ts": datetime.fromtimestamp(
            snapshot_time, tz=timezone.utc
        ).isoformat(),
        # --- Position ---
        "lon": sv[5],
        "lat": sv[6],
        "baro_altitude_m": sv[7],
        "geo_altitude_m": sv[13],
        "on_ground": bool(sv[8]),
        # --- Velocity ---
        "velocity_ms": sv[9],
        "true_track_deg": sv[10],
        "vertical_rate_ms": sv[11],
        # --- Transponder ---
        "squawk": sv[14],
        "spi": bool(sv[15]),
        # --- Meta ---
        "position_source": sv[16],
        "category": category,
        "last_contact": sv[4],
        # --- Idempotency key (precomputed for downstream) ---
        "idem_key": f"{icao24.lower().strip()}:{event_time}",
    }

    return record


def normalize_batch(
    snapshot_time: int, states: list[list]
) -> list[dict]:
    """Normalize a batch of state vectors from one snapshot.

    Skips malformed records and logs warnings.
    Returns list of valid landing-format dicts.
    """
    results = []
    for sv in states:
        record = normalize_state_vector(snapshot_time, sv)
        if record is not None:
            results.append(record)
    return results
