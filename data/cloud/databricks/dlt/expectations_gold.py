"""
data/cloud/databricks/dlt/expectations_gold.py
W4.3 / ds-03 — DLT data-quality expectations for the four gold tables.

Contract sources (shared/contracts/):
  gold_airport_congestion.schema.json  v1.0.0
  gold_sector_load.schema.json         v1.0.0
  gold_emergency_events.schema.json    v1.0.0
  gold_routing_stats.schema.json       v1.0.0

ADR-0006: airport_congestion EXCLUDES no-airport records; there is NO
expectation that manufactures or allows an "UNKNOWN" airport bucket.

Guard: the ``dlt`` import is wrapped so the file passes ``python -m
py_compile`` and unit-tests offline without a Databricks runtime.
"""

# ---------------------------------------------------------------------------
# Offline-safe dlt import guard (same pattern as expectations_silver.py)
# ---------------------------------------------------------------------------
try:
    import dlt  # type: ignore[import]  # noqa: F401
    _ON_DATABRICKS = True
except ModuleNotFoundError:
    _ON_DATABRICKS = False

    class _DltStub:
        @staticmethod
        def expect_all_or_drop(expectations: dict):
            def decorator(fn):
                return fn
            return decorator

        @staticmethod
        def table(name: str = "", **kwargs):
            def decorator(fn):
                return fn
            return decorator

        @staticmethod
        def read_stream(table_name: str):
            return None

    dlt = _DltStub()  # type: ignore[assignment]


# ===========================================================================
# gold_airport_congestion expectations
# Schema: gold_airport_congestion.schema.json
#
# required: airport_icao, window_start, window_end, aircraft_count,
#           ground_count, airborne_count
#
# Range rules:
#   aircraft_count: minimum 0
#   ground_count:   minimum 0
#   airborne_count: minimum 0
#
# ADR-0006: airport_icao IS NOT NULL (records with no airport are excluded
#           upstream; no UNKNOWN sentinel must exist in gold).
# ===========================================================================

GOLD_CONGESTION_EXPECTATIONS: dict[str, str] = {
    # NOT NULL — required fields
    "congestion_airport_icao_not_null":   "airport_icao IS NOT NULL",
    "congestion_window_start_not_null":   "window_start IS NOT NULL",
    "congestion_window_end_not_null":     "window_end IS NOT NULL",
    "congestion_aircraft_count_not_null": "aircraft_count IS NOT NULL",
    "congestion_ground_count_not_null":   "ground_count IS NOT NULL",
    "congestion_airborne_count_not_null": "airborne_count IS NOT NULL",

    # Range — properties.aircraft_count minimum 0
    "congestion_aircraft_count_non_negative": "aircraft_count >= 0",

    # Range — properties.ground_count minimum 0
    "congestion_ground_count_non_negative":   "ground_count >= 0",

    # Range — properties.airborne_count minimum 0
    "congestion_airborne_count_non_negative": "airborne_count >= 0",
}


# ===========================================================================
# gold_sector_load expectations
# Schema: gold_sector_load.schema.json
#
# required: h3_r4, window_start, window_end, aircraft_count
#
# Range rules:
#   aircraft_count: minimum 0
# ===========================================================================

GOLD_SECTOR_EXPECTATIONS: dict[str, str] = {
    # NOT NULL — required fields
    "sector_h3_r4_not_null":           "h3_r4 IS NOT NULL",
    "sector_window_start_not_null":    "window_start IS NOT NULL",
    "sector_window_end_not_null":      "window_end IS NOT NULL",
    "sector_aircraft_count_not_null":  "aircraft_count IS NOT NULL",

    # Range — properties.aircraft_count minimum 0
    "sector_aircraft_count_non_negative": "aircraft_count >= 0",
}


# ===========================================================================
# gold_emergency_events expectations
# Schema: gold_emergency_events.schema.json
#
# required: icao24, squawk, first_seen_ts, last_seen_ts, lat, lon,
#           origin_country, duration_s
#
# Range rules:
#   lat:        minimum -90, maximum 90
#   lon:        minimum -180, maximum 180
#   duration_s: minimum 0
#
# Enum rule:
#   squawk: enum ["7500", "7600", "7700"]
#
# Pattern rule:
#   icao24: pattern ^[0-9a-f]{6}$
# ===========================================================================

GOLD_EMERGENCY_EXPECTATIONS: dict[str, str] = {
    # NOT NULL — required fields
    "emergency_icao24_not_null":        "icao24 IS NOT NULL",
    "emergency_squawk_not_null":        "squawk IS NOT NULL",
    "emergency_first_seen_ts_not_null": "first_seen_ts IS NOT NULL",
    "emergency_last_seen_ts_not_null":  "last_seen_ts IS NOT NULL",
    "emergency_lat_not_null":           "lat IS NOT NULL",
    "emergency_lon_not_null":           "lon IS NOT NULL",
    "emergency_origin_country_not_null": "origin_country IS NOT NULL",
    "emergency_duration_s_not_null":    "duration_s IS NOT NULL",

    # Enum — properties.squawk enum ["7500","7600","7700"]
    "emergency_squawk_valid_code":
        "squawk IN ('7500', '7600', '7700')",

    # Pattern — properties.icao24 pattern ^[0-9a-f]{6}$
    "emergency_icao24_format":
        "icao24 RLIKE '^[0-9a-f]{6}$'",

    # Range — properties.lat minimum -90, maximum 90
    "emergency_lat_range":
        "lat BETWEEN -90 AND 90",

    # Range — properties.lon minimum -180, maximum 180
    "emergency_lon_range":
        "lon BETWEEN -180 AND 180",

    # Range — properties.duration_s minimum 0
    "emergency_duration_s_non_negative":
        "duration_s >= 0",
}


# ===========================================================================
# gold_routing_stats expectations
# Schema: gold_routing_stats.schema.json
#
# required: icao24, window_start, window_end, origin_lat, origin_lon,
#           destination_lat, destination_lon, ping_count
#
# Range rules:
#   origin_lat:      minimum -90.0, maximum 90.0
#   origin_lon:      minimum -180.0, maximum 180.0
#   destination_lat: minimum -90.0, maximum 90.0
#   destination_lon: minimum -180.0, maximum 180.0
#   ping_count:      minimum 1
#
# Pattern rule:
#   icao24: pattern ^[0-9a-f]{6}$
# ===========================================================================

GOLD_ROUTING_EXPECTATIONS: dict[str, str] = {
    # NOT NULL — required fields
    "routing_icao24_not_null":           "icao24 IS NOT NULL",
    "routing_window_start_not_null":     "window_start IS NOT NULL",
    "routing_window_end_not_null":       "window_end IS NOT NULL",
    "routing_origin_lat_not_null":       "origin_lat IS NOT NULL",
    "routing_origin_lon_not_null":       "origin_lon IS NOT NULL",
    "routing_destination_lat_not_null":  "destination_lat IS NOT NULL",
    "routing_destination_lon_not_null":  "destination_lon IS NOT NULL",
    "routing_ping_count_not_null":       "ping_count IS NOT NULL",

    # Pattern — properties.icao24 pattern ^[0-9a-f]{6}$
    "routing_icao24_format":
        "icao24 RLIKE '^[0-9a-f]{6}$'",

    # Range — origin_lat minimum -90.0, maximum 90.0
    "routing_origin_lat_range":
        "origin_lat BETWEEN -90.0 AND 90.0",

    # Range — origin_lon minimum -180.0, maximum 180.0
    "routing_origin_lon_range":
        "origin_lon BETWEEN -180.0 AND 180.0",

    # Range — destination_lat minimum -90.0, maximum 90.0
    "routing_destination_lat_range":
        "destination_lat BETWEEN -90.0 AND 90.0",

    # Range — destination_lon minimum -180.0, maximum 180.0
    "routing_destination_lon_range":
        "destination_lon BETWEEN -180.0 AND 180.0",

    # Range — ping_count minimum 1
    "routing_ping_count_positive":
        "ping_count >= 1",
}


# ===========================================================================
# DLT table functions — one per gold table
# ===========================================================================

@dlt.table(  # type: ignore[misc]
    name="gold_airport_congestion_validated",
    comment=(
        "Gold airport congestion with DLT quality expectations (ds-03). "
        "ADR-0006: airport_icao IS NOT NULL; no UNKNOWN bucket. "
        "Violating rows are dropped and counted in expectation metrics."
    ),
)
@dlt.expect_all_or_drop(GOLD_CONGESTION_EXPECTATIONS)  # type: ignore[misc]
def gold_airport_congestion_validated():
    if not _ON_DATABRICKS:
        raise RuntimeError("DLT table function — not callable offline.")
    return dlt.read_stream("gold_airport_congestion")  # type: ignore[union-attr]


@dlt.table(  # type: ignore[misc]
    name="gold_sector_load_validated",
    comment=(
        "Gold sector load with DLT quality expectations (ds-03). "
        "Violating rows are dropped and counted in expectation metrics."
    ),
)
@dlt.expect_all_or_drop(GOLD_SECTOR_EXPECTATIONS)  # type: ignore[misc]
def gold_sector_load_validated():
    if not _ON_DATABRICKS:
        raise RuntimeError("DLT table function — not callable offline.")
    return dlt.read_stream("gold_sector_load")  # type: ignore[union-attr]


@dlt.table(  # type: ignore[misc]
    name="gold_emergency_events_validated",
    comment=(
        "Gold emergency events with DLT quality expectations (ds-03). "
        "squawk must be in (7500, 7600, 7700). "
        "Violating rows are dropped and counted in expectation metrics."
    ),
)
@dlt.expect_all_or_drop(GOLD_EMERGENCY_EXPECTATIONS)  # type: ignore[misc]
def gold_emergency_events_validated():
    if not _ON_DATABRICKS:
        raise RuntimeError("DLT table function — not callable offline.")
    return dlt.read_stream("gold_emergency_events")  # type: ignore[union-attr]


@dlt.table(  # type: ignore[misc]
    name="gold_routing_stats_validated",
    comment=(
        "Gold routing stats with DLT quality expectations (ds-03). "
        "ping_count >= 1; all required coord fields non-null. "
        "Violating rows are dropped and counted in expectation metrics."
    ),
)
@dlt.expect_all_or_drop(GOLD_ROUTING_EXPECTATIONS)  # type: ignore[misc]
def gold_routing_stats_validated():
    if not _ON_DATABRICKS:
        raise RuntimeError("DLT table function — not callable offline.")
    return dlt.read_stream("gold_routing_stats")  # type: ignore[union-attr]
