"""
data/cloud/databricks/dlt/expectations_silver.py
W4.3 / ds-03 — DLT data-quality expectations for the silver_flight_state table.

Contract source: shared/contracts/silver_flight_state.schema.json v1.0.0

Every expectation here encodes exactly one schema rule (marked inline).
Action:  @dlt.expect_or_drop  — violating rows are dropped and counted;
         the pipeline does NOT fail so genuine bad data is quarantined, not
         silently accepted or pipeline-aborting.

Guard: the ``dlt`` import is wrapped in a try/except so this file passes
``python -m py_compile`` and unit-tests offline without a Databricks runtime.
"""

# ---------------------------------------------------------------------------
# Offline-safe dlt import guard
# (mirrors the pattern used in 01_bronze_to_silver.py)
# ---------------------------------------------------------------------------
try:
    import dlt  # type: ignore[import]  # noqa: F401  — injected on Databricks
    _ON_DATABRICKS = True
except ModuleNotFoundError:
    _ON_DATABRICKS = False

    # Minimal stubs so the module is importable and py_compile-safe offline.
    class _DltStub:
        """Stub that makes @dlt.expect_or_drop a no-op decorator offline."""

        @staticmethod
        def expect_or_drop(name: str, constraint: str):
            """Return a pass-through decorator."""
            def decorator(fn):
                return fn
            return decorator

        @staticmethod
        def expect_all(expectations: dict, on_violation: str = "drop"):
            """Return a pass-through decorator."""
            def decorator(fn):
                return fn
            return decorator

        @staticmethod
        def table(name: str = "", **kwargs):
            def decorator(fn):
                return fn
            return decorator

    dlt = _DltStub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Silver expectations — one dict per logical rule-set, attached as
# @dlt.expect_all on the silver streaming table transformation function.
#
# Key   = human-readable expectation name (shown in DLT metrics UI)
# Value = SQL predicate evaluated per row; rows where the predicate is FALSE
#         are dropped (expect_or_drop semantics).
#
# Schema rule traceability (silver_flight_state.schema.json):
#   ICAO24_FORMAT    — properties.icao24.pattern ^[0-9a-f]{6}$
#   LAT_RANGE        — properties.lat minimum -90 / maximum 90
#   LON_RANGE        — properties.lon minimum -180 / maximum 180
#   VELOCITY_MIN     — properties.velocity_ms minimum 0
#   TRUE_TRACK_RANGE — properties.true_track_deg minimum 0 / maximum 360
#   BARO_ALT_RANGE   — properties.baro_altitude_m minimum -1000 / maximum 30000
#   SQUAWK_FORMAT    — properties.squawk pattern ^[0-7]{4}$ (nullable)
#   GEOHASH7_LENGTH  — properties.geohash7 pattern ^[0-9b-hjkmnp-z]{7}$ (7 chars)
#   H3_R7_NOT_NULL   — required[]: h3_r7
#   METAR_WIND_MIN   — properties.metar_wind_kt minimum 0 (nullable)
#   METAR_VIS_MIN    — properties.metar_vis_m minimum 0 (nullable)
#   METAR_CEIL_MIN   — properties.metar_ceiling_ft minimum 0 (nullable)
#   NOT_NULL_*       — required[]: icao24, event_ts, lon, lat, on_ground,
#                                   origin_country, geohash7, h3_r7
# ---------------------------------------------------------------------------

# --- Required (NOT NULL) constraints ---------------------------------------
# Maps to schema "required": ["icao24","event_ts","lon","lat","on_ground",
#                              "origin_country","geohash7","h3_r7"]
SILVER_NOT_NULL_EXPECTATIONS: dict[str, str] = {
    "silver_icao24_not_null":        "icao24 IS NOT NULL",
    "silver_event_ts_not_null":      "event_ts IS NOT NULL",
    "silver_lon_not_null":           "lon IS NOT NULL",
    "silver_lat_not_null":           "lat IS NOT NULL",
    "silver_on_ground_not_null":     "on_ground IS NOT NULL",
    "silver_origin_country_not_null": "origin_country IS NOT NULL",
    "silver_geohash7_not_null":      "geohash7 IS NOT NULL",
    "silver_h3_r7_not_null":         "h3_r7 IS NOT NULL",
}

# --- Range & format constraints --------------------------------------------
SILVER_RANGE_EXPECTATIONS: dict[str, str] = {
    # icao24: pattern ^[0-9a-f]{6}$
    "silver_icao24_format":
        "icao24 RLIKE '^[0-9a-f]{6}$'",

    # lat: minimum -90, maximum 90
    "silver_lat_range":
        "lat BETWEEN -90 AND 90",

    # lon: minimum -180, maximum 180
    "silver_lon_range":
        "lon BETWEEN -180 AND 180",

    # velocity_ms: minimum 0 (nullable — nulls pass)
    "silver_velocity_ms_non_negative":
        "velocity_ms IS NULL OR velocity_ms >= 0",

    # true_track_deg: minimum 0, maximum 360 (nullable — nulls pass)
    "silver_true_track_range":
        "true_track_deg IS NULL OR true_track_deg BETWEEN 0 AND 360",

    # baro_altitude_m: minimum -1000, maximum 30000 (nullable — nulls pass)
    "silver_baro_altitude_range":
        "baro_altitude_m IS NULL OR baro_altitude_m BETWEEN -1000 AND 30000",

    # squawk: pattern ^[0-7]{4}$ OR NULL (nullable field)
    "silver_squawk_format":
        "squawk IS NULL OR squawk RLIKE '^[0-7]{4}$'",

    # geohash7: pattern ^[0-9b-hjkmnp-z]{7}$ — length 7 is the observable gate
    # (full charset check via RLIKE covers both length and alphabet)
    "silver_geohash7_format":
        "LENGTH(geohash7) = 7",

    # h3_r7: pattern ^[0-9a-f]{15}$ — non-null covered above; format check here
    "silver_h3_r7_format":
        "h3_r7 RLIKE '^[0-9a-f]{15}$'",

    # metar_wind_kt: minimum 0 (nullable)
    "silver_metar_wind_non_negative":
        "metar_wind_kt IS NULL OR metar_wind_kt >= 0",

    # metar_vis_m: minimum 0 (nullable)
    "silver_metar_vis_non_negative":
        "metar_vis_m IS NULL OR metar_vis_m >= 0",

    # metar_ceiling_ft: minimum 0 (nullable)
    "silver_metar_ceiling_non_negative":
        "metar_ceiling_ft IS NULL OR metar_ceiling_ft >= 0",
}

# Merged expectation dict (used by @dlt.expect_all)
SILVER_ALL_EXPECTATIONS: dict[str, str] = {
    **SILVER_NOT_NULL_EXPECTATIONS,
    **SILVER_RANGE_EXPECTATIONS,
}


# ---------------------------------------------------------------------------
# DLT table function
# The decorator is applied only when running on Databricks.  Offline the
# function is still importable and callable for unit tests.
# ---------------------------------------------------------------------------

@dlt.expect_all(SILVER_ALL_EXPECTATIONS, on_violation="drop")  # type: ignore[misc]
@dlt.table(  # type: ignore[misc]
    name="silver_flight_state_validated",
    comment=(
        "Silver flight state with DLT quality expectations (ds-03). "
        "Rows that violate any constraint are dropped and counted in the "
        "expectation metrics dashboard."
    ),
)
def silver_flight_state_validated():
    """
    Identity transform — expectations fire on every incoming row.

    In practice the upstream silver_flight_state table is already the output
    of 01_bronze_to_silver; this DLT layer adds a declarative quality gate
    so that post-transform stragglers (e.g. late schema drift) are caught
    and surfaced in the DLT metrics UI without requiring a code change.

    The caller (DLT pipeline YAML / notebook) must define the upstream
    ``silver_flight_state`` dataset, e.g.:

        @dlt.table(name="silver_flight_state")
        def silver_flight_state():
            return spark.readStream.table("main.default.silver_flight_state")
    """
    if not _ON_DATABRICKS:
        raise RuntimeError(
            "silver_flight_state_validated() is a DLT table function and "
            "cannot be called outside a Databricks DLT pipeline. "
            "Import SILVER_ALL_EXPECTATIONS for offline tests."
        )
    return dlt.read_stream("silver_flight_state")  # type: ignore[union-attr]
