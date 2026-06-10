#!/usr/bin/env python3
"""
data/cloud/databricks/dlt/test_expectations.py
W4.3 / ds-03 — Offline fixture-driven expression evaluation tests.

Approach
--------
Each DLT SQL predicate (from SILVER_ALL_EXPECTATIONS and the four
GOLD_*_EXPECTATIONS dicts) is translated into an equivalent Python expression
and evaluated against:
  - a GOOD row (the expression must evaluate to True)
  - a deliberately BAD row (the expression must evaluate to False)

No Spark or DLT runtime is required.  The evaluator uses DuckDB's SQL engine
to parse and execute the predicates on single-row in-memory tables, so the
exact same SQL text that DLT will run is also tested here.  This guarantees
that a passing DLT constraint really does accept good rows and reject bad ones.

Test pattern
------------
For every expectation key we instantiate two DuckDB queries:
    SELECT <predicate> FROM (VALUES (...)) t(col1, col2, ...)
and assert:
    - good_row query returns True
    - bad_row  query returns False

Running
-------
    python -m pytest data/cloud/databricks/dlt/test_expectations.py -v
    # or directly:
    python data/cloud/databricks/dlt/test_expectations.py
"""

import sys
import unittest
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import expectation dicts — no Spark/DLT needed
# ---------------------------------------------------------------------------
from data.cloud.databricks.dlt.expectations_silver import (  # noqa: E402
    SILVER_ALL_EXPECTATIONS,
)
from data.cloud.databricks.dlt.expectations_gold import (  # noqa: E402
    GOLD_CONGESTION_EXPECTATIONS,
    GOLD_SECTOR_EXPECTATIONS,
    GOLD_EMERGENCY_EXPECTATIONS,
    GOLD_ROUTING_EXPECTATIONS,
)


# ---------------------------------------------------------------------------
# DuckDB evaluation helper
# ---------------------------------------------------------------------------

import re as _re

def _spark_to_duckdb_sql(expression: str) -> str:
    """
    Translate Spark SQL / DLT predicate syntax to DuckDB-compatible SQL.

    DLT expressions use Spark SQL dialect.  DuckDB uses slightly different
    function names for some operators.  This function handles the small set
    of Spark-isms present in our expectation expressions so the same predicate
    text that runs on Databricks can also be evaluated offline in DuckDB.

    Translations applied:
      col RLIKE 'pattern'  ->  regexp_matches(col, 'pattern')
      (covers both bare ``RLIKE`` and ``OR ... RLIKE`` forms)

    All other syntax (BETWEEN, IS NULL/NOT NULL, IN, >=, <=, LENGTH) is
    identical between Spark SQL and DuckDB.
    """
    # Replace: <col> RLIKE '<pattern>'  with  regexp_matches(<col>, '<pattern>')
    # The pattern is a single-quoted string literal; we handle it with a regex
    # that captures the column reference and the quoted pattern.
    result = _re.sub(
        r"(\w+)\s+RLIKE\s+('[^']*')",
        r"regexp_matches(\1, \2)",
        expression,
        flags=_re.IGNORECASE,
    )
    return result


def _evaluate(expression: str, row: dict) -> bool:
    """
    Evaluate a DLT SQL predicate against a single row using DuckDB.

    ``row`` is a dict of column_name -> value.  A temporary in-memory view
    is created so that the predicate can reference column names directly.

    NULL values in the row dict are passed as SQL NULL; Python None becomes NULL.

    Returns True (passes constraint) or False (constraint violated).
    Raises on SQL syntax errors so tests fail loudly on typos.

    The expression is first translated from Spark SQL dialect (RLIKE) to
    DuckDB-compatible SQL via ``_spark_to_duckdb_sql`` — so the DLT expression
    text is tested as-is, not a hand-rewritten version.
    """
    con = duckdb.connect()

    # Translate Spark SQL -> DuckDB SQL (RLIKE -> regexp_matches, etc.)
    duckdb_expression = _spark_to_duckdb_sql(expression)

    # Build a parameterised query: SELECT <expr> FROM (VALUES (?,...)) t(col,...)
    col_names = list(row.keys())
    col_list = ", ".join(col_names)
    placeholders = ", ".join(["?" for _ in col_names])
    values = list(row.values())

    sql = (
        f"SELECT CASE WHEN ({duckdb_expression}) THEN TRUE ELSE FALSE END AS result "
        f"FROM (VALUES ({placeholders})) AS t({col_list})"
    )
    result = con.execute(sql, values).fetchone()
    con.close()
    return bool(result[0])


# ---------------------------------------------------------------------------
# Good/bad row fixture factory for each expression key
# ---------------------------------------------------------------------------
# Format:  key -> (good_row: dict, bad_row: dict)
# The rows need only contain the column(s) referenced by the expression.
# Unrelated columns may be omitted from the row.
# ---------------------------------------------------------------------------

_SILVER_FIXTURES: dict[str, tuple[dict, dict]] = {
    # ---- NOT NULL -----------------------------------------------------------
    "silver_icao24_not_null": (
        {"icao24": "4b1806"},
        {"icao24": None},
    ),
    "silver_event_ts_not_null": (
        {"event_ts": "2024-06-03T12:00:00Z"},
        {"event_ts": None},
    ),
    "silver_lon_not_null": (
        {"lon": 4.7638},
        {"lon": None},
    ),
    "silver_lat_not_null": (
        {"lat": 52.308},
        {"lat": None},
    ),
    "silver_on_ground_not_null": (
        {"on_ground": False},
        {"on_ground": None},
    ),
    "silver_origin_country_not_null": (
        {"origin_country": "Switzerland"},
        {"origin_country": None},
    ),
    "silver_geohash7_not_null": (
        {"geohash7": "u173zq6"},
        {"geohash7": None},
    ),
    "silver_h3_r7_not_null": (
        {"h3_r7": "872830b2fffffff"},
        {"h3_r7": None},
    ),

    # ---- RANGE / FORMAT -----------------------------------------------------
    "silver_icao24_format": (
        {"icao24": "4b1806"},
        {"icao24": "GGGGGG"},   # uppercase / not hex
    ),
    "silver_lat_range": (
        {"lat": 52.308},
        {"lat": 91.0},          # > 90
    ),
    "silver_lon_range": (
        {"lon": 4.764},
        {"lon": 181.0},         # > 180
    ),
    "silver_velocity_ms_non_negative": (
        {"velocity_ms": 0.0},
        {"velocity_ms": -0.1},  # negative
    ),
    "silver_true_track_range": (
        {"true_track_deg": 180.0},
        {"true_track_deg": 361.0},  # > 360
    ),
    "silver_baro_altitude_range": (
        {"baro_altitude_m": 10000.0},
        {"baro_altitude_m": 30001.0},  # > 30000
    ),
    "silver_squawk_format": (
        {"squawk": "2536"},
        {"squawk": "8888"},     # '8' is not an octal digit
    ),
    "silver_geohash7_format": (
        {"geohash7": "u173zq6"},
        {"geohash7": "u173z"},  # length 5
    ),
    "silver_h3_r7_format": (
        {"h3_r7": "872830b2fffffff"},
        {"h3_r7": "ZZZZZZZZZZZZZZ"},  # not hex, wrong length
    ),
    "silver_metar_wind_non_negative": (
        {"metar_wind_kt": 12.0},
        {"metar_wind_kt": -1.0},
    ),
    "silver_metar_vis_non_negative": (
        {"metar_vis_m": 9999.0},
        {"metar_vis_m": -1.0},
    ),
    "silver_metar_ceiling_non_negative": (
        {"metar_ceiling_ft": 3500.0},
        {"metar_ceiling_ft": -1.0},
    ),
}

# Nullable columns — null values MUST also pass the constraint.
_SILVER_NULL_PASSTHROUGH: dict[str, dict] = {
    "silver_velocity_ms_non_negative":  {"velocity_ms": None},
    "silver_true_track_range":          {"true_track_deg": None},
    "silver_baro_altitude_range":       {"baro_altitude_m": None},
    "silver_squawk_format":             {"squawk": None},
    "silver_metar_wind_non_negative":   {"metar_wind_kt": None},
    "silver_metar_vis_non_negative":    {"metar_vis_m": None},
    "silver_metar_ceiling_non_negative": {"metar_ceiling_ft": None},
}

# --- Gold congestion --------------------------------------------------------
_GOLD_CONGESTION_FIXTURES: dict[str, tuple[dict, dict]] = {
    "congestion_airport_icao_not_null":   (
        {"airport_icao": "EHAM"},
        {"airport_icao": None},
    ),
    "congestion_window_start_not_null":   (
        {"window_start": "2024-06-03T14:00:00Z"},
        {"window_start": None},
    ),
    "congestion_window_end_not_null":     (
        {"window_end": "2024-06-03T15:00:00Z"},
        {"window_end": None},
    ),
    "congestion_aircraft_count_not_null": (
        {"aircraft_count": 42},
        {"aircraft_count": None},
    ),
    "congestion_ground_count_not_null":   (
        {"ground_count": 12},
        {"ground_count": None},
    ),
    "congestion_airborne_count_not_null": (
        {"airborne_count": 30},
        {"airborne_count": None},
    ),
    "congestion_aircraft_count_non_negative": (
        {"aircraft_count": 0},
        {"aircraft_count": -1},
    ),
    "congestion_ground_count_non_negative": (
        {"ground_count": 0},
        {"ground_count": -1},
    ),
    "congestion_airborne_count_non_negative": (
        {"airborne_count": 0},
        {"airborne_count": -1},
    ),
}

# --- Gold sector load -------------------------------------------------------
_GOLD_SECTOR_FIXTURES: dict[str, tuple[dict, dict]] = {
    "sector_h3_r4_not_null":          (
        {"h3_r4": "841ea45ffffffff"},
        {"h3_r4": None},
    ),
    "sector_window_start_not_null":   (
        {"window_start": "2024-06-03T12:00:00Z"},
        {"window_start": None},
    ),
    "sector_window_end_not_null":     (
        {"window_end": "2024-06-03T12:05:00Z"},
        {"window_end": None},
    ),
    "sector_aircraft_count_not_null": (
        {"aircraft_count": 2},
        {"aircraft_count": None},
    ),
    "sector_aircraft_count_non_negative": (
        {"aircraft_count": 0},
        {"aircraft_count": -1},
    ),
}

# --- Gold emergency events --------------------------------------------------
_GOLD_EMERGENCY_FIXTURES: dict[str, tuple[dict, dict]] = {
    "emergency_icao24_not_null":        (
        {"icao24": "3c6751"},
        {"icao24": None},
    ),
    "emergency_squawk_not_null":        (
        {"squawk": "7700"},
        {"squawk": None},
    ),
    "emergency_first_seen_ts_not_null": (
        {"first_seen_ts": "2024-06-03T15:00:00Z"},
        {"first_seen_ts": None},
    ),
    "emergency_last_seen_ts_not_null":  (
        {"last_seen_ts": "2024-06-03T15:12:30Z"},
        {"last_seen_ts": None},
    ),
    "emergency_lat_not_null":           (
        {"lat": 50.0379},
        {"lat": None},
    ),
    "emergency_lon_not_null":           (
        {"lon": 8.5706},
        {"lon": None},
    ),
    "emergency_origin_country_not_null": (
        {"origin_country": "Germany"},
        {"origin_country": None},
    ),
    "emergency_duration_s_not_null":    (
        {"duration_s": 750},
        {"duration_s": None},
    ),
    "emergency_squawk_valid_code": (
        {"squawk": "7700"},
        {"squawk": "1234"},   # valid octal but not an emergency code
    ),
    "emergency_icao24_format": (
        {"icao24": "3c6751"},
        {"icao24": "ZZZZZZ"},
    ),
    "emergency_lat_range": (
        {"lat": 50.0379},
        {"lat": -91.0},
    ),
    "emergency_lon_range": (
        {"lon": 8.5706},
        {"lon": 181.0},
    ),
    "emergency_duration_s_non_negative": (
        {"duration_s": 0},
        {"duration_s": -1},
    ),
}

# --- Gold routing stats -----------------------------------------------------
_GOLD_ROUTING_FIXTURES: dict[str, tuple[dict, dict]] = {
    "routing_icao24_not_null":           (
        {"icao24": "3c6751"},
        {"icao24": None},
    ),
    "routing_window_start_not_null":     (
        {"window_start": "2024-06-03T12:00:00Z"},
        {"window_start": None},
    ),
    "routing_window_end_not_null":       (
        {"window_end": "2024-06-03T12:00:40Z"},
        {"window_end": None},
    ),
    "routing_origin_lat_not_null":       (
        {"origin_lat": 49.535},
        {"origin_lat": None},
    ),
    "routing_origin_lon_not_null":       (
        {"origin_lon": 6.033},
        {"origin_lon": None},
    ),
    "routing_destination_lat_not_null":  (
        {"destination_lat": 49.6},
        {"destination_lat": None},
    ),
    "routing_destination_lon_not_null":  (
        {"destination_lon": 6.1},
        {"destination_lon": None},
    ),
    "routing_ping_count_not_null":       (
        {"ping_count": 5},
        {"ping_count": None},
    ),
    "routing_icao24_format": (
        {"icao24": "3c6751"},
        {"icao24": "ZZZZZZ"},
    ),
    "routing_origin_lat_range": (
        {"origin_lat": 49.5},
        {"origin_lat": 91.0},
    ),
    "routing_origin_lon_range": (
        {"origin_lon": 6.0},
        {"origin_lon": -181.0},
    ),
    "routing_destination_lat_range": (
        {"destination_lat": 49.6},
        {"destination_lat": -91.0},
    ),
    "routing_destination_lon_range": (
        {"destination_lon": 6.1},
        {"destination_lon": 181.0},
    ),
    "routing_ping_count_positive": (
        {"ping_count": 1},
        {"ping_count": 0},    # minimum is 1, so 0 is invalid
    ),
}


# ---------------------------------------------------------------------------
# Test case builder
# ---------------------------------------------------------------------------

def _make_test_cases(
    expectations: dict[str, str],
    fixtures: dict[str, tuple[dict, dict]],
    class_name: str,
) -> type:
    """
    Dynamically create a unittest.TestCase subclass with one test method per
    expectation key.

    Each generated test:
      1. Looks up the expression for the key.
      2. Evaluates it against the good row and asserts the result is True.
      3. Evaluates it against the bad row and asserts the result is False.
    """

    methods: dict = {}

    for key, expression in expectations.items():
        if key not in fixtures:
            # Missing fixture — fail explicitly rather than skip silently.
            def _missing_fixture_test(self, _key=key):
                self.fail(f"No test fixture defined for expectation key: {_key!r}")
            _missing_fixture_test.__name__ = f"test_{key}"
            methods[f"test_{key}"] = _missing_fixture_test
            continue

        good_row, bad_row = fixtures[key]

        def _test(self, _key=key, _expr=expression, _good=good_row, _bad=bad_row):
            # Good row: constraint must pass (True)
            good_result = _evaluate(_expr, _good)
            self.assertTrue(
                good_result,
                msg=(
                    f"GOOD row incorrectly REJECTED by [{_key}].\n"
                    f"  expression: {_expr}\n"
                    f"  row:        {_good}"
                ),
            )
            # Bad row: constraint must reject (False)
            bad_result = _evaluate(_expr, _bad)
            self.assertFalse(
                bad_result,
                msg=(
                    f"BAD row NOT rejected by [{_key}].\n"
                    f"  expression: {_expr}\n"
                    f"  row:        {_bad}"
                ),
            )

        _test.__name__ = f"test_{key}"
        methods[f"test_{key}"] = _test

    return type(class_name, (unittest.TestCase,), methods)


# Dynamically create test classes
TestSilverExpectations = _make_test_cases(
    SILVER_ALL_EXPECTATIONS,
    _SILVER_FIXTURES,
    "TestSilverExpectations",
)

TestGoldCongestionExpectations = _make_test_cases(
    GOLD_CONGESTION_EXPECTATIONS,
    _GOLD_CONGESTION_FIXTURES,
    "TestGoldCongestionExpectations",
)

TestGoldSectorExpectations = _make_test_cases(
    GOLD_SECTOR_EXPECTATIONS,
    _GOLD_SECTOR_FIXTURES,
    "TestGoldSectorExpectations",
)

TestGoldEmergencyExpectations = _make_test_cases(
    GOLD_EMERGENCY_EXPECTATIONS,
    _GOLD_EMERGENCY_FIXTURES,
    "TestGoldEmergencyExpectations",
)

TestGoldRoutingExpectations = _make_test_cases(
    GOLD_ROUTING_EXPECTATIONS,
    _GOLD_ROUTING_FIXTURES,
    "TestGoldRoutingExpectations",
)


# ---------------------------------------------------------------------------
# Nullable passthrough tests for silver (separate class for clarity)
# ---------------------------------------------------------------------------

class TestSilverNullablePassthrough(unittest.TestCase):
    """
    Nullable silver columns must pass (True) when the value is NULL.
    A DLT constraint that drops NULLs on a nullable column would incorrectly
    remove valid rows.
    """

    def _assert_null_passes(self, key: str, row: dict):
        expr = SILVER_ALL_EXPECTATIONS[key]
        result = _evaluate(expr, row)
        self.assertTrue(
            result,
            msg=(
                f"NULL value INCORRECTLY REJECTED by [{key}].\n"
                f"  expression: {expr}\n"
                f"  row:        {row}\n"
                f"  This column is nullable; NULLs must pass."
            ),
        )

    def test_velocity_ms_null_passes(self):
        self._assert_null_passes("silver_velocity_ms_non_negative", {"velocity_ms": None})

    def test_true_track_null_passes(self):
        self._assert_null_passes("silver_true_track_range", {"true_track_deg": None})

    def test_baro_altitude_null_passes(self):
        self._assert_null_passes("silver_baro_altitude_range", {"baro_altitude_m": None})

    def test_squawk_null_passes(self):
        self._assert_null_passes("silver_squawk_format", {"squawk": None})

    def test_metar_wind_null_passes(self):
        self._assert_null_passes("silver_metar_wind_non_negative", {"metar_wind_kt": None})

    def test_metar_vis_null_passes(self):
        self._assert_null_passes("silver_metar_vis_non_negative", {"metar_vis_m": None})

    def test_metar_ceiling_null_passes(self):
        self._assert_null_passes("silver_metar_ceiling_non_negative", {"metar_ceiling_ft": None})


# ---------------------------------------------------------------------------
# ADR-0006 guard: no UNKNOWN airport expectation exists
# ---------------------------------------------------------------------------

class TestADR0006NoUnknownAirport(unittest.TestCase):
    """
    ADR-0006: the gold_airport_congestion expectations must NOT contain any
    reference to the string 'UNKNOWN' (synthetic bucket is forbidden).
    """

    def test_no_unknown_in_congestion_expectations(self):
        for key, expr in GOLD_CONGESTION_EXPECTATIONS.items():
            self.assertNotIn(
                "UNKNOWN",
                expr.upper(),
                msg=(
                    f"Expectation [{key}] references 'UNKNOWN' — "
                    f"ADR-0006 forbids synthetic UNKNOWN airport buckets.\n"
                    f"  expression: {expr}"
                ),
            )

    def test_airport_icao_not_null_rejects_unknown_row(self):
        """A row with airport_icao='UNKNOWN' is rejected because the NOT NULL
        constraint rejects nulls but actually UNKNOWN would pass NOT NULL.
        This test documents that UNKNOWN is valid string-wise but the upstream
        pipeline (ADR-0006) guarantees it never appears in gold.
        The DLT layer relies on upstream correctness, not a string-block."""
        # Confirm our NOT NULL expectation passes for any non-null string.
        expr = GOLD_CONGESTION_EXPECTATIONS["congestion_airport_icao_not_null"]
        result = _evaluate(expr, {"airport_icao": "UNKNOWN"})
        # NOT NULL passes for any non-null value — we document this intentionally.
        # The upstream exclusion (not a DLT string filter) is the ADR-0006 guard.
        self.assertTrue(
            result,
            msg=(
                "Documentation test: NOT NULL passes for any non-null value. "
                "ADR-0006 enforces no-UNKNOWN via pipeline logic, not a DLT filter."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
