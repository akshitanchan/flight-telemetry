# Databricks notebook source
# =============================================================================
# 02_silver_to_gold.py
# Wave W4.3 · Task ds-04
#
# Reads silver Delta and builds all four gold Delta tables:
#
#   gold_emergency_events   — squawk 7500/7600/7700 incidents (no window partition)
#   gold_sector_load        — distinct aircraft per H3 r4 cell per 5-min window
#   gold_airport_congestion — ground/airborne counts + avg altitude per airport / 5m
#   gold_routing_stats      — route stats per (icao24, callsign) (no window partition)
#
# Sections
# --------
#   A. Full recompute — overwrites all four gold tables from silver Delta.
#   B. Incremental MERGE INTO — gold_airport_congestion (primary incremental target).
#   C. Delta time-travel demo — VERSION AS OF / TIMESTAMP AS OF.
#
# Mirrors: data/transforms/silver_to_gold.py (authoritative local version)
# Target schemas: data/cloud/ddl/delta/gold_*.sql
# Parity anchor: data/cloud/databricks/lib/gold_logic.py
#
# UC parameterisation:
#   silver_table              — fully-qualified source Delta table
#   gold_emergency_table      — target: gold_emergency_events
#   gold_sector_table         — target: gold_sector_load
#   gold_congestion_table     — target: gold_airport_congestion
#   gold_routing_table        — target: gold_routing_stats
#   incremental_window_start  — ISO timestamp: lower bound for incremental batch
#   incremental_window_end    — ISO timestamp: upper bound for incremental batch
# =============================================================================

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Set these as **Job task parameters** (preferred) or via the Widgets panel
# MAGIC when running interactively on a Free-Edition cluster.
# MAGIC
# MAGIC | Parameter | Description | Example |
# MAGIC |---|---|---|
# MAGIC | `silver_table` | Fully-qualified source Delta table | `main.default.silver_flight_state` |
# MAGIC | `gold_emergency_table` | Target: emergency events | `main.default.gold_emergency_events` |
# MAGIC | `gold_sector_table` | Target: sector load | `main.default.gold_sector_load` |
# MAGIC | `gold_congestion_table` | Target: airport congestion | `main.default.gold_airport_congestion` |
# MAGIC | `gold_routing_table` | Target: routing stats | `main.default.gold_routing_stats` |
# MAGIC | `incremental_window_start` | Lower bound for incremental batch (ISO) | `2024-06-03T12:00:00` |
# MAGIC | `incremental_window_end` | Upper bound for incremental batch (ISO) | `2024-06-03T23:59:59` |

# COMMAND ----------

# Guard: dbutils / spark exist on Databricks; fall back to stubs so the file
# passes `python -m py_compile` and import checks offline.
try:
    dbutils  # type: ignore[name-defined]  # noqa: F821
    spark    # type: ignore[name-defined]  # noqa: F821
    _ON_DATABRICKS = True
except NameError:
    _ON_DATABRICKS = False

    class _DbutilsWidgetsStub:
        """Minimal stub so py_compile and offline import succeed."""
        def get(self, name: str, default: str = "") -> str:
            return default

        def text(self, name: str, default: str, label: str = "") -> None:
            pass

    class _DbutilsStub:
        widgets = _DbutilsWidgetsStub()

    dbutils = _DbutilsStub()  # type: ignore[assignment]
    spark = None              # type: ignore[assignment]

# COMMAND ----------

# Widget / job-param declarations (idempotent on Databricks).
if _ON_DATABRICKS:
    dbutils.widgets.text(  # type: ignore[union-attr]
        "silver_table",
        "flight_telemetry.silver_flight_state",
        "Source Delta table (catalog.schema.table)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "gold_emergency_table",
        "flight_telemetry.gold_emergency_events",
        "Target: gold_emergency_events",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "gold_sector_table",
        "flight_telemetry.gold_sector_load",
        "Target: gold_sector_load",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "gold_congestion_table",
        "flight_telemetry.gold_airport_congestion",
        "Target: gold_airport_congestion",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "gold_routing_table",
        "flight_telemetry.gold_routing_stats",
        "Target: gold_routing_stats",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "incremental_window_start",
        "",
        "Incremental batch lower bound (ISO timestamp, optional)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "incremental_window_end",
        "",
        "Incremental batch upper bound (ISO timestamp, optional)",
    )

# COMMAND ----------

# Resolve parameter values.
SILVER_TABLE: str = dbutils.widgets.get("silver_table")
GOLD_EMERGENCY_TABLE: str = dbutils.widgets.get("gold_emergency_table")
GOLD_SECTOR_TABLE: str = dbutils.widgets.get("gold_sector_table")
GOLD_CONGESTION_TABLE: str = dbutils.widgets.get("gold_congestion_table")
GOLD_ROUTING_TABLE: str = dbutils.widgets.get("gold_routing_table")
INCREMENTAL_WINDOW_START: str = dbutils.widgets.get("incremental_window_start")
INCREMENTAL_WINDOW_END: str = dbutils.widgets.get("incremental_window_end")

if _ON_DATABRICKS and not SILVER_TABLE:
    raise ValueError(
        "Widget 'silver_table' is empty. "
        "Set it to a fully-qualified table name, e.g. main.default.silver_flight_state"
    )

print(f"silver_table             : {SILVER_TABLE!r}")
print(f"gold_emergency_table     : {GOLD_EMERGENCY_TABLE!r}")
print(f"gold_sector_table        : {GOLD_SECTOR_TABLE!r}")
print(f"gold_congestion_table    : {GOLD_CONGESTION_TABLE!r}")
print(f"gold_routing_table       : {GOLD_ROUTING_TABLE!r}")
print(f"incremental_window_start : {INCREMENTAL_WINDOW_START!r}")
print(f"incremental_window_end   : {INCREMENTAL_WINDOW_END!r}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Library import
# MAGIC
# MAGIC `lib/gold_logic.py` is the pure-Python parity anchor (importable offline).
# MAGIC The notebook's Spark SQL implementations mirror its logic exactly.  The
# MAGIC shared constants `WINDOW_MINUTES` and `EMERGENCY_GAP_S` are imported here
# MAGIC for documentation cross-reference only; the Spark SQL below hard-references
# MAGIC the numeric values for clarity inside the SQL strings.

# COMMAND ----------

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.cloud.databricks.lib.gold_logic import (
    WINDOW_MINUTES,
    EMERGENCY_GAP_S,
    incremental_merge_emergency,
    incremental_merge_routing,
)

print(f"WINDOW_MINUTES  : {WINDOW_MINUTES}")   # 5
print(f"EMERGENCY_GAP_S : {EMERGENCY_GAP_S}")   # 1800

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION A — Full Recompute
# MAGIC
# MAGIC Overwrites all four gold tables from the full silver Delta table.
# MAGIC
# MAGIC Strategy per table:
# MAGIC
# MAGIC | Table | Grain | Write strategy |
# MAGIC |---|---|---|
# MAGIC | `gold_emergency_events` | (icao24, squawk, first_seen_ts) | `overwrite` (full) |
# MAGIC | `gold_sector_load` | (h3_r4, window_start) | `replaceWhere` on `window_date` |
# MAGIC | `gold_airport_congestion` | (airport_icao, window_start) | `replaceWhere` on `window_date` |
# MAGIC | `gold_routing_stats` | (icao24, callsign, window_start) | `overwrite` (full) |
# MAGIC
# MAGIC Running this section twice produces identical gold tables — the overwrite /
# MAGIC replaceWhere pattern is idempotent by design.

# COMMAND ----------
# MAGIC %md
# MAGIC ### A1 — Ensure gold tables exist

# COMMAND ----------

if _ON_DATABRICKS:
    # gold_emergency_events — no partition
    spark.sql(f"""  # type: ignore[union-attr]
        CREATE TABLE IF NOT EXISTS {GOLD_EMERGENCY_TABLE} (
            icao24          STRING    NOT NULL,
            callsign        STRING,
            squawk          STRING    NOT NULL,
            first_seen_ts   TIMESTAMP NOT NULL,
            last_seen_ts    TIMESTAMP NOT NULL,
            lat             DOUBLE    NOT NULL,
            lon             DOUBLE    NOT NULL,
            origin_country  STRING    NOT NULL,
            nearest_airport STRING,
            duration_s      BIGINT    NOT NULL
        )
        USING DELTA
        TBLPROPERTIES (
            'delta.minReaderVersion' = '1',
            'delta.minWriterVersion' = '2',
            '_contract_version'      = '1.0.0'
        )
    """)

    # gold_sector_load — partitioned by window_date
    spark.sql(f"""  # type: ignore[union-attr]
        CREATE TABLE IF NOT EXISTS {GOLD_SECTOR_TABLE} (
            h3_r4          STRING    NOT NULL,
            window_start   TIMESTAMP NOT NULL,
            window_end     TIMESTAMP NOT NULL,
            aircraft_count BIGINT    NOT NULL,
            window_date    DATE      NOT NULL
        )
        USING DELTA
        PARTITIONED BY (window_date)
        TBLPROPERTIES (
            'delta.minReaderVersion' = '1',
            'delta.minWriterVersion' = '2',
            '_contract_version'      = '1.0.0'
        )
    """)

    # gold_airport_congestion — partitioned by window_date
    spark.sql(f"""  # type: ignore[union-attr]
        CREATE TABLE IF NOT EXISTS {GOLD_CONGESTION_TABLE} (
            airport_icao   STRING    NOT NULL,
            window_start   TIMESTAMP NOT NULL,
            window_end     TIMESTAMP NOT NULL,
            aircraft_count BIGINT    NOT NULL,
            avg_altitude_m DOUBLE,
            ground_count   BIGINT    NOT NULL,
            airborne_count BIGINT    NOT NULL,
            window_date    DATE      NOT NULL
        )
        USING DELTA
        PARTITIONED BY (window_date)
        TBLPROPERTIES (
            'delta.minReaderVersion' = '1',
            'delta.minWriterVersion' = '2',
            '_contract_version'      = '1.0.0'
        )
    """)

    # gold_routing_stats — no partition
    spark.sql(f"""  # type: ignore[union-attr]
        CREATE TABLE IF NOT EXISTS {GOLD_ROUTING_TABLE} (
            icao24           STRING    NOT NULL,
            callsign         STRING,
            window_start     TIMESTAMP NOT NULL,
            window_end       TIMESTAMP NOT NULL,
            origin_lat       DOUBLE    NOT NULL,
            origin_lon       DOUBLE    NOT NULL,
            destination_lat  DOUBLE    NOT NULL,
            destination_lon  DOUBLE    NOT NULL,
            max_altitude_m   DOUBLE,
            avg_velocity_mps DOUBLE,
            ping_count       BIGINT    NOT NULL
        )
        USING DELTA
        TBLPROPERTIES (
            'delta.minReaderVersion' = '1',
            'delta.minWriterVersion' = '2',
            '_contract_version'      = '1.0.0'
        )
    """)
    print("All gold tables ensured.")
else:
    print("Offline mode — CREATE TABLE skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### A2 — Full recompute: gold_emergency_events
# MAGIC
# MAGIC Emergency events span arbitrary time ranges so they don't partition cleanly
# MAGIC by window_date.  A full `overwrite` is used for recompute.  The incremental
# MAGIC path (Section B) uses upsert by `(icao24, squawk, first_seen_ts)`.
# MAGIC
# MAGIC The gap-split logic (EMERGENCY_GAP_S = 1800 s) is implemented via a
# MAGIC session-window function with a gap threshold.  Databricks SQL's
# MAGIC `session_window()` applies a maximum inactivity gap to merge consecutive
# MAGIC observations of the same (icao24, squawk) into one session, while naturally
# MAGIC splitting on gaps > 1800 s — matching the Python loop in silver_to_gold.py.
# MAGIC
# MAGIC Only squawk 7500 / 7600 / 7700 rows are included (all others are filtered).

# COMMAND ----------

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # type: ignore[import]

    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]
    _silver_count = silver_df.count()
    print(f"Silver records read: {_silver_count:,}")

    # Filter to emergency squawks only.
    emg_silver = silver_df.filter(F.col("squawk").isin("7500", "7600", "7700"))

    # Use Spark session_window to replicate the Python gap-split loop.
    # session_window groups consecutive rows with gap < gapDuration into one
    # session — identical semantics to the EMERGENCY_GAP_S python logic when
    # gapDuration = EMERGENCY_GAP_S seconds.
    emg_df = (
        emg_silver
        .groupBy(
            "icao24",
            "squawk",
            F.session_window("event_ts", f"{EMERGENCY_GAP_S} seconds").alias("session"),
        )
        .agg(
            # Prefer the first non-null callsign seen in the session.
            F.first("callsign", ignorenulls=True).alias("callsign"),
            F.min("event_ts").alias("first_seen_ts"),
            F.max("event_ts").alias("last_seen_ts"),
            # lat/lon at first observation (origin of the emergency event).
            F.first("lat").alias("lat"),
            F.first("lon").alias("lon"),
            F.first("origin_country", ignorenulls=True).alias("origin_country"),
            F.first("nearest_airport", ignorenulls=True).alias("nearest_airport"),
        )
        .withColumn(
            "duration_s",
            (F.unix_timestamp("last_seen_ts") - F.unix_timestamp("first_seen_ts"))
            .cast("bigint"),
        )
        .drop("session")
    )

    (
        emg_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_EMERGENCY_TABLE)
    )
    _emg_count = spark.table(GOLD_EMERGENCY_TABLE).count()  # type: ignore[union-attr]
    print(f"gold_emergency_events written: {_emg_count:,} rows")
else:
    print("Offline mode — emergency recompute skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### A3 — Full recompute: gold_sector_load
# MAGIC
# MAGIC Distinct aircraft per H3 res-4 cell per 5-minute window.
# MAGIC
# MAGIC `date_trunc` + arithmetic floors event_ts to the 5-minute window bucket,
# MAGIC exactly matching `_floor_to_window` in the Python reference:
# MAGIC
# MAGIC ```
# MAGIC floor(minute / 5) * 5  →  window_start
# MAGIC window_start + INTERVAL 5 MINUTES  →  window_end
# MAGIC ```
# MAGIC
# MAGIC Recompute uses `replaceWhere` on all window_date values in the silver table
# MAGIC so partial re-runs for specific date ranges remain idempotent.

# COMMAND ----------

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # noqa: F811

    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]

    sector_df = (
        silver_df
        .filter(F.col("h3_r7").isNotNull())
        # Derive h3_r4 from h3_r7 using a Spark UDF backed by the h3 library.
        # cell_to_parent is pure-Python and cheap; it runs once per partition.
        .withColumn(
            "h3_r4",
            F.udf(lambda cell: __import__("h3").cell_to_parent(cell, 4),
                  returnType=__import__("pyspark.sql.types", fromlist=["StringType"]).StringType()
                  )("h3_r7"),
        )
        # Floor event_ts to 5-minute window.
        .withColumn(
            "window_start",
            F.date_trunc("minute", "event_ts") - F.expr(
                "INTERVAL " + str(WINDOW_MINUTES) + " minutes"
                " * (minute(event_ts) % " + str(WINDOW_MINUTES) + ")"
            ),
        )
        .withColumn(
            "window_end",
            F.col("window_start") + F.expr(f"INTERVAL {WINDOW_MINUTES} minutes"),
        )
        .groupBy("h3_r4", "window_start", "window_end")
        .agg(F.countDistinct("icao24").alias("aircraft_count"))
        .withColumn("window_date", F.to_date("window_start"))
    )

    _dates = sector_df.select(
        F.min("window_date").alias("min_d"),
        F.max("window_date").alias("max_d"),
    ).first()
    _sector_predicate = (
        f"window_date >= '{_dates['min_d']}' AND window_date <= '{_dates['max_d']}'"
    )

    (
        sector_df.write
        .format("delta")
        .mode("overwrite")
        .option("replaceWhere", _sector_predicate)
        .saveAsTable(GOLD_SECTOR_TABLE)
    )
    _sector_count = spark.table(GOLD_SECTOR_TABLE).count()  # type: ignore[union-attr]
    print(f"gold_sector_load written: {_sector_count:,} rows ({_sector_predicate})")
else:
    print("Offline mode — sector recompute skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### A4 — Full recompute: gold_airport_congestion
# MAGIC
# MAGIC Ground/airborne counts and avg altitude near airports per 5-minute window.
# MAGIC
# MAGIC Design decisions matching the Python reference (ADR-0006):
# MAGIC
# MAGIC * Records with `nearest_airport IS NULL` are excluded — no "UNKNOWN" bucket.
# MAGIC * Each distinct `icao24` contributes once per `(airport_icao, window_start)`.
# MAGIC   `on_ground` status is taken from the **first** observation per aircraft
# MAGIC   per window (matching the Python loop which skips icao24 after it's been
# MAGIC   added to `agg["icaos"]`).
# MAGIC * `avg_altitude_m` is computed as `SUM(baro_altitude_m) / COUNT(baro_altitude_m)`
# MAGIC   over the deduplicated set — equivalent to `alt_sum / alt_count` in Python.
# MAGIC   This is the PRIMARY incremental target in Section B.
# MAGIC
# MAGIC avg_altitude idempotency note: storing `alt_sum` and `alt_count` as
# MAGIC intermediate columns and deriving `avg_altitude_m` at the final SELECT
# MAGIC ensures that MERGE INTO (Section B) can recompute the average from
# MAGIC components, never by averaging pre-computed averages.

# COMMAND ----------

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # noqa: F811
    from pyspark.sql.window import Window  # type: ignore[import]

    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]

    # Floor to window, filter nulls (ADR-0006).
    cong_base = (
        silver_df
        .filter(F.col("nearest_airport").isNotNull())
        .withColumn(
            "window_start",
            F.date_trunc("minute", "event_ts") - F.expr(
                "INTERVAL " + str(WINDOW_MINUTES) + " minutes"
                " * (minute(event_ts) % " + str(WINDOW_MINUTES) + ")"
            ),
        )
        .withColumn(
            "window_end",
            F.col("window_start") + F.expr(f"INTERVAL {WINDOW_MINUTES} minutes"),
        )
    )

    # Deduplicate: keep only the FIRST ping per (airport, window, icao24).
    # This mirrors the Python loop which skips icao24 after first insertion.
    _row_num_w = Window.partitionBy(
        "nearest_airport", "window_start", "icao24"
    ).orderBy("event_ts")
    cong_dedup = (
        cong_base
        .withColumn("_rn", F.row_number().over(_row_num_w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    cong_df = (
        cong_dedup
        .groupBy("nearest_airport", "window_start", "window_end")
        .agg(
            F.count("icao24").alias("aircraft_count"),
            F.sum(F.when(F.col("on_ground"), 1).otherwise(0)).alias("ground_count"),
            F.sum(F.when(~F.col("on_ground"), 1).otherwise(0)).alias("airborne_count"),
            # Store sum and count components — derive avg at SELECT time.
            F.sum("baro_altitude_m").alias("_alt_sum"),
            F.count(F.col("baro_altitude_m")).alias("_alt_count"),
        )
        .withColumn(
            "avg_altitude_m",
            F.when(
                F.col("_alt_count") > 0,
                F.round(F.col("_alt_sum") / F.col("_alt_count"), 2),
            ).otherwise(F.lit(None).cast("double")),
        )
        .withColumnRenamed("nearest_airport", "airport_icao")
        .withColumn("window_date", F.to_date("window_start"))
        .drop("_alt_sum", "_alt_count")
    )

    _dates = cong_df.select(
        F.min("window_date").alias("min_d"),
        F.max("window_date").alias("max_d"),
    ).first()
    _cong_predicate = (
        f"window_date >= '{_dates['min_d']}' AND window_date <= '{_dates['max_d']}'"
    )

    (
        cong_df.write
        .format("delta")
        .mode("overwrite")
        .option("replaceWhere", _cong_predicate)
        .saveAsTable(GOLD_CONGESTION_TABLE)
    )
    _cong_count = spark.table(GOLD_CONGESTION_TABLE).count()  # type: ignore[union-attr]
    print(f"gold_airport_congestion written: {_cong_count:,} rows ({_cong_predicate})")
else:
    print("Offline mode — congestion recompute skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### A5 — Full recompute: gold_routing_stats
# MAGIC
# MAGIC Route-level statistics per (icao24, normalised callsign).
# MAGIC
# MAGIC * `callsign` is normalised: `TRIM(callsign)` and empty → NULL (audit M18).
# MAGIC * `origin` = lat/lon at the chronologically **first** event per route.
# MAGIC * `destination` = lat/lon at the chronologically **last** event per route.
# MAGIC * `window_start` = min(event_ts), `window_end` = max(event_ts).
# MAGIC * `max_altitude_m` = max(baro_altitude_m).
# MAGIC * `avg_velocity_mps` = sum(velocity_ms) / count(velocity_ms), rounded 2 dp.
# MAGIC * `ping_count` = total observations for the route.
# MAGIC
# MAGIC Routing stats span unbounded time so a full `overwrite` is used for
# MAGIC recompute.  The incremental path upserts by `(icao24, callsign, window_start)`.

# COMMAND ----------

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # noqa: F811
    from pyspark.sql.window import Window  # noqa: F811

    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]

    # Normalise callsign: trim whitespace, empty → NULL (audit M18).
    routing_base = (
        silver_df
        .filter(F.col("icao24").isNotNull())
        .withColumn(
            "callsign",
            F.when(
                F.trim(F.col("callsign")) == "",
                F.lit(None).cast("string"),
            ).otherwise(F.trim(F.col("callsign"))),
        )
    )

    # Use window functions to pick first/last lat/lon per route.
    _route_w_asc = Window.partitionBy("icao24", "callsign").orderBy("event_ts")
    _route_w_desc = Window.partitionBy("icao24", "callsign").orderBy(F.col("event_ts").desc())

    routing_enriched = (
        routing_base
        .withColumn("_first_lat", F.first("lat").over(_route_w_asc))
        .withColumn("_first_lon", F.first("lon").over(_route_w_asc))
        .withColumn("_last_lat", F.first("lat").over(_route_w_desc))
        .withColumn("_last_lon", F.first("lon").over(_route_w_desc))
    )

    routing_df = (
        routing_enriched
        .groupBy("icao24", "callsign")
        .agg(
            F.min("event_ts").alias("window_start"),
            F.max("event_ts").alias("window_end"),
            F.first("_first_lat").alias("origin_lat"),
            F.first("_first_lon").alias("origin_lon"),
            F.first("_last_lat").alias("destination_lat"),
            F.first("_last_lon").alias("destination_lon"),
            F.max("baro_altitude_m").alias("max_altitude_m"),
            # avg_velocity: sum/count to keep components computable.
            F.sum("velocity_ms").alias("_vel_sum"),
            F.count(F.col("velocity_ms")).alias("_vel_count"),
            F.count("icao24").alias("ping_count"),
        )
        .withColumn(
            "avg_velocity_mps",
            F.when(
                F.col("_vel_count") > 0,
                F.round(F.col("_vel_sum") / F.col("_vel_count"), 2),
            ).otherwise(F.lit(None).cast("double")),
        )
        .drop("_vel_sum", "_vel_count")
    )

    (
        routing_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_ROUTING_TABLE)
    )
    _routing_count = spark.table(GOLD_ROUTING_TABLE).count()  # type: ignore[union-attr]
    print(f"gold_routing_stats written: {_routing_count:,} rows")
else:
    print("Offline mode — routing recompute skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### A6 — Post-load data quality assertions (full recompute)
# MAGIC
# MAGIC These checks run after all four writes.  A failure raises immediately so
# MAGIC downstream consumers are not exposed to bad data.

# COMMAND ----------

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # noqa: F811

    _dq_failures = []

    # gold_emergency_events: mandatory NOT NULL columns.
    for _col in ("icao24", "squawk", "first_seen_ts", "last_seen_ts", "duration_s"):
        _n = spark.table(GOLD_EMERGENCY_TABLE).filter(F.col(_col).isNull()).count()  # type: ignore[union-attr]
        if _n:
            _dq_failures.append(f"{GOLD_EMERGENCY_TABLE}.{_col}: {_n} nulls")

    # gold_sector_load: mandatory columns + aircraft_count > 0.
    _sl = spark.table(GOLD_SECTOR_TABLE)  # type: ignore[union-attr]
    for _col in ("h3_r4", "window_start", "window_end", "aircraft_count", "window_date"):
        _n = _sl.filter(F.col(_col).isNull()).count()
        if _n:
            _dq_failures.append(f"{GOLD_SECTOR_TABLE}.{_col}: {_n} nulls")
    _n = _sl.filter(F.col("aircraft_count") < 1).count()
    if _n:
        _dq_failures.append(f"{GOLD_SECTOR_TABLE}.aircraft_count < 1: {_n} rows")

    # gold_airport_congestion: mandatory columns + count consistency.
    _ac = spark.table(GOLD_CONGESTION_TABLE)  # type: ignore[union-attr]
    for _col in ("airport_icao", "window_start", "window_end",
                 "aircraft_count", "ground_count", "airborne_count", "window_date"):
        _n = _ac.filter(F.col(_col).isNull()).count()
        if _n:
            _dq_failures.append(f"{GOLD_CONGESTION_TABLE}.{_col}: {_n} nulls")
    _n = _ac.filter(
        F.col("ground_count") + F.col("airborne_count") != F.col("aircraft_count")
    ).count()
    if _n:
        _dq_failures.append(
            f"{GOLD_CONGESTION_TABLE}: ground+airborne != aircraft_count in {_n} rows"
        )

    # gold_routing_stats: mandatory columns + ping_count >= 1.
    _rs = spark.table(GOLD_ROUTING_TABLE)  # type: ignore[union-attr]
    for _col in ("icao24", "window_start", "window_end",
                 "origin_lat", "origin_lon", "destination_lat", "destination_lon",
                 "ping_count"):
        _n = _rs.filter(F.col(_col).isNull()).count()
        if _n:
            _dq_failures.append(f"{GOLD_ROUTING_TABLE}.{_col}: {_n} nulls")

    if _dq_failures:
        raise RuntimeError(
            "Gold DQ checks failed:\n" + "\n".join(f"  - {e}" for e in _dq_failures)
        )
    print("All gold DQ checks passed.")
else:
    print("Offline mode — DQ checks skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION B — Incremental MERGE INTO
# MAGIC
# MAGIC ### Why gold_airport_congestion is the primary incremental target
# MAGIC
# MAGIC `gold_airport_congestion` is keyed on `(airport_icao, window_start)` — a
# MAGIC finite, stable composite key aligned with the 5-minute window floor.  This
# MAGIC makes MERGE INTO clean:
# MAGIC
# MAGIC * **Match clause**: `target.airport_icao = source.airport_icao AND
# MAGIC   target.window_start = source.window_start` — one row per key in the target.
# MAGIC * **WHEN MATCHED THEN UPDATE**: replaces all columns atomically.
# MAGIC * **WHEN NOT MATCHED THEN INSERT**: adds new windows not yet in the target.
# MAGIC
# MAGIC Re-running the MERGE on the same batch is idempotent: the source aggregation
# MAGIC produces exactly the same rows, so MATCHED rows get the same values written
# MAGIC and NOT MATCHED rows are still inserted only once (Delta's upsert semantics
# MAGIC prevent double-insertion).
# MAGIC
# MAGIC ### avg_altitude_m idempotency
# MAGIC
# MAGIC `avg_altitude_m` in the source aggregate is always derived as
# MAGIC `SUM(baro_altitude_m) / COUNT(baro_altitude_m)` over the deduplicated
# MAGIC aircraft set for that window — never as an average of previously stored
# MAGIC averages.  This means:
# MAGIC
# MAGIC * A re-run of the incremental batch recomputes from the raw silver rows →
# MAGIC   the source value is identical → MATCHED UPDATE writes the same value.
# MAGIC * No floating-point drift accumulates across re-runs.
# MAGIC
# MAGIC ### Emergency events and routing stats
# MAGIC
# MAGIC Their natural timestamps can shift when earlier observations arrive, so a
# MAGIC naive timestamp-keyed MERGE is unsafe. Sections B3 and B4 use keyed partial
# MAGIC recompute: identify affected entities, rebuild them from full silver history,
# MAGIC delete their stale gold rows, and insert the corrected aggregates.

# COMMAND ----------
# MAGIC %md
# MAGIC ### B1 — Build incremental source batch for gold_airport_congestion
# MAGIC
# MAGIC The incremental batch is scoped by `incremental_window_start` and
# MAGIC `incremental_window_end` (job parameters).  If these are empty the cell
# MAGIC skips gracefully — only Section A (full recompute) is active.

# COMMAND ----------

_RUN_INCREMENTAL = bool(INCREMENTAL_WINDOW_START and INCREMENTAL_WINDOW_END)

if _ON_DATABRICKS and _RUN_INCREMENTAL:
    from pyspark.sql import functions as F  # noqa: F811
    from pyspark.sql.window import Window  # noqa: F811

    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]

    # Scope to the incremental window.
    inc_silver = silver_df.filter(
        (F.col("event_ts") >= F.lit(INCREMENTAL_WINDOW_START).cast("timestamp"))
        & (F.col("event_ts") < F.lit(INCREMENTAL_WINDOW_END).cast("timestamp"))
    )
    _inc_silver_count = inc_silver.count()
    print(f"Incremental silver records: {_inc_silver_count:,}")

    # Same aggregation logic as A4: filter null airports, dedup, aggregate.
    inc_base = (
        inc_silver
        .filter(F.col("nearest_airport").isNotNull())
        .withColumn(
            "window_start",
            F.date_trunc("minute", "event_ts") - F.expr(
                "INTERVAL " + str(WINDOW_MINUTES) + " minutes"
                " * (minute(event_ts) % " + str(WINDOW_MINUTES) + ")"
            ),
        )
        .withColumn(
            "window_end",
            F.col("window_start") + F.expr(f"INTERVAL {WINDOW_MINUTES} minutes"),
        )
    )

    _rn_w = Window.partitionBy(
        "nearest_airport", "window_start", "icao24"
    ).orderBy("event_ts")
    inc_dedup = (
        inc_base
        .withColumn("_rn", F.row_number().over(_rn_w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    inc_cong_df = (
        inc_dedup
        .groupBy("nearest_airport", "window_start", "window_end")
        .agg(
            F.count("icao24").alias("aircraft_count"),
            F.sum(F.when(F.col("on_ground"), 1).otherwise(0)).alias("ground_count"),
            F.sum(F.when(~F.col("on_ground"), 1).otherwise(0)).alias("airborne_count"),
            F.sum("baro_altitude_m").alias("_alt_sum"),
            F.count(F.col("baro_altitude_m")).alias("_alt_count"),
        )
        .withColumn(
            "avg_altitude_m",
            F.when(
                F.col("_alt_count") > 0,
                F.round(F.col("_alt_sum") / F.col("_alt_count"), 2),
            ).otherwise(F.lit(None).cast("double")),
        )
        .withColumnRenamed("nearest_airport", "airport_icao")
        .withColumn("window_date", F.to_date("window_start"))
        .drop("_alt_sum", "_alt_count")
    )

    # Register as a temp view so the MERGE SQL below can reference it.
    inc_cong_df.createOrReplaceTempView("_inc_cong_source")
    _inc_source_count = inc_cong_df.count()
    print(f"Incremental source rows: {_inc_source_count:,}")
elif _ON_DATABRICKS:
    print("incremental_window_start / incremental_window_end not set — skipping incremental.")
else:
    print("Offline mode — incremental source skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### B2 — MERGE INTO gold_airport_congestion
# MAGIC
# MAGIC The MERGE statement:
# MAGIC * Matches on `(airport_icao, window_start)` — the natural compound key for
# MAGIC   this windowed aggregate.
# MAGIC * On match: updates all columns (full row replacement — no partial update).
# MAGIC   Because the source re-derives `avg_altitude_m` from components, the
# MAGIC   updated value is always the correct recomputed average for that window,
# MAGIC   not a stale accumulation.
# MAGIC * On no match: inserts the new window row.
# MAGIC
# MAGIC Idempotency proof: re-running this MERGE with the same `_inc_cong_source`
# MAGIC rows yields no net change.  Every MATCHED row receives the same value it
# MAGIC already holds; no new NOT MATCHED rows exist since they were inserted on
# MAGIC the first run.

# COMMAND ----------

if _ON_DATABRICKS and _RUN_INCREMENTAL:
    spark.sql(f"""  # type: ignore[union-attr]
        MERGE INTO {GOLD_CONGESTION_TABLE} AS target
        USING _inc_cong_source AS source
        ON target.airport_icao = source.airport_icao
           AND target.window_start = source.window_start
        WHEN MATCHED THEN UPDATE SET
            target.window_end     = source.window_end,
            target.aircraft_count = source.aircraft_count,
            target.ground_count   = source.ground_count,
            target.airborne_count = source.airborne_count,
            target.avg_altitude_m = source.avg_altitude_m,
            target.window_date    = source.window_date
        WHEN NOT MATCHED THEN INSERT (
            airport_icao, window_start, window_end,
            aircraft_count, ground_count, airborne_count,
            avg_altitude_m, window_date
        ) VALUES (
            source.airport_icao, source.window_start, source.window_end,
            source.aircraft_count, source.ground_count, source.airborne_count,
            source.avg_altitude_m, source.window_date
        )
    """)
    _merged_count = spark.table(GOLD_CONGESTION_TABLE).count()  # type: ignore[union-attr]
    print(f"MERGE complete. gold_airport_congestion total rows: {_merged_count:,}")
elif _ON_DATABRICKS:
    print("Skipping MERGE — incremental parameters not set.")
else:
    print("Offline mode — MERGE INTO skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### B3 — Incremental keyed partial-recompute: gold_emergency_events
# MAGIC
# MAGIC **Overlapping-batch safe.**  The superseded approach keyed the MERGE on
# MAGIC `(icao24, squawk, first_seen_ts)`, which is only correct for non-overlapping
# MAGIC append-only batches.  When an incoming batch contains an earlier observation
# MAGIC of an ongoing event, the true `first_seen_ts` shifts earlier — a naive MERGE
# MAGIC on the old key inserts a duplicate event row instead of extending the existing
# MAGIC one.
# MAGIC
# MAGIC The fix is a **keyed partial-recompute**:
# MAGIC
# MAGIC 1. Identify affected `(icao24, squawk)` entities in the new batch.
# MAGIC 2. For those entities, fetch ALL their silver rows from the full silver table
# MAGIC    (existing history + new batch observations).
# MAGIC 3. Recompute emergency event sessions from the combined silver rows — this
# MAGIC    re-derives `first_seen_ts` as the true global minimum, correctly handles
# MAGIC    gap splits, and is idempotent.
# MAGIC 4. DELETE the stale gold rows for affected entities, then INSERT the
# MAGIC    recomputed rows.  Unaffected entities are untouched.
# MAGIC
# MAGIC Idempotency: running the same batch twice produces no net change (DELETE
# MAGIC targets the same entities; INSERT recomputes to the same rows).
# MAGIC Order-independence: batch A then B yields the same gold as batch B then A,
# MAGIC both equal to a full recompute over (A + B).

# COMMAND ----------

if _ON_DATABRICKS and _RUN_INCREMENTAL:
    from pyspark.sql import functions as F  # noqa: F811

    # Step 1: Identify affected (icao24, squawk) entities from the new batch.
    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]
    inc_emg_new = (
        silver_df
        .filter(F.col("squawk").isin("7500", "7600", "7700"))
        .filter(
            (F.col("event_ts") >= F.lit(INCREMENTAL_WINDOW_START).cast("timestamp"))
            & (F.col("event_ts") < F.lit(INCREMENTAL_WINDOW_END).cast("timestamp"))
        )
        .select("icao24", "squawk")
        .distinct()
    )
    # Persist affected entities as a temp view for the DELETE predicate.
    inc_emg_new.createOrReplaceTempView("_inc_emg_affected_entities")
    _n_affected = inc_emg_new.count()
    print(f"Affected emergency entities: {_n_affected:,}")

    # Step 2: For affected entities, collect ALL their silver rows (full history).
    # This is the union of existing silver + new batch for those entities only.
    full_emg_silver_for_affected = (
        silver_df
        .filter(F.col("squawk").isin("7500", "7600", "7700"))
        .join(inc_emg_new, on=["icao24", "squawk"], how="inner")
    )

    # Step 3: Recompute emergency sessions from the combined silver for affected
    # entities.  Uses the same session_window aggregation as Section A2.
    recomputed_emg_df = (
        full_emg_silver_for_affected
        .groupBy(
            "icao24",
            "squawk",
            F.session_window("event_ts", f"{EMERGENCY_GAP_S} seconds").alias("session"),
        )
        .agg(
            F.first("callsign", ignorenulls=True).alias("callsign"),
            F.min("event_ts").alias("first_seen_ts"),
            F.max("event_ts").alias("last_seen_ts"),
            F.first("lat").alias("lat"),
            F.first("lon").alias("lon"),
            F.first("origin_country", ignorenulls=True).alias("origin_country"),
            F.first("nearest_airport", ignorenulls=True).alias("nearest_airport"),
        )
        .withColumn(
            "duration_s",
            (F.unix_timestamp("last_seen_ts") - F.unix_timestamp("first_seen_ts"))
            .cast("bigint"),
        )
        .drop("session")
    )
    recomputed_emg_df.createOrReplaceTempView("_inc_emg_recomputed")
    _n_recomputed = recomputed_emg_df.count()
    print(f"Recomputed emergency gold rows for affected entities: {_n_recomputed:,}")

    # Step 4a: DELETE stale gold rows for affected (icao24, squawk) entities.
    # This removes all old rows — including those with now-stale first_seen_ts
    # values that would become duplicates if we only inserted.
    spark.sql(f"""  # type: ignore[union-attr]
        DELETE FROM {GOLD_EMERGENCY_TABLE}
        WHERE (icao24, squawk) IN (
            SELECT icao24, squawk FROM _inc_emg_affected_entities
        )
    """)

    # Step 4b: INSERT the freshly recomputed rows for affected entities.
    (
        recomputed_emg_df.write  # type: ignore[union-attr]
        .format("delta")
        .mode("append")
        .saveAsTable(GOLD_EMERGENCY_TABLE)
    )
    _emg_total = spark.table(GOLD_EMERGENCY_TABLE).count()  # type: ignore[union-attr]
    print(f"Emergency events incremental complete. Total rows: {_emg_total:,}")
elif _ON_DATABRICKS:
    print("Skipping emergency incremental — incremental parameters not set.")
else:
    print("Offline mode — emergency incremental skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### B4 — Incremental keyed partial-recompute: gold_routing_stats
# MAGIC
# MAGIC **Overlapping-batch safe.**  The previous approach keyed the MERGE on
# MAGIC `(icao24, callsign, window_start)`, but `window_start` is the derived
# MAGIC `min(event_ts)` of the route — it shifts earlier when a new batch brings
# MAGIC observations that predate existing ones.  A naive MERGE would insert a
# MAGIC duplicate row instead of extending the route.
# MAGIC
# MAGIC The fix mirrors B3: **keyed partial-recompute**.
# MAGIC
# MAGIC 1. Identify affected `(icao24, normalised_callsign)` entities in the batch.
# MAGIC 2. Fetch ALL silver rows for those entities from the full silver table.
# MAGIC 3. Recompute routing stats from the combined set — re-derives `window_start`
# MAGIC    as the true global `min(event_ts)`, `origin_lat/lon` from the earliest
# MAGIC    observation, etc.
# MAGIC 4. DELETE stale gold rows for affected entities, INSERT recomputed rows.

# COMMAND ----------

if _ON_DATABRICKS and _RUN_INCREMENTAL:
    from pyspark.sql import functions as F  # noqa: F811
    from pyspark.sql.window import Window  # noqa: F811

    # Step 1: Identify affected (icao24, normalised callsign) entities.
    silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]
    inc_routing_new = (
        silver_df
        .filter(F.col("icao24").isNotNull())
        .filter(
            (F.col("event_ts") >= F.lit(INCREMENTAL_WINDOW_START).cast("timestamp"))
            & (F.col("event_ts") < F.lit(INCREMENTAL_WINDOW_END).cast("timestamp"))
        )
        .withColumn(
            "callsign",
            F.when(
                F.trim(F.col("callsign")) == "",
                F.lit(None).cast("string"),
            ).otherwise(F.trim(F.col("callsign"))),
        )
        .select("icao24", "callsign")
        .distinct()
    )
    inc_routing_new.createOrReplaceTempView("_inc_routing_affected_entities")
    _n_routing_affected = inc_routing_new.count()
    print(f"Affected routing entities: {_n_routing_affected:,}")

    # Step 2: Collect ALL silver rows for affected entities (full history).
    full_routing_silver_for_affected = (
        silver_df
        .filter(F.col("icao24").isNotNull())
        .withColumn(
            "callsign",
            F.when(
                F.trim(F.col("callsign")) == "",
                F.lit(None).cast("string"),
            ).otherwise(F.trim(F.col("callsign"))),
        )
        .join(inc_routing_new, on=["icao24", "callsign"], how="inner")
    )

    # Step 3: Recompute routing stats from combined silver for affected entities.
    # Uses the same window-function aggregation as Section A5.
    _w_asc = Window.partitionBy("icao24", "callsign").orderBy("event_ts")
    _w_desc = Window.partitionBy("icao24", "callsign").orderBy(F.col("event_ts").desc())
    routing_enriched = (
        full_routing_silver_for_affected
        .withColumn("_first_lat", F.first("lat").over(_w_asc))
        .withColumn("_first_lon", F.first("lon").over(_w_asc))
        .withColumn("_last_lat", F.first("lat").over(_w_desc))
        .withColumn("_last_lon", F.first("lon").over(_w_desc))
    )

    recomputed_routing_df = (
        routing_enriched
        .groupBy("icao24", "callsign")
        .agg(
            F.min("event_ts").alias("window_start"),
            F.max("event_ts").alias("window_end"),
            F.first("_first_lat").alias("origin_lat"),
            F.first("_first_lon").alias("origin_lon"),
            F.first("_last_lat").alias("destination_lat"),
            F.first("_last_lon").alias("destination_lon"),
            F.max("baro_altitude_m").alias("max_altitude_m"),
            F.sum("velocity_ms").alias("_vel_sum"),
            F.count(F.col("velocity_ms")).alias("_vel_count"),
            F.count("icao24").alias("ping_count"),
        )
        .withColumn(
            "avg_velocity_mps",
            F.when(
                F.col("_vel_count") > 0,
                F.round(F.col("_vel_sum") / F.col("_vel_count"), 2),
            ).otherwise(F.lit(None).cast("double")),
        )
        .drop("_vel_sum", "_vel_count")
    )
    recomputed_routing_df.createOrReplaceTempView("_inc_routing_recomputed")
    _n_routing_recomputed = recomputed_routing_df.count()
    print(f"Recomputed routing gold rows for affected entities: {_n_routing_recomputed:,}")

    # Step 4a: DELETE stale gold rows for affected entities.
    # Uses COALESCE to handle NULL callsign in the IN predicate safely.
    spark.sql(f"""  # type: ignore[union-attr]
        DELETE FROM {GOLD_ROUTING_TABLE}
        WHERE (icao24, COALESCE(callsign, '')) IN (
            SELECT icao24, COALESCE(callsign, '')
            FROM _inc_routing_affected_entities
        )
    """)

    # Step 4b: INSERT recomputed rows for affected entities.
    (
        recomputed_routing_df.write  # type: ignore[union-attr]
        .format("delta")
        .mode("append")
        .saveAsTable(GOLD_ROUTING_TABLE)
    )
    _routing_total = spark.table(GOLD_ROUTING_TABLE).count()  # type: ignore[union-attr]
    print(f"Routing stats incremental complete. Total rows: {_routing_total:,}")
elif _ON_DATABRICKS:
    print("Skipping routing incremental — incremental parameters not set.")
else:
    print("Offline mode — routing incremental skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION C — Delta Time-Travel Demo
# MAGIC
# MAGIC This section demonstrates Delta Lake's time-travel capability on
# MAGIC `gold_airport_congestion`.  It is designed to be run **after** both a full
# MAGIC recompute (Section A) and at least one incremental MERGE (Section B) have
# MAGIC completed, so that the table has at least two Delta versions.
# MAGIC
# MAGIC Two access methods are shown:
# MAGIC * `VERSION AS OF <n>` — explicit version number (stable, deterministic).
# MAGIC * `TIMESTAMP AS OF <ts>` — wall-clock snapshot at a given time.
# MAGIC
# MAGIC The demo also shows a before/after diff using set operations so the exact
# MAGIC rows added or changed by the MERGE are visible in the notebook output.

# COMMAND ----------
# MAGIC %md
# MAGIC ### C1 — Inspect table history

# COMMAND ----------

if _ON_DATABRICKS:
    print(f"Delta history for: {GOLD_CONGESTION_TABLE}")
    spark.sql(f"DESCRIBE HISTORY {GOLD_CONGESTION_TABLE}").show(10, truncate=False)  # type: ignore[union-attr]
else:
    print("Offline mode — DESCRIBE HISTORY skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### C2 — Read a previous version with VERSION AS OF
# MAGIC
# MAGIC Replace `_TT_VERSION` with the version number shown in C1 that corresponds
# MAGIC to the state **before** the last incremental MERGE.  Typically this is
# MAGIC `current_version - 1`.

# COMMAND ----------

if _ON_DATABRICKS:
    # Determine the current Delta version programmatically.
    _history = spark.sql(  # type: ignore[union-attr]
        f"DESCRIBE HISTORY {GOLD_CONGESTION_TABLE} LIMIT 2"
    ).collect()
    _current_version = int(_history[0]["version"])
    _prev_version = max(0, _current_version - 1)

    print(f"Current version : {_current_version}")
    print(f"Previous version: {_prev_version}")

    # Read the state at version N-1 (before the last MERGE).
    _before_df = spark.read.format("delta").option(  # type: ignore[union-attr]
        "versionAsOf", _prev_version
    ).table(GOLD_CONGESTION_TABLE)
    _before_count = _before_df.count()

    # Read the current state (after the last MERGE).
    _after_df = spark.table(GOLD_CONGESTION_TABLE)  # type: ignore[union-attr]
    _after_count = _after_df.count()

    print(f"\nRow counts — before (v{_prev_version}): {_before_count:,}")
    print(f"Row counts — after  (v{_current_version}): {_after_count:,}")
    print(f"Net change: {_after_count - _before_count:+,} rows")
else:
    print("Offline mode — VERSION AS OF skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### C3 — Read a snapshot with TIMESTAMP AS OF
# MAGIC
# MAGIC `TIMESTAMP AS OF` retrieves the table as it existed at a wall-clock time.
# MAGIC Useful for point-in-time debugging when you know the approximate time of a
# MAGIC write but not the version number.

# COMMAND ----------

if _ON_DATABRICKS:
    # Retrieve the timestamp of the version just before the current one.
    _prev_ts = str(_history[1]["timestamp"]) if len(_history) > 1 else None

    if _prev_ts:
        _ts_df = spark.read.format("delta").option(  # type: ignore[union-attr]
            "timestampAsOf", _prev_ts
        ).table(GOLD_CONGESTION_TABLE)
        _ts_count = _ts_df.count()
        print(f"TIMESTAMP AS OF {_prev_ts!r}: {_ts_count:,} rows")
    else:
        print("Only one version in history — TIMESTAMP AS OF demo requires >= 2 versions.")
else:
    print("Offline mode — TIMESTAMP AS OF skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### C4 — Before/after diff: rows added or updated by the incremental MERGE
# MAGIC
# MAGIC Uses set-difference and join to isolate the exact rows that changed.
# MAGIC Output is limited to 20 rows for notebook readability.

# COMMAND ----------

if _ON_DATABRICKS and _RUN_INCREMENTAL:
    from pyspark.sql import functions as F  # noqa: F811

    # Rows in AFTER that are not in BEFORE (new windows).
    _new_rows = _after_df.exceptAll(_before_df)
    _new_count = _new_rows.count()

    # Rows whose airport_icao/window_start existed in both but have different values.
    _before_keyed = _before_df.select(
        "airport_icao", "window_start",
        F.col("aircraft_count").alias("before_aircraft_count"),
        F.col("avg_altitude_m").alias("before_avg_altitude_m"),
    )
    _after_keyed = _after_df.select(
        "airport_icao", "window_start",
        F.col("aircraft_count").alias("after_aircraft_count"),
        F.col("avg_altitude_m").alias("after_avg_altitude_m"),
    )
    _changed_rows = (
        _before_keyed
        .join(_after_keyed, on=["airport_icao", "window_start"])
        .filter(
            (F.col("before_aircraft_count") != F.col("after_aircraft_count"))
            | (F.col("before_avg_altitude_m") != F.col("after_avg_altitude_m"))
        )
    )
    _changed_count = _changed_rows.count()

    print(f"\nTime-travel diff summary:")
    print(f"  New rows inserted  : {_new_count:,}")
    print(f"  Rows updated       : {_changed_count:,}")
    print("\nSample of new rows:")
    _new_rows.show(20, truncate=False)
    print("\nSample of updated rows (before vs after):")
    _changed_rows.show(20, truncate=False)
elif _ON_DATABRICKS:
    print("No incremental MERGE was run — before/after diff skipped.")
else:
    print("Offline mode — diff skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### C5 — Results template (fill in after cloud run)
# MAGIC
# MAGIC ```
# MAGIC ============================================================
# MAGIC RESULTS TEMPLATE — 02_silver_to_gold (ds-04)
# MAGIC ============================================================
# MAGIC Run date            : YYYY-MM-DD HH:MM UTC
# MAGIC silver_table        : <value>
# MAGIC
# MAGIC --- SECTION A: Full recompute ---
# MAGIC gold_emergency_events   :        ??? rows
# MAGIC gold_sector_load        :        ??? rows
# MAGIC gold_airport_congestion :        ??? rows
# MAGIC gold_routing_stats      :        ??? rows
# MAGIC
# MAGIC --- SECTION B: Incremental MERGE (if run) ---
# MAGIC incremental_window_start : <value>
# MAGIC incremental_window_end   : <value>
# MAGIC inc silver records       :        ??? rows
# MAGIC inc cong source rows     :        ??? rows
# MAGIC MERGE result (total rows):        ??? rows
# MAGIC
# MAGIC --- SECTION C: Time-travel demo ---
# MAGIC table                    : gold_airport_congestion
# MAGIC version before MERGE     :        ???
# MAGIC version after MERGE      :        ???
# MAGIC rows before MERGE        :        ???
# MAGIC rows after MERGE         :        ???
# MAGIC net change               :      +/- ???
# MAGIC new rows inserted        :        ???
# MAGIC rows updated             :        ???
# MAGIC ============================================================
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Runbook (ds-08)
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC 1. Notebook `01_bronze_to_silver` has completed and `silver_flight_state`
# MAGIC    is populated.
# MAGIC 2. All gold target tables exist (created by A1 on first run, or via DDL in
# MAGIC    `data/cloud/ddl/delta/gold_*.sql`).
# MAGIC 3. The `h3` Python library is installed on the cluster
# MAGIC    (`pip install h3` or cluster init script).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 1 — Full recompute (baseline or recovery)
# MAGIC
# MAGIC Set job parameters:
# MAGIC
# MAGIC | Parameter | Value |
# MAGIC |---|---|
# MAGIC | `silver_table` | `<catalog>.<schema>.silver_flight_state` |
# MAGIC | `gold_emergency_table` | `<catalog>.<schema>.gold_emergency_events` |
# MAGIC | `gold_sector_table` | `<catalog>.<schema>.gold_sector_load` |
# MAGIC | `gold_congestion_table` | `<catalog>.<schema>.gold_airport_congestion` |
# MAGIC | `gold_routing_table` | `<catalog>.<schema>.gold_routing_stats` |
# MAGIC | `incremental_window_start` | *(leave empty)* |
# MAGIC | `incremental_window_end` | *(leave empty)* |
# MAGIC
# MAGIC Run Section A only (A1–A6).  The DQ assertions in A6 will raise if any
# MAGIC mandatory column has nulls or if `ground_count + airborne_count != aircraft_count`.
# MAGIC
# MAGIC Expected output: row counts in the Results Summary cell.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 2 — Incremental MERGE on a held-back batch
# MAGIC
# MAGIC 1. Identify a time window of silver data NOT already processed (e.g. the most
# MAGIC    recent 6 hours).
# MAGIC 2. Set `incremental_window_start` and `incremental_window_end` to ISO
# MAGIC    timestamps bounding that window.
# MAGIC 3. Run Section B only (B1–B4).
# MAGIC
# MAGIC To test idempotency: run Section B twice with identical parameters.  The
# MAGIC second run's MERGE should show 0 new rows inserted and 0 rows updated.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 3 — Time-travel demo
# MAGIC
# MAGIC 1. Run Section A (full recompute) to establish version N.
# MAGIC 2. Run Section B (incremental MERGE) to advance to version N+1.
# MAGIC 3. Run Section C (C1–C4).
# MAGIC
# MAGIC C1 shows the full DESCRIBE HISTORY output.
# MAGIC C2 reads version N (before MERGE) and version N+1 (after MERGE) and prints
# MAGIC row counts.
# MAGIC C3 reads by timestamp (the commit time of version N).
# MAGIC C4 shows the exact new/changed rows, filling in the Results Template.
# MAGIC
# MAGIC Fill in the Results Template in cell C5 and attach to the ds-08 runbook.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Failure and recovery
# MAGIC
# MAGIC | Symptom | Resolution |
# MAGIC |---|---|
# MAGIC | DQ assertion failure in A6 | Inspect the specific column; re-run silver notebook if source data is missing |
# MAGIC | MERGE target lock timeout | Retry; increase cluster size or reduce batch scope |
# MAGIC | `ModuleNotFoundError: h3` | Install h3 on cluster: `%pip install h3` in a preceding cell or via init script |
# MAGIC | Partial write left target corrupt | Delta's transaction log guarantees atomicity; re-run the full Section A to recover |
# MAGIC | Wrong avg_altitude after MERGE | Verify source recomputes from `alt_sum / alt_count`, not from a cached avg column |
# MAGIC
# MAGIC ---
# MAGIC *End of notebook 02_silver_to_gold — W4.3 / ds-04*
