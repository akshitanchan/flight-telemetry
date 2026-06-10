# Databricks notebook source
# =============================================================================
# 01_bronze_to_silver.py
# Wave W4.3 · Task ds-02
#
# Reads bronze JSONL files from a parameterized Unity Catalog Volume path,
# applies the C1 silver_flight_state transform (field validation, spatial
# indices, dedup, nearest-airport enrichment), and writes the result as a
# partitioned Delta table.
#
# Mirrors: data/transforms/bronze_to_silver.py (authoritative local version)
# Target schema: data/cloud/ddl/delta/silver_flight_state.sql
# Contract: shared/contracts/silver_flight_state.schema.json  v1.0.0
#
# UC-Volume parameterisation:
#   bronze_volume_path  — UC Volume path containing *.jsonl bronze files
#   silver_table        — fully-qualified Delta table (<catalog>.<schema>.<table>)
#   airports_ref_path   — UC Volume path to airports.json reference (optional)
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
# MAGIC | `bronze_volume_path` | UC Volume path containing `*.jsonl` bronze files | `/Volumes/main/default/bronze_vol/bronze/flight_state/` |
# MAGIC | `silver_table` | Fully-qualified target Delta table | `main.default.silver_flight_state` |
# MAGIC | `airports_ref_path` | UC Volume path to `airports.json` (optional) | `/Volumes/main/default/bronze_vol/reference/airports.json` |

# COMMAND ----------

# Guard: dbutils / spark exist on Databricks; fall back to stubs so the file
# passes `python -m py_compile` and import checks offline.
try:
    # On Databricks these are injected into the global scope automatically.
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
    spark = None  # type: ignore[assignment]

# COMMAND ----------

# Widget / job-param declarations — safe to re-run (idempotent on Databricks).
if _ON_DATABRICKS:
    dbutils.widgets.text(  # type: ignore[union-attr]
        "bronze_volume_path",
        "",
        "Bronze UC Volume path (*.jsonl files)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "silver_table",
        "flight_telemetry.silver_flight_state",
        "Target Delta table (catalog.schema.table)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "airports_ref_path",
        "",
        "Airports JSON reference path (UC Volume, optional)",
    )

# COMMAND ----------

# Resolve parameter values.
BRONZE_VOLUME_PATH: str = dbutils.widgets.get("bronze_volume_path")
SILVER_TABLE: str = dbutils.widgets.get("silver_table")
AIRPORTS_REF_PATH: str = dbutils.widgets.get("airports_ref_path")

# Validate mandatory parameters at run-time so failures are immediately clear.
if _ON_DATABRICKS:
    if not BRONZE_VOLUME_PATH:
        raise ValueError(
            "Widget 'bronze_volume_path' is empty. "
            "Set it to the UC Volume path that contains the bronze JSONL files, "
            "e.g. /Volumes/<catalog>/<schema>/<volume>/bronze/flight_state/"
        )
    if not SILVER_TABLE:
        raise ValueError(
            "Widget 'silver_table' is empty. "
            "Set it to a fully-qualified table name, "
            "e.g. main.default.silver_flight_state"
        )

print(f"bronze_volume_path : {BRONZE_VOLUME_PATH!r}")
print(f"silver_table       : {SILVER_TABLE!r}")
print(f"airports_ref_path  : {AIRPORTS_REF_PATH!r}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Library import
# MAGIC
# MAGIC `lib/transforms.py` contains the pure-Python row logic and is importable
# MAGIC offline.  In the notebook we add the repo root to `sys.path` so that both
# MAGIC the `data.cloud.databricks.lib` package path and a direct relative import
# MAGIC work regardless of how the Databricks Repo is mounted.

# COMMAND ----------

import sys
import os

# When running inside a Databricks Repo the CWD is typically the repo root.
# Add the repo root explicitly so `data.cloud.databricks.lib` is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.cloud.databricks.lib.transforms import (
    transform_record,
    nearest_airport,
    load_airports_reference,
    idem_key_for,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 — Read bronze JSONL from UC Volume

# COMMAND ----------

# Read all JSONL files under the bronze volume path.
# spark.read.json handles multi-file JSONL natively; schema inference runs on
# the first pass.  We use PERMISSIVE mode so corrupt records are captured in
# _corrupt_record rather than aborting the job.

if _ON_DATABRICKS:
    bronze_df = (
        spark.read  # type: ignore[union-attr]
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .json(BRONZE_VOLUME_PATH + "*.jsonl")
    )
    _bronze_count = bronze_df.count()
    print(f"Bronze records read: {_bronze_count:,}")
else:
    bronze_df = None
    _bronze_count = 0
    print("Offline mode — Spark read skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 — Load airports reference (broadcast)

# COMMAND ----------

# The airports reference is small (<10 k rows).  Load it into a Python list on
# the driver and broadcast it so every executor has a copy without shuffle.

if _ON_DATABRICKS and AIRPORTS_REF_PATH:
    try:
        import json as _json

        # /dbfs/ prefix lets the driver open UC Volume paths via FUSE.
        _dbfs_path = AIRPORTS_REF_PATH.replace("/Volumes/", "/dbfs/Volumes/")
        with open(_dbfs_path) as _f:
            _airports_list = _json.load(_f)
        _airports_bc = spark.sparkContext.broadcast(_airports_list)  # type: ignore[union-attr]
        print(f"Airports reference loaded: {len(_airports_list):,} entries (broadcast).")
    except FileNotFoundError:
        _airports_bc = spark.sparkContext.broadcast([])  # type: ignore[union-attr]
        print("Airports reference not found — nearest_airport will be null.")
else:
    _airports_bc = None
    print("No airports_ref_path provided — nearest_airport will be null.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Apply transform via mapInPandas
# MAGIC
# MAGIC `mapInPandas` is used rather than a row-level Python UDF to avoid
# MAGIC per-row serialisation overhead and to keep the dedup set local to each
# MAGIC partition.  Each partition performs its own dedup; cross-partition
# MAGIC duplicates are handled by the `MERGE`/`REPLACE WHERE` write strategy in
# MAGIC Step 5.

# COMMAND ----------

if _ON_DATABRICKS:
    import pandas as _pd
    from pyspark.sql import functions as F  # type: ignore[import]
    from pyspark.sql.types import (  # type: ignore[import]
        StructType, StructField,
        StringType, DoubleType, BooleanType, DateType, TimestampType,
    )

    # Output schema matching silver_flight_state DDL + event_date partition col.
    _SILVER_SCHEMA = StructType([
        StructField("icao24",           StringType(),    False),
        StructField("callsign",         StringType(),    True),
        StructField("event_ts",         StringType(),    False),  # kept as string; cast below
        StructField("lon",              DoubleType(),    False),
        StructField("lat",              DoubleType(),    False),
        StructField("baro_altitude_m",  DoubleType(),    True),
        StructField("velocity_ms",      DoubleType(),    True),
        StructField("true_track_deg",   DoubleType(),    True),
        StructField("vertical_rate_ms", DoubleType(),    True),
        StructField("on_ground",        BooleanType(),   False),
        StructField("squawk",           StringType(),    True),
        StructField("origin_country",   StringType(),    False),
        StructField("nearest_airport",  StringType(),    True),
        StructField("geohash7",         StringType(),    False),
        StructField("h3_r7",            StringType(),    False),
        StructField("metar_wind_kt",    DoubleType(),    True),
        StructField("metar_vis_m",      DoubleType(),    True),
        StructField("metar_ceiling_ft", DoubleType(),    True),
    ])

    def _transform_partition(iterator):
        """Apply transform_record to every pandas chunk in a partition."""
        # Resolve broadcast; fall back to empty list if not set.
        airports = _airports_bc.value if _airports_bc is not None else []
        seen_keys: set = set()

        for pdf in iterator:
            output_rows = []
            for _, row in pdf.iterrows():
                record = row.dropna().to_dict()

                # Dedup within partition
                key = idem_key_for(record)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                silver = transform_record(record)
                if silver is None:
                    continue

                # Nearest-airport enrichment
                if airports:
                    silver["nearest_airport"] = nearest_airport(
                        silver["lat"], silver["lon"], airports
                    )

                output_rows.append(silver)

            if output_rows:
                yield _pd.DataFrame(output_rows)
            else:
                yield _pd.DataFrame(columns=[f.name for f in _SILVER_SCHEMA.fields])

    silver_df_raw = bronze_df.mapInPandas(_transform_partition, schema=_SILVER_SCHEMA)

    # Cast event_ts to TIMESTAMP and add event_date partition column.
    silver_df = (
        silver_df_raw
        .withColumn("event_ts", F.to_timestamp("event_ts"))
        .withColumn("event_date", F.to_date("event_ts"))
    )

    _silver_count = silver_df.count()
    print(f"Silver records after transform: {_silver_count:,}")
    print(f"Records dropped/deduped: {_bronze_count - _silver_count:,}")
else:
    silver_df = None
    _silver_count = 0
    print("Offline mode — transform skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 — Post-load data quality assertions
# MAGIC
# MAGIC These checks run on the transformed DataFrame **before** the Delta write.
# MAGIC A failure here halts the job and prevents bad data from reaching the
# MAGIC silver table.

# COMMAND ----------

if _ON_DATABRICKS and silver_df is not None:
    from pyspark.sql import functions as F  # noqa: F811 — already imported above

    _dq_failures = []

    # 1. No nulls in NOT NULL columns.
    for col_name in ("icao24", "event_ts", "lon", "lat", "on_ground", "origin_country",
                     "geohash7", "h3_r7"):
        null_count = silver_df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            _dq_failures.append(f"NOT NULL violation: {col_name} has {null_count} nulls")

    # 2. Lon/lat in range.
    bad_lon = silver_df.filter((F.col("lon") < -180) | (F.col("lon") > 180)).count()
    if bad_lon:
        _dq_failures.append(f"lon out of range: {bad_lon} rows")

    bad_lat = silver_df.filter((F.col("lat") < -90) | (F.col("lat") > 90)).count()
    if bad_lat:
        _dq_failures.append(f"lat out of range: {bad_lat} rows")

    # 3. Velocity non-negative.
    bad_vel = silver_df.filter(F.col("velocity_ms") < 0).count()
    if bad_vel:
        _dq_failures.append(f"velocity_ms negative: {bad_vel} rows")

    # 4. geohash7 length.
    bad_gh = silver_df.filter(F.length(F.col("geohash7")) != 7).count()
    if bad_gh:
        _dq_failures.append(f"geohash7 wrong length: {bad_gh} rows")

    # 5. h3_r7 length.
    bad_h3 = silver_df.filter(F.length(F.col("h3_r7")) != 15).count()
    if bad_h3:
        _dq_failures.append(f"h3_r7 wrong length: {bad_h3} rows")

    if _dq_failures:
        raise RuntimeError(
            "Data quality checks failed — aborting before Delta write:\n"
            + "\n".join(f"  - {e}" for e in _dq_failures)
        )
    print(f"All DQ checks passed ({_silver_count:,} rows).")
else:
    print("Offline mode — DQ checks skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5 — Write to Delta (idempotent REPLACE WHERE partition)
# MAGIC
# MAGIC Strategy: **replaceWhere** on `event_date` partitions covered by the
# MAGIC current batch.  This is idempotent — re-running for the same date range
# MAGIC overwrites exactly those partitions and leaves all others untouched.
# MAGIC
# MAGIC The target table must already exist (created by `spark.sql()` running the
# MAGIC DDL in `data/cloud/ddl/delta/silver_flight_state.sql`).  The notebook
# MAGIC creates it if absent so the first run is self-contained.

# COMMAND ----------

if _ON_DATABRICKS and silver_df is not None:
    from pyspark.sql import functions as F  # noqa: F811

    # Ensure the target table exists (idempotent CREATE IF NOT EXISTS).
    spark.sql(f"""  # type: ignore[union-attr]
        CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
            icao24           STRING    NOT NULL,
            callsign         STRING,
            event_ts         TIMESTAMP NOT NULL,
            lon              DOUBLE    NOT NULL,
            lat              DOUBLE    NOT NULL,
            baro_altitude_m  DOUBLE,
            velocity_ms      DOUBLE,
            true_track_deg   DOUBLE,
            vertical_rate_ms DOUBLE,
            on_ground        BOOLEAN   NOT NULL,
            squawk           STRING,
            origin_country   STRING    NOT NULL,
            nearest_airport  STRING,
            geohash7         STRING    NOT NULL,
            h3_r7            STRING    NOT NULL,
            metar_wind_kt    DOUBLE,
            metar_vis_m      DOUBLE,
            metar_ceiling_ft DOUBLE,
            event_date       DATE      NOT NULL
        )
        USING DELTA
        PARTITIONED BY (event_date)
        TBLPROPERTIES (
            'delta.minReaderVersion' = '1',
            'delta.minWriterVersion' = '2',
            '_contract_version'      = '1.0.0'
        )
    """)

    # Collect the date range in this batch to scope the replaceWhere predicate.
    _dates = silver_df.select(
        F.min("event_date").alias("min_d"),
        F.max("event_date").alias("max_d"),
    ).first()
    _min_date = str(_dates["min_d"])
    _max_date = str(_dates["max_d"])
    _replace_predicate = f"event_date >= '{_min_date}' AND event_date <= '{_max_date}'"
    print(f"Writing partitions: {_replace_predicate}")

    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .option("replaceWhere", _replace_predicate)
        .saveAsTable(SILVER_TABLE)
    )

    print(f"Write complete. Table: {SILVER_TABLE}")
else:
    print("Offline mode — Delta write skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6 — Row-count summary (results capture)

# COMMAND ----------

if _ON_DATABRICKS:
    _written = spark.table(SILVER_TABLE).count()  # type: ignore[union-attr]
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  bronze_records_in  : {_bronze_count:>12,}")
    print(f"  silver_records_out : {_written:>12,}")
    print(f"  dropped_deduped    : {_bronze_count - _silver_count:>12,}")
    print(f"  target_table       : {SILVER_TABLE}")
    print(f"  partitions_written : {_replace_predicate}")
    print("=" * 60)
else:
    print("Offline mode — summary skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC *End of notebook 01_bronze_to_silver — W4.3 / ds-02*
