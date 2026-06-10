# Databricks notebook source
# =============================================================================
# 03_refresh_experiment.py
# Wave W4.3 · Task ds-05
#
# Measures incremental-MERGE vs full-recompute latency on gold_airport_congestion
# at multiple row-count scales on real Delta / Spark.
#
# Sections
# --------
#   A. Widget / parameter setup (sizes, inc_pct, silver_table, gold_table).
#   B. Scale-data generation — inflates the silver Delta table to each target
#      size using Spark CROSS JOIN + LIMIT (bounded for Free Edition quota).
#   C. Experiment loop — for each scale:
#        C1. Full recompute: GROUP BY over all rows → overwrite gold.
#        C2. Incremental MERGE: aggregate incremental batch → MERGE INTO gold.
#        C3. Record timings in results list.
#   D. Results table — prints a formatted summary and stores to Delta.
#
# Free Edition quota note
# -----------------------
# Databricks Free Edition clusters are limited in memory/compute.  Widget
# default sizes are kept deliberately modest (10_000 / 50_000 / 200_000 rows).
# Do NOT exceed 1_000_000 rows on a single-node Free Edition cluster — use a
# real cluster (Standard_DS3_v2 or equivalent) for higher scales.
#
# Offline / py_compile guard
# --------------------------
# All spark / dbutils calls are inside `if _ON_DATABRICKS:` blocks.  The file
# passes `python -m py_compile` and import checks without a Spark runtime.
#
# Cross-reference
# ---------------
# MERGE statement mirrors Section B2 of 02_silver_to_gold.py exactly.
# Aggregation logic mirrors Section A4 of 02_silver_to_gold.py exactly.
# Offline harness: data/cloud/databricks/experiments/harness.py (DuckDB proxy).
# Results template: data/cloud/databricks/experiments/results_template.md
# =============================================================================

# COMMAND ----------
# MAGIC %md
# MAGIC ## 03_refresh_experiment — Incremental MERGE vs Full-Recompute Latency
# MAGIC
# MAGIC **Task**: ds-05  **Wave**: W4.3
# MAGIC
# MAGIC This notebook measures the latency of two refresh strategies for
# MAGIC `gold_airport_congestion` at increasing row-count scales:
# MAGIC
# MAGIC | Strategy | Description |
# MAGIC |---|---|
# MAGIC | **Full recompute** | GROUP BY over all silver rows, overwrite gold with `replaceWhere` |
# MAGIC | **Incremental MERGE** | Aggregate only the new batch, `MERGE INTO` the existing gold table |
# MAGIC
# MAGIC Speedup = full_recompute_s / incremental_merge_s.  A speedup > 1x means
# MAGIC incremental MERGE was faster; < 1x means full recompute was cheaper.
# MAGIC
# MAGIC The offline DuckDB proxy for this experiment lives at
# MAGIC `data/cloud/databricks/experiments/harness.py`.  Run it locally before
# MAGIC bringing up a Databricks cluster to verify the experiment shape.
# MAGIC
# MAGIC **Free Edition quota**: keep widget `sizes` at or below `200000` on a
# MAGIC single-node cluster.  Raise to `500000` on Standard_DS3_v2 or better.

# COMMAND ----------

# Guard: dbutils / spark exist on Databricks; fall back to stubs so the file
# passes `python -m py_compile` and offline import checks.
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
        "gold_congestion_table",
        "flight_telemetry.gold_airport_congestion",
        "Target Delta table for gold_airport_congestion",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "experiment_results_table",
        "flight_telemetry.refresh_experiment_results",
        "Delta table to persist timing results (created if missing)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "sizes",
        "10000,50000,200000",
        "Comma-separated target row counts (max 200000 on Free Edition)",
    )
    dbutils.widgets.text(  # type: ignore[union-attr]
        "inc_pct",
        "0.10",
        "Fraction of rows treated as the incremental batch (0 < inc_pct < 1)",
    )

# COMMAND ----------

# Resolve parameter values.
SILVER_TABLE: str = dbutils.widgets.get("silver_table")
GOLD_CONGESTION_TABLE: str = dbutils.widgets.get("gold_congestion_table")
EXPERIMENT_RESULTS_TABLE: str = dbutils.widgets.get("experiment_results_table")
_sizes_raw: str = dbutils.widgets.get("sizes")
_inc_pct_raw: str = dbutils.widgets.get("inc_pct")

SIZES: list[int] = [int(s.strip()) for s in _sizes_raw.split(",") if s.strip()]
INC_PCT: float = float(_inc_pct_raw) if _inc_pct_raw else 0.10

if _ON_DATABRICKS and not SILVER_TABLE:
    raise ValueError(
        "Widget 'silver_table' is empty. "
        "Set it to a fully-qualified table name, e.g. "
        "main.default.silver_flight_state"
    )

print(f"silver_table              : {SILVER_TABLE!r}")
print(f"gold_congestion_table     : {GOLD_CONGESTION_TABLE!r}")
print(f"experiment_results_table  : {EXPERIMENT_RESULTS_TABLE!r}")
print(f"sizes                     : {SIZES}")
print(f"inc_pct                   : {INC_PCT}")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION B — Scale-Data Generation
# MAGIC
# MAGIC For each target size the silver table is inflated via CROSS JOIN + LIMIT into
# MAGIC a temporary view `_exp_silver_<size>`.  The CROSS JOIN multiplier is computed
# MAGIC so that `multiplier * actual_silver_rows >= target_size`.
# MAGIC
# MAGIC Synthetic rows differ from the originals in `icao24` (suffixed `_<seq>`) but
# MAGIC share the same `nearest_airport` distribution as the real silver data.
# MAGIC Records with `nearest_airport IS NULL` are excluded from the congestion
# MAGIC aggregate (ADR-0006), so experiment rows without an airport are non-contributing
# MAGIC padding.
# MAGIC
# MAGIC **Free Edition safety**: each call creates a temporary view, not a persisted
# MAGIC table, so no Delta storage is consumed by the generated data.

# COMMAND ----------

import time as _time

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # type: ignore[import]

    _silver_df = spark.table(SILVER_TABLE)  # type: ignore[union-attr]
    _actual_silver_count = _silver_df.count()
    print(f"Actual silver rows: {_actual_silver_count:,}")

    _scale_views: dict[int, str] = {}

    for _target_size in SIZES:
        _multiplier = (_target_size // max(_actual_silver_count, 1)) + 1
        _view_name = f"_exp_silver_{_target_size}"

        _inflated = (
            _silver_df
            .crossJoin(
                spark  # type: ignore[union-attr]
                .range(1, _multiplier + 1)
                .withColumnRenamed("id", "_seq")
            )
            .withColumn(
                "icao24",
                F.concat(F.col("icao24"), F.lit("_"), F.col("_seq").cast("string")),
            )
            .drop("_seq")
            .limit(_target_size)
        )
        _inflated.createOrReplaceTempView(_view_name)
        _actual = spark.table(_view_name).count()  # type: ignore[union-attr]
        _scale_views[_target_size] = _view_name
        print(f"  View '{_view_name}': {_actual:,} rows (target {_target_size:,})")
else:
    _scale_views = {}
    print("Offline mode — scale views not created.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION C — Experiment Loop
# MAGIC
# MAGIC For each target size:
# MAGIC
# MAGIC 1. **Split** into base (90%) and incremental (10%) using `row_number()`.
# MAGIC 2. **Full recompute** (Strategy A): aggregate all rows with GROUP BY,
# MAGIC    overwrite `gold_congestion_table` using `replaceWhere`.  Timed.
# MAGIC 3. **Incremental MERGE** (Strategy B):
# MAGIC    a. Pre-build the base gold table from the base silver partition (un-timed).
# MAGIC    b. Aggregate only the incremental batch (timed).
# MAGIC    c. MERGE INTO the base gold table (timed, contiguous with step b).
# MAGIC
# MAGIC The MERGE statement is identical to Section B2 of 02_silver_to_gold.py.
# MAGIC Match key: `(airport_icao, window_start)`.  On MATCHED: full row
# MAGIC replacement.  On NOT MATCHED: INSERT.
# MAGIC
# MAGIC `avg_altitude_m` is always derived as `alt_sum / alt_count` — never
# MAGIC averaged-of-averages — so the MERGE is idempotent.

# COMMAND ----------

_WINDOW_MINUTES = 5  # mirrors WINDOW_MINUTES from lib/gold_logic.py

_results: list[dict] = []

if _ON_DATABRICKS:
    from pyspark.sql import functions as F  # noqa: F811
    from pyspark.sql.window import Window  # type: ignore[import]

    def _aggregate_congestion(df, label: str):  # type: ignore[no-untyped-def]
        """Aggregate a silver DataFrame into gold_airport_congestion shape.

        Mirrors Section A4 of 02_silver_to_gold.py exactly:
        - Filter nearest_airport IS NOT NULL (ADR-0006).
        - Floor event_ts to 5-minute window.
        - Deduplicate: first ping per (airport, window, icao24).
        - GROUP BY (nearest_airport, window_start, window_end).
        - avg_altitude_m = SUM / COUNT (never average-of-averages).
        """
        _cong_base = (
            df
            .filter(F.col("nearest_airport").isNotNull())
            .withColumn(
                "window_start",
                F.date_trunc("minute", "event_ts") - F.expr(
                    f"INTERVAL {_WINDOW_MINUTES} minutes"
                    f" * (minute(event_ts) % {_WINDOW_MINUTES})"
                ),
            )
            .withColumn(
                "window_end",
                F.col("window_start") + F.expr(f"INTERVAL {_WINDOW_MINUTES} minutes"),
            )
        )

        _rn_w = Window.partitionBy(
            "nearest_airport", "window_start", "icao24"
        ).orderBy("event_ts")
        _dedup = (
            _cong_base
            .withColumn("_rn", F.row_number().over(_rn_w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

        return (
            _dedup
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

    for _target_size in SIZES:
        _view = _scale_views[_target_size]
        _all_df = spark.table(_view)  # type: ignore[union-attr]
        _base_size = int(_target_size * (1.0 - INC_PCT))
        _inc_size = _target_size - _base_size

        print(
            f"\n--- Scale: {_target_size:,} rows "
            f"(base: {_base_size:,}, incremental: {_inc_size:,}) ---"
        )

        # Split base / incremental using row_number over the view.
        _rn_split_w = Window.orderBy(F.monotonically_increasing_id())
        _split_df = _all_df.withColumn("_split_rn", F.row_number().over(_rn_split_w))
        _base_df = _split_df.filter(F.col("_split_rn") <= _base_size).drop("_split_rn")
        _inc_df = _split_df.filter(F.col("_split_rn") > _base_size).drop("_split_rn")

        # ------------------------------------------------------------------
        # C1 — Full Recompute (timed)
        # ------------------------------------------------------------------
        _t_full_start = _time.monotonic()

        _full_cong = _aggregate_congestion(_all_df, "full")
        # Determine date range for idempotent replaceWhere.
        _full_dates = _full_cong.select(
            F.min("window_date").alias("min_d"),
            F.max("window_date").alias("max_d"),
        ).first()
        _full_predicate = (
            f"window_date >= '{_full_dates['min_d']}' "  # type: ignore[index]
            f"AND window_date <= '{_full_dates['max_d']}'"  # type: ignore[index]
        )

        (
            _full_cong.write
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", _full_predicate)
            .option("overwriteSchema", "true")
            .saveAsTable(GOLD_CONGESTION_TABLE)
        )

        _full_recompute_s = _time.monotonic() - _t_full_start
        _full_rows = spark.table(GOLD_CONGESTION_TABLE).count()  # type: ignore[union-attr]
        print(
            f"  Full recompute : {_full_recompute_s:.3f}s "
            f"({_full_rows:,} gold rows written)"
        )

        # ------------------------------------------------------------------
        # C2 — Incremental MERGE (timed)
        # Pre-step (un-timed): build base gold from _base_df.
        # ------------------------------------------------------------------
        _base_cong = _aggregate_congestion(_base_df, "base")
        (
            _base_cong.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(GOLD_CONGESTION_TABLE)
        )

        # Timed: aggregate incremental batch + MERGE INTO.
        _t_merge_start = _time.monotonic()

        _inc_cong = _aggregate_congestion(_inc_df, "inc")
        _inc_cong.createOrReplaceTempView("_exp_inc_cong_source")

        spark.sql(f"""  # type: ignore[union-attr]
            MERGE INTO {GOLD_CONGESTION_TABLE} AS target
            USING _exp_inc_cong_source AS source
            ON  target.airport_icao = source.airport_icao
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

        _incremental_merge_s = _time.monotonic() - _t_merge_start
        _merged_rows = spark.table(GOLD_CONGESTION_TABLE).count()  # type: ignore[union-attr]
        print(
            f"  Incremental MERGE : {_incremental_merge_s:.3f}s "
            f"({_merged_rows:,} gold rows after MERGE)"
        )

        _speedup = round(_full_recompute_s / max(_incremental_merge_s, 1e-6), 2)
        print(f"  Speedup : {_speedup}x")

        _results.append({
            "target_size": _target_size,
            "base_rows": _base_size,
            "inc_rows": _inc_size,
            "full_recompute_s": round(_full_recompute_s, 3),
            "incremental_merge_s": round(_incremental_merge_s, 3),
            "speedup_factor": _speedup,
        })
else:
    print("Offline mode — experiment loop skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## SECTION D — Results Table
# MAGIC
# MAGIC Prints a formatted summary and persists the timing data to a Delta table
# MAGIC (`experiment_results_table` widget) for downstream analysis.

# COMMAND ----------

if _ON_DATABRICKS and _results:
    from pyspark.sql.types import (  # type: ignore[import]
        StructType, StructField,
        IntegerType, DoubleType, LongType,
    )

    _schema = StructType([
        StructField("target_size",          LongType(),   nullable=False),
        StructField("base_rows",            LongType(),   nullable=False),
        StructField("inc_rows",             LongType(),   nullable=False),
        StructField("full_recompute_s",     DoubleType(), nullable=False),
        StructField("incremental_merge_s",  DoubleType(), nullable=False),
        StructField("speedup_factor",       DoubleType(), nullable=False),
    ])

    _results_df = spark.createDataFrame(_results, schema=_schema)  # type: ignore[union-attr]
    (
        _results_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(EXPERIMENT_RESULTS_TABLE)
    )
    print(f"Results persisted to: {EXPERIMENT_RESULTS_TABLE}")

    # Print ASCII summary.
    print()
    print("=== Experiment Results: Incremental MERGE vs Full Recompute ===")
    print(
        f"{'Total Rows':<14} | {'Full Recompute (s)':<20} | "
        f"{'Incremental MERGE (s)':<23} | {'Speedup':<8}"
    )
    print("-" * 75)
    for _r in _results:
        print(
            f"{_r['target_size']:<14} | {_r['full_recompute_s']:<20} | "
            f"{_r['incremental_merge_s']:<23} | {_r['speedup_factor']}x"
        )
    print("-" * 75)
    print()
    print(
        "NOTE: Fill results_template.md with these numbers + cluster spec "
        "for the ds-08 runbook."
    )
elif _ON_DATABRICKS:
    print("No results collected — ensure SECTION C ran without errors.")
else:
    print("Offline mode — results table skipped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Runbook (ds-08)
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC 1. Notebook `01_bronze_to_silver` has completed and
# MAGIC    `<catalog>.<schema>.silver_flight_state` is populated.
# MAGIC 2. A Unity Catalog schema (e.g. `flight_telemetry`) is accessible from
# MAGIC    the cluster.
# MAGIC 3. The `h3` Python library is installed on the cluster
# MAGIC    (`%pip install h3` or via cluster init script).
# MAGIC
# MAGIC ### Cluster recommendation
# MAGIC
# MAGIC | Edition | Recommended node | Max `sizes` widget |
# MAGIC |---|---|---|
# MAGIC | Free Edition (single-node) | DBR 13.3 LTS | 200_000 rows |
# MAGIC | Standard (multi-node) | Standard_DS3_v2 × 2 workers | 1_000_000 rows |
# MAGIC | Performance (multi-node) | Standard_DS4_v2 × 4 workers | 5_000_000+ rows |
# MAGIC
# MAGIC ### Step 1 — Configure widgets
# MAGIC
# MAGIC | Widget | Recommended value |
# MAGIC |---|---|
# MAGIC | `silver_table` | `<catalog>.<schema>.silver_flight_state` |
# MAGIC | `gold_congestion_table` | `<catalog>.<schema>.gold_airport_congestion` |
# MAGIC | `experiment_results_table` | `<catalog>.<schema>.refresh_experiment_results` |
# MAGIC | `sizes` | `10000,50000,200000` (Free Edition) or `50000,200000,500000` (Standard) |
# MAGIC | `inc_pct` | `0.10` |
# MAGIC
# MAGIC ### Step 2 — Run all sections
# MAGIC
# MAGIC Use **Run All** (Ctrl+Shift+Enter) or run sections A→D in order.
# MAGIC Expected total runtime: < 3 minutes on Free Edition (200k rows).
# MAGIC
# MAGIC ### Step 3 — Capture results
# MAGIC
# MAGIC After Section D prints the results table:
# MAGIC
# MAGIC 1. Copy the `target_size / full_recompute_s / incremental_merge_s / speedup`
# MAGIC    values into `data/cloud/databricks/experiments/results_template.md`.
# MAGIC 2. Fill in the cluster spec (node type, DBR version, worker count).
# MAGIC 3. Attach the completed template to the ds-08 runbook PR.
# MAGIC
# MAGIC ### Step 4 — Idempotency verification
# MAGIC
# MAGIC Re-run Section C with identical widget values.  The second MERGE run should
# MAGIC produce the same gold row counts as the first — verifying the MERGE is
# MAGIC idempotent.
# MAGIC
# MAGIC ### Failure and recovery
# MAGIC
# MAGIC | Symptom | Resolution |
# MAGIC |---|---|
# MAGIC | `silver_table not found` | Run `01_bronze_to_silver` first |
# MAGIC | `MERGE target lock timeout` | Retry; reduce `sizes` or increase cluster |
# MAGIC | `AnalysisException: nearest_airport` | Silver schema mismatch; check `01_bronze_to_silver` output |
# MAGIC | `ModuleNotFoundError: h3` | `%pip install h3` in a preceding cell or via init script |
# MAGIC | Section B CROSS JOIN OOM | Reduce the largest entry in `sizes` widget |
# MAGIC | Partial write left gold corrupt | Re-run Section C1 (full recompute) to recover |
# MAGIC
# MAGIC ---
# MAGIC *End of notebook 03_refresh_experiment — W4.3 / ds-05*
