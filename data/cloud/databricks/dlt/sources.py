"""Lakeflow SDP source aliases for validating existing silver and gold tables."""

from __future__ import annotations

import dlt  # type: ignore[import-not-found]


SOURCE_CATALOG = spark.conf.get("flight_telemetry.source_catalog", "main")  # type: ignore[name-defined]  # noqa: F821
SOURCE_SCHEMA = spark.conf.get("flight_telemetry.source_schema", "default")  # type: ignore[name-defined]  # noqa: F821


def _source_table(table_name: str):
    qualified = f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"
    return spark.readStream.table(qualified)  # type: ignore[name-defined]  # noqa: F821


@dlt.view(name="silver_flight_state")
def silver_flight_state():
    return _source_table("silver_flight_state")


@dlt.view(name="gold_airport_congestion")
def gold_airport_congestion():
    return _source_table("gold_airport_congestion")


@dlt.view(name="gold_sector_load")
def gold_sector_load():
    return _source_table("gold_sector_load")


@dlt.view(name="gold_emergency_events")
def gold_emergency_events():
    return _source_table("gold_emergency_events")


@dlt.view(name="gold_routing_stats")
def gold_routing_stats():
    return _source_table("gold_routing_stats")
