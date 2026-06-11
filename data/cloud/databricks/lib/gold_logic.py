"""
Pure-Python aggregation mirror of data/transforms/silver_to_gold.py.

This module is intentionally free of Spark / PySpark / Delta so it can be
imported and unit-tested offline without a cluster.  The Databricks notebook
imports this module and reuses the constants for documentation/cross-reference;
the actual Spark aggregations in the notebook are written as native Spark SQL
to push computation to the engine.

All four aggregation functions here are EXACT mirrors of the authoritative
local implementations in data/transforms/silver_to_gold.py:

1. aggregate_emergency_events  — squawk 7500/7600/7700 incident merge with
                                  EMERGENCY_GAP_S gap-split logic (audit M17).
2. aggregate_sector_load        — distinct aircraft per H3 r4 cell per 5m window.
3. aggregate_airport_congestion — ground/airborne counts + avg altitude near
                                  airports per 5m window (ADR-0006: no UNKNOWN
                                  bucket for aircraft with no nearest_airport).
4. aggregate_routing_stats      — route-level stats per (icao24, normalised
                                  callsign) with origin=first/destination=last,
                                  max_altitude, avg_velocity, ping_count.

avg_altitude_m is ALWAYS derived as alt_sum / alt_count from the raw
accumulator — never as an average-of-averages — so it remains recomputable
from components and MERGE INTO operations stay idempotent.

Incremental merge functions (overlapping-batch safe)
-----------------------------------------------------
5. incremental_merge_emergency  — keyed partial-recompute for gold_emergency_events.
6. incremental_merge_routing    — keyed partial-recompute for gold_routing_stats.

Both functions implement the same pattern:

  (a) Identify affected entities in new_batch (the (icao24, squawk) or
      (icao24, normalised_callsign) pairs that appear in the incoming data).
  (b) For those entities only, collect ALL silver rows from the union of
      existing_silver and new_batch.  This ensures that any earlier
      observations arriving in new_batch are included in the recomputation
      so first_seen_ts / window_start is always the global minimum.
  (c) Recompute the gold aggregate over the combined silver rows — using
      the same aggregate_emergency_events / aggregate_routing_stats logic —
      so the result is byte-identical to a full recompute over the union.
  (d) Return the gold rows for unaffected entities untouched, and replace
      the rows for affected entities with the freshly recomputed ones.

Correctness guarantees:
  * Idempotent: applying the same new_batch twice produces the same result.
  * Order-independent: batch_A then batch_B == batch_B then batch_A ==
    full recompute over (batch_A + batch_B).
  * Converges to full-recompute: for any partition of silver rows into
    batches, the incremental result equals aggregate_*(union of all batches).
"""

import h3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Shared constants (authoritative in silver_to_gold.py; duplicated here so
# this module is self-contained and testable without importing from transforms).
# ---------------------------------------------------------------------------

WINDOW_MINUTES: int = 5

# Two observations of the same (icao24, squawk) separated by more than this
# many seconds are treated as separate emergency events (audit M17).
EMERGENCY_GAP_S: int = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _floor_to_window(dt: datetime, minutes: int) -> datetime:
    """Floor a UTC-aware or naive datetime to the nearest window bucket.

    Mirrors data/transforms/silver_to_gold._floor_to_window exactly.
    """
    delta_mins = dt.minute - (dt.minute % minutes)
    return dt.replace(minute=delta_mins, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# 1. Emergency events
# ---------------------------------------------------------------------------

def aggregate_emergency_events(records: list[dict]) -> list[dict]:
    """Identify and aggregate emergency squawk events.

    Consecutive observations of the same (icao24, squawk) are merged into one
    event.  A gap larger than EMERGENCY_GAP_S between observations splits them
    into separate events so two incidents hours apart are not merged into one
    huge-duration record (audit M17).

    Mirrors data/transforms/silver_to_gold.aggregate_emergency_events exactly.
    """
    open_event: dict[tuple[str, str], dict] = {}
    gold_records: list[dict] = []

    def _close(evt: dict) -> None:
        first = datetime.fromisoformat(evt["first_seen_ts"])
        last = datetime.fromisoformat(evt["last_seen_ts"])
        evt["duration_s"] = max(0, int((last - first).total_seconds()))
        gold_records.append(evt)

    for rec in sorted(records, key=lambda r: r.get("event_ts", "")):
        squawk = rec.get("squawk")
        if squawk not in ("7500", "7600", "7700"):
            continue

        icao24 = rec["icao24"]
        event_ts = rec["event_ts"]
        key = (icao24, squawk)

        evt = open_event.get(key)
        if evt is not None:
            gap = (
                datetime.fromisoformat(event_ts)
                - datetime.fromisoformat(evt["last_seen_ts"])
            ).total_seconds()
            if gap <= EMERGENCY_GAP_S:
                evt["last_seen_ts"] = event_ts
                if not evt["callsign"] and rec.get("callsign"):
                    evt["callsign"] = rec["callsign"]
                continue
            # Gap too large: close current event, start fresh.
            _close(evt)

        open_event[key] = {
            "icao24": icao24,
            "callsign": rec.get("callsign"),
            "squawk": squawk,
            "first_seen_ts": event_ts,
            "last_seen_ts": event_ts,
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "origin_country": rec.get("origin_country"),
            "nearest_airport": rec.get("nearest_airport"),
        }

    for evt in open_event.values():
        _close(evt)

    return gold_records


# ---------------------------------------------------------------------------
# 2. Sector load
# ---------------------------------------------------------------------------

def aggregate_sector_load(records: list[dict]) -> list[dict]:
    """Aggregate distinct aircraft per H3 res-4 sector per 5-minute window.

    Mirrors data/transforms/silver_to_gold.aggregate_sector_load exactly.
    """
    # key: (h3_r4, window_start_iso) → set of icao24
    sectors: dict[tuple[str, str], set[str]] = defaultdict(set)

    for rec in records:
        h3_r7 = rec.get("h3_r7")
        if not h3_r7:
            continue
        try:
            h3_r4 = h3.cell_to_parent(h3_r7, 4)
        except Exception:
            continue

        ts_str = rec.get("event_ts")
        if not ts_str:
            continue

        dt = datetime.fromisoformat(ts_str)
        w_start = _floor_to_window(dt, WINDOW_MINUTES)
        key = (h3_r4, w_start.isoformat())
        sectors[key].add(rec["icao24"])

    gold_records = []
    for (h3_r4, w_start_iso), icao_set in sectors.items():
        w_start = datetime.fromisoformat(w_start_iso)
        w_end = w_start + timedelta(minutes=WINDOW_MINUTES)
        gold_records.append({
            "h3_r4": h3_r4,
            "window_start": w_start_iso,
            "window_end": w_end.isoformat(),
            "aircraft_count": len(icao_set),
        })

    return gold_records


# ---------------------------------------------------------------------------
# 3. Airport congestion
# ---------------------------------------------------------------------------

def aggregate_airport_congestion(records: list[dict]) -> list[dict]:
    """Aggregate ground/airborne counts and avg altitude near airports per 5m window.

    Aircraft with no nearest_airport are silently dropped — no synthetic
    "UNKNOWN" bucket is created (ADR-0006, audit C7).

    avg_altitude_m is derived as alt_sum / alt_count so the raw accumulator
    components (alt_sum, alt_count) are always available for idempotent MERGE.

    Mirrors data/transforms/silver_to_gold.aggregate_airport_congestion exactly.
    """
    # key: (airport_icao, window_start_iso)
    # val: { icaos: set, ground: int, airborne: int, alt_sum: float, alt_count: int }
    congestion: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "icaos": set(), "ground": 0, "airborne": 0, "alt_sum": 0.0, "alt_count": 0
    })

    for rec in records:
        airport = rec.get("nearest_airport")
        if not airport:
            # ADR-0006: no UNKNOWN bucket.
            continue

        ts_str = rec.get("event_ts")
        if not ts_str:
            continue

        dt = datetime.fromisoformat(ts_str)
        w_start = _floor_to_window(dt, WINDOW_MINUTES)
        key = (airport, w_start.isoformat())

        agg = congestion[key]
        if rec["icao24"] not in agg["icaos"]:
            agg["icaos"].add(rec["icao24"])
            if rec.get("on_ground", False):
                agg["ground"] += 1
            else:
                agg["airborne"] += 1

            alt = rec.get("baro_altitude_m")
            if alt is not None:
                agg["alt_sum"] += alt
                agg["alt_count"] += 1

    gold_records = []
    for (airport, w_start_iso), agg in congestion.items():
        w_start = datetime.fromisoformat(w_start_iso)
        w_end = w_start + timedelta(minutes=WINDOW_MINUTES)

        avg_alt: Optional[float] = None
        if agg["alt_count"] > 0:
            avg_alt = round(agg["alt_sum"] / agg["alt_count"], 2)

        gold_records.append({
            "airport_icao": airport,
            "window_start": w_start_iso,
            "window_end": w_end.isoformat(),
            "aircraft_count": len(agg["icaos"]),
            "ground_count": agg["ground"],
            "airborne_count": agg["airborne"],
            "avg_altitude_m": avg_alt,
        })

    return gold_records


# ---------------------------------------------------------------------------
# 4. Routing stats
# ---------------------------------------------------------------------------

def aggregate_routing_stats(records: list[dict]) -> list[dict]:
    """Aggregate route-level statistics per aircraft over a window.

    Groups by (icao24, normalised callsign).  Trailing/leading whitespace and
    empty-string callsign variants are normalised to None before keying so
    they don't fan out into duplicate routes (audit M18).

    origin = first observed lat/lon; destination = last observed lat/lon.
    window_start = first event_ts; window_end = last event_ts.
    avg_velocity_mps is derived as vel_sum / vel_count.

    Mirrors data/transforms/silver_to_gold.aggregate_routing_stats exactly.
    """
    routes: dict[tuple, dict] = {}

    for rec in sorted(records, key=lambda r: r.get("event_ts", "")):
        icao24 = rec.get("icao24")
        if not icao24:
            continue
        callsign = rec.get("callsign")
        if isinstance(callsign, str):
            callsign = callsign.strip() or None

        event_ts = rec.get("event_ts")
        key = (icao24, callsign)

        if key not in routes:
            routes[key] = {
                "icao24": icao24,
                "callsign": callsign,
                "first_seen_ts": event_ts,
                "last_seen_ts": event_ts,
                "origin_lat": rec.get("lat"),
                "origin_lon": rec.get("lon"),
                "destination_lat": rec.get("lat"),
                "destination_lon": rec.get("lon"),
                "max_altitude_m": rec.get("baro_altitude_m"),
                "vel_sum": 0.0,
                "vel_count": 0,
                "ping_count": 0,
            }

        rt = routes[key]
        rt["last_seen_ts"] = event_ts
        rt["destination_lat"] = rec.get("lat")
        rt["destination_lon"] = rec.get("lon")
        rt["ping_count"] += 1

        alt = rec.get("baro_altitude_m")
        if alt is not None:
            if rt["max_altitude_m"] is None or alt > rt["max_altitude_m"]:
                rt["max_altitude_m"] = alt

        vel = rec.get("velocity_ms")
        if vel is not None:
            rt["vel_sum"] += vel
            rt["vel_count"] += 1

    gold_records = []
    for rt in routes.values():
        avg_vel: Optional[float] = None
        if rt["vel_count"] > 0:
            avg_vel = round(rt["vel_sum"] / rt["vel_count"], 2)

        gold_records.append({
            "icao24": rt["icao24"],
            "callsign": rt["callsign"],
            "window_start": rt["first_seen_ts"],
            "window_end": rt["last_seen_ts"],
            "origin_lat": rt["origin_lat"],
            "origin_lon": rt["origin_lon"],
            "destination_lat": rt["destination_lat"],
            "destination_lon": rt["destination_lon"],
            "max_altitude_m": rt["max_altitude_m"],
            "avg_velocity_mps": avg_vel,
            "ping_count": rt["ping_count"],
        })

    return gold_records


# ---------------------------------------------------------------------------
# 5. Incremental merge — gold_emergency_events (overlapping-batch safe)
# ---------------------------------------------------------------------------

def incremental_merge_emergency(
    existing_gold: list[dict],
    existing_silver: list[dict],
    new_batch_silver: list[dict],
) -> list[dict]:
    """Keyed partial-recompute incremental merge for gold_emergency_events.

    Correct for overlapping and out-of-order batches.  The result is
    identical to aggregate_emergency_events(existing_silver + new_batch_silver)
    regardless of batch ordering or overlap.

    Parameters
    ----------
    existing_gold:
        Current gold_emergency_events rows (all rows currently stored).
    existing_silver:
        All silver rows that were used to build existing_gold (full history
        for emergency-squawk records held in the silver store).
    new_batch_silver:
        Incoming silver rows for the new incremental batch.

    Returns
    -------
    New complete gold_emergency_events row list.  Rows for entities NOT
    touched by new_batch are carried forward unchanged.  Rows for affected
    entities are replaced with a fresh recomputation over the union.

    Design
    ------
    Affected entity key: (icao24, squawk).  Only squawk 7500/7600/7700 rows
    can produce emergency events; non-emergency rows in new_batch are ignored
    for entity identification (they don't affect emergency gold).

    For each affected (icao24, squawk):
      - Gather ALL silver rows for that entity from existing_silver + new_batch.
      - Run aggregate_emergency_events over that subset.
      - Replace gold rows keyed to that entity with the recomputed result.

    This guarantees first_seen_ts is always the global minimum across all
    batches, so overlapping batches that carry earlier observations converge
    correctly instead of inserting duplicate gold rows.
    """
    EMERGENCY_SQUAWKS = ("7500", "7600", "7700")

    # Step 1: Identify affected (icao24, squawk) entities from the new batch.
    affected: set[tuple[str, str]] = set()
    for rec in new_batch_silver:
        squawk = rec.get("squawk")
        icao24 = rec.get("icao24")
        if squawk in EMERGENCY_SQUAWKS and icao24:
            affected.add((icao24, squawk))

    # If no emergency records in the batch, existing gold is unchanged.
    if not affected:
        return list(existing_gold)

    # Step 2: Collect silver rows for affected entities from the full union.
    # Deduplicate by (icao24, event_ts) — the silver primary key — so that
    # rows already present in existing_silver are not double-counted when
    # new_batch_silver is applied a second time (idempotency guarantee).
    seen_silver_keys: set[tuple] = set()
    union_silver: list[dict] = []
    for r in existing_silver + new_batch_silver:
        key = (r.get("icao24"), r.get("event_ts"))
        if key not in seen_silver_keys:
            seen_silver_keys.add(key)
            union_silver.append(r)

    affected_silver: list[dict] = [
        r for r in union_silver
        if r.get("squawk") in EMERGENCY_SQUAWKS
        and (r.get("icao24"), r.get("squawk")) in affected
    ]

    # Step 3: Recompute gold rows for affected entities over the combined set.
    recomputed: list[dict] = aggregate_emergency_events(affected_silver)

    # Step 4: Carry forward gold rows for unaffected (icao24, squawk) entities,
    # then append the freshly recomputed rows for affected entities.
    unaffected_gold: list[dict] = [
        g for g in existing_gold
        if (g.get("icao24"), g.get("squawk")) not in affected
    ]

    return unaffected_gold + recomputed


# ---------------------------------------------------------------------------
# 6. Incremental merge — gold_routing_stats (overlapping-batch safe)
# ---------------------------------------------------------------------------

def _normalise_callsign(callsign) -> Optional[str]:
    """Normalise callsign: strip whitespace, empty string → None (audit M18)."""
    if isinstance(callsign, str):
        return callsign.strip() or None
    return callsign


def incremental_merge_routing(
    existing_gold: list[dict],
    existing_silver: list[dict],
    new_batch_silver: list[dict],
) -> list[dict]:
    """Keyed partial-recompute incremental merge for gold_routing_stats.

    Correct for overlapping and out-of-order batches.  The result is
    identical to aggregate_routing_stats(existing_silver + new_batch_silver)
    regardless of batch ordering or overlap.

    Parameters
    ----------
    existing_gold:
        Current gold_routing_stats rows (all rows currently stored).
    existing_silver:
        All silver rows that were used to build existing_gold (full history
        held in the silver store for routing records).
    new_batch_silver:
        Incoming silver rows for the new incremental batch.

    Returns
    -------
    New complete gold_routing_stats row list.  Rows for entities NOT touched
    by new_batch are carried forward unchanged.  Rows for affected entities
    are replaced with a fresh recomputation over the union.

    Design
    ------
    Affected entity key: (icao24, normalised_callsign).  For each affected
    entity, gather ALL silver rows from existing_silver + new_batch_silver,
    run aggregate_routing_stats over that subset, and replace the gold rows
    for that entity.

    This guarantees window_start (= first_seen_ts of the route) is always
    the global minimum across all batches, eliminating the duplicate-row
    problem that occurs when a new batch contains earlier observations of an
    ongoing route.
    """
    # Step 1: Identify affected (icao24, normalised_callsign) entities.
    affected: set[tuple] = set()
    for rec in new_batch_silver:
        icao24 = rec.get("icao24")
        if not icao24:
            continue
        callsign = _normalise_callsign(rec.get("callsign"))
        affected.add((icao24, callsign))

    # If batch has no valid records, existing gold is unchanged.
    if not affected:
        return list(existing_gold)

    # Step 2: Collect silver rows for affected entities from the full union.
    # Deduplicate by (icao24, event_ts) — the silver primary key — so that
    # rows already present in existing_silver are not double-counted when
    # new_batch_silver is applied a second time (idempotency guarantee).
    seen_silver_keys: set[tuple] = set()
    union_silver: list[dict] = []
    for r in existing_silver + new_batch_silver:
        key = (r.get("icao24"), r.get("event_ts"))
        if key not in seen_silver_keys:
            seen_silver_keys.add(key)
            union_silver.append(r)

    affected_silver: list[dict] = []
    for r in union_silver:
        icao24 = r.get("icao24")
        if not icao24:
            continue
        callsign = _normalise_callsign(r.get("callsign"))
        if (icao24, callsign) in affected:
            affected_silver.append(r)

    # Step 3: Recompute gold rows for affected entities.
    recomputed: list[dict] = aggregate_routing_stats(affected_silver)

    # Step 4: Carry forward gold rows for unaffected entities, then append
    # freshly recomputed rows for affected entities.
    unaffected_gold: list[dict] = [
        g for g in existing_gold
        if (g.get("icao24"), _normalise_callsign(g.get("callsign"))) not in affected
    ]

    return unaffected_gold + recomputed
