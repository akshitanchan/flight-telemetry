#!/usr/bin/env python3
"""
Local bounded transforms: Silver -> Gold.

Aggregates silver flight states into three gold tables:
1. gold_emergency_events (squawk 7500/7600/7700 incidents)
2. gold_sector_load (distinct aircraft per H3 res 4 cell per 5m window)
3. gold_airport_congestion (ground vs airborne counts near airports per 5m window)
"""

import h3
import json
import logging
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

# JSON schema validators (optional but recommended if jsonschema is installed)
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

logger = logging.getLogger(__name__)

WINDOW_MINUTES = 5

# Emergency observations of the same (icao24, squawk) farther apart than this
# start a new event, so two incidents hours apart are not merged (audit M17).
EMERGENCY_GAP_S = 1800  # 30 minutes

def _floor_to_window(dt: datetime, minutes: int) -> datetime:
    """Floor a datetime to the nearest window bucket."""
    delta_mins = dt.minute - (dt.minute % minutes)
    return dt.replace(minute=delta_mins, second=0, microsecond=0)

def aggregate_emergency_events(records: list[dict]) -> list[dict]:
    """Identify and aggregate emergency squawk events.

    Consecutive observations of the same (icao24, squawk) are merged into one
    event. A gap larger than EMERGENCY_GAP_S between observations splits them
    into separate events, so two incidents hours apart are not merged into one
    huge-duration record (audit M17).
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
            gap = (datetime.fromisoformat(event_ts)
                   - datetime.fromisoformat(evt["last_seen_ts"])).total_seconds()
            if gap <= EMERGENCY_GAP_S:
                evt["last_seen_ts"] = event_ts
                if not evt["callsign"] and rec.get("callsign"):
                    evt["callsign"] = rec["callsign"]
                continue
            # Gap too large: close the current event and start a fresh one.
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

def aggregate_sector_load(records: list[dict]) -> list[dict]:
    """Aggregate distinct aircraft per H3_r4 sector per 5m window."""
    # key: (h3_r4, window_start_iso)
    # val: set of icao24
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
            "aircraft_count": len(icao_set)
        })

    return gold_records

def aggregate_airport_congestion(records: list[dict]) -> list[dict]:
    """Aggregate ground/airborne counts and avg altitude near airports per 5m window."""
    # key: (airport_icao, window_start_iso)
    # val: { "icaos": set(), "ground": int, "airborne": int, "alt_sum": float, "alt_count": int }
    congestion: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "icaos": set(), "ground": 0, "airborne": 0, "alt_sum": 0.0, "alt_count": 0
    })

    for rec in records:
        airport = rec.get("nearest_airport")
        if not airport:
            # No airport within the association radius — this aircraft is not near
            # any airport and does not contribute to airport congestion. We do NOT
            # invent a synthetic "UNKNOWN" bucket (audit C7 / ADR-0006).
            continue

        ts_str = rec.get("event_ts")
        if not ts_str:
            continue

        dt = datetime.fromisoformat(ts_str)
        w_start = _floor_to_window(dt, WINDOW_MINUTES)
        key = (airport, w_start.isoformat())

        agg = congestion[key]
        
        # Deduplicate per window if an aircraft sends multiple pings
        # But we must decide if we count *every* ping's on_ground status or 
        # just latest. For a 5-minute window batch, counting pings is standard.
        # But let's count distinct aircraft for the `aircraft_count` field.
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
        
        avg_alt = None
        if agg["alt_count"] > 0:
            avg_alt = round(agg["alt_sum"] / agg["alt_count"], 2)

        gold_records.append({
            "airport_icao": airport,
            "window_start": w_start_iso,
            "window_end": w_end.isoformat(),
            "aircraft_count": len(agg["icaos"]),
            "ground_count": agg["ground"],
            "airborne_count": agg["airborne"],
            "avg_altitude_m": avg_alt
        })

    return gold_records

def aggregate_routing_stats(records: list[dict]) -> list[dict]:
    """Aggregate route-level statistics per aircraft over a window."""
    # group by (icao24, callsign), then collect metrics
    routes = {}

    for rec in sorted(records, key=lambda r: r.get("event_ts", "")):
        icao24 = rec.get("icao24")
        if not icao24:
            continue
        callsign = rec.get("callsign")
        # Normalise callsign before keying so trailing-whitespace / empty variants
        # of the same flight don't fan out into duplicate routes (audit M18).
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
                "ping_count": 0
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
    # For daily windows or arbitrary bounds, we bucket by start of window. 
    # For this sample, we just use the bounds of the actual flight as the window.
    for rt in routes.values():
        avg_vel = None
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
            "ping_count": rt["ping_count"]
        })

    return gold_records

def load_schema(schema_path: Path) -> dict:
    if not schema_path.exists():
        return {}
    with open(schema_path) as f:
        return json.load(f)

def write_gold_table(output_path: Path, records: list[dict], schema: dict | None, validate: bool = True) -> int:
    """Write records to JSONL, optionally validating against schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic output order so gold diffs are stable across runs (audit m15).
    records = sorted(records, key=lambda r: json.dumps(r, sort_keys=True))

    validator = None
    if validate and HAS_JSONSCHEMA and schema:
        validator = jsonschema.Draft202012Validator(schema)

    written = 0
    with open(output_path, "w") as f:
        for rec in records:
            if validator:
                try:
                    validator.validate(rec)
                except jsonschema.ValidationError as e:
                    logger.warning("Validation error in %s: %s", output_path.name, e.message)
                    continue
            
            f.write(json.dumps(rec) + "\n")
            written += 1
            
    return written

def run_silver_to_gold(
    silver_input: Path,
    out_dir: Path,
    contracts_dir: Path,
    validate: bool = True
) -> dict:
    """Main execution of all three gold aggregates."""
    start_ts = time_mod.monotonic()

    # 1. Read silver records
    records = []
    with open(silver_input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 2. Compute aggregates
    emergencies = aggregate_emergency_events(records)
    sectors = aggregate_sector_load(records)
    congestion = aggregate_airport_congestion(records)
    routing = aggregate_routing_stats(records)

    # 3. Write and validate
    schemas = {
        "emergency": load_schema(contracts_dir / "gold_emergency_events.schema.json"),
        "sector": load_schema(contracts_dir / "gold_sector_load.schema.json"),
        "congestion": load_schema(contracts_dir / "gold_airport_congestion.schema.json"),
        "routing": load_schema(contracts_dir / "gold_routing_stats.schema.json"),
    }

    w_em = write_gold_table(out_dir / "gold_emergency_events.jsonl", emergencies, schemas["emergency"], validate)
    w_sec = write_gold_table(out_dir / "gold_sector_load.jsonl", sectors, schemas["sector"], validate)
    w_con = write_gold_table(out_dir / "gold_airport_congestion.jsonl", congestion, schemas["congestion"], validate)
    w_rte = write_gold_table(out_dir / "gold_routing_stats.jsonl", routing, schemas["routing"], validate)

    elapsed = time_mod.monotonic() - start_ts

    stats = {
        "input_file": silver_input.name,
        "silver_records_read": len(records),
        "gold_emergency_written": w_em,
        "gold_sector_written": w_sec,
        "gold_congestion_written": w_con,
        "gold_routing_written": w_rte,
        "elapsed_s": round(elapsed, 3),
    }
    logger.info("Silver->Gold complete: %s", json.dumps(stats, indent=2))
    return stats
