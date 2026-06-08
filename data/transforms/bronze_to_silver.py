#!/usr/bin/env python3
"""
Bronze-to-silver transform — converts landing-format JSONL records into
silver_flight_state records validated against the shared contract.

Transform steps:
  1. Read landing JSONL (bronze)
  2. Map and rename fields to silver schema
  3. Compute derived spatial indices (geohash7, h3_r7)
  4. Apply type/range validation
  5. Deduplicate on (icao24, event_ts)
  6. Stub nullable enrichment fields (nearest_airport, METAR)
  7. Validate output against the silver contract
  8. Write silver JSONL

This is the local development version. Production would run as a
Databricks/Spark job or dbt model.
"""

import json
import logging
import sys
import time
from pathlib import Path

import h3
import pygeohash

logger = logging.getLogger(__name__)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load the silver schema for validation
SCHEMA_PATH = PROJECT_ROOT / "shared" / "contracts" / "silver_flight_state.schema.json"


def _load_schema() -> dict:
    """Load the silver_flight_state JSON Schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validate_range(value, field: str, minimum=None, maximum=None) -> bool:
    """Check if a value is within the allowed range."""
    if value is None:
        return True  # nullable fields are OK when null
    if minimum is not None and value < minimum:
        logger.debug("Range check failed: %s=%s < %s", field, value, minimum)
        return False
    if maximum is not None and value > maximum:
        logger.debug("Range check failed: %s=%s > %s", field, value, maximum)
        return False
    return True


def transform_record(record: dict) -> dict | None:
    """Transform a single landing record into silver format.

    Returns a silver-compliant dict, or None if the record fails
    validation and should be dropped.
    """
    # --- Required field checks ---
    icao24 = record.get("icao24")
    if not icao24 or not isinstance(icao24, str) or len(icao24) != 6:
        logger.debug("Dropping record: invalid icao24 %r", icao24)
        return None

    event_ts = record.get("event_ts")
    if not event_ts:
        logger.debug("Dropping record: missing event_ts")
        return None

    lon = record.get("lon")
    lat = record.get("lat")
    if lon is None or lat is None:
        logger.debug("Dropping record: missing lon/lat")
        return None

    # --- Range validation ---
    if not _validate_range(lon, "lon", -180, 180):
        return None
    if not _validate_range(lat, "lat", -90, 90):
        return None
    if not _validate_range(record.get("velocity_ms"), "velocity_ms", 0):
        return None
    if not _validate_range(record.get("true_track_deg"), "true_track_deg", 0, 360):
        return None

    # --- Squawk validation (octal digits only) ---
    squawk = record.get("squawk")
    if squawk is not None:
        if not isinstance(squawk, str) or len(squawk) != 4:
            squawk = None
        elif not all(c in "01234567" for c in squawk):
            squawk = None

    # --- Compute derived spatial indices ---
    geohash7 = pygeohash.encode(lat, lon, precision=7)
    h3_r7 = h3.latlng_to_cell(lat, lon, 7)

    # --- Build silver record ---
    silver = {
        "icao24": icao24,
        "callsign": record.get("callsign"),
        "event_ts": event_ts,
        "lon": round(lon, 6),
        "lat": round(lat, 6),
        "baro_altitude_m": record.get("baro_altitude_m"),
        "velocity_ms": record.get("velocity_ms"),
        "true_track_deg": record.get("true_track_deg"),
        "vertical_rate_ms": record.get("vertical_rate_ms"),
        "on_ground": bool(record.get("on_ground", False)),
        "squawk": squawk,
        "origin_country": record.get("origin_country", "Unknown"),
        # --- Derived fields ---
        "geohash7": geohash7,
        "h3_r7": h3_r7,
        # --- Enrichment stubs (will be populated when reference joins are added) ---
        "nearest_airport": None,
        "metar_wind_kt": None,
        "metar_vis_m": None,
        "metar_ceiling_ft": None,
    }

    return silver


def run_transform(
    input_path: Path,
    output_path: Path,
    validate: bool = True,
) -> dict:
    """Run the full bronze-to-silver transform.

    Args:
        input_path: Path to landing JSONL file (bronze).
        output_path: Path to write silver JSONL output.
        validate: If True, validate each output record against the silver schema.

    Returns:
        Summary dict with statistics.
    """
    start = time.monotonic()
    schema = _load_schema() if validate else None

    # Optional: use jsonschema for full contract validation
    validator = None
    if validate:
        try:
            import jsonschema
            validator = jsonschema.Draft202012Validator(schema)
        except ImportError:
            logger.warning("jsonschema not installed — skipping full contract validation")

    seen_keys: set[str] = set()
    total_read = 0
    total_written = 0
    total_dropped = 0
    total_deduped = 0
    validation_errors = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_num, e)
                total_dropped += 1
                continue

            # Deduplicate
            idem_key = record.get("idem_key", f"{record.get('icao24')}:{record.get('event_ts')}")
            if idem_key in seen_keys:
                total_deduped += 1
                continue
            seen_keys.add(idem_key)

            # Transform
            silver = transform_record(record)
            if silver is None:
                total_dropped += 1
                continue

            # Contract validation
            if validator:
                errors = list(validator.iter_errors(silver))
                if errors:
                    validation_errors += 1
                    for err in errors:
                        logger.warning(
                            "Validation error on line %d: %s at %s",
                            line_num,
                            err.message,
                            list(err.absolute_path),
                        )
                    total_dropped += 1
                    continue

            fout.write(json.dumps(silver) + "\n")
            total_written += 1

    elapsed = time.monotonic() - start

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "records_read": total_read,
        "records_written": total_written,
        "records_dropped": total_dropped,
        "records_deduped": total_deduped,
        "validation_errors": validation_errors,
        "elapsed_s": round(elapsed, 3),
    }

    logger.info("Transform complete: %s", json.dumps(summary, indent=2))
    return summary
