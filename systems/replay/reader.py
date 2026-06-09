#!/usr/bin/env python3
"""
Replay reader — reads OpenSky-format JSONL snapshot files.

Each line in the input is a JSON object matching the OpenSky /states/all
response format:

    {"time": <unix_ts>, "states": [[icao24, callsign, ...], ...]}

The reader yields individual state-vector arrays with their snapshot
timestamp attached, suitable for downstream normalization.
"""

import json
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def read_snapshots(path: Path) -> Generator[dict, None, None]:
    """Yield raw API-response dicts from a JSONL file.

    Each yielded dict has keys: "time" (int) and "states" (list of lists).
    """
    logger.info("Opening replay file: %s", path)
    line_count = 0
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_num, e)
                continue

            if "time" not in snapshot or "states" not in snapshot:
                logger.warning("Skipping line %d: missing 'time' or 'states'", line_num)
                continue

            line_count += 1
            yield snapshot

    logger.info("Read %d snapshots from %s", line_count, path.name)
