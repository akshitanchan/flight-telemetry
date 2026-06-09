#!/usr/bin/env python3
"""
Abstract interface for spatiotemporal index backends.

Every index strategy must implement this interface so the benchmark
runner can swap strategies without changing query or measurement code.

Plan §5.1: "Three interchangeable index backends behind one interface."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def parse_iso_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime.

    Handles a trailing 'Z' (UTC). Used for correct chronological comparison
    instead of fragile lexicographic string comparison, which breaks across
    mixed 'Z' vs '+00:00' suffixes (audit M12/M13).
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box for spatial range queries."""
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def contains(self, lon: float, lat: float) -> bool:
        return (self.lon_min <= lon <= self.lon_max and
                self.lat_min <= lat <= self.lat_max)


@dataclass(frozen=True)
class TimeWindow:
    """Time range for temporal filtering."""
    start: str   # ISO 8601 timestamp
    end: str     # ISO 8601 timestamp


@dataclass(frozen=True)
class SpatiotemporalQuery:
    """A spatiotemporal range query: all records in bbox B during window W."""
    bbox: BBox
    time_window: TimeWindow
    query_id: int = 0


class SpatiotemporalIndex(ABC):
    """Abstract base for pluggable spatiotemporal index backends.

    Lifecycle:
        1. build(records) — ingest all records and build the index
        2. query(query) — run a spatiotemporal range query, return matching records
        3. stats() — return index metadata (build time, size, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this index strategy."""
        ...

    @abstractmethod
    def build(self, records: list[dict]) -> None:
        """Build the index from a list of silver-format records.

        Each record has at minimum: icao24, event_ts, lon, lat, geohash7, h3_r7.
        """
        ...

    @abstractmethod
    def query(self, q: SpatiotemporalQuery) -> list[dict]:
        """Execute a spatiotemporal range query.

        Returns all records within the bbox AND time window.
        """
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return index statistics.

        Must include at minimum:
            - record_count: int
            - build_time_s: float
            - index_size_bytes: int (estimated in-memory size)
        """
        ...
