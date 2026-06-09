#!/usr/bin/env python3
"""Structured analytics tool over the gold tables.

Exposes a small set of **named, validated** operations — never free-form or
destructive SQL (plan §5.4) — so the AI layer answers operational questions with
deterministic, grounded results.

Every operation returns a dict with:
  - ``answer``  : the primary result (scalar / list / dict) the eval checks read
  - ``rows``    : the supporting gold rows
  - ``sources`` : the gold table(s) the answer was derived from (for citation)
"""

import json
from pathlib import Path

GOLD_FILES = {
    "congestion": "gold_airport_congestion.jsonl",
    "sector": "gold_sector_load.jsonl",
    "emergency": "gold_emergency_events.jsonl",
    "routing": "gold_routing_stats.jsonl",
}


class AnalyticsError(ValueError):
    """Raised for an unknown operation or invalid parameters."""


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class AnalyticsTool:
    """Validated, parameterized query interface over the four gold tables."""

    def __init__(self, gold_dir):
        self.gold_dir = Path(gold_dir)
        self._tables = {
            name: _load_jsonl(self.gold_dir / fname)
            for name, fname in GOLD_FILES.items()
        }

    @property
    def operations(self) -> dict:
        """Whitelist of callable operations (no arbitrary code/SQL)."""
        return {
            "count_emergencies": self.count_emergencies,
            "list_emergency_aircraft": self.list_emergency_aircraft,
            "airport_congestion": self.airport_congestion,
            "count_active_airports": self.count_active_airports,
            "count_active_sectors": self.count_active_sectors,
            "total_flights_tracked": self.total_flights_tracked,
            "flight_summary": self.flight_summary,
            "highest_altitude_flight": self.highest_altitude_flight,
        }

    def call(self, operation: str, **params):
        """Dispatch a named operation. Rejects anything not on the whitelist."""
        ops = self.operations
        if operation not in ops:
            raise AnalyticsError(
                f"unknown operation {operation!r}; allowed: {sorted(ops)}"
            )
        return ops[operation](**params)

    # ---- emergency events ----
    def count_emergencies(self, squawk=None) -> dict:
        rows = self._tables["emergency"]
        if squawk is not None:
            squawk = str(squawk)
            rows = [r for r in rows if r.get("squawk") == squawk]
        return {"answer": len(rows), "rows": rows, "sources": ["gold_emergency_events"]}

    def list_emergency_aircraft(self, squawk=None) -> dict:
        rows = self._tables["emergency"]
        if squawk is not None:
            squawk = str(squawk)
            rows = [r for r in rows if r.get("squawk") == squawk]
        ids = sorted({r["icao24"] for r in rows})
        return {"answer": ids, "rows": rows, "sources": ["gold_emergency_events"]}

    # ---- airport congestion ----
    def airport_congestion(self, airport_icao=None) -> dict:
        if not airport_icao:
            raise AnalyticsError("airport_icao is required")
        rows = [r for r in self._tables["congestion"]
                if r.get("airport_icao") == airport_icao]
        total = sum(r.get("aircraft_count", 0) for r in rows)
        return {
            "answer": {"airport_icao": airport_icao,
                       "aircraft_count": total,
                       "windows": len(rows)},
            "rows": rows,
            "sources": ["gold_airport_congestion"],
        }

    def count_active_airports(self) -> dict:
        n = len({r["airport_icao"] for r in self._tables["congestion"]})
        return {"answer": n, "rows": self._tables["congestion"],
                "sources": ["gold_airport_congestion"]}

    # ---- sector load ----
    def count_active_sectors(self) -> dict:
        n = len({r["h3_r4"] for r in self._tables["sector"]})
        return {"answer": n, "rows": self._tables["sector"],
                "sources": ["gold_sector_load"]}

    # ---- routing ----
    def total_flights_tracked(self) -> dict:
        n = len({r["icao24"] for r in self._tables["routing"]})
        return {"answer": n, "rows": self._tables["routing"],
                "sources": ["gold_routing_stats"]}

    def flight_summary(self, icao24=None) -> dict:
        if not icao24:
            raise AnalyticsError("icao24 is required")
        rows = [r for r in self._tables["routing"] if r.get("icao24") == icao24]
        if not rows:
            return {"answer": None, "rows": [], "sources": ["gold_routing_stats"]}
        r = rows[0]
        return {
            "answer": {"icao24": icao24, "callsign": r.get("callsign"),
                       "max_altitude_m": r.get("max_altitude_m"),
                       "avg_velocity_mps": r.get("avg_velocity_mps"),
                       "ping_count": r.get("ping_count")},
            "rows": rows,
            "sources": ["gold_routing_stats"],
        }

    def highest_altitude_flight(self) -> dict:
        rows = [r for r in self._tables["routing"]
                if r.get("max_altitude_m") is not None]
        if not rows:
            return {"answer": None, "rows": [], "sources": ["gold_routing_stats"]}
        top = max(rows, key=lambda r: r["max_altitude_m"])
        return {
            "answer": {"icao24": top["icao24"], "max_altitude_m": top["max_altitude_m"]},
            "rows": [top],
            "sources": ["gold_routing_stats"],
        }
