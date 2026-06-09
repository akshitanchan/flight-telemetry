#!/usr/bin/env python3
"""Deterministic rule-based answer strategy (W3.2 baseline).

Routes a natural-language question to a tool operation using keyword/intent
patterns and entity extraction (squawk code, airport ICAO, icao24), executes the
tool, and formats a grounded answer with a source citation. No LLM — fully
deterministic and offline, so it runs in CI without keys. This is "strategy #1";
later gates add alternative strategies behind the same interface.
"""

import re

from ai.agent.base import Answer, AnswerStrategy

_SQUAWK = re.compile(r"\b(7500|7600|7700)\b")
_ICAO = re.compile(r"\b[A-Z]{4}\b")          # airport ICAO, case-sensitive
_ICAO24 = re.compile(r"\b[0-9a-f]{6}\b")      # transponder hex id


class RuleBasedStrategy(AnswerStrategy):
    name = "rule_based_v1"

    def __init__(self, analytics, retrieval):
        self.analytics = analytics
        self.retrieval = retrieval

    def answer(self, question: str) -> Answer:
        q = question.lower()
        m = _SQUAWK.search(question)
        squawk = m.group(1) if m else None
        m = _ICAO.search(question)
        icao = m.group(0) if m else None
        m = _ICAO24.search(q)
        icao24 = m.group(0) if m else None

        # --- retrieval intents take precedence (corpus questions) ---
        if any(k in q for k in ("metar", "decode", "weather")):
            return self._retrieval(question)
        if "squawk" in q and any(k in q for k in ("mean", "indicate", "stand for", "stands for")):
            return self._retrieval(question)
        if any(k in q for k in ("report", "incident", "failure", "accident")):
            return self._retrieval(question)

        # --- emergency analytics ---
        if "squawk" in q or "emergency" in q:
            if any(k in q for k in ("which", "list", "what aircraft", "who")):
                return self._analytics(
                    question, "list_emergency_aircraft", {"squawk": squawk},
                    lambda r: (f"Aircraft that squawked {squawk}: "
                               f"{', '.join(r['answer']) or 'none'}." if squawk
                               else f"Emergency aircraft: {', '.join(r['answer']) or 'none'}."))
            return self._analytics(
                question, "count_emergencies", {"squawk": squawk},
                lambda r: (f"{r['answer']} flight(s) squawked {squawk}." if squawk
                           else f"There were {r['answer']} emergency squawk event(s)."))

        # --- airport congestion for a specific airport ---
        if icao and any(k in q for k in ("airport", "congestion", "aircraft")):
            return self._analytics(
                question, "airport_congestion", {"airport_icao": icao},
                lambda r: (f"{icao} had {r['answer']['aircraft_count']} aircraft "
                           f"across {r['answer']['windows']} window(s)."))

        # --- count operations ---
        if "how many" in q and "airport" in q:
            return self._analytics(
                question, "count_active_airports", {},
                lambda r: f"{r['answer']} distinct airports had associated traffic.")
        if "sector" in q:
            return self._analytics(
                question, "count_active_sectors", {},
                lambda r: f"{r['answer']} distinct H3 sectors were active.")
        if "flight" in q and "how many" in q and not icao24:
            return self._analytics(
                question, "total_flights_tracked", {},
                lambda r: f"{r['answer']} distinct flights were tracked.")

        # --- altitude / flight summary ---
        if icao24 and any(k in q for k in ("altitude", "summary", "max", "speed", "velocity")):
            return self._analytics(
                question, "flight_summary", {"icao24": icao24},
                lambda r: (f"Flight {icao24} ({r['answer']['callsign']}) reached "
                           f"{r['answer']['max_altitude_m']} m over "
                           f"{r['answer']['ping_count']} pings."))
        if any(k in q for k in ("highest", "maximum")) and "altitude" in q:
            return self._analytics(
                question, "highest_altitude_flight", {},
                lambda r: (f"The highest-altitude flight was {r['answer']['icao24']} "
                           f"at {r['answer']['max_altitude_m']} m."))

        # --- fallback: free-text retrieval ---
        return self._retrieval(question)

    # ---- helpers ----
    def _analytics(self, question, operation, params, fmt) -> Answer:
        result = self.analytics.call(operation, **params)
        cites = ", ".join(result.get("sources", [])) or "none"
        return Answer(
            question=question,
            answer_text=f"{fmt(result)} (source: {cites})",
            result=result,
            route=f"analytics:{operation}",
            strategy=self.name,
        )

    def _retrieval(self, question) -> Answer:
        result = self.retrieval.search(question)
        top = result["results"][0]["snippet"] if result["results"] else "No relevant document found."
        cites = ", ".join(result.get("sources", [])) or "none"
        return Answer(
            question=question,
            answer_text=f"{top} (source: {cites})",
            result=result,
            route="retrieval:search",
            strategy=self.name,
        )
