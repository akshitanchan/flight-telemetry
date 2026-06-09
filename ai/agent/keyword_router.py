#!/usr/bin/env python3
"""Second deterministic answer strategy (W3.3): keyword-scoring intent router.

A different routing *mechanism* from the ordered-rules baseline: it scores the
question against weighted keyword sets per operation (with entity gating for
operations that need an airport ICAO or an icao24) and picks the highest-scoring
operation. Retrieval intents short-circuit first. Fully deterministic / offline.

Provides a genuine comparison point against ``rule_based_v1`` at the same
(near-zero) latency and zero token cost.
"""

import re

from ai.agent.base import Answer, AnswerStrategy

_TOK = re.compile(r"[a-z0-9]+")
_SQUAWK = re.compile(r"\b(7500|7600|7700)\b")
_ICAO = re.compile(r"\b[A-Z]{4}\b")
_ICAO24 = re.compile(r"\b[0-9a-f]{6}\b")

_RETRIEVAL_NOUNS = {"metar", "decode", "weather", "report", "incident", "failure", "accident"}
_MEANING = ("mean", "means", "indicate", "indicates", "stand for", "stands for")


class KeywordScoreStrategy(AnswerStrategy):
    name = "keyword_score_v1"

    def __init__(self, analytics, retrieval):
        self.analytics = analytics
        self.retrieval = retrieval

    def answer(self, question: str) -> Answer:
        q = question.lower()
        toks = set(_TOK.findall(q))
        m = _SQUAWK.search(question)
        squawk = m.group(1) if m else None
        m = _ICAO.search(question)
        icao = m.group(0) if m else None
        m = _ICAO24.search(q)
        icao24 = m.group(0) if m else None

        # --- retrieval intents short-circuit ---
        if toks & _RETRIEVAL_NOUNS:
            return self._retrieval(question)
        if "squawk" in q and any(p in q for p in _MEANING):
            return self._retrieval(question)

        how_many = "how many" in q
        scores = {
            "count_emergencies":
                3 * len(toks & {"emergency", "emergencies"})
                + len(toks & {"events", "event", "incidents"})
                + (1 if how_many else 0),
            "list_emergency_aircraft":
                2 * len(toks & {"squawked"})
                + len(toks & {"emergency", "emergencies"})
                + (2 if (any(w in q for w in ("which", "who", "list"))
                         and ("aircraft" in q or "squawk" in q)) else 0),
            "count_active_airports":
                len(toks & {"airports", "traffic"}) + (1 if how_many else 0),
            "count_active_sectors":
                2 * len(toks & {"sector", "sectors"}),
            "total_flights_tracked":
                2 * len(toks & {"flights"}) + len(toks & {"tracked"})
                + (1 if how_many else 0),
            "highest_altitude_flight":
                len(toks & {"altitude"})
                + (2 if any(w in q for w in ("highest", "tallest")) else 0)
                + (1 if ("maximum" in q and not icao24) else 0),
        }
        # Entity-gated operations only score when their entity is present.
        if icao:
            scores["airport_congestion"] = 2 + len(toks & {"airport", "congestion", "aircraft"})
        if icao24:
            scores["flight_summary"] = 2 + len(toks & {"altitude", "speed", "velocity", "summary", "max", "maximum"})

        best = max(scores, key=lambda k: scores[k])
        if scores[best] <= 0:
            return self._retrieval(question)

        params = {}
        if best in ("count_emergencies", "list_emergency_aircraft") and squawk:
            params["squawk"] = squawk
        elif best == "airport_congestion":
            params["airport_icao"] = icao
        elif best == "flight_summary":
            params["icao24"] = icao24
        return self._analytics(question, best, params)

    # ---- helpers ----
    def _analytics(self, question, operation, params) -> Answer:
        result = self.analytics.call(operation, **params)
        cites = ", ".join(result.get("sources", [])) or "none"
        return Answer(question=question,
                      answer_text=f"{result['answer']} (source: {cites})",
                      result=result, route=f"analytics:{operation}", strategy=self.name)

    def _retrieval(self, question) -> Answer:
        result = self.retrieval.search(question)
        cites = ", ".join(result.get("sources", [])) or "none"
        top = result["results"][0]["snippet"] if result["results"] else "No relevant document found."
        return Answer(question=question, answer_text=f"{top} (source: {cites})",
                      result=result, route="retrieval:search", strategy=self.name)
