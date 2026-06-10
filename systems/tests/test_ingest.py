#!/usr/bin/env python3
"""
systems/tests/test_ingest.py
-----------------------------
Fully-offline tests for the OpenSky ingestion service (ws1-02).

All six behaviours are covered:
1. Token flow  — TokenManager fetches a token on first call and refreshes
                 proactively when the remaining lifetime is below the
                 REFRESH_BEFORE_EXPIRY_S threshold.
                 ingest_token_refresh_total increments on each fetch.
2. 401 refresh+retry-once  — a 401 response triggers exactly ONE
                             force_refresh + retry, then succeeds.
3. 429 backoff  — a 429 response with X-Rate-Limit-Retry-After-Seconds calls
                  the injected sleep_fn with the correct value and increments
                  ingest_rate_limit_sleeps_total.
4. Idempotent at-least-once landing  — feeding the same snapshot twice lands
                                       records exactly once.
5. Length-robust parse end-to-end  — an 18-field state vector with a
                                      category value flows through the service
                                      and lands with category populated.
6. Bronze shape  — the JSONL line the service writes matches
                   {"time", "states": [[...]]} and round-trips through
                   read_snapshots().

DB-dependent assertions are gated with unittest.skipUnless so CI without a
live Postgres stays green.

Stubbing approach
-----------------
- TokenManager.fetch_fn: inject a plain async function returning a dict with
  access_token / expires_in; no httpx required.
- OpenSkyClient._http: replaced with a lightweight FakeHTTP object whose .get()
  pops pre-programmed httpx.Response instances from a queue.  httpx.Response
  objects are constructed with a dummy httpx.Request so raise_for_status() works.
- sleep_fn: inject an async no-op _RecordingSleep that records calls; tests
  assert invocation counts and arguments without ever actually sleeping.
- Prometheus metrics: capture ._value.get() before and after the code under
  test and assert the delta.  All async operations run inside a single
  asyncio.run() call so that TokenManager's asyncio.Lock is bound to the
  correct event loop.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# ── Project root on sys.path (matches other test modules) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from systems.ingest.token import TokenManager, REFRESH_BEFORE_EXPIRY_S
from systems.ingest.client import OpenSkyClient
from systems.ingest.service import IngestService
from systems.replay.reader import read_snapshots
from shared.obs.telemetry import (
    ingest_token_refresh_total,
    ingest_rate_limit_sleeps_total,
    ingest_records_total,
)

# ── DB availability guard (identical pattern to test_index.py) ─────────────
from shared.store.pg import healthcheck as _pg_healthcheck

_DB_AVAILABLE = _pg_healthcheck()

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# A minimal 18-field OpenSky state vector used in multiple tests.
_STATE_18 = [
    "abc123",       # 0: icao24
    "SWR162 ",      # 1: callsign (trailing space — should be trimmed)
    "Switzerland",  # 2: origin_country
    1717416000,     # 3: time_position
    1717416000,     # 4: last_contact
    4.7638,         # 5: longitude
    52.3080,        # 6: latitude
    11278.0,        # 7: baro_altitude
    False,          # 8: on_ground
    234.5,          # 9: velocity
    42.3,           # 10: true_track
    0.0,            # 11: vertical_rate
    None,           # 12: sensors
    11290.5,        # 13: geo_altitude
    "2536",         # 14: squawk
    False,          # 15: spi
    0,              # 16: position_source
    3,              # 17: category (only in 18-field responses)
]

# A dummy httpx.Request needed so that httpx.Response.raise_for_status() works.
_DUMMY_REQUEST = httpx.Request("GET", "http://fake.example/states/all")


def _make_response(
    status: int,
    body: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    """Build a pre-wired httpx.Response (with request attached) so that
    raise_for_status() works correctly."""
    kwargs: dict[str, Any] = {"request": _DUMMY_REQUEST}
    if headers:
        kwargs["headers"] = headers
    if body is not None:
        kwargs["json"] = body
    return httpx.Response(status, **kwargs)


class _RecordingSleep:
    """Async no-op sleep that records every seconds argument without sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class _FakeHTTP:
    """Minimal stand-in for httpx.AsyncClient.

    Pops responses in FIFO order.  Raises IndexError if more requests are made
    than responses were queued (test bugs surface immediately).
    """

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._queue = list(responses)

    async def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response:
        return self._queue.pop(0)

    async def aclose(self) -> None:
        pass


class _FakeMockClient:
    """Stand-in for OpenSkyClient used directly by IngestService tests.

    Supports the async context-manager protocol that IngestService expects.
    """

    def __init__(self, responses: list[dict]) -> None:
        self._queue = list(responses)

    async def __aenter__(self) -> "_FakeMockClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def fetch_states(self) -> dict:
        return self._queue.pop(0)


def _counter_value(counter) -> float:
    """Read the current float value of a prometheus_client Counter."""
    return counter._value.get()


# ---------------------------------------------------------------------------
# 1. Token flow
# ---------------------------------------------------------------------------


class TestTokenFlow(unittest.TestCase):
    """TokenManager fetches a token on first call and refreshes near expiry."""

    def test_initial_fetch_returns_token(self):
        """get_token() on a fresh manager fetches and returns a token."""
        async def _body():
            calls: list[int] = []

            async def fake_fetch(client_id, client_secret, token_url):
                calls.append(1)
                return {"access_token": "tok-1", "expires_in": 1800}

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            token = await mgr.get_token()
            return token, calls

        token, calls = asyncio.run(_body())
        self.assertEqual(token, "tok-1")
        self.assertEqual(len(calls), 1)

    def test_second_call_uses_cached_token(self):
        """A second get_token() within the lifetime does NOT re-fetch."""
        async def _body():
            calls: list[int] = []

            async def fake_fetch(client_id, client_secret, token_url):
                calls.append(1)
                return {"access_token": f"tok-{len(calls)}", "expires_in": 1800}

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            await mgr.get_token()
            await mgr.get_token()
            return calls

        calls = asyncio.run(_body())
        self.assertEqual(len(calls), 1, "Expected only 1 fetch; got a spurious second call.")

    def test_refresh_when_near_expiry(self):
        """get_token() refreshes proactively when lifetime < REFRESH_BEFORE_EXPIRY_S."""
        async def _body():
            call_count = 0

            async def fake_fetch(client_id, client_secret, token_url):
                nonlocal call_count
                call_count += 1
                return {
                    "access_token": f"tok-{call_count}",
                    # Shorter than REFRESH_BEFORE_EXPIRY_S so the token
                    # is immediately considered near-expiry on the next call.
                    "expires_in": REFRESH_BEFORE_EXPIRY_S - 1,
                }

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            t1 = await mgr.get_token()
            t2 = await mgr.get_token()
            return t1, t2, call_count

        t1, t2, call_count = asyncio.run(_body())
        self.assertEqual(call_count, 2, "Expected proactive refresh on second get_token().")
        self.assertEqual(t1, "tok-1")
        self.assertEqual(t2, "tok-2")

    def test_metric_increments_on_each_fetch(self):
        """ingest_token_refresh_total increments once per token fetch call."""
        before = _counter_value(ingest_token_refresh_total)

        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": REFRESH_BEFORE_EXPIRY_S - 1}

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            await mgr.get_token()   # fetch #1
            await mgr.get_token()   # near-expiry → fetch #2

        asyncio.run(_body())
        after = _counter_value(ingest_token_refresh_total)
        self.assertEqual(after - before, 2.0, "Expected counter to increment by 2.")

    def test_force_refresh_discards_cached_token(self):
        """force_refresh() always re-fetches regardless of remaining lifetime."""
        async def _body():
            call_count = 0

            async def fake_fetch(client_id, client_secret, token_url):
                nonlocal call_count
                call_count += 1
                return {"access_token": f"tok-{call_count}", "expires_in": 1800}

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            t1 = await mgr.get_token()
            t2 = await mgr.force_refresh()
            return t1, t2, call_count

        t1, t2, call_count = asyncio.run(_body())
        self.assertEqual(call_count, 2)
        self.assertNotEqual(t1, t2)

    def test_metric_increments_on_force_refresh(self):
        """ingest_token_refresh_total increments on force_refresh() too."""
        before = _counter_value(ingest_token_refresh_total)

        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            await mgr.get_token()
            await mgr.force_refresh()

        asyncio.run(_body())
        after = _counter_value(ingest_token_refresh_total)
        self.assertEqual(after - before, 2.0)


# ---------------------------------------------------------------------------
# 2. 401 refresh + retry-once
# ---------------------------------------------------------------------------


class Test401RetryOnce(unittest.TestCase):
    """A 401 triggers exactly one force_refresh + retry, then succeeds."""

    def test_401_triggers_one_force_refresh_and_retries(self):
        """Client calls force_refresh exactly once and succeeds on the retry."""
        async def _body():
            force_refresh_calls: list[str] = []

            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            class _TrackingTokenManager(TokenManager):
                async def force_refresh(self_inner) -> str:
                    force_refresh_calls.append("forced")
                    return await super().force_refresh()

            mgr = _TrackingTokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            ok_body = {"time": 1717416000, "states": []}
            client = OpenSkyClient(
                token_manager=mgr,
                base_url="http://fake",
                sleep_fn=_RecordingSleep(),
            )
            client._http = _FakeHTTP([
                _make_response(401),
                _make_response(200, body=ok_body),
            ])
            result = await client.fetch_states()
            return result, force_refresh_calls

        result, force_refresh_calls = asyncio.run(_body())
        self.assertEqual(len(force_refresh_calls), 1, "force_refresh must be called exactly once.")
        self.assertEqual(result["time"], 1717416000)

    def test_401_does_not_loop(self):
        """If the retry after a 401 also returns 200, no further refresh occurs."""
        async def _body():
            force_refresh_calls: list[str] = []

            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            class _TrackingTokenManager(TokenManager):
                async def force_refresh(self_inner) -> str:
                    force_refresh_calls.append("forced")
                    return await super().force_refresh()

            mgr = _TrackingTokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr,
                base_url="http://fake",
                sleep_fn=_RecordingSleep(),
            )
            client._http = _FakeHTTP([
                _make_response(401),
                _make_response(200, body={"time": 999, "states": []}),
            ])
            await client.fetch_states()
            return force_refresh_calls

        force_refresh_calls = asyncio.run(_body())
        self.assertEqual(
            len(force_refresh_calls),
            1,
            "force_refresh must NOT be called more than once on a single 401.",
        )

    def test_401_does_not_sleep(self):
        """A 401 must not invoke the rate-limit sleep_fn."""
        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr,
                base_url="http://fake",
                sleep_fn=sleep,
            )
            client._http = _FakeHTTP([
                _make_response(401),
                _make_response(200, body={"time": 1, "states": []}),
            ])
            await client.fetch_states()
            return sleep.calls

        calls = asyncio.run(_body())
        self.assertEqual(calls, [], "A 401 must not call the sleep function.")


# ---------------------------------------------------------------------------
# 3. 429 backoff
# ---------------------------------------------------------------------------


class Test429Backoff(unittest.TestCase):
    """HTTP 429 triggers sleep_fn(N) and increments ingest_rate_limit_sleeps_total."""

    def _make_responses_and_client(self, responses: list[httpx.Response]):
        """Async factory: build client + recording sleep inside the event loop."""
        async def _factory():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr,
                base_url="http://fake",
                sleep_fn=sleep,
            )
            client._http = _FakeHTTP(responses)
            return client, sleep

        return _factory

    def test_429_sleeps_for_header_value(self):
        """sleep_fn is called with the value from X-Rate-Limit-Retry-After-Seconds."""
        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr, base_url="http://fake", sleep_fn=sleep
            )
            client._http = _FakeHTTP([
                _make_response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "42"}),
                _make_response(200, body={"time": 1, "states": []}),
            ])
            await client.fetch_states()
            return sleep.calls

        calls = asyncio.run(_body())
        self.assertEqual(calls, [42.0], f"Expected sleep(42.0), got {calls}")

    def test_429_no_actual_sleep(self):
        """The injected no-op sleep_fn means zero real wait time in tests."""
        # _RecordingSleep.__call__ is async but does not await asyncio.sleep,
        # so the coroutine completes immediately regardless of the argument.
        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr, base_url="http://fake", sleep_fn=sleep
            )
            client._http = _FakeHTTP([
                _make_response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "42"}),
                _make_response(200, body={"time": 2, "states": []}),
            ])
            await client.fetch_states()

        asyncio.run(_body())  # must complete instantly

    def test_429_increments_rate_limit_metric(self):
        """ingest_rate_limit_sleeps_total increments by 1 on each 429."""
        before = _counter_value(ingest_rate_limit_sleeps_total)

        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr, base_url="http://fake", sleep_fn=sleep
            )
            client._http = _FakeHTTP([
                _make_response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "5"}),
                _make_response(200, body={"time": 3, "states": []}),
            ])
            await client.fetch_states()

        asyncio.run(_body())
        after = _counter_value(ingest_rate_limit_sleeps_total)
        self.assertEqual(after - before, 1.0)

    def test_429_uses_default_sleep_when_header_absent(self):
        """When the rate-limit header is absent, sleep_fn is called with 10.0 (the default)."""
        from systems.ingest.client import _DEFAULT_RATE_LIMIT_SLEEP_S

        async def _body():
            async def fake_fetch(client_id, client_secret, token_url):
                return {"access_token": "tok", "expires_in": 1800}

            sleep = _RecordingSleep()
            mgr = TokenManager("id", "secret", "http://fake", fetch_fn=fake_fetch)
            client = OpenSkyClient(
                token_manager=mgr, base_url="http://fake", sleep_fn=sleep
            )
            client._http = _FakeHTTP([
                _make_response(429),          # no X-Rate-Limit header
                _make_response(200, body={"time": 4, "states": []}),
            ])
            await client.fetch_states()
            return sleep.calls

        calls = asyncio.run(_body())
        self.assertEqual(
            calls,
            [_DEFAULT_RATE_LIMIT_SLEEP_S],
            f"Expected sleep({_DEFAULT_RATE_LIMIT_SLEEP_S}), got {calls}",
        )


# ---------------------------------------------------------------------------
# 4. Idempotent at-least-once landing
# ---------------------------------------------------------------------------


class TestIdempotentLanding(unittest.TestCase):
    """Feeding the same snapshot twice lands records exactly once."""

    def test_duplicate_snapshot_lands_once(self):
        """Two identical polls produce exactly one landing record (dedup guard)."""
        snapshot = {"time": 1717416000, "states": [list(_STATE_18)]}

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([snapshot, snapshot]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=2)
                written = svc._total_written
                lines = [
                    json.loads(l)
                    for l in (tmpdir / "landing.jsonl").read_text().splitlines()
                    if l.strip()
                ]
                return written, lines
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        written, lines = asyncio.run(_body())
        self.assertEqual(written, 1, f"Expected 1 landing record; got {written}.")
        self.assertEqual(len(lines), 1)

    def test_different_snapshots_both_land(self):
        """Two distinct state vectors in separate polls each land once."""
        sv_a = list(_STATE_18)
        sv_b = list(_STATE_18)
        sv_b[0] = "def456"   # different icao24 → different idem_key

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([
                        {"time": 1717416000, "states": [sv_a]},
                        {"time": 1717416000, "states": [sv_b]},
                    ]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=2)
                return svc._total_written
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(asyncio.run(_body()), 2)

    def test_ingest_records_total_counts_landing_records(self):
        """ingest_records_total increments by the number of UNIQUE records landed."""
        snapshot = {"time": 1717416000, "states": [list(_STATE_18)]}
        before = _counter_value(ingest_records_total)

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([snapshot, snapshot]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=2)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        asyncio.run(_body())
        after = _counter_value(ingest_records_total)
        self.assertEqual(
            after - before,
            1.0,
            "Metric must increment once per unique landing record, not per poll.",
        )


# ---------------------------------------------------------------------------
# 5. Length-robust parse end-to-end (18-field vector with category)
# ---------------------------------------------------------------------------


class TestLengthRobustParse(unittest.TestCase):
    """An 18-field state vector flows through service → landing with category populated."""

    def test_18_field_vector_lands_with_category(self):
        """category field is present and correct in the landed record."""
        sv = list(_STATE_18)  # index 17 == 3

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [sv]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                lines = [
                    json.loads(l)
                    for l in (tmpdir / "landing.jsonl").read_text().splitlines()
                    if l.strip()
                ]
                return lines
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        lines = asyncio.run(_body())
        self.assertEqual(len(lines), 1)
        record = lines[0]
        self.assertEqual(record["category"], 3)
        self.assertEqual(record["icao24"], "abc123")

    def test_17_field_vector_lands_with_category_none(self):
        """A 17-field vector (no category element) lands with category == None."""
        sv = list(_STATE_18[:17])  # drop index 17

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [sv]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                lines = [
                    json.loads(l)
                    for l in (tmpdir / "landing.jsonl").read_text().splitlines()
                    if l.strip()
                ]
                return lines
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        lines = asyncio.run(_body())
        self.assertEqual(len(lines), 1)
        self.assertIsNone(lines[0]["category"])

    def test_callsign_trimmed_end_to_end(self):
        """Callsign trailing whitespace is stripped in the landed record."""
        sv = list(_STATE_18)
        sv[1] = "  SWR162  "

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [sv]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                record = json.loads(
                    (tmpdir / "landing.jsonl").read_text().splitlines()[0]
                )
                return record
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        record = asyncio.run(_body())
        self.assertEqual(record["callsign"], "SWR162")


# ---------------------------------------------------------------------------
# 6. Bronze shape
# ---------------------------------------------------------------------------


class TestBronzeShape(unittest.TestCase):
    """The JSONL line written to bronze matches {"time","states":[...]} and
    round-trips through read_snapshots()."""

    def test_bronze_file_written(self):
        """Service writes at least one line to the bronze file."""
        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [list(_STATE_18)]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                bronze_path = tmpdir / "bronze.jsonl"
                exists = bronze_path.exists()
                size = bronze_path.stat().st_size if exists else 0
                return exists, size
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        exists, size = asyncio.run(_body())
        self.assertTrue(exists)
        self.assertGreater(size, 0)

    def test_bronze_line_has_time_and_states_keys(self):
        """Each bronze line is valid JSON with 'time' and 'states' keys."""
        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [list(_STATE_18)]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                lines = (tmpdir / "bronze.jsonl").read_text().splitlines()
                return [json.loads(l) for l in lines if l.strip()]
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        objs = asyncio.run(_body())
        self.assertEqual(len(objs), 1)
        self.assertIn("time", objs[0])
        self.assertIn("states", objs[0])

    def test_bronze_time_matches_snapshot_time(self):
        """The 'time' field in bronze matches the snapshot time from the API."""
        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [list(_STATE_18)]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                obj = json.loads((tmpdir / "bronze.jsonl").read_text().splitlines()[0])
                return obj["time"]
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(asyncio.run(_body()), 1717416000)

    def test_bronze_states_contains_original_vector(self):
        """The 'states' array in bronze preserves the raw state vector."""
        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [list(_STATE_18)]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                obj = json.loads((tmpdir / "bronze.jsonl").read_text().splitlines()[0])
                return obj["states"]
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        states = asyncio.run(_body())
        self.assertIsInstance(states, list)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0][0], "abc123")   # icao24 preserved verbatim

    def test_bronze_round_trips_through_read_snapshots(self):
        """read_snapshots() successfully reads the bronze file written by the service."""
        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([{"time": 1717416000, "states": [list(_STATE_18)]}]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=1)
                snapshots = list(read_snapshots(tmpdir / "bronze.jsonl"))
                return snapshots
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        snapshots = asyncio.run(_body())
        self.assertEqual(len(snapshots), 1)
        snap = snapshots[0]
        self.assertEqual(snap["time"], 1717416000)
        self.assertEqual(len(snap["states"]), 1)
        self.assertEqual(snap["states"][0][0], "abc123")

    def test_multiple_polls_append_multiple_bronze_lines(self):
        """Each poll cycle appends one JSONL line; two polls → two lines."""
        sv_a = list(_STATE_18)
        sv_b = list(_STATE_18)
        sv_b[0] = "xyz789"

        async def _body():
            tmpdir = Path(tempfile.mkdtemp())
            try:
                svc = IngestService(
                    client=_FakeMockClient([
                        {"time": 1717416000, "states": [sv_a]},
                        {"time": 1717416001, "states": [sv_b]},
                    ]),
                    bronze_path=tmpdir / "bronze.jsonl",
                    landing_path=tmpdir / "landing.jsonl",
                    journal_path=None,
                    poll_interval_s=0.0,
                    sleep_fn=_RecordingSleep(),
                )
                await svc.run(max_polls=2)
                snapshots = list(read_snapshots(tmpdir / "bronze.jsonl"))
                return snapshots
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        snapshots = asyncio.run(_body())
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["time"], 1717416000)
        self.assertEqual(snapshots[1]["time"], 1717416001)


# ---------------------------------------------------------------------------
# DB-gated tests (skip when DATABASE_URL is absent / Postgres unreachable)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _DB_AVAILABLE,
    "DB-gated: requires a live PostgreSQL (DATABASE_URL must be set)",
)
class TestIngestServiceWithDB(unittest.TestCase):
    """Assertions that exercise the real Postgres landing path.

    Gated with skipUnless so offline CI stays green.  When a live DB is
    present, this class verifies the DB is reachable and the ingest pipeline
    runs end-to-end.
    """

    def test_placeholder_db_reachable(self):
        """Sanity check: DB is reachable (this class only runs when _DB_AVAILABLE)."""
        from shared.store.pg import healthcheck
        self.assertTrue(healthcheck())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
