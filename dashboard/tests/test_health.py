#!/usr/bin/env python3
"""Tests for dashboard.health — platform-health back-end.

Conventions:
- unittest.TestCase subclasses, runnable via ``python -m pytest dashboard/tests/test_health.py -v``
  or ``python -m unittest discover -s dashboard/tests -v``.
- NO streamlit import anywhere in this file.
- Fully offline: all tests pass without any running services or network access.

Test groups:
1. TestParseMetricsText — feed a Prometheus text-exposition fixture and assert
   that the parsed dict contains expected metric names and values.
2. TestScrapeServiceOffline — point scrape_service / fetch_all_services at
   unreachable URLs and assert the "not available" contract (no exception, correct
   ServiceStatus shape).
3. TestQueryPrometheusOffline — same offline contract for query_prometheus.
4. TestNoStreamlitImport — guard that dashboard.health never imports streamlit.
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Dashboard.health must be importable with NO services running and NO streamlit.
from dashboard.health import (  # noqa: E402
    ServiceStatus,
    fetch_all_services,
    parse_metrics_text,
    query_prometheus,
    scrape_service,
    check_prometheus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A realistic Prometheus text-exposition blob covering counters, histograms,
# and a gauge.  Uses the C6 <service>_<noun>_<unit> naming convention.
_METRICS_FIXTURE = """\
# HELP ingest_records_total Total number of flight-state records ingested.
# TYPE ingest_records_total counter
ingest_records_total 42.0
# HELP ingest_poll_latency_seconds Elapsed time for a single poll cycle.
# TYPE ingest_poll_latency_seconds histogram
ingest_poll_latency_seconds_bucket{le="0.05"} 3.0
ingest_poll_latency_seconds_bucket{le="0.1"} 5.0
ingest_poll_latency_seconds_bucket{le="+Inf"} 7.0
ingest_poll_latency_seconds_sum 1.25
ingest_poll_latency_seconds_count 7.0
# HELP ingest_rate_limit_sleeps_total Number of rate-limit sleeps.
# TYPE ingest_rate_limit_sleeps_total counter
ingest_rate_limit_sleeps_total 3.0
# HELP ml_predict_latency_seconds End-to-end prediction latency.
# TYPE ml_predict_latency_seconds histogram
ml_predict_latency_seconds_bucket{le="0.001"} 0.0
ml_predict_latency_seconds_bucket{le="0.01"} 12.0
ml_predict_latency_seconds_bucket{le="+Inf"} 15.0
ml_predict_latency_seconds_sum 0.135
ml_predict_latency_seconds_count 15.0
# HELP http_requests_total Total HTTP requests handled.
# TYPE http_requests_total counter
http_requests_total{route="/predict",status="200"} 100.0
http_requests_total{route="/health",status="200"} 200.0
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 52428800.0
"""

# An unreachable URL — using localhost on a port that is never bound in CI.
_UNREACHABLE_URL = "http://127.0.0.1:19999/metrics"
_UNREACHABLE_PROM = "http://127.0.0.1:19998"


# ===========================================================================
# 1.  Text-exposition parser tests
# ===========================================================================


class TestParseMetricsText(unittest.TestCase):
    """parse_metrics_text must return expected metric names and values."""

    @classmethod
    def setUpClass(cls):
        cls.parsed = parse_metrics_text(_METRICS_FIXTURE)

    # ---- return type ----

    def test_returns_dict(self):
        self.assertIsInstance(self.parsed, dict)

    # ---- counter metrics ----

    def test_ingest_records_total(self):
        self.assertIn("ingest_records_total", self.parsed)
        self.assertEqual(self.parsed["ingest_records_total"], 42.0)

    def test_ingest_rate_limit_sleeps_total(self):
        self.assertIn("ingest_rate_limit_sleeps_total", self.parsed)
        self.assertEqual(self.parsed["ingest_rate_limit_sleeps_total"], 3.0)

    # ---- histogram metrics (sum + count + bucket) ----

    def test_ingest_poll_latency_sum(self):
        self.assertIn("ingest_poll_latency_seconds_sum", self.parsed)
        self.assertAlmostEqual(self.parsed["ingest_poll_latency_seconds_sum"], 1.25)

    def test_ingest_poll_latency_count(self):
        self.assertIn("ingest_poll_latency_seconds_count", self.parsed)
        self.assertEqual(self.parsed["ingest_poll_latency_seconds_count"], 7.0)

    def test_ingest_poll_latency_bucket_inf(self):
        self.assertIn("ingest_poll_latency_seconds_bucket", self.parsed)

    def test_ml_predict_latency_sum(self):
        self.assertIn("ml_predict_latency_seconds_sum", self.parsed)
        self.assertAlmostEqual(self.parsed["ml_predict_latency_seconds_sum"], 0.135)

    def test_ml_predict_latency_count(self):
        self.assertIn("ml_predict_latency_seconds_count", self.parsed)
        self.assertEqual(self.parsed["ml_predict_latency_seconds_count"], 15.0)

    # ---- labelled counter (last label set wins) ----

    def test_http_requests_total_present(self):
        self.assertIn("http_requests_total", self.parsed)
        # Value is the last sample processed (200.0 for /health)
        self.assertGreater(self.parsed["http_requests_total"], 0)

    # ---- gauge ----

    def test_process_resident_memory_bytes(self):
        self.assertIn("process_resident_memory_bytes", self.parsed)
        self.assertEqual(self.parsed["process_resident_memory_bytes"], 52428800.0)

    # ---- empty text ----

    def test_empty_string_returns_empty_dict(self):
        result = parse_metrics_text("")
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    # ---- malformed text does not raise ----

    def test_malformed_text_does_not_raise(self):
        result = parse_metrics_text("not valid prometheus text !!!\n###\n")
        self.assertIsInstance(result, dict)


# ===========================================================================
# 2.  scrape_service offline / unreachable-endpoint tests
# ===========================================================================


class TestScrapeServiceOffline(unittest.TestCase):
    """scrape_service must return ServiceStatus(available=False) for unreachable URLs,
    NEVER raise an exception.
    """

    def test_returns_service_status(self):
        result = scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        self.assertIsInstance(result, ServiceStatus)

    def test_available_is_false(self):
        result = scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        self.assertFalse(result.available)

    def test_error_message_is_non_empty(self):
        result = scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        self.assertIsInstance(result.error, str)
        self.assertGreater(len(result.error), 0)

    def test_metrics_dict_is_empty(self):
        result = scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        self.assertEqual(result.metrics, {})

    def test_service_name_preserved(self):
        result = scrape_service("my-service", _UNREACHABLE_URL, timeout=1.0)
        self.assertEqual(result.service, "my-service")

    def test_url_preserved(self):
        result = scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        self.assertEqual(result.url, _UNREACHABLE_URL)

    def test_does_not_raise(self):
        """The most critical contract: must not propagate any exception."""
        try:
            scrape_service("test-svc", _UNREACHABLE_URL, timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"scrape_service raised unexpectedly: {exc}")


# ===========================================================================
# 3.  fetch_all_services offline tests
# ===========================================================================


class TestFetchAllServicesOffline(unittest.TestCase):
    """fetch_all_services with unreachable defaults must return a list of
    ServiceStatus objects, all available=False, without raising.
    """

    @classmethod
    def setUpClass(cls):
        import os

        # Override all endpoints to unreachable ports so the test is fast and
        # does not depend on anything running.
        os.environ["ML_SERVE_METRICS_URL"] = "http://127.0.0.1:19997/metrics"
        os.environ["INGEST_METRICS_URL"] = "http://127.0.0.1:19996/metrics"
        os.environ["PROMETHEUS_URL"] = "http://127.0.0.1:19995"
        cls.statuses = fetch_all_services(timeout=1.0)

    @classmethod
    def tearDownClass(cls):
        import os

        for key in ("ML_SERVE_METRICS_URL", "INGEST_METRICS_URL", "PROMETHEUS_URL"):
            os.environ.pop(key, None)

    def test_returns_list(self):
        self.assertIsInstance(self.statuses, list)

    def test_three_statuses_returned(self):
        self.assertEqual(len(self.statuses), 3)

    def test_all_unavailable(self):
        for svc in self.statuses:
            self.assertFalse(svc.available, f"{svc.service} should be unavailable")

    def test_all_have_service_names(self):
        names = {s.service for s in self.statuses}
        self.assertIn("ml-serve", names)
        self.assertIn("ingestion", names)
        self.assertIn("prometheus", names)

    def test_no_exception_raised(self):
        try:
            fetch_all_services(timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"fetch_all_services raised unexpectedly: {exc}")


# ===========================================================================
# 4.  query_prometheus offline tests
# ===========================================================================


class TestQueryPrometheusOffline(unittest.TestCase):
    """query_prometheus must return a dict with available=False when Prometheus
    is unreachable, never raising.
    """

    def test_returns_dict(self):
        result = query_prometheus("ingest_records_total", _UNREACHABLE_PROM, timeout=1.0)
        self.assertIsInstance(result, dict)

    def test_available_is_false(self):
        result = query_prometheus("ingest_records_total", _UNREACHABLE_PROM, timeout=1.0)
        self.assertFalse(result["available"])

    def test_value_is_none(self):
        result = query_prometheus("ingest_records_total", _UNREACHABLE_PROM, timeout=1.0)
        self.assertIsNone(result["value"])

    def test_error_is_non_empty(self):
        result = query_prometheus("ingest_records_total", _UNREACHABLE_PROM, timeout=1.0)
        self.assertGreater(len(result["error"]), 0)

    def test_does_not_raise(self):
        try:
            query_prometheus("ingest_records_total", _UNREACHABLE_PROM, timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"query_prometheus raised unexpectedly: {exc}")


# ===========================================================================
# 5.  check_prometheus offline tests
# ===========================================================================


class TestCheckPrometheusOffline(unittest.TestCase):
    """check_prometheus must return ServiceStatus(available=False) when Prometheus
    is unreachable, never raising.
    """

    def test_returns_service_status(self):
        result = check_prometheus(_UNREACHABLE_PROM, timeout=1.0)
        self.assertIsInstance(result, ServiceStatus)

    def test_available_is_false(self):
        result = check_prometheus(_UNREACHABLE_PROM, timeout=1.0)
        self.assertFalse(result.available)

    def test_service_name_is_prometheus(self):
        result = check_prometheus(_UNREACHABLE_PROM, timeout=1.0)
        self.assertEqual(result.service, "prometheus")

    def test_does_not_raise(self):
        try:
            check_prometheus(_UNREACHABLE_PROM, timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"check_prometheus raised unexpectedly: {exc}")


# ===========================================================================
# 6.  No-streamlit import guard
# ===========================================================================


class TestNoStreamlitImport(unittest.TestCase):
    """Guard: dashboard.health must never import streamlit."""

    def test_streamlit_not_imported_by_health_module(self):
        import dashboard.health as health_module

        source_file = Path(health_module.__file__).read_text()
        import_pattern = re.compile(
            r"^\s*(import streamlit|from streamlit\b)", re.MULTILINE
        )
        matches = import_pattern.findall(source_file)
        self.assertEqual(
            matches,
            [],
            f"dashboard/health.py must not contain a streamlit import; found: {matches}",
        )


if __name__ == "__main__":
    unittest.main()
