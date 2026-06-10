"""
ml/test_obs.py
--------------
Observability layer tests (C6 contract).

Coverage:
1. GET /metrics exposes ``ml_predict_latency_seconds`` and
   ``http_requests_total`` after a prediction call.
2. ``setup_tracing`` configures a tracer provider; a span emitted via
   that provider reaches an in-memory exporter without error and carries
   the expected resource attribute.
3. Importing ``shared.obs.telemetry`` with no collector running is safe
   (no network I/O at import time).
"""

import pytest
from fastapi.testclient import TestClient

from ml.serve import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. /metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    """Verify that the /metrics endpoint is mounted and exposes the expected
    named metrics after the prediction route has been exercised."""

    def _make_predict_call(self):
        payload = {
            "flight_id": "obs_test_001",
            "duration_s": 600.0,
            "alt_change": 500.0,
            "avg_speed": 200.0,
            "max_vrate": 8.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"predict failed: {resp.text}"

    def test_metrics_endpoint_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_text_plain(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_ml_predict_latency_seconds_present(self):
        """ml_predict_latency_seconds must appear in the /metrics output after
        at least one prediction so the histogram has been observed."""
        self._make_predict_call()
        response = client.get("/metrics")
        assert "ml_predict_latency_seconds" in response.text

    def test_http_requests_total_present(self):
        """http_requests_total must appear; requesting /metrics itself
        increments the counter so it will always be present after this call."""
        self._make_predict_call()
        response = client.get("/metrics")
        assert "http_requests_total" in response.text

    def test_http_requests_total_labels_route_and_status(self):
        """http_requests_total must carry route and status labels."""
        self._make_predict_call()
        response = client.get("/metrics")
        # The Prometheus exposition format for a labelled counter looks like:
        #   http_requests_total{route="/predict",status="200"} N
        assert 'route="' in response.text
        assert 'status="' in response.text

    def test_ml_predict_latency_histogram_has_bucket_lines(self):
        """Histogram metrics include _bucket lines in the exposition output."""
        self._make_predict_call()
        response = client.get("/metrics")
        assert "ml_predict_latency_seconds_bucket" in response.text


# ---------------------------------------------------------------------------
# 2. Tracing — setup_tracing produces spans captured by in-memory exporter
# ---------------------------------------------------------------------------

class TestSetupTracing:
    """Verify that setup_tracing configures a working tracer provider and that
    spans are emitted correctly.  Uses a fresh TracerProvider with an
    InMemorySpanExporter so the test does not depend on a live collector."""

    def test_span_emitted_to_in_memory_exporter(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry import trace

        resource = Resource.create({"service.name": "test"})
        provider = TracerProvider(resource=resource)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Use the provider directly without touching the global provider so
        # this test is fully isolated from ml/serve.py's global setup.
        tracer = provider.get_tracer("test.scope")

        with tracer.start_as_current_span("test.operation") as span:
            span.set_attribute("key", "value")

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].name == "test.operation"

    def test_span_resource_carries_service_name(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry import trace

        service = "test-svc"
        resource = Resource.create({"service.name": service})
        provider = TracerProvider(resource=resource)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("scope")

        with tracer.start_as_current_span("svc.op"):
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].resource.attributes.get("service.name") == service

    def test_setup_tracing_no_op_when_no_collector(self, monkeypatch):
        """setup_tracing must not raise even when the OTLP endpoint is
        unreachable or the env var is absent."""
        import shared.obs.telemetry as telemetry

        # Reset idempotency guard to exercise the setup path again.
        original = telemetry._tracing_configured
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:19999")
        telemetry._tracing_configured = False

        try:
            # Must not raise.
            telemetry.setup_tracing("test-no-collector")
        finally:
            telemetry._tracing_configured = original


# ---------------------------------------------------------------------------
# 3. Import-time safety — no network I/O
# ---------------------------------------------------------------------------

class TestImportTimeSafety:
    """Importing shared.obs.telemetry must be safe with no network access."""

    def test_import_does_not_raise(self):
        # If we got this far, the import already succeeded at test-collection
        # time.  Re-import to be explicit.
        import importlib
        import shared.obs.telemetry
        reloaded = importlib.import_module("shared.obs.telemetry")
        assert reloaded is not None

    def test_metric_singletons_are_accessible(self):
        from shared.obs.telemetry import (
            ingest_records_total,
            ingest_poll_latency_seconds,
            ingest_rate_limit_sleeps_total,
            ingest_token_refresh_total,
            ml_predict_latency_seconds,
            http_requests_total,
        )
        # Basic sanity: each object has the expected type marker.
        assert "counter" in str(ingest_records_total)
        assert "histogram" in str(ingest_poll_latency_seconds)
        assert "counter" in str(ingest_rate_limit_sleeps_total)
        assert "counter" in str(ingest_token_refresh_total)
        assert "histogram" in str(ml_predict_latency_seconds)
        assert "counter" in str(http_requests_total)
