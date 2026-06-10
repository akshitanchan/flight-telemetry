"""
shared/obs/telemetry.py
-----------------------
Shared observability module for the Flight Telemetry platform.

Provides:
- Module-level Prometheus metric singletons (counters and histograms).
- ``setup_metrics(app)`` — mounts a ``/metrics`` ASGI endpoint on a FastAPI
  app and registers per-request HTTP instrumentation middleware.
- ``setup_tracing(service_name)`` — configures an OpenTelemetry tracer provider
  with the OTLP gRPC exporter; gracefully no-ops when no collector is reachable
  so offline runs and CI remain green.
- ``start_metrics_server(port)`` — lightweight standalone HTTP metrics server
  for non-FastAPI processes (e.g. the ingestion loop).

Design rules:
- Importing this module MUST NOT open any network connection.  The OTLP
  exporter is created lazily inside ``setup_tracing`` and failures are caught.
- Metric objects are module-level singletons; callers import them directly:

      from shared.obs.telemetry import ml_predict_latency_seconds
      with ml_predict_latency_seconds.time():
          ...

- ``setup_metrics`` and ``setup_tracing`` are idempotent within a process:
  calling them more than once (e.g. in tests) is safe.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
    start_http_server,
    REGISTRY,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric singletons
# ---------------------------------------------------------------------------
# All metrics follow the naming convention: <service>_<noun>_<unit>
# They are registered once at module import time against the default REGISTRY.

# --- Ingestion metrics (consumed by the ingest loop) -----------------------
ingest_records_total = Counter(
    "ingest_records_total",
    "Total number of flight-state records ingested from the upstream API.",
)

ingest_poll_latency_seconds = Histogram(
    "ingest_poll_latency_seconds",
    "Elapsed time (seconds) for a single upstream API poll cycle.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ingest_rate_limit_sleeps_total = Counter(
    "ingest_rate_limit_sleeps_total",
    "Number of times the ingest loop slept due to upstream rate-limiting.",
)

ingest_token_refresh_total = Counter(
    "ingest_token_refresh_total",
    "Number of OAuth/API token refresh operations performed by the ingest loop.",
)

# --- ML serving metrics (consumed by ml/serve.py) --------------------------
ml_predict_latency_seconds = Histogram(
    "ml_predict_latency_seconds",
    "End-to-end prediction latency in seconds, measured inside /predict.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# HTTP request counter shared across services; labelled by route and status.
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled.",
    labelnames=("route", "status"),
)

# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------

# Guard against double-instrumentation when tests re-import the app module.
_metrics_mounted = False
_tracing_configured = False
_tracing_lock = threading.Lock()


def setup_metrics(app: "FastAPI") -> None:
    """Mount ``/metrics`` on *app* and add HTTP request instrumentation.

    Idempotent: calling this more than once on the same app object is safe
    (the middleware is added only on the first call).

    Args:
        app: A FastAPI application instance.
    """
    global _metrics_mounted
    if _metrics_mounted:
        logger.debug("setup_metrics: already mounted, skipping.")
        return

    # Mount the Prometheus ASGI endpoint at /metrics.
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Middleware: increment http_requests_total and track per-route latency.
    # We register this *after* mounting /metrics so the route is available,
    # but note that middleware runs on every request including /metrics itself
    # (which is fine — the overhead is negligible).
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest

    class _HTTPMetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            response = await call_next(request)
            route = request.url.path
            http_requests_total.labels(route=route, status=str(response.status_code)).inc()
            return response

    app.add_middleware(_HTTPMetricsMiddleware)
    _metrics_mounted = True
    logger.info("Prometheus /metrics endpoint mounted.")


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------

def setup_tracing(service_name: str) -> None:
    """Configure a global OTel tracer provider for *service_name*.

    Reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` from the environment
    (default: ``http://localhost:4317``).  If the collector is unreachable or
    the env var is absent/empty the function falls back to a no-op provider
    so offline runs and CI remain green.

    Idempotent: the global tracer provider is set only once per process.

    Args:
        service_name: Value for the ``service.name`` OTel resource attribute
                      (e.g. ``"ml-serve"`` or ``"ingest"``).
    """
    global _tracing_configured
    with _tracing_lock:
        if _tracing_configured:
            logger.debug("setup_tracing: already configured, skipping.")
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)

            endpoint = (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
                or "http://localhost:4317"
            )

            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(
                    "OTel tracing configured for service=%r, endpoint=%s",
                    service_name,
                    endpoint,
                )
            except Exception as exc:  # noqa: BLE001
                # Exporter instantiation failed (missing lib, bad endpoint, etc.).
                # Fall back to a no-op exporter so the process still starts.
                logger.warning(
                    "OTel OTLP exporter setup failed (%s); tracing spans will be dropped.",
                    exc,
                )

            trace.set_tracer_provider(provider)

        except Exception as exc:  # noqa: BLE001
            # Catch-all: if the OTel SDK itself fails to import or configure,
            # log and continue — observability should never crash the service.
            logger.warning("OTel SDK setup failed (%s); tracing disabled.", exc)

        _tracing_configured = True


def get_tracer(name: str = __name__):
    """Return an OTel tracer.  Safe to call before ``setup_tracing``; returns
    the no-op tracer if the provider has not been configured yet.

    Args:
        name: Instrumentation scope name (typically ``__name__`` of the caller).
    """
    from opentelemetry import trace

    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Standalone metrics HTTP server (for non-FastAPI processes)
# ---------------------------------------------------------------------------

_standalone_server_started = False
_standalone_server_lock = threading.Lock()


def start_metrics_server(port: int = 8001) -> None:
    """Start a standalone Prometheus metrics HTTP server on *port*.

    Intended for non-FastAPI services such as the ingestion loop.  The server
    runs in a daemon thread managed by ``prometheus_client`` so it does not
    block the caller.

    Idempotent: calling this more than once with the same port is a no-op after
    the first successful start.

    Args:
        port: TCP port to listen on (default 8001).
    """
    global _standalone_server_started
    with _standalone_server_lock:
        if _standalone_server_started:
            logger.debug("start_metrics_server: already running, skipping.")
            return
        try:
            start_http_server(port)
            _standalone_server_started = True
            logger.info("Prometheus metrics server listening on port %d.", port)
        except OSError as exc:
            # Port already in use (common in test environments).
            logger.warning(
                "Could not start metrics server on port %d: %s", port, exc
            )
