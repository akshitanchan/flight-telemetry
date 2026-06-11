"""dashboard/health.py
---------------------
Platform-health back-end for the flight-telemetry dashboard.

Two public capabilities:
1. ``parse_metrics_text(text)`` — parse a Prometheus text-exposition blob into
   a flat ``{metric_name: float}`` dict.  Uses
   ``prometheus_client.parser.text_string_to_metric_families`` so it handles
   all standard exposition types (counter, gauge, histogram, summary).

2. ``scrape_service(url, timeout)`` — HTTP-fetch a ``/metrics`` endpoint and
   return a ``ServiceStatus`` describing availability + parsed metrics.

3. ``query_prometheus(metric_name, prometheus_url, timeout)`` — query the
   Prometheus HTTP API (``/api/v1/query``) for an instant-vector value.

4. ``fetch_all_services()`` — scrape all configured service endpoints (or
   query Prometheus) and return a list of ``ServiceStatus`` objects.

Configuration (env vars, all have sensible localhost defaults):
    PROMETHEUS_URL          Prometheus base URL (default http://localhost:9090)
    ML_SERVE_METRICS_URL    ml-serve /metrics URL (default http://localhost:8000/metrics)
    INGEST_METRICS_URL      ingestion /metrics URL (default http://localhost:8001/metrics)

Availability-gating contract
-----------------------------
Every network call is wrapped so that a refused connection, a DNS failure, a
timeout, or any unexpected exception returns a ``ServiceStatus(available=False)``
object.  Nothing in this module raises an exception to the caller.  This means
the dashboard can call this code unconditionally and just check
``status.available`` — it will never crash the page.

Import contract
---------------
This module MUST NOT import streamlit (it must be import-testable without a
running Streamlit server).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default service endpoints (match docker-compose host-port mappings)
# ---------------------------------------------------------------------------
_DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
_DEFAULT_ML_SERVE_METRICS_URL = "http://localhost:8000/metrics"
_DEFAULT_INGEST_METRICS_URL = "http://localhost:8001/metrics"

# Key metrics from C6 (shared/obs/telemetry.py) naming convention:
# <service>_<noun>_<unit>
_KEY_METRICS = [
    "ingest_records_total",
    "ingest_poll_latency_seconds",
    "ingest_rate_limit_sleeps_total",
    "ingest_token_refresh_total",
    "ml_predict_latency_seconds",
    "http_requests_total",
]

# ---------------------------------------------------------------------------
# Public status data class
# ---------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    """Result object returned by every fetch function.

    Attributes:
        service:    Human-readable service name, e.g. ``"ml-serve"``.
        url:        The endpoint that was (or would have been) contacted.
        available:  ``True`` if the endpoint responded with HTTP 2xx.
        error:      Short error description when ``available=False``; empty
                    string when everything is fine.
        metrics:    Flat ``{metric_name: float}`` dict of parsed metrics
                    (empty when ``available=False``).
        raw_labels: Optional extra metadata (e.g. Prometheus job labels).
    """

    service: str
    url: str
    available: bool = False
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    raw_labels: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Text-exposition parser
# ---------------------------------------------------------------------------


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse a Prometheus text-exposition blob into ``{metric_name: float}``.

    For histograms and summaries the ``_sum``, ``_count``, and ``_bucket``
    series are stored with their full name (e.g. ``ingest_poll_latency_seconds_sum``).
    Gauge and counter samples are stored under their base name.

    Label-sets are collapsed: when a metric has multiple label combinations the
    **last** sample value wins (sufficient for monitoring cardinality typical in
    this platform).

    Args:
        text: Raw Prometheus text exposition string.

    Returns:
        A flat dict mapping metric name (string) to its float value.
    """
    result: dict[str, float] = {}
    try:
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                result[sample.name] = sample.value
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse Prometheus text exposition: %s", exc)
    return result


# ---------------------------------------------------------------------------
# 2. Scrape a service /metrics endpoint
# ---------------------------------------------------------------------------


def scrape_service(
    service: str,
    url: str,
    timeout: float = 3.0,
) -> ServiceStatus:
    """Fetch *url* (a Prometheus ``/metrics`` endpoint) and return a ``ServiceStatus``.

    Connection errors, timeouts, and non-2xx HTTP responses all result in
    ``ServiceStatus(available=False)`` — never an exception.

    Args:
        service: Human-readable service name used in the status object.
        url:     Full URL of the ``/metrics`` endpoint.
        timeout: Request timeout in seconds (default 3.0).

    Returns:
        A ``ServiceStatus`` with ``available=True`` and parsed ``metrics`` on
        success, or ``available=False`` with a descriptive ``error`` string on
        any failure.
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=False)
        if response.status_code >= 400:
            return ServiceStatus(
                service=service,
                url=url,
                available=False,
                error=f"HTTP {response.status_code}",
            )
        metrics = parse_metrics_text(response.text)
        return ServiceStatus(
            service=service,
            url=url,
            available=True,
            metrics=metrics,
        )
    except httpx.ConnectError as exc:
        logger.debug("scrape_service(%s): connection refused — %s", service, exc)
        return ServiceStatus(
            service=service,
            url=url,
            available=False,
            error="connection refused",
        )
    except httpx.TimeoutException as exc:
        logger.debug("scrape_service(%s): timed out — %s", service, exc)
        return ServiceStatus(
            service=service,
            url=url,
            available=False,
            error="timed out",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape_service(%s): unexpected error — %s", service, exc)
        return ServiceStatus(
            service=service,
            url=url,
            available=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# 3. Query the Prometheus HTTP API
# ---------------------------------------------------------------------------


def query_prometheus(
    metric_name: str,
    prometheus_url: str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Query the Prometheus HTTP API for an instant-vector value.

    Calls ``<prometheus_url>/api/v1/query?query=<metric_name>`` and returns a
    dict with keys ``"available"`` (bool), ``"value"`` (float | None), and
    ``"error"`` (str).  Never raises.

    Args:
        metric_name:    PromQL expression / metric name to query.
        prometheus_url: Base URL of the Prometheus server (env ``PROMETHEUS_URL``
                        or ``http://localhost:9090`` by default).
        timeout:        Request timeout in seconds.

    Returns:
        ``{"available": bool, "value": float | None, "error": str}``
    """
    base = (
        prometheus_url
        or os.environ.get("PROMETHEUS_URL", "").strip()
        or _DEFAULT_PROMETHEUS_URL
    )
    api_url = f"{base.rstrip('/')}/api/v1/query"
    try:
        response = httpx.get(
            api_url,
            params={"query": metric_name},
            timeout=timeout,
            follow_redirects=False,
        )
        if response.status_code >= 400:
            return {
                "available": False,
                "value": None,
                "error": f"HTTP {response.status_code}",
            }
        payload = response.json()
        results = payload.get("data", {}).get("result", [])
        if results:
            raw_value = results[0].get("value", [None, None])[1]
            try:
                return {"available": True, "value": float(raw_value), "error": ""}
            except (TypeError, ValueError):
                return {"available": True, "value": None, "error": "non-numeric value"}
        return {"available": True, "value": None, "error": "no data"}
    except httpx.ConnectError as exc:
        logger.debug("query_prometheus(%s): connection refused — %s", metric_name, exc)
        return {"available": False, "value": None, "error": "connection refused"}
    except httpx.TimeoutException as exc:
        logger.debug("query_prometheus(%s): timed out — %s", metric_name, exc)
        return {"available": False, "value": None, "error": "timed out"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("query_prometheus(%s): unexpected error — %s", metric_name, exc)
        return {"available": False, "value": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# 4. Prometheus server liveness check
# ---------------------------------------------------------------------------


def check_prometheus(
    prometheus_url: str | None = None,
    timeout: float = 3.0,
) -> ServiceStatus:
    """Check whether the Prometheus server itself is reachable.

    Hits ``/-/ready`` and returns a ``ServiceStatus``.  Never raises.

    Args:
        prometheus_url: Base URL of the Prometheus server.
        timeout:        Request timeout in seconds.

    Returns:
        ``ServiceStatus`` with ``available=True`` when Prometheus is up.
    """
    base = (
        prometheus_url
        or os.environ.get("PROMETHEUS_URL", "").strip()
        or _DEFAULT_PROMETHEUS_URL
    )
    url = f"{base.rstrip('/')}/-/ready"
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=False)
        if response.status_code < 400:
            return ServiceStatus(
                service="prometheus",
                url=base,
                available=True,
            )
        return ServiceStatus(
            service="prometheus",
            url=base,
            available=False,
            error=f"HTTP {response.status_code}",
        )
    except httpx.ConnectError as exc:
        logger.debug("check_prometheus: connection refused — %s", exc)
        return ServiceStatus(
            service="prometheus",
            url=base,
            available=False,
            error="connection refused",
        )
    except httpx.TimeoutException as exc:
        logger.debug("check_prometheus: timed out — %s", exc)
        return ServiceStatus(
            service="prometheus",
            url=base,
            available=False,
            error="timed out",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("check_prometheus: unexpected error — %s", exc)
        return ServiceStatus(
            service="prometheus",
            url=base,
            available=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# 5. Fetch all configured services
# ---------------------------------------------------------------------------


def fetch_all_services(timeout: float = 3.0) -> list[ServiceStatus]:
    """Scrape all platform service metrics endpoints.

    Reads endpoint URLs from environment variables with localhost defaults
    that match docker-compose host-port mappings:

    - ml-serve  → ``ML_SERVE_METRICS_URL``  (default :8000/metrics)
    - ingestion → ``INGEST_METRICS_URL``    (default :8001/metrics)
    - prometheus → ``PROMETHEUS_URL``       (default :9090)

    Returns a list with one ``ServiceStatus`` per service (three total).
    Never raises.

    Args:
        timeout: Per-request timeout in seconds.

    Returns:
        List of ``ServiceStatus`` objects, one per service.
    """
    ml_url = (
        os.environ.get("ML_SERVE_METRICS_URL", "").strip()
        or _DEFAULT_ML_SERVE_METRICS_URL
    )
    ingest_url = (
        os.environ.get("INGEST_METRICS_URL", "").strip()
        or _DEFAULT_INGEST_METRICS_URL
    )

    statuses = [
        scrape_service("ml-serve", ml_url, timeout=timeout),
        scrape_service("ingestion", ingest_url, timeout=timeout),
        check_prometheus(timeout=timeout),
    ]
    return statuses
