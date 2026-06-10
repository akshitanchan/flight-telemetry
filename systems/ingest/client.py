"""
systems/ingest/client.py
------------------------
Async HTTP client for the OpenSky Network /states/all endpoint.

Design rules:
- Importing this module opens NO network connection.
- ``OpenSkyClient`` is instantiated without opening a socket; the underlying
  ``httpx.AsyncClient`` is created lazily on the first request (or via the
  async context manager).
- On HTTP 401: refresh the token exactly once and retry.
- On HTTP 429: read ``X-Rate-Limit-Retry-After-Seconds``, sleep, then
  retry.  The sleep function is injectable so tests never actually wait.
- Increments ``ingest_rate_limit_sleeps_total`` on every rate-limit sleep.
- Each poll is wrapped in an OTel span named ``ingestion.poll``.

Environment variables (consumed via defaults; caller may override):
    OPENSKY_BASE_URL   — API base URL
                         (default: https://opensky-network.org/api)
    OPENSKY_BBOX_LAMIN — bounding box south latitude  (default: 47.0)
    OPENSKY_BBOX_LAMAX — bounding box north latitude  (default: 55.0)
    OPENSKY_BBOX_LOMIN — bounding box west longitude  (default: 5.0)
    OPENSKY_BBOX_LOMAX — bounding box east longitude  (default: 15.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://opensky-network.org/api"

# Rate-limit response header name (exact, case-insensitive via httpx).
_RATE_LIMIT_HEADER = "X-Rate-Limit-Retry-After-Seconds"

# Default fallback sleep when the header is absent.
_DEFAULT_RATE_LIMIT_SLEEP_S: float = 10.0

# Request timeout for /states/all (seconds).
_REQUEST_TIMEOUT_S: float = 30.0


class OpenSkyClient:
    """Async client for OpenSky /states/all.

    Args:
        token_manager: A :class:`~systems.ingest.token.TokenManager` instance
                       used to obtain Bearer tokens.
        base_url: OpenSky API base URL.  Defaults to ``OPENSKY_BASE_URL`` env
                  var, then ``https://opensky-network.org/api``.
        bbox: ``(lamin, lamax, lomin, lomax)`` bounding box tuple.  Defaults
              to env vars ``OPENSKY_BBOX_*``.
        sleep_fn: Async callable ``(seconds: float) -> None``.  Defaults to
                  ``asyncio.sleep``.  Inject a no-op coroutine in tests to
                  skip real waits.
    """

    def __init__(
        self,
        token_manager,  # TokenManager; typed loosely to avoid circular import
        base_url: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._token_manager = token_manager
        self._base_url = (
            base_url
            or os.environ.get("OPENSKY_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._bbox = bbox or _bbox_from_env()
        self._sleep_fn: Callable[[float], Awaitable[None]] = (
            sleep_fn if sleep_fn is not None else asyncio.sleep
        )
        self._http: Any = None  # httpx.AsyncClient, created lazily

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the underlying httpx.AsyncClient.

        Called automatically by ``__aenter__``.  Idempotent.
        """
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S)
            logger.debug("OpenSkyClient: httpx.AsyncClient opened.")

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient.

        Called automatically by ``__aexit__``.  Safe to call even if
        ``start()`` was never called.
        """
        if self._http is not None:
            await self._http.aclose()
            self._http = None
            logger.debug("OpenSkyClient: httpx.AsyncClient closed.")

    async def __aenter__(self) -> "OpenSkyClient":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_states(self) -> dict:
        """Fetch the current state vectors from /states/all.

        Handles:
        - Auth: attaches a Bearer token from the token manager.
        - 401: refreshes the token and retries exactly once.
        - 429: sleeps for the duration in the rate-limit header, then retries.
        - Wraps the full operation in an OTel span ``ingestion.poll``.

        Returns:
            Parsed JSON dict with keys ``"time"`` and ``"states"``.
            If OpenSky returns an empty response (no aircraft in bbox),
            ``"states"`` will be an empty list.

        Raises:
            httpx.HTTPStatusError: On non-retried HTTP errors.
            httpx.RequestError: On connection/timeout errors.
            RuntimeError: If the client was not started.
        """
        from shared.obs.telemetry import (
            ingest_poll_latency_seconds,
            get_tracer,
        )

        if self._http is None:
            raise RuntimeError(
                "OpenSkyClient not started. Call start() or use async context manager."
            )

        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("ingestion.poll"):
            with ingest_poll_latency_seconds.time():
                return await self._fetch_with_retry()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_with_retry(self) -> dict:
        """Inner fetch with 401-retry and 429-backoff logic."""
        token = await self._token_manager.get_token()
        response = await self._send_request(token)

        if response.status_code == 401:
            logger.warning("HTTP 401 received; refreshing token and retrying once.")
            token = await self._token_manager.force_refresh()
            response = await self._send_request(token)

        if response.status_code == 429:
            await self._handle_rate_limit(response)
            # After sleeping, get a fresh token (it may have been refreshed
            # proactively during the long sleep).
            token = await self._token_manager.get_token()
            response = await self._send_request(token)

        response.raise_for_status()
        body = response.json()
        # Normalise: OpenSky returns null for states when no aircraft present.
        if body.get("states") is None:
            body["states"] = []
        return body

    async def _send_request(self, token: str):
        """Build and send a single GET request to /states/all."""
        lamin, lamax, lomin, lomax = self._bbox
        params = {
            "lamin": lamin,
            "lamax": lamax,
            "lomin": lomin,
            "lomax": lomax,
        }
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{self._base_url}/states/all"

        logger.debug(
            "GET %s params=%s", url, params
        )
        return await self._http.get(url, params=params, headers=headers)

    async def _handle_rate_limit(self, response: Any) -> None:
        """Sleep for the duration specified by the rate-limit header.

        Increments ``ingest_rate_limit_sleeps_total`` once per backoff.

        Args:
            response: The httpx Response with status 429.
        """
        from shared.obs.telemetry import ingest_rate_limit_sleeps_total

        raw = response.headers.get(_RATE_LIMIT_HEADER)
        try:
            sleep_seconds = float(raw) if raw is not None else _DEFAULT_RATE_LIMIT_SLEEP_S
        except ValueError:
            sleep_seconds = _DEFAULT_RATE_LIMIT_SLEEP_S

        logger.warning(
            "HTTP 429 rate-limited; sleeping %.1f s (header %s=%r).",
            sleep_seconds,
            _RATE_LIMIT_HEADER,
            raw,
        )
        ingest_rate_limit_sleeps_total.inc()
        await self._sleep_fn(sleep_seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_from_env() -> tuple[float, float, float, float]:
    """Read bounding-box env vars and return ``(lamin, lamax, lomin, lomax)``."""
    return (
        float(os.environ.get("OPENSKY_BBOX_LAMIN", "47.0")),
        float(os.environ.get("OPENSKY_BBOX_LAMAX", "55.0")),
        float(os.environ.get("OPENSKY_BBOX_LOMIN", "5.0")),
        float(os.environ.get("OPENSKY_BBOX_LOMAX", "15.0")),
    )
