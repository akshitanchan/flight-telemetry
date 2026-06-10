"""
systems/ingest/token.py
-----------------------
OAuth2 client-credentials token manager for the OpenSky Network API.

Design rules:
- Importing this module opens NO network connection.
- Tokens are fetched lazily on first ``get_token()`` call.
- Proactive refresh: re-fetches the token when fewer than
  ``REFRESH_BEFORE_EXPIRY_S`` seconds remain on the current one
  (default 60 s before the 1800 s / 30-min lifetime).
- On HTTP 401 from downstream callers, ``force_refresh()`` must be called
  to discard the cached token and re-fetch unconditionally.
- Every token fetch (initial or refresh) increments
  ``ingest_token_refresh_total`` from shared.obs.telemetry.
- ``_fetch_token()`` is a coroutine; tests patch it via
  ``monkeypatch.setattr`` or dependency injection on ``_fetch_fn``.

Environment variables:
    OPENSKY_CLIENT_ID       — OAuth2 client ID (required for live ops)
    OPENSKY_CLIENT_SECRET   — OAuth2 client secret (required for live ops)
    OPENSKY_TOKEN_URL       — Token endpoint
                              (default: https://auth.opensky-network.org/auth/
                               realms/opensky-network/protocol/openid-connect/token)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Default OpenSky OAuth2 token endpoint (Keycloak-backed).
# Override via OPENSKY_TOKEN_URL env var to point at a fixture in tests.
_DEFAULT_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

# How many seconds before expiry to proactively refresh the token.
REFRESH_BEFORE_EXPIRY_S: int = 60

# Assumed token lifetime in seconds when the server does not send
# ``expires_in`` (OpenSky returns 1800).
DEFAULT_TOKEN_LIFETIME_S: int = 1800


class TokenManager:
    """Manage an OAuth2 client-credentials token.

    Args:
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        token_url: Full URL of the OAuth2 token endpoint.
        fetch_fn: Async callable ``(client_id, client_secret, token_url)
                  -> dict`` that performs the actual HTTP POST.  Defaults
                  to the built-in ``_fetch_token`` coroutine.  Inject a
                  mock here in tests to avoid real network calls.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str | None = None,
        fetch_fn: Callable[..., Awaitable[dict]] | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url or os.environ.get(
            "OPENSKY_TOKEN_URL", _DEFAULT_TOKEN_URL
        )
        self._fetch_fn = fetch_fn or _fetch_token

        self._access_token: str | None = None
        self._expires_at: float = 0.0  # monotonic clock
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_token(self) -> str:
        """Return a valid access token, refreshing proactively if needed.

        Thread-/task-safe: concurrent callers share a single asyncio Lock
        so the token endpoint is not hammered on simultaneous expirations.

        Returns:
            A valid Bearer access token string.

        Raises:
            RuntimeError: If the token fetch fails and no cached token exists.
        """
        async with self._lock:
            if self._needs_refresh():
                await self._do_refresh()
            return self._access_token  # type: ignore[return-value]

    async def force_refresh(self) -> str:
        """Discard the cached token and fetch a fresh one unconditionally.

        Call this when the upstream API returns HTTP 401 to recover from
        out-of-band token revocation or clock skew.

        Returns:
            A freshly issued access token string.
        """
        async with self._lock:
            self._access_token = None
            self._expires_at = 0.0
            await self._do_refresh()
            return self._access_token  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _needs_refresh(self) -> bool:
        """Return True if the token is absent or about to expire."""
        if self._access_token is None:
            return True
        remaining = self._expires_at - time.monotonic()
        return remaining < REFRESH_BEFORE_EXPIRY_S

    async def _do_refresh(self) -> None:
        """Invoke the fetch function and update the cached token.

        Increments ``ingest_token_refresh_total`` on every call.
        Wraps the operation in an OTel span named ``ingestion.token_refresh``.
        """
        # Import here to keep module-level import free of network side-effects.
        from shared.obs.telemetry import ingest_token_refresh_total, get_tracer

        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("ingestion.token_refresh"):
            logger.debug(
                "Fetching OAuth2 token from %s (client_id=%s)",
                self._token_url,
                self._client_id,
            )
            try:
                data = await self._fetch_fn(
                    self._client_id, self._client_secret, self._token_url
                )
            except Exception as exc:
                logger.error("Token fetch failed: %s", exc)
                raise

            access_token = data.get("access_token")
            if not access_token:
                raise RuntimeError(
                    f"Token response missing 'access_token': {data!r}"
                )

            lifetime = int(data.get("expires_in", DEFAULT_TOKEN_LIFETIME_S))
            self._access_token = access_token
            self._expires_at = time.monotonic() + lifetime

            ingest_token_refresh_total.inc()
            logger.info(
                "OAuth2 token refreshed; expires in %d s.", lifetime
            )


# ---------------------------------------------------------------------------
# Default low-level fetch implementation
# ---------------------------------------------------------------------------

async def _fetch_token(
    client_id: str,
    client_secret: str,
    token_url: str,
) -> dict:
    """POST client-credentials grant to *token_url* and return the JSON body.

    This function is the sole network-touching code in token.py.  It is
    deliberately separated so tests can inject a replacement via the
    ``fetch_fn`` constructor argument of :class:`TokenManager`.

    Args:
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        token_url: Full token endpoint URL.

    Returns:
        Parsed JSON response dict.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
        httpx.RequestError: On connection/timeout errors.
    """
    import httpx

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Convenience factory: read credentials from environment
# ---------------------------------------------------------------------------

def token_manager_from_env(fetch_fn: Callable[..., Awaitable[dict]] | None = None) -> TokenManager:
    """Build a :class:`TokenManager` from environment variables.

    Args:
        fetch_fn: Optional injectable fetch coroutine for testing.

    Returns:
        A configured TokenManager instance (no network call made yet).
    """
    return TokenManager(
        client_id=os.environ.get("OPENSKY_CLIENT_ID", ""),
        client_secret=os.environ.get("OPENSKY_CLIENT_SECRET", ""),
        token_url=os.environ.get("OPENSKY_TOKEN_URL", _DEFAULT_TOKEN_URL),
        fetch_fn=fetch_fn,
    )
