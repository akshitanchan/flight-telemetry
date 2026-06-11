"""Platform Health view — service health and key Prometheus metrics.

Renders per-service health status cards and a table of key
``<service>_<noun>_<unit>`` metrics scraped from live service endpoints.
When nothing is reachable (the default offline/dev state) it shows a clean
"not available" message rather than crashing or showing partial data.
"""

import streamlit as st

from dashboard.health import (
    ServiceStatus,
    fetch_all_services,
)

# ---------------------------------------------------------------------------
# Key metrics to surface in the UI, grouped by service.
# Names follow the C6 convention: <service>_<noun>_<unit>.
# ---------------------------------------------------------------------------
_METRIC_DISPLAY: dict[str, list[tuple[str, str]]] = {
    "ml-serve": [
        ("ml_predict_latency_seconds_sum", "Predict latency total (s)"),
        ("ml_predict_latency_seconds_count", "Predict requests"),
        ("http_requests_total", "HTTP requests"),
    ],
    "ingestion": [
        ("ingest_records_total", "Records ingested"),
        ("ingest_poll_latency_seconds_sum", "Poll latency total (s)"),
        ("ingest_poll_latency_seconds_count", "Poll cycles"),
        ("ingest_rate_limit_sleeps_total", "Rate-limit sleeps"),
        ("ingest_token_refresh_total", "Token refreshes"),
    ],
}

# Status badge colours
_STATUS_OK = "normal"
_STATUS_DOWN = "off"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_badge(available: bool) -> str:
    if available:
        return "Running"
    return "Down"


def _render_service_card(status: ServiceStatus) -> None:
    """Render a single service status card with metric details."""
    label = status.service.upper()
    badge = _status_badge(status.available)

    if status.available:
        st.success(f"**{label}** — {badge}", icon="✅")
    else:
        reason = f": {status.error}" if status.error else ""
        st.error(f"**{label}** — {badge}{reason}", icon="🔴")

    key_metrics = _METRIC_DISPLAY.get(status.service, [])
    if status.available and key_metrics:
        rows: list[dict] = []
        for metric_key, metric_label in key_metrics:
            value = status.metrics.get(metric_key)
            rows.append(
                {
                    "Metric": metric_label,
                    "Raw key": metric_key,
                    "Value": f"{value:,.4g}" if value is not None else "—",
                }
            )
        if rows:
            import pandas as pd

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
    elif status.available:
        st.caption("No key metrics configured for this service.")


def _render_prometheus_card(status: ServiceStatus) -> None:
    """Render the Prometheus server status card."""
    label = "PROMETHEUS"
    if status.available:
        st.success(f"**{label}** — Running  ({status.url})", icon="✅")
    else:
        reason = f": {status.error}" if status.error else ""
        st.error(f"**{label}** — Down{reason}  ({status.url})", icon="🔴")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Platform Health view."""
    st.header("Platform Health")
    st.caption(
        "Service status and key Prometheus metrics scraped from live endpoints. "
        "Endpoints are read from ``ML_SERVE_METRICS_URL``, ``INGEST_METRICS_URL``, "
        "and ``PROMETHEUS_URL`` (localhost defaults match docker-compose ports)."
    )

    # Fetch with a brief spinner so the user knows a network call is happening.
    with st.spinner("Checking service health…"):
        statuses = fetch_all_services(timeout=3.0)

    # Summarise
    n_up = sum(1 for s in statuses if s.available)
    n_total = len(statuses)

    if n_up == 0:
        st.warning(
            "No platform services are currently reachable. "
            "Start the docker-compose stack (``docker compose up -d``) to see live metrics.",
            icon="⚠️",
        )
    elif n_up < n_total:
        st.warning(
            f"{n_up}/{n_total} services reachable. Some services are down.",
            icon="⚠️",
        )
    else:
        st.success(f"All {n_total} services are reachable.", icon="✅")

    st.divider()

    # ---------------------------------------------------------------------------
    # Per-service cards
    # ---------------------------------------------------------------------------
    # Split into two groups: application services (scraped directly) + Prometheus
    app_statuses = [s for s in statuses if s.service != "prometheus"]
    prom_status = next((s for s in statuses if s.service == "prometheus"), None)

    # Application services — two columns
    if app_statuses:
        st.subheader("Application Services")
        cols = st.columns(len(app_statuses))
        for col, svc_status in zip(cols, app_statuses):
            with col:
                _render_service_card(svc_status)

    st.divider()

    # Prometheus card
    st.subheader("Observability")
    if prom_status is not None:
        _render_prometheus_card(prom_status)
        if prom_status.available:
            prom_base = prom_status.url
            st.markdown(
                f"Open the Prometheus UI: [{prom_base}]({prom_base})"
            )

    st.divider()

    # ---------------------------------------------------------------------------
    # Full metrics expander (available services only)
    # ---------------------------------------------------------------------------
    available_app = [s for s in app_statuses if s.available]
    if available_app:
        with st.expander("All scraped metrics (raw)", expanded=False):
            for svc_status in available_app:
                st.markdown(f"**{svc_status.service}** — `{svc_status.url}`")
                if svc_status.metrics:
                    import pandas as pd

                    df = pd.DataFrame(
                        [
                            {"Metric": k, "Value": v}
                            for k, v in sorted(svc_status.metrics.items())
                        ]
                    )
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No metrics parsed.")

    # ---------------------------------------------------------------------------
    # Refresh button
    # ---------------------------------------------------------------------------
    st.divider()
    if st.button("Refresh"):
        st.rerun()
