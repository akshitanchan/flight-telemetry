"""Operational view — live gold-table metrics for the flight-telemetry platform.

Renders the four gold tables (airport congestion, sector load, emergency
events, routing stats) with summary KPI cards and sortable data tables.
"""

import streamlit as st

from dashboard.data import load_gold


def render() -> None:
    """Render the operational dashboard view."""
    st.header("Operational Overview")
    st.caption("Source: gold tables from data/processed/ (or Postgres if DATABASE_URL is set)")

    gold = load_gold()

    # ------------------------------------------------------------------
    # Top-level KPI row
    # ------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    active_airports = (
        gold.congestion["airport_icao"].nunique()
        if not gold.congestion.empty
        else 0
    )
    active_sectors = (
        gold.sector["h3_r4"].nunique()
        if not gold.sector.empty
        else 0
    )
    emergency_count = len(gold.emergency)
    tracked_flights = (
        gold.routing["icao24"].nunique()
        if not gold.routing.empty
        else 0
    )

    kpi1.metric("Active Airports", active_airports)
    kpi2.metric("Active Sectors", active_sectors)
    kpi3.metric("Emergency Events", emergency_count)
    kpi4.metric("Tracked Flights", tracked_flights)

    st.divider()

    # ------------------------------------------------------------------
    # Airport Congestion
    # ------------------------------------------------------------------
    st.subheader("Airport Congestion")
    if gold.congestion.empty:
        st.info("No congestion data available.")
    else:
        # Summary chart: aircraft count per airport (aggregate across windows)
        agg = (
            gold.congestion.groupby("airport_icao", as_index=False)["aircraft_count"]
            .sum()
            .sort_values("aircraft_count", ascending=False)
        )
        st.bar_chart(agg.set_index("airport_icao")["aircraft_count"])

        with st.expander("Raw congestion table", expanded=False):
            st.dataframe(
                gold.congestion.sort_values(
                    ["airport_icao", "window_start"], ascending=[True, False]
                ),
                use_container_width=True,
            )

    st.divider()

    # ------------------------------------------------------------------
    # Sector Load
    # ------------------------------------------------------------------
    st.subheader("Sector Load")
    if gold.sector.empty:
        st.info("No sector-load data available.")
    else:
        top_sectors = (
            gold.sector.groupby("h3_r4", as_index=False)["aircraft_count"]
            .sum()
            .sort_values("aircraft_count", ascending=False)
            .head(20)
        )
        st.bar_chart(top_sectors.set_index("h3_r4")["aircraft_count"])

        with st.expander("Raw sector table (top 200 rows)", expanded=False):
            st.dataframe(
                gold.sector.sort_values("aircraft_count", ascending=False).head(200),
                use_container_width=True,
            )

    st.divider()

    # ------------------------------------------------------------------
    # Emergency Events
    # ------------------------------------------------------------------
    st.subheader("Emergency Events")
    if gold.emergency.empty:
        st.info("No emergency events recorded.")
    else:
        # Squawk breakdown
        squawk_counts = gold.emergency["squawk"].value_counts().reset_index()
        squawk_counts.columns = ["squawk", "count"]
        squawk_labels = {"7500": "7500 Hijack", "7600": "7600 Radio Failure", "7700": "7700 General Emergency"}
        squawk_counts["squawk_label"] = squawk_counts["squawk"].map(
            lambda s: squawk_labels.get(s, s)
        )

        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            st.bar_chart(
                squawk_counts.set_index("squawk_label")["count"],
                color="#ff4b4b",
            )
        with col_table:
            display_cols = [
                c for c in ["icao24", "callsign", "squawk", "origin_country",
                             "nearest_airport", "first_seen_ts", "duration_s"]
                if c in gold.emergency.columns
            ]
            st.dataframe(
                gold.emergency[display_cols].sort_values("first_seen_ts", ascending=False),
                use_container_width=True,
            )

    st.divider()

    # ------------------------------------------------------------------
    # Routing Stats
    # ------------------------------------------------------------------
    st.subheader("Routing Stats")
    if gold.routing.empty:
        st.info("No routing data available.")
    else:
        col_alt, col_vel = st.columns(2)
        with col_alt:
            st.caption("Max Altitude (m) per flight")
            alt_data = (
                gold.routing.dropna(subset=["max_altitude_m"])
                .sort_values("max_altitude_m", ascending=False)
                .head(20)
            )
            if not alt_data.empty:
                label_col = "callsign" if alt_data["callsign"].notna().any() else "icao24"
                st.bar_chart(alt_data.set_index(label_col)["max_altitude_m"])

        with col_vel:
            st.caption("Avg Velocity (m/s) per flight")
            vel_data = (
                gold.routing.dropna(subset=["avg_velocity_mps"])
                .sort_values("avg_velocity_mps", ascending=False)
                .head(20)
            )
            if not vel_data.empty:
                label_col = "callsign" if vel_data["callsign"].notna().any() else "icao24"
                st.bar_chart(vel_data.set_index(label_col)["avg_velocity_mps"])

        with st.expander("Raw routing table", expanded=False):
            st.dataframe(
                gold.routing.sort_values("max_altitude_m", ascending=False),
                use_container_width=True,
            )
