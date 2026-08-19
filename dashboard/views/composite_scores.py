"""Phase 2c: composite indices and scores."""

import streamlit as st

from dashboard import charts
from dashboard.components import render_or_caveat
from dashboard.data_loader import load_analytical


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header("Composite Scores")
    st.caption(load_analytical("composite_indices").get("methodology_note", ""))

    result = load_analytical("composite_indices")
    per_variety = result.get("per_variety", {}).get(variety, {})
    market_wide = result.get("market_wide", {})

    st.subheader(f"{variety} scores")
    cols = st.columns(4)

    def _gauge(col, data, key, title):
        if data.get("status") == "ok":
            value = data.get(key) if key in data else (data.get("index") or data.get("score"))
            col.plotly_chart(charts.gauge_chart(value, title=title), width="stretch")
        else:
            col.info(f"ℹ️ {title}: insufficient evidence")

    _gauge(cols[0], per_variety.get("market_strength_index", {}), "index", "Market Strength")
    _gauge(cols[1], per_variety.get("price_stability_index", {}), "index", "Price Stability")
    _gauge(cols[2], per_variety.get("risk_score", {}), "score", "Risk Score")
    _gauge(cols[3], per_variety.get("composite_commodity_index", {}), "index", "Composite Commodity Index")

    bullish_bearish = per_variety.get("bullish_bearish", {})

    def _render_bb(data):
        c1, c2 = st.columns(2)
        c1.metric("Bullish score", f"{data['bullish_score']:.0f}/100")
        c2.metric("Bearish score", f"{data['bearish_score']:.0f}/100")
        st.write("Signals:")
        st.json(data["signals"])

    render_or_caveat(bullish_bearish, _render_bb)

    confidence = per_variety.get("confidence_score", {})
    if confidence.get("status") == "ok":
        st.caption(f"Confidence score: {confidence['score']:.0f}/100 (availability={confidence['components']['component_availability_pct']}%, sample adequacy={confidence['components']['sample_size_adequacy_pct']}%)")

    st.subheader("Market-wide")
    cols2 = st.columns(3)
    _gauge(cols2[0], market_wide.get("supply_pressure_index", {}), "index", "Supply Pressure")
    _gauge(cols2[1], market_wide.get("arrival_pressure_index", {}), "index", "Arrival Pressure")
    _gauge(cols2[2], market_wide.get("demand_index", {}), "index", "Demand Index")
