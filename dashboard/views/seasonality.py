"""Module 4: Seasonality."""

import streamlit as st

from dashboard import charts
from dashboard.components import render_or_caveat
from dashboard.data_loader import load_analytical


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header("Seasonality")

    result = load_analytical("seasonality")
    price_seasonality = result.get("price_seasonality", {}).get(variety, {})

    st.subheader(f"Seasonal Price Index -- {variety}")

    def _render_price_seasonality(data):
        seasonal_index = data.get("seasonal_price_index", {})
        if seasonal_index.get("status") == "ok":
            st.plotly_chart(charts.seasonal_index_bar(seasonal_index["index_by_month"], title=f"{variety} seasonal price index (100 = neutral)"), width="stretch")
            if seasonal_index.get("months_with_insufficient_evidence"):
                st.caption(f"Months with fewer than the minimum years of history: {seasonal_index['months_with_insufficient_evidence']}")
        else:
            st.info(f"ℹ️ Seasonal index insufficient evidence: {seasonal_index.get('reason')}")

        if data.get("workbook_reported_monthly_avg_by_month"):
            with st.expander("Workbook's own 'Seasonality index' sheet (reference only -- not a normalized index, see caveat)"):
                st.json(data["workbook_reported_monthly_avg_by_month"])

        yoy = data.get("yoy_seasonal_comparison", {})
        if yoy.get("status") == "ok":
            st.subheader("Actual vs historical seasonal average (last 12 months)")
            rows = [
                {"month": k, **v} for k, v in sorted(yoy["monthly_vs_historical"].items())
            ]
            st.dataframe(rows, width="stretch")

        for c in data.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")

    render_or_caveat(price_seasonality, _render_price_seasonality)

    st.subheader("Seasonal Arrival Index -- Guntur")
    arrival_seasonality = result.get("arrival_seasonality", {})

    def _render_arrival_seasonality(data):
        seasonal_index = data.get("seasonal_arrival_index", {})
        if seasonal_index.get("status") == "ok":
            st.plotly_chart(charts.seasonal_index_bar(seasonal_index["index_by_month"], title="Seasonal arrival index (100 = neutral)"), width="stretch")

    render_or_caveat(arrival_seasonality, _render_arrival_seasonality)

    st.subheader("Harvest vs Off-Season")
    harvest = result.get("harvest_vs_offseason", {})

    def _render_harvest(data):
        st.write(f"Harvest months (highest seasonal arrivals): **{data['harvest_months']}**")
        st.write(f"Off-season months (lowest seasonal arrivals): **{data['offseason_months']}**")
        st.caption(data.get("note", ""))
        behavior = data.get("price_behavior_by_variety", {}).get(variety)
        if behavior:
            st.write(
                f"{variety}: avg price index during harvest = **{behavior['avg_price_index_during_harvest_months']}**, "
                f"during off-season = **{behavior['avg_price_index_during_offseason_months']}**"
            )

    render_or_caveat(harvest, _render_harvest)
