"""Modules 7-10: Export Analysis, Balance Sheet, USD/INR, Cold Storage."""

import streamlit as st

from dashboard import charts
from dashboard.components import metric_row, render_or_caveat
from dashboard.data_loader import load_analytical


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header("Export, Balance Sheet, FX & Cold Storage")

    st.subheader("Export Analysis")
    export_result = load_analytical("export_analysis")

    def _render_export(data):
        metric_row(
            [
                ("CAGR", f"{data['cagr']['cagr_pct']}%" if data["cagr"].get("status") == "ok" else "n/a"),
                ("YoY growth", f"{data['yoy_growth_pct_latest']:.1f}%" if data.get("yoy_growth_pct_latest") is not None else "n/a"),
                ("MoM growth", f"{data['mom_growth_pct_latest']:.1f}%" if data.get("mom_growth_pct_latest") is not None else "n/a"),
            ]
        )
        seasonal = data.get("seasonal_export_index", {})
        if seasonal.get("status") == "ok":
            st.plotly_chart(charts.seasonal_index_bar(seasonal["index_by_month"], title="Seasonal export index"), width="stretch")
        vs_price = data.get("vs_price", {}).get(variety, {})
        if vs_price.get("pearson_correlation", {}).get("status") == "ok":
            st.caption(f"Correlation with {variety} price: r={vs_price['pearson_correlation']['r']:.3f} (n={vs_price['pearson_correlation']['n']} months)")
        for c in data.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")

    render_or_caveat(export_result, _render_export)

    st.subheader("Balance Sheet (Lakh Tons)")
    balance_result = load_analytical("balance_sheet")
    per_category = balance_result.get("per_category", {})
    if per_category:
        category = st.selectbox("Category", sorted(per_category.keys()))
        data = per_category[category]

        def _render_category(d):
            years = sorted(d["annual_values_lakh_tons"].keys())
            values = [d["annual_values_lakh_tons"][y] for y in years]
            fig = charts.bar_chart(years, values, title=f"{category} (actuals)", y_title="Lakh Tons")
            st.plotly_chart(fig, width="stretch")
            metric_row(
                [
                    ("Latest actual", f"{d['latest_actual_value']:,.2f} ({d['latest_actual_year']})"),
                    ("2026 estimate (workbook's own)", f"{d['estimate_2026']:,.2f}" if d.get("estimate_2026") is not None else "n/a"),
                    ("YoY change", f"{d['yoy_change_pct_latest']:.1f}%" if d.get("yoy_change_pct_latest") is not None else "n/a"),
                ]
            )
            corr = d.get("correlation_with_price", {}).get(variety, {})
            if corr.get("status") == "ok":
                st.caption(f"Correlation with {variety} annual price: r={corr['r']:.3f} (n={corr['n']} years -- annual data, directional signal only)")

        render_or_caveat(data, _render_category)

        surplus = balance_result.get("surplus_deficit_total_supply_minus_demand", {})
        if surplus.get("status") == "ok":
            years = sorted(surplus["annual_values_lakh_tons"].keys())
            values = [surplus["annual_values_lakh_tons"][y] for y in years]
            st.plotly_chart(charts.bar_chart(years, values, title="Supply surplus/deficit (Total Supply - Total Demand)", y_title="Lakh Tons"), width="stretch")
        for c in balance_result.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")
    else:
        st.info("ℹ️ No balance sheet data available yet.")

    st.subheader("USD/INR Exchange Rate")
    fx_result = load_analytical("fx_analysis")

    def _render_fx(data):
        metric_row(
            [
                ("Latest rate", f"₹{data['latest_rate']:.2f}"),
                ("CAGR", f"{data['cagr']['cagr_pct']}%" if data["cagr"].get("status") == "ok" else "n/a"),
                ("SMA 30", f"₹{data['moving_averages']['sma_30']:.2f}" if data["moving_averages"].get("sma_30") is not None else "n/a"),
            ]
        )
        vs_price = data.get("vs_price", {}).get(variety, {})
        if vs_price.get("pearson_correlation", {}).get("status") == "ok":
            st.caption(f"Correlation with {variety} price: r={vs_price['pearson_correlation']['r']:.3f}")
            for c in vs_price.get("data_quality_caveats", []):
                st.caption(f"ℹ️ {c}")

    render_or_caveat(fx_result, _render_fx)

    st.subheader("Cold Storage")
    cold_result = load_analytical("cold_storage")
    per_region = cold_result.get("per_region", {})
    if per_region:
        rows = [
            {"region": region, "latest_known_value": d.get("latest_known_value"), "as_of": d.get("latest_known_date"), "status": d.get("status")}
            for region, d in per_region.items()
        ]
        st.dataframe(rows, width="stretch")
        for c in cold_result.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")
    else:
        st.info("ℹ️ No cold storage data available yet.")
