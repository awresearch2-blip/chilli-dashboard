"""Modules 1-3: Price Analysis, Arrival Analysis, Price vs Arrivals."""

import streamlit as st

from analytics.arrival_module import SHEET_NAME as ARRIVAL_SHEET
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from dashboard import charts
from dashboard.components import metric_row, render_or_caveat
from dashboard.data_loader import load_analytical, load_clean_sheet


def _variety_price_series(variety: str, metric: str = "avg_price"):
    from analytics.data_access import date_indexed_series

    df = load_clean_sheet(PRICE_SHEET)
    return date_indexed_series(df, "value", {"variety": variety, "metric": metric})


def _clip_range(series, date_range):
    if series is None or date_range is None:
        return series
    start, end = date_range
    return series.loc[str(start):str(end)]


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header("Price, Arrivals & Price-vs-Arrivals")

    price_result = load_analytical("price_analysis").get(variety, {})
    arrival_result = load_analytical("arrival_analysis")
    pva_result = load_analytical("price_vs_arrivals").get(variety, {})

    st.subheader(f"Price -- {variety}")

    def _render_price(data):
        metric_row(
            [
                ("Latest avg price", f"₹{data['latest_avg_price']:,.0f}"),
                ("CAGR", f"{data['cagr']['cagr_pct']}%" if data["cagr"].get("status") == "ok" else "n/a"),
                ("30d volatility (CV)", f"{data['volatility_cv_30d']:.3f}" if data.get("volatility_cv_30d") is not None else "n/a"),
                ("Trend regime", data["trend_strength_90obs"].get("regime", "n/a") if data["trend_strength_90obs"].get("status") == "ok" else "n/a"),
            ]
        )
        avg = _clip_range(_variety_price_series(variety, "avg_price"), date_range)
        sma_30 = avg.rolling(30, min_periods=30).mean() if avg is not None else None
        ema_30 = avg.ewm(span=30, adjust=False, min_periods=30).mean() if avg is not None else None
        fig = charts.time_series_chart({"Avg price": avg, "SMA 30": sma_30, "EMA 30": ema_30}, title=f"{variety} avg price, Guntur", y_title="Rs/quintal")
        st.plotly_chart(fig, width="stretch")
        if data.get("data_quality_caveats"):
            for c in data["data_quality_caveats"]:
                st.caption(f"ℹ️ {c}")

    render_or_caveat(price_result, _render_price)

    st.subheader("Arrivals -- Guntur (total market)")

    def _render_arrivals(field_key, label, unit):
        data = arrival_result.get(field_key, {})

        def _do_render(d):
            metric_row(
                [
                    ("Latest daily", f"{d['latest_daily']:,.0f} {unit}"),
                    ("YoY growth", f"{d['yoy_growth_pct_latest']:.1f}%" if d.get("yoy_growth_pct_latest") is not None else "n/a"),
                    ("MoM growth", f"{d['mom_growth_pct_latest']:.1f}%" if d.get("mom_growth_pct_latest") is not None else "n/a"),
                ]
            )
            from analytics.arrival_module import field_series

            df = load_clean_sheet(ARRIVAL_SHEET)
            series = _clip_range(field_series(df, field_key), date_range)
            rolling = series.rolling(30, min_periods=30).mean() if series is not None else None
            fig = charts.time_series_chart({label: series, "30d rolling mean": rolling}, title=f"{label}, Guntur", y_title=unit)
            st.plotly_chart(fig, width="stretch")

            peak_lean = d.get("seasonal_peak_lean", {})
            if peak_lean.get("status") == "ok":
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                avgs = peak_lean["monthly_avg_arrivals_bags"]
                cats = [month_names[int(m) - 1] for m in sorted(avgs, key=int)]
                vals = [avgs[m] for m in sorted(avgs, key=int)]
                st.plotly_chart(charts.bar_chart(cats, vals, title=f"Seasonal average {label.lower()}", y_title=unit), width="stretch")
                st.caption(f"Peak months: {peak_lean['peak_months']} | Lean months: {peak_lean['lean_months']}")

        render_or_caveat(data, _do_render)

    col1, col2 = st.columns(2)
    with col1:
        _render_arrivals("arrivals_bags", "Arrivals", "bags")
    with col2:
        _render_arrivals("offtake_bags", "Offtake", "bags")

    st.subheader(f"Price vs Arrivals -- {variety}")

    def _render_pva(data):
        metric_row(
            [
                ("Pearson r (full history)", f"{data['pearson_correlation']['r']:.3f}" if data["pearson_correlation"].get("status") == "ok" else "n/a"),
                ("Rolling 90obs corr (now)", f"{data['rolling_90obs_correlation_latest']:.3f}" if data.get("rolling_90obs_correlation_latest") is not None else "n/a"),
                ("Elasticity", f"{data['elasticity_price_wrt_arrivals']['elasticity']:.3f}" if data["elasticity_price_wrt_arrivals"].get("status") == "ok" else "n/a"),
            ]
        )
        lag = data.get("lag_correlogram_30obs", {})
        if lag.get("status") == "ok":
            st.plotly_chart(charts.bar_chart(lag["lags"], lag["correlations"], title="Lag correlogram (price vs arrivals)", y_title="correlation"), width="stretch")
            st.caption(f"Best lag: {lag['best_lag']} observations (best |r|={lag['best_corr']})")
        for c in data.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")

    render_or_caveat(pva_result, _render_pva)
