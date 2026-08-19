"""Phase 3a/3b: forecasts with full explainability."""

import streamlit as st

from analytics.price_module import SHEET_NAME as PRICE_SHEET, variety_series
from dashboard import charts
from dashboard.components import metric_row, render_or_caveat
from dashboard.data_loader import load_clean_sheet, load_forecast, load_realized_accuracy


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header(f"Forecast -- {variety}")

    forecast_result = load_forecast(variety)

    def _render(data):
        st.caption(f"As of {data['as_of']} | current price: ₹{data['current_price']:,.0f}")

        key_drivers = data.get("key_drivers", {})
        if key_drivers.get("status") == "ok":
            st.write(f"**Key drivers** (bullish score {key_drivers.get('bullish_score', 0):.0f}/100, bearish score {key_drivers.get('bearish_score', 0):.0f}/100):")
            for statement in key_drivers["statements"]:
                st.write(f"- {statement}")

        forecasts = data.get("forecasts_by_horizon_days", {})
        horizon_key = str(horizon)
        fc = forecasts.get(horizon_key, {})

        def _render_horizon(hd):
            price_series = variety_series(load_clean_sheet(PRICE_SHEET), variety, "avg_price")
            history = price_series.dropna().tail(365)
            fig = charts.forecast_fan_chart(history, hd["target_date"], hd["point_forecast"], hd["lower_ci"], hd["upper_ci"], title=f"{variety} {horizon}-day forecast")
            st.plotly_chart(fig, width="stretch")

            metric_row(
                [
                    ("Point forecast", f"₹{hd['point_forecast']:,.0f}"),
                    ("Confidence interval", f"₹{hd['lower_ci']:,.0f} - ₹{hd['upper_ci']:,.0f}" if hd.get("lower_ci") is not None else "n/a"),
                    ("Model used", hd["model_used"]),
                    ("Backtested MAPE", f"{hd['backtest_accuracy']['mape']:.1f}%"),
                ]
            )
            if hd.get("ci_is_empirical_from_backtest_residuals"):
                st.caption("Confidence interval is empirical (from this model/horizon's real backtest residuals), not a parametric assumption.")

            probability = hd.get("probability", {})
            if probability.get("status") == "ok":
                c1, c2 = st.columns(2)
                c1.plotly_chart(charts.gauge_chart(probability["bullish_probability_pct"], title="Bullish probability"), width="stretch")
                c2.plotly_chart(charts.gauge_chart(probability["bearish_probability_pct"], title="Bearish probability"), width="stretch")
                st.caption(probability["method"])
            else:
                st.info(f"ℹ️ Probability: {probability.get('reason', 'insufficient evidence')}")

            with st.expander("Model explanation"):
                explanation = hd.get("model_explanation", {})
                if explanation.get("status") == "ok":
                    st.write(explanation.get("description", ""))
                    extra = {k: v for k, v in explanation.items() if k not in ("status", "type", "description")}
                    if extra:
                        st.json(extra)
                else:
                    st.info("ℹ️ No model explanation available.")

            with st.expander("Similar historical periods"):
                similar = hd.get("similar_historical_periods", {})
                if similar.get("status") == "ok":
                    if similar.get("note"):
                        st.caption(similar["note"])
                    st.dataframe(similar["similar_periods"], width="stretch")
                else:
                    st.info(f"ℹ️ {similar.get('reason', 'insufficient evidence')}")

            risks = hd.get("risks_and_limitations", [])
            if risks:
                st.write("**Risks & limitations:**")
                for risk in risks:
                    st.write(f"- {risk}")

        render_or_caveat(fc, _render_horizon)

    render_or_caveat(forecast_result, _render)

    st.subheader("Historical accuracy")
    accuracy_result = load_realized_accuracy()
    if accuracy_result.get("status") == "ok":
        st.json(accuracy_result["realized_accuracy"])
        st.caption(f"{accuracy_result['forecasts_scored']} forecasts scored, {accuracy_result['forecasts_pending']} still pending their target date.")
    else:
        st.info(f"ℹ️ {accuracy_result.get('reason', 'No realized accuracy data yet.')}")
