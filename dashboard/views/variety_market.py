"""Modules 5-6: Variety Analysis, Market Comparison."""

import streamlit as st

from dashboard import charts
from dashboard.components import metric_row, render_or_caveat
from dashboard.data_loader import load_analytical


def render(variety: str, market: str, date_range, horizon: int) -> None:
    st.header("Variety Analysis & Market Comparison")

    variety_result = load_analytical("variety_analysis")
    st.subheader(f"Variety Analysis (benchmark: {variety_result.get('benchmark', 'Teja')})")

    per_variety = variety_result.get("per_variety_vs_benchmark", {})
    if per_variety:
        rows = []
        for other_variety, data in per_variety.items():
            if data.get("status") != "ok":
                continue
            rows.append(
                {
                    "variety": other_variety,
                    "premium/discount %": data["latest_premium_discount_pct"],
                    "ratio to benchmark": data["latest_ratio_to_benchmark"],
                    "correlation to benchmark": data["correlation_to_benchmark"]["r"] if data["correlation_to_benchmark"].get("status") == "ok" else None,
                    "30d momentum (ratio) %": data["relative_momentum_roc_pct"]["roc_30"],
                }
            )
        st.dataframe(rows, width="stretch")
    else:
        st.info("ℹ️ No variety comparison data available yet.")

    matrix = variety_result.get("pairwise_correlation_matrix", {})
    if matrix:
        all_varieties = sorted({v for pair in matrix.keys() for v in pair.split(" vs ")})
        st.plotly_chart(charts.correlation_heatmap(matrix, all_varieties, title="Pairwise price correlation (recomputed from daily prices)"), width="stretch")
        for c in variety_result.get("data_quality_caveats", []):
            st.caption(f"ℹ️ {c}")

    st.subheader("Market Comparison (Teja, Rs/quintal)")
    market_result = load_analytical("market_comparison")
    pairs = market_result.get("pairs", {})

    pair_key_map = {
        "Guntur vs Warangal": "guntur_vs_warangal",
        "Guntur vs Khammam (non-cold storage)": "guntur_vs_khammam_noncold",
        "Warangal vs Khammam (non-cold storage)": "warangal_vs_khammam_noncold",
        "Guntur vs Khammam (cold storage)": "guntur_vs_khammam_cold",
    }

    for label, key in pair_key_map.items():
        data = pairs.get(key, {})

        def _render_pair(d, label=label):
            metric_row(
                [
                    ("Latest spread", f"₹{d['latest_spread']:,.0f}"),
                    ("Premium % of B", f"{d['latest_premium_pct_of_b']:.1f}%"),
                    ("Convergence/divergence", d["convergence_divergence"].get("label", "n/a")),
                    ("Spread anomaly", "⚠️ yes" if d.get("spread_anomaly_flag") else "no"),
                ]
            )

        with st.expander(label, expanded=(key == "guntur_vs_warangal")):
            render_or_caveat(data, _render_pair)

    for c in market_result.get("data_quality_caveats", []):
        st.caption(f"ℹ️ {c}")
