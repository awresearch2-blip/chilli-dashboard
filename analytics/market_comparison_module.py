"""Module 6 -- Market Comparison: Guntur vs Warangal vs Khammam (Teja).

Prices are Rs/quintal in every sheet, so the differing bag sizes noted in
Phase 1 (Guntur 45kg, Warangal 40kg, Khammam 38kg) do NOT affect this
module's price comparisons -- bag size only matters for arrival *volume*
comparisons, which this module doesn't make.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet
from analytics.price_module import SHEET_NAME as GUNTUR_PRICE_SHEET

WARANGAL_SHEET = "Warangal Teja Price& Arrivals"
KHAMMAM_SHEET = "Khammam Teja Price& Arrivals"

ANOMALY_CAVEAT = (
    "'Spread anomaly' flags a price gap wider than this pair's own recent "
    "history (|rolling z-score| > 2), not a validated arbitrage opportunity "
    "-- no transport-cost or quality-differential data exists to determine "
    "whether it is actually exploitable."
)
KHAMMAM_CAVEAT = (
    "Khammam has the highest missingness of all price sheets (Phase 1 "
    "finding); pairs involving Khammam are correspondingly less reliable."
)


def _latest(series: pd.Series, ndigits: int = 2):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), ndigits)


def analyze_pair(series_a: pd.Series, series_b: pd.Series, window: int = 90) -> dict:
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if len(aligned) < 30:
        return {
            "status": "insufficient_evidence",
            "reason": "Not enough overlapping valid observations",
            "available_n": len(aligned),
            "required_n": 30,
        }

    spread = aligned["a"] - aligned["b"]
    premium_pct_of_b = spread / aligned["b"] * 100
    trend_window = min(window, len(spread))
    convergence = ts.trend_strength(spread.abs(), window=trend_window, min_n=10)
    convergence_label = None
    if convergence.get("status") == "ok":
        regime = convergence["regime"]
        convergence_label = {"uptrend": "diverging", "downtrend": "converging", "range_bound": "stable"}[regime]

    zscore_latest = _latest(ts.rolling_zscore(spread, window), 3)

    return {
        "status": "ok",
        "latest_spread": _latest(spread),
        "latest_premium_pct_of_b": _latest(premium_pct_of_b),
        "avg_spread_90d": _latest(spread.rolling(90, min_periods=90).mean()),
        "convergence_divergence": {**convergence, "label": convergence_label},
        "spread_anomaly_zscore_90d": zscore_latest,
        "spread_anomaly_flag": zscore_latest is not None and abs(zscore_latest) > 2,
        "observations_used": len(aligned),
    }


def run(guntur_df: pd.DataFrame = None, warangal_df: pd.DataFrame = None, khammam_df: pd.DataFrame = None) -> dict:
    if guntur_df is None:
        guntur_df = load_clean_sheet(GUNTUR_PRICE_SHEET)
    if warangal_df is None:
        warangal_df = load_clean_sheet(WARANGAL_SHEET)
    if khammam_df is None:
        khammam_df = load_clean_sheet(KHAMMAM_SHEET)

    guntur = date_indexed_series(guntur_df, "value", {"variety": "Teja", "metric": "avg_price"})
    warangal = date_indexed_series(warangal_df, "avg_price")
    khammam_noncold = date_indexed_series(khammam_df, "value", {"metric": "avg_price_non_cold_storage"})
    khammam_cold = date_indexed_series(khammam_df, "value", {"metric": "avg_price_cold_storage"})

    pairs = {
        "guntur_vs_warangal": analyze_pair(guntur, warangal),
        "guntur_vs_khammam_noncold": analyze_pair(guntur, khammam_noncold),
        "warangal_vs_khammam_noncold": analyze_pair(warangal, khammam_noncold),
        "guntur_vs_khammam_cold": analyze_pair(guntur, khammam_cold),
    }
    return {"pairs": pairs, "data_quality_caveats": [ANOMALY_CAVEAT, KHAMMAM_CAVEAT]}
