"""Module 3 -- Price vs Arrivals, Guntur market.

Pairs each variety's price against Guntur's TOTAL-market daily arrivals,
because the workbook has no variety-specific arrivals series. This is a
documented approximation, not a hidden one -- see CAVEAT below, echoed into
every result this module produces.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.arrival_module import SHEET_NAME as ARRIVAL_SHEET
from analytics.arrival_module import field_series
from analytics.data_access import load_clean_sheet
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series

CAVEAT = (
    "Guntur Daily arrivals is a total-market figure across all varieties, not "
    "variety-specific. This pairs each variety's own price against total "
    "market supply (not that variety's own supply) because the workbook has "
    "no variety-specific arrivals series."
)


def _latest(series: pd.Series):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), 4)


def analyze_variety(price_df: pd.DataFrame, arrivals_df: pd.DataFrame, variety: str) -> dict:
    price = variety_series(price_df, variety, "avg_price")
    arrivals = field_series(arrivals_df, "arrivals_bags")

    if price.dropna().empty or arrivals.dropna().empty:
        return {"status": "insufficient_evidence", "reason": "Missing price or arrivals series"}

    return {
        "status": "ok",
        "pearson_correlation": ts.pearson_corr(price, arrivals),
        "spearman_correlation": ts.spearman_corr(price, arrivals),
        "rolling_90obs_correlation_latest": _latest(ts.rolling_corr(price, arrivals, 90)),
        "lag_correlogram_30obs": ts.lag_correlogram(price, arrivals, max_lag=30),
        "elasticity_price_wrt_arrivals": ts.log_log_elasticity(arrivals, price),
        "data_quality_caveats": [CAVEAT],
    }


def run(price_df: pd.DataFrame = None, arrivals_df: pd.DataFrame = None) -> dict:
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)
    if arrivals_df is None:
        arrivals_df = load_clean_sheet(ARRIVAL_SHEET)
    return {variety: analyze_variety(price_df, arrivals_df, variety) for variety in VARIETIES}
