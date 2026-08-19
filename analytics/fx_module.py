"""Module 9 -- USD/INR exchange rate."""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series

FX_SHEET = "USD to INR exchange rate"

TREND_CORRELATION_CAVEAT = (
    "Both USD/INR and chilli prices have trended upward for most of this "
    "12-year history -- the full-history correlation above is partly (or "
    "mostly) a shared-trend artifact, not necessarily real short-term "
    "co-movement. The rolling and lag correlations are more informative "
    "about actual sensitivity than the single full-sample r."
)


def _latest(series: pd.Series, ndigits: int = 4):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), ndigits)


def analyze_vs_price(fx_series: pd.Series, price_series: pd.Series) -> dict:
    return {
        "pearson_correlation": ts.pearson_corr(fx_series, price_series),
        "spearman_correlation": ts.spearman_corr(fx_series, price_series),
        "rolling_90d_correlation_latest": _latest(ts.rolling_corr(fx_series, price_series, 90)),
        "lag_correlogram_30d": ts.lag_correlogram(fx_series, price_series, max_lag=30),
        "price_elasticity_wrt_usd_inr": ts.log_log_elasticity(fx_series, price_series),
        "data_quality_caveats": [TREND_CORRELATION_CAVEAT],
    }


def run(fx_df: pd.DataFrame = None, price_df: pd.DataFrame = None) -> dict:
    if fx_df is None:
        fx_df = load_clean_sheet(FX_SHEET)
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)

    fx_series = date_indexed_series(fx_df, "usd_inr_rate")
    if fx_series.dropna().empty:
        return {"status": "insufficient_evidence", "reason": "No USD/INR data"}

    result = {
        "status": "ok",
        "as_of": str(pd.Timestamp(fx_series.dropna().index[-1]).date()),
        "latest_rate": _latest(fx_series),
        "cagr": ts.cagr(fx_series),
        "moving_averages": {
            "sma_30": _latest(fx_series.rolling(30, min_periods=30).mean()),
            "sma_90": _latest(fx_series.rolling(90, min_periods=90).mean()),
            "ema_30": _latest(fx_series.ewm(span=30, adjust=False, min_periods=30).mean()),
        },
        "vs_price": {},
    }

    for variety in VARIETIES:
        price_series = variety_series(price_df, variety, "avg_price")
        if price_series.dropna().empty:
            result["vs_price"][variety] = {"status": "insufficient_evidence", "reason": f"No price data for '{variety}'"}
            continue
        result["vs_price"][variety] = analyze_vs_price(fx_series, price_series)

    return result
