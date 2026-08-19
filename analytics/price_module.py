"""Module 1 -- Price Analysis for LCA334 ("334" in the workbook) and Teja,
Guntur market.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet

SHEET_NAME = "Guntur Varietywise daily price"
VARIETIES = ["Teja", "334"]


def variety_series(df: pd.DataFrame, variety: str, metric: str) -> pd.Series:
    return date_indexed_series(df, "value", {"variety": variety, "metric": metric})


def _latest(series: pd.Series):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), 2)


def _caveats(variety: str) -> list:
    notes = []
    if variety == "334":
        notes.append(
            "Referred to as 'LCA334' in the brief; the workbook labels this variety column '334'."
        )
    return notes


def analyze_variety(df: pd.DataFrame, variety: str) -> dict:
    avg = variety_series(df, variety, "avg_price")
    low = variety_series(df, variety, "low_price")
    high = variety_series(df, variety, "high_price")

    if avg.dropna().empty:
        return {"status": "insufficient_evidence", "reason": f"No avg_price data found for variety '{variety}'"}

    latest_date = avg.dropna().index[-1]

    return {
        "status": "ok",
        "as_of": str(pd.Timestamp(latest_date).date()),
        "latest_avg_price": _latest(avg),
        "latest_range": {"low": _latest(low), "high": _latest(high)},
        "cagr": ts.cagr(avg),
        "moving_averages": {
            "sma_7": _latest(ts.sma(avg, 7)),
            "sma_30": _latest(ts.sma(avg, 30)),
            "sma_90": _latest(ts.sma(avg, 90)),
            "ema_7": _latest(ts.ema(avg, 7)),
            "ema_30": _latest(ts.ema(avg, 30)),
        },
        "volatility_cv_30d": _latest(ts.rolling_cv(avg, 30)),
        "drawdown": ts.drawdown(avg),
        "momentum_roc_pct": {
            "roc_7": _latest(ts.momentum_roc(avg, 7)),
            "roc_30": _latest(ts.momentum_roc(avg, 30)),
            "roc_90": _latest(ts.momentum_roc(avg, 90)),
        },
        "trend_strength_90obs": ts.trend_strength(avg, window=90),
        "percentile_distribution": {
            "trailing_1y": ts.percentile_rank(avg, window=365),
            "full_history": ts.percentile_rank(avg),
        },
        "observations_used": int(avg.dropna().shape[0]),
        "data_quality_caveats": _caveats(variety),
    }


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_clean_sheet(SHEET_NAME)
    return {variety: analyze_variety(df, variety) for variety in VARIETIES}
