"""Module 8 -- Balance Sheet.

At most ~9 real annual observations (2017-2025) exist per category -- the
default min_n=30 used elsewhere would make every correlation here
permanently insufficient_evidence. This module deliberately uses a lower,
documented min_n=6 instead, and always reports n alongside r so the reader
can judge reliability themselves; a standing caveat says these are
directional signals only, never statistically robust at this sample size.

The workbook's own "2026(exp)" column is its own forecast, not an actual --
excluded from every trend/correlation calculation and reported separately.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import load_clean_sheet
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series

BALANCE_SHEET = "Red Chilli Balance sheet"
ANNUAL_MIN_N = 6
ESTIMATE_YEAR = "2026(exp)"

ANNUAL_CORRELATION_CAVEAT = (
    "Balance sheet correlations are based on at most ~9 annual observations "
    "-- always reported with n alongside r. These are directional signals "
    "only, never treated as statistically robust at this sample size."
)


def _category_series(df: pd.DataFrame, category: str) -> pd.Series:
    subset = df[(df["category"] == category) & (~df["is_estimate"])].copy()
    subset["year_int"] = pd.to_numeric(subset["year"], errors="coerce")
    subset = subset.dropna(subset=["year_int"])
    idx = pd.to_datetime(subset["year_int"].astype(int).astype(str) + "-01-01")
    s = pd.Series(subset["value_lakh_tons"].to_numpy(), index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _estimate_value(df: pd.DataFrame, category: str):
    row = df[(df["category"] == category) & (df["is_estimate"])]
    if row.empty or pd.isna(row["value_lakh_tons"].iloc[0]):
        return None
    return round(float(row["value_lakh_tons"].iloc[0]), 2)


def _series_to_annual_dict(s: pd.Series) -> dict:
    return {str(idx.year): round(float(v), 2) for idx, v in s.items() if pd.notna(v)}


def analyze_category(df: pd.DataFrame, category: str, annual_price_by_variety: dict) -> dict:
    s = _category_series(df, category)
    if s.dropna().empty:
        return {"status": "insufficient_evidence", "reason": f"No actual (non-estimate) data for '{category}'"}

    yoy_latest = None
    if len(s) > 1:
        pct = s.pct_change().iloc[-1]
        yoy_latest = round(float(pct) * 100, 2) if pd.notna(pct) else None

    correlation_with_price = {
        variety: ts.pearson_corr(s, annual_price, min_n=ANNUAL_MIN_N)
        for variety, annual_price in annual_price_by_variety.items()
    }

    return {
        "status": "ok",
        "annual_values_lakh_tons": _series_to_annual_dict(s),
        "latest_actual_year": s.index[-1].year,
        "latest_actual_value": round(float(s.iloc[-1]), 2),
        "estimate_2026": _estimate_value(df, category),
        "cagr": ts.cagr(s, min_n=ANNUAL_MIN_N),
        "trend": ts.trend_strength(s, window=len(s), min_n=5),
        "yoy_change_pct_latest": yoy_latest,
        "correlation_with_price": correlation_with_price,
    }


def _surplus_deficit(df: pd.DataFrame) -> dict:
    supply = _category_series(df, "Total Supply")
    demand = _category_series(df, "Total Demand/Usage")
    aligned = pd.concat([supply.rename("supply"), demand.rename("demand")], axis=1).dropna()
    if aligned.empty:
        return {"status": "insufficient_evidence", "reason": "Total Supply / Total Demand/Usage not both available"}
    surplus_deficit = aligned["supply"] - aligned["demand"]
    return {
        "status": "ok",
        "annual_values_lakh_tons": _series_to_annual_dict(surplus_deficit),
        "latest": round(float(surplus_deficit.iloc[-1]), 2),
    }


def run(balance_df: pd.DataFrame = None, price_df: pd.DataFrame = None) -> dict:
    if balance_df is None:
        balance_df = load_clean_sheet(BALANCE_SHEET)
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)

    annual_price_by_variety = {}
    for variety in VARIETIES:
        price_series = variety_series(price_df, variety, "avg_price")
        if not price_series.dropna().empty:
            annual_price_by_variety[variety] = ts.resample_agg(price_series, "YS", how="mean")

    categories = sorted(balance_df["category"].dropna().unique())
    per_category = {cat: analyze_category(balance_df, cat, annual_price_by_variety) for cat in categories}

    return {
        "per_category": per_category,
        "surplus_deficit_total_supply_minus_demand": _surplus_deficit(balance_df),
        "estimate_year_excluded_from_actuals": ESTIMATE_YEAR,
        "data_quality_caveats": [ANNUAL_CORRELATION_CAVEAT],
    }
