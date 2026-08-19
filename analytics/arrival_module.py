"""Module 2 -- Arrival Analysis, Guntur market (total, all varieties combined --
this sheet doesn't break arrivals down by variety)."""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet

SHEET_NAME = "Guntur Daily arrivals"
FIELDS = {"arrivals_bags": "arrivals_bags", "offtake_bags": "offtake_bags"}


def _insufficient(reason, available_n=None, required_n=None):
    result = {"status": "insufficient_evidence", "reason": reason}
    if available_n is not None:
        result["available_n"] = int(available_n)
    if required_n is not None:
        result["required_n"] = int(required_n)
    return result


def field_series(df: pd.DataFrame, column: str) -> pd.Series:
    return date_indexed_series(df, column)


def _latest(series: pd.Series):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), 1)


def _peak_lean_months(monthly_sum: pd.Series, min_years: int = 2) -> dict:
    monthly_sum = monthly_sum.dropna()
    if monthly_sum.empty:
        return _insufficient("No monthly arrival totals available")
    grouped = monthly_sum.groupby(monthly_sum.index.month)
    means = grouped.mean()
    years_used = grouped.apply(lambda s: s.index.year.nunique())
    if (years_used < min_years).all():
        return _insufficient("Fewer than the minimum years of history in every calendar month", int(years_used.max()), min_years)
    ranked = means.sort_values(ascending=False)
    return {
        "status": "ok",
        "monthly_avg_arrivals_bags": {int(m): round(float(v), 0) for m, v in means.items()},
        "peak_months": [int(m) for m in ranked.index[:3]],
        "lean_months": [int(m) for m in ranked.index[-3:]],
        "years_used_per_month": {int(m): int(y) for m, y in years_used.items()},
    }


def analyze_field(df: pd.DataFrame, column: str) -> dict:
    daily = field_series(df, column)
    if daily.dropna().empty:
        return {"status": "insufficient_evidence", "reason": f"No data found for '{column}'"}

    weekly = ts.resample_agg(daily, "W", how="sum")
    monthly = ts.resample_agg(daily, "MS", how="sum")

    return {
        "status": "ok",
        "as_of": str(pd.Timestamp(daily.dropna().index[-1]).date()),
        "latest_daily": _latest(daily),
        "latest_weekly_total": _latest(weekly),
        "latest_monthly_total": _latest(monthly),
        "rolling_30d_mean": _latest(daily.rolling(30, min_periods=30).mean()),
        "rolling_90d_mean": _latest(daily.rolling(90, min_periods=90).mean()),
        "volatility_cv_30d": _latest(ts.rolling_cv(daily, 30)),
        "yoy_growth_pct_latest": _latest(ts.yoy_growth(monthly)),
        "mom_growth_pct_latest": _latest(ts.mom_growth(monthly)),
        "seasonal_peak_lean": _peak_lean_months(monthly),
        "observations_used": int(daily.dropna().shape[0]),
    }


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_clean_sheet(SHEET_NAME)
    return {label: analyze_field(df, column) for label, column in FIELDS.items()}
