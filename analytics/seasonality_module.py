"""Module 4 -- Seasonality, Guntur market.

Important finding baked into this module: the workbook's own "Seasonality
index_Guntur Teja" sheet does NOT contain a normalized seasonal index
despite its name -- its values match the plain monthly-average Teja price
almost exactly (e.g. Jun-2015 sheet value 9629 vs actual daily-average
9629.41; Dec-2020: 15752 vs 15752.38). It's monthly average price levels,
not a deseasonalized 100=neutral index. This module reports that sheet's
values as-is for reference (never silently "corrects" real workbook data),
but computes its own true normalized seasonal index independently from
daily prices rather than treating the sheet's label as authoritative --
this is exactly the kind of thing the brief's "never fabricate seasonality"
rule is meant to guard against: assuming a label describes its content
without checking.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.arrival_module import SHEET_NAME as ARRIVAL_SHEET
from analytics.arrival_module import field_series
from analytics.data_access import load_clean_sheet
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series

WORKBOOK_SEASONALITY_SHEET = "Seasonality index_Guntur Teja"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

WORKBOOK_SHEET_CAVEAT = (
    "The workbook's own 'Seasonality index_Guntur Teja' sheet contains "
    "monthly average Teja PRICE LEVELS by year, not a normalized seasonal "
    "index, despite its name (verified: its values match the actual daily "
    "average almost exactly for the same month/year). Reported below as-is "
    "for reference under 'workbook_reported_monthly_avg_by_month' -- this "
    "module's own 'seasonal_price_index' is the actual normalized index."
)


def _workbook_reference(seas_df: pd.DataFrame) -> dict:
    if seas_df.empty or "seasonality_index" not in seas_df.columns:
        return {}
    grouped = seas_df.dropna(subset=["seasonality_index"]).groupby("period_label")["seasonality_index"].mean()
    return {m: round(float(grouped[m]), 1) for m in MONTH_ORDER if m in grouped.index}


def _yoy_seasonal_comparison(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"status": "insufficient_evidence", "reason": "No valid observations"}
    monthly = s.resample("MS").mean()
    if len(monthly) < 13:
        return {"status": "insufficient_evidence", "reason": "Need at least 13 months of history", "available_n": len(monthly), "required_n": 13}

    latest_12 = monthly.tail(12)
    comparisons = {}
    for period, value in latest_12.items():
        historical = monthly[(monthly.index.month == period.month) & (monthly.index < period)]
        if historical.empty:
            continue
        hist_mean = float(historical.mean())
        comparisons[str(period.date())] = {
            "actual": round(float(value), 2),
            "historical_avg_same_month": round(hist_mean, 2),
            "deviation_pct": round((value - hist_mean) / hist_mean * 100, 2) if hist_mean else None,
            "years_of_history": int(historical.shape[0]),
        }
    if not comparisons:
        return {"status": "insufficient_evidence", "reason": "No prior-year history available for any of the last 12 months"}
    return {"status": "ok", "monthly_vs_historical": comparisons}


def _harvest_vs_offseason(arrival_index: dict, price_index_by_variety: dict) -> dict:
    if arrival_index.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": "Arrival seasonal index unavailable"}
    by_month = arrival_index["index_by_month"]
    ranked = sorted(by_month.items(), key=lambda kv: kv[1]["index"], reverse=True)
    harvest_months = [m for m, _ in ranked[:3]]
    offseason_months = [m for m, _ in ranked[-3:]]

    price_behavior = {}
    for variety, price_result in price_index_by_variety.items():
        if price_result.get("status") != "ok":
            continue
        price_by_month = price_result["seasonal_price_index"]["index_by_month"]
        harvest_vals = [price_by_month[m]["index"] for m in harvest_months if m in price_by_month]
        offseason_vals = [price_by_month[m]["index"] for m in offseason_months if m in price_by_month]
        if harvest_vals and offseason_vals:
            price_behavior[variety] = {
                "avg_price_index_during_harvest_months": round(sum(harvest_vals) / len(harvest_vals), 1),
                "avg_price_index_during_offseason_months": round(sum(offseason_vals) / len(offseason_vals), 1),
            }

    return {
        "status": "ok",
        "harvest_months": harvest_months,
        "offseason_months": offseason_months,
        "note": "Harvest/off-season defined by arrival volume (highest/lowest seasonal arrival index), not price.",
        "price_behavior_by_variety": price_behavior,
    }


def run(price_df: pd.DataFrame = None, arrivals_df: pd.DataFrame = None, seasonality_sheet_df: pd.DataFrame = None) -> dict:
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)
    if arrivals_df is None:
        arrivals_df = load_clean_sheet(ARRIVAL_SHEET)
    if seasonality_sheet_df is None:
        seasonality_sheet_df = load_clean_sheet(WORKBOOK_SEASONALITY_SHEET)

    price_seasonality = {}
    for variety in VARIETIES:
        series = variety_series(price_df, variety, "avg_price")
        if series.dropna().empty:
            price_seasonality[variety] = {"status": "insufficient_evidence", "reason": f"No price data for '{variety}'"}
            continue
        entry = {
            "status": "ok",
            "seasonal_price_index": ts.seasonal_index(series),
            "yoy_seasonal_comparison": _yoy_seasonal_comparison(series),
        }
        if variety == "Teja":
            entry["workbook_reported_monthly_avg_by_month"] = _workbook_reference(seasonality_sheet_df)
            entry["data_quality_caveats"] = [WORKBOOK_SHEET_CAVEAT]
        price_seasonality[variety] = entry

    arrivals_series = field_series(arrivals_df, "arrivals_bags")
    arrival_seasonality = (
        {"status": "insufficient_evidence", "reason": "No arrivals data"}
        if arrivals_series.dropna().empty
        else {"status": "ok", "seasonal_arrival_index": ts.seasonal_index(arrivals_series)}
    )

    arrival_index_for_harvest = arrival_seasonality.get("seasonal_arrival_index", {"status": "insufficient_evidence"})
    harvest_vs_offseason = _harvest_vs_offseason(arrival_index_for_harvest, price_seasonality)

    return {
        "price_seasonality": price_seasonality,
        "arrival_seasonality": arrival_seasonality,
        "harvest_vs_offseason": harvest_vs_offseason,
    }
