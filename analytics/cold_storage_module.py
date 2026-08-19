"""Module 10 -- Cold Storage.

Flagged in Phase 1 as a sparse work-in-progress tracker (6 rows, each
populating only 1-2 of 6 regional columns). That's still true. This module
reports whatever real latest values exist per region but explicitly returns
insufficient_evidence for trend or impact-on-arrivals/price analysis rather
than computing something misleadingly precise from a handful of points.
"""

import pandas as pd

from analytics.data_access import date_indexed_series, load_clean_sheet

SHEET_NAME = "cold storage"
REGIONS = {
    "ap_stock_bags": "Andhra Pradesh",
    "telangana_stock_bags": "Telangana",
    "karnataka_stock_bags": "Karnataka",
    "guntur_stock_bags": "Guntur",
    "warangal_stock_bags": "Warangal",
    "khammam_stock_bags": "Khammam",
}
SPARSE_REASON = (
    "Only a handful of sparse rows exist in this sheet as of the current "
    "workbook (Phase 1 finding) -- not enough real history for trend or "
    "impact-on-arrivals/price analysis."
)


def analyze_region(df: pd.DataFrame, column: str) -> dict:
    s = date_indexed_series(df, column).dropna()
    if s.empty:
        return {"status": "insufficient_evidence", "reason": "No data recorded for this region yet"}
    return {
        "status": "insufficient_evidence_for_trend",
        "latest_known_value": round(float(s.iloc[-1]), 0),
        "latest_known_date": str(pd.Timestamp(s.index[-1]).date()),
        "observations_available": len(s),
        "reason": SPARSE_REASON,
    }


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_clean_sheet(SHEET_NAME)

    per_region = {label: analyze_region(df, column) for column, label in REGIONS.items()}

    return {
        "per_region": per_region,
        "impact_on_arrivals": {"status": "insufficient_evidence", "reason": SPARSE_REASON},
        "impact_on_price": {"status": "insufficient_evidence", "reason": SPARSE_REASON},
        "data_quality_caveats": [SPARSE_REASON],
    }
