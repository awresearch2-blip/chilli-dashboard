"""Module 7 -- Export Analysis.

The workbook never labels the export figure's unit (Phase 1 finding) --
carried through here as a standing caveat rather than guessed at.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series

EXPORT_SHEET = "Red chilli exports"

UNIT_CAVEAT = "Export volume's unit is not labeled anywhere in the workbook -- treated as the workbook's native unit, not guessed at."


def _latest(series: pd.Series, ndigits: int = 2):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), ndigits)


def analyze_vs_price(export_series: pd.Series, price_series: pd.Series) -> dict:
    monthly_price = ts.resample_agg(price_series, "MS", how="mean")
    aligned_n = pd.concat([export_series.rename("e"), monthly_price.rename("p")], axis=1).dropna().shape[0]
    return {
        "pearson_correlation": ts.pearson_corr(export_series, monthly_price),
        "spearman_correlation": ts.spearman_corr(export_series, monthly_price),
        "lag_correlogram_6mo": ts.lag_correlogram(export_series, monthly_price, max_lag=6),
        "aligned_months_used": aligned_n,
    }


def run(exports_df: pd.DataFrame = None, price_df: pd.DataFrame = None) -> dict:
    if exports_df is None:
        exports_df = load_clean_sheet(EXPORT_SHEET)
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)

    export_series = date_indexed_series(exports_df, "export_volume")
    if export_series.dropna().empty:
        return {"status": "insufficient_evidence", "reason": "No export volume data"}

    result = {
        "status": "ok",
        "cagr": ts.cagr(export_series),
        "yoy_growth_pct_latest": _latest(ts.yoy_growth(export_series)),
        "mom_growth_pct_latest": _latest(ts.mom_growth(export_series)),
        "seasonal_export_index": ts.seasonal_index(export_series),
        "vs_price": {},
        "observations_used": int(export_series.dropna().shape[0]),
        "data_quality_caveats": [UNIT_CAVEAT],
    }

    for variety in VARIETIES:
        price_series = variety_series(price_df, variety, "avg_price")
        if price_series.dropna().empty:
            result["vs_price"][variety] = {"status": "insufficient_evidence", "reason": f"No price data for '{variety}'"}
            continue
        result["vs_price"][variety] = analyze_vs_price(export_series, price_series)

    return result
