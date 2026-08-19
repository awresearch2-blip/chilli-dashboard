"""Finds historical dates whose feature vector most closely resembles
"now" (standardized Euclidean nearest-neighbor over the same lag/rolling/
calendar features forecasting/feature_engineering.py already builds), and
reports what the price *actually* did over the following `horizon_days` --
every number here is a real historical outcome, never a projection.
"""

import numpy as np
import pandas as pd

from forecasting.feature_engineering import FEATURE_COLUMNS
from forecasting.utils import nearest_value

EXCLUDE_RECENT_DAYS = 120  # avoid trivially "matching" the recent past to itself
MIN_CANDIDATES = 10
TOP_N = 3
MAX_FALLBACK_DAYS = 10  # how far back to look for a complete row if the exact origin's own lag_1 is gapped


def _nearest_complete_row_date(valid_rows_index, origin_date, max_back: int = MAX_FALLBACK_DAYS):
    """The single-day lag_1 feature (unlike the rolling stats) has no
    partial-window fallback -- if the specific date one real observation
    before the origin is itself a gap (a "Closed" mandi day, a data-entry
    miss), the origin's own row is genuinely incomplete. Rather than go
    dark on the single day that matters most (the current forecast), fall
    back to the most recent earlier date that IS complete -- clearly
    labeled as such, never disguised as the exact origin date.
    """
    for days_back in range(0, max_back + 1):
        candidate = origin_date - pd.Timedelta(days=days_back)
        if candidate in valid_rows_index:
            return candidate, days_back
    return None, None


def find_similar_periods(features_df: pd.DataFrame, price_series: pd.Series, origin_date, horizon_days: int, top_n: int = TOP_N) -> dict:
    origin_date = pd.Timestamp(origin_date)
    valid_rows = features_df.dropna(subset=FEATURE_COLUMNS)
    if len(valid_rows) < 50:
        return {"status": "insufficient_evidence", "reason": "Not enough historical feature rows to search"}

    used_date, days_back = _nearest_complete_row_date(valid_rows.index, origin_date)
    if used_date is None:
        return {
            "status": "insufficient_evidence",
            "reason": f"Origin date's own features are incomplete (a recent data gap) and no complete row was found within {MAX_FALLBACK_DAYS} days before it",
        }

    means, stds = valid_rows[FEATURE_COLUMNS].mean(), valid_rows[FEATURE_COLUMNS].std().replace(0, 1)
    standardized = (valid_rows[FEATURE_COLUMNS] - means) / stds
    origin_vec = standardized.loc[used_date]

    last_known_date = price_series.dropna().index[-1]
    eligible = standardized.index[
        (standardized.index <= used_date - pd.Timedelta(days=EXCLUDE_RECENT_DAYS))
        & (standardized.index + pd.Timedelta(days=horizon_days) <= last_known_date)
    ]
    if len(eligible) < MIN_CANDIDATES:
        return {
            "status": "insufficient_evidence",
            "reason": "Not enough eligible historical candidates (need trailing history and a known outcome after horizon_days)",
            "available_n": len(eligible),
            "required_n": MIN_CANDIDATES,
        }

    distances = ((standardized.loc[eligible] - origin_vec) ** 2).sum(axis=1).pow(0.5)
    nearest = distances.nsmallest(top_n)

    periods = []
    for date, distance in nearest.items():
        price_then = price_series.get(date)
        price_after = nearest_value(price_series, date + pd.Timedelta(days=horizon_days), tolerance_days=2)
        if price_then is None or pd.isna(price_then) or price_after is None:
            continue
        periods.append(
            {
                "date": str(date.date()),
                "similarity_distance": round(float(distance), 3),
                "price_then": round(float(price_then), 2),
                "price_after_horizon": round(float(price_after), 2),
                "realized_change_pct": round((price_after - float(price_then)) / float(price_then) * 100, 2),
            }
        )

    if not periods:
        return {"status": "insufficient_evidence", "reason": "No candidate analog periods had a fully known outcome"}

    result = {"status": "ok", "similar_periods": periods}
    if days_back > 0:
        result["note"] = f"Origin date's own features had a recent data gap; used the nearest complete row {days_back} day(s) earlier ({used_date.date()}) instead."
    return result
