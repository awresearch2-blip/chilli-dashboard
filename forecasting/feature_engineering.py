"""Builds the ML model's feature matrix -- used only by the XGBoost model;
the classical models (Seasonal Naive, Holt-Winters, SARIMA) forecast
directly from the raw price series.

Every feature at row t is built only from information available at or
before t -- no future leakage. The target (added separately per horizon via
`add_target`) is the only forward-looking value, and it's looked up by
*exact calendar date* (t + horizon_days), not by row position, since the
price series has occasional gaps and "the price 30 days from now" must mean
30 calendar days, not 30 rows.
"""

import pandas as pd

PRICE_LAGS = [1, 7, 14, 30, 90]
ROLLING_WINDOWS = [7, 30, 90]
HARVEST_MONTHS = {2, 3, 4}  # Guntur's highest-arrival months per Phase 2b's seasonal arrival index

FEATURE_COLUMNS = (
    ["price_lag_0"]
    + [f"price_lag_{lag}" for lag in PRICE_LAGS]
    + [f"price_rollmean_{w}" for w in ROLLING_WINDOWS]
    + [f"price_rollstd_{w}" for w in ROLLING_WINDOWS]
    + ["month", "day_of_year", "is_harvest_month", "arrivals_lag_1", "arrivals_rollmean_30", "fx_lag_1", "fx_rollmean_30"]
)


def _min_periods(window: int) -> int:
    """A single scattered gap (a "Closed" mandi day, a data-entry miss) is a
    permanent, expected feature of this workbook (Phase 1 finding), not a
    rare edge case -- requiring a completely full window (pandas' rolling()
    default) means one gap silently disables the rolling stat for the next
    `window` rows, including possibly the most recent (most important) row.
    Tolerating a partial window is still computed purely from real observed
    values -- nothing is imputed -- just not thrown away over one missing day."""
    return max(3, window // 2)


def build_feature_frame(price_series: pd.Series, arrivals_series: pd.Series, fx_series: pd.Series) -> pd.DataFrame:
    """NOTE: every real caller always passes real arrivals_series/fx_series --
    Guntur Daily arrivals and USD/INR are permanent sheets in this workbook.
    The `is not None` branches below are defensive, but they degrade to
    an all-NaN column, not an all-NaN-free frame missing those columns --
    every row then fails any consumer's `dropna(subset=FEATURE_COLUMNS)`
    (XGBoost training, similar_periods search). That's fine today because
    this path is never actually exercised, but it means "pass None" is not
    a working way to run without exogenous data -- don't rely on it.
    """
    df = pd.DataFrame(index=price_series.index)
    df["price_lag_0"] = price_series
    for lag in PRICE_LAGS:
        df[f"price_lag_{lag}"] = price_series.shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = price_series.shift(1)
        df[f"price_rollmean_{window}"] = shifted.rolling(window, min_periods=_min_periods(window)).mean()
        df[f"price_rollstd_{window}"] = shifted.rolling(window, min_periods=_min_periods(window)).std()

    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_harvest_month"] = df["month"].isin(HARVEST_MONTHS).astype(int)

    arrivals_aligned = arrivals_series.reindex(df.index) if arrivals_series is not None else None
    if arrivals_aligned is not None:
        shifted_arrivals = arrivals_aligned.shift(1)
        df["arrivals_lag_1"] = shifted_arrivals
        df["arrivals_rollmean_30"] = shifted_arrivals.rolling(30, min_periods=_min_periods(30)).mean()
    else:
        df["arrivals_lag_1"] = None
        df["arrivals_rollmean_30"] = None

    # USD/INR has its own, different trading calendar than the mandi price
    # sheet (2277 FX dates vs ~2621 price dates) -- a plain reindex leaves
    # most rows without a same-day FX quote, which cascades into ~75% of
    # 30-day rolling windows (and even some individual lag values, including
    # at the most recent date) coming out NaN. Forward-filling is the
    # correct fix here, not a data-quality compromise: a currency rate
    # genuinely persists across a day with no fresh quote, so "yesterday's
    # last known rate" is real, causally-valid information as of that date
    # -- unlike price or arrivals, where a gap usually means "the mandi was
    # closed that day" and there is no real value to carry forward.
    fx_aligned = fx_series.reindex(df.index).ffill() if fx_series is not None else None
    if fx_aligned is not None:
        shifted_fx = fx_aligned.shift(1)
        df["fx_lag_1"] = shifted_fx
        df["fx_rollmean_30"] = shifted_fx.rolling(30).mean()
    else:
        df["fx_lag_1"] = None
        df["fx_rollmean_30"] = None

    return df


def add_target(features_df: pd.DataFrame, price_series: pd.Series, horizon_days: int) -> pd.DataFrame:
    """Target = the actual price at (row date + horizon_days calendar days),
    looked up by exact date -- rows where that exact date doesn't exist in
    the price series (a gap) get NaN and are dropped by the caller, not
    approximated."""
    shifted_dates = features_df.index + pd.Timedelta(days=horizon_days)
    target = price_series.reindex(shifted_dates)
    target.index = features_df.index
    return features_df.assign(target=target)
