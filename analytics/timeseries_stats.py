"""Generic, reusable statistical primitives shared by every analytics module.

Two kinds of function here:
- Transformations (sma, ema, rolling_cv, momentum_roc, resample_agg, ...)
  return a pandas Series/DataFrame. Insufficient history shows up as NaN at
  the affected points (via min_periods) -- that's the honest, standard
  pandas behavior, not something to paper over.
- Summary statistics that produce a single number from a whole series or
  pair of series (cagr, correlations, elasticity, seasonal_index, ...) take
  a `min_n` and return an explicit {"status": "insufficient_evidence", ...}
  shape below that threshold, instead of a precise-looking number computed
  from too few real observations.

Nothing here imputes, interpolates, or estimates a missing value. Every
function operates on `.dropna()`'d data unless documented otherwise.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _insufficient(reason: str, available_n: int = None, required_n: int = None) -> dict:
    result = {"status": "insufficient_evidence", "reason": reason}
    if available_n is not None:
        result["available_n"] = int(available_n)
    if required_n is not None:
        result["required_n"] = int(required_n)
    return result


def cagr(series: pd.Series, min_n: int = 30) -> dict:
    s = series.dropna()
    if len(s) < min_n:
        return _insufficient("Not enough valid observations to compute a reliable CAGR", len(s), min_n)
    start_date, end_date = s.index[0], s.index[-1]
    start_value, end_value = float(s.iloc[0]), float(s.iloc[-1])
    years = (end_date - start_date).days / 365.25
    if years <= 0 or start_value <= 0:
        return _insufficient("Non-positive start value or non-positive time span", len(s))
    value = (end_value / start_value) ** (1 / years) - 1
    return {
        "status": "ok",
        "cagr_pct": round(value * 100, 2),
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "start_value": start_value,
        "end_value": end_value,
        "years_spanned": round(years, 2),
    }


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_cv(series: pd.Series, window: int) -> pd.Series:
    """Rolling coefficient of variation (std/mean) -- volatility measured in a
    way appropriate for a commodity mandi price, not an equity annualized-vol
    formula that assumes daily exchange trading."""
    roll = series.rolling(window, min_periods=window)
    mean = roll.mean()
    std = roll.std()
    return (std / mean).where(mean != 0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """How many rolling-std-deviations the current value is from its own
    rolling mean -- used to flag "unusually wide relative to its own recent
    history", not any absolute notion of normal."""
    roll = series.rolling(window, min_periods=window)
    mean = roll.mean()
    std = roll.std()
    return ((series - mean) / std).where(std != 0)


def drawdown(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return _insufficient("No valid observations", 0)
    running_max = s.cummax()
    dd = (s - running_max) / running_max
    max_dd_date = dd.idxmin()
    max_dd_value = float(dd.loc[max_dd_date])
    peak_date = s.loc[:max_dd_date].idxmax()

    recovered = False
    recovery_date = None
    peak_value = float(s.loc[peak_date])
    after_trough = s.loc[max_dd_date:]
    recovered_mask = after_trough >= peak_value
    if recovered_mask.any():
        recovered = True
        recovery_date = recovered_mask.idxmax()

    result = {
        "status": "ok",
        "max_drawdown_pct": round(max_dd_value * 100, 2),
        "peak_date": str(pd.Timestamp(peak_date).date()),
        "trough_date": str(pd.Timestamp(max_dd_date).date()),
        "recovered": recovered,
    }
    if recovered:
        result["recovery_date"] = str(pd.Timestamp(recovery_date).date())
        result["days_to_recover"] = int((recovery_date - max_dd_date).days)
    return result


def momentum_roc(series: pd.Series, window: int) -> pd.Series:
    """Rate of change over `window` valid observations (not calendar days --
    gaps in the source series mean this isn't always a fixed calendar span)."""
    s = series.dropna()
    return ((s - s.shift(window)) / s.shift(window)).reindex(series.index)


def trend_strength(series: pd.Series, window: int, min_n: int = 10) -> dict:
    s = series.dropna().tail(window)
    if len(s) < min_n:
        return _insufficient("Not enough valid observations in the lookback window", len(s), min_n)
    x = np.arange(len(s))
    y = s.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    slope_pct_per_period = (slope / y.mean()) * 100 if y.mean() != 0 else 0.0

    if r_squared >= 0.3 and slope_pct_per_period > 0.05:
        regime = "uptrend"
    elif r_squared >= 0.3 and slope_pct_per_period < -0.05:
        regime = "downtrend"
    else:
        regime = "range_bound"

    return {
        "status": "ok",
        "slope_pct_per_period": round(slope_pct_per_period, 4),
        "r_squared": round(float(r_squared), 3),
        "regime": regime,
        "observations_used": len(s),
    }


def percentile_rank(series: pd.Series, value: float = None, window: int = None) -> dict:
    s = series.dropna()
    if window is not None:
        s = s.tail(window)
    if s.empty:
        return _insufficient("No valid observations", 0)
    if value is None:
        value = float(s.iloc[-1])
    rank = float((s <= value).mean() * 100)
    return {
        "status": "ok",
        "value": value,
        "percentile": round(rank, 1),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
        "observations_used": len(s),
    }


def resample_agg(series: pd.Series, freq: str, how: str = "mean") -> pd.Series:
    resampler = series.dropna().resample(freq)
    return resampler.sum() if how == "sum" else resampler.mean()


def yoy_growth(period_series: pd.Series) -> pd.Series:
    return period_series.pct_change(periods=12) * 100


def mom_growth(period_series: pd.Series) -> pd.Series:
    return period_series.pct_change(periods=1) * 100


def _align(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return df


def _constant_input_reason(df: pd.DataFrame) -> str:
    if df["x"].nunique() <= 1:
        return "'x' series is constant (zero variance) over the aligned period -- correlation is undefined"
    if df["y"].nunique() <= 1:
        return "'y' series is constant (zero variance) over the aligned period -- correlation is undefined"
    return None


def pearson_corr(x: pd.Series, y: pd.Series, min_n: int = 30) -> dict:
    df = _align(x, y)
    if len(df) < min_n:
        return _insufficient("Not enough aligned, valid observation pairs", len(df), min_n)
    constant_reason = _constant_input_reason(df)
    if constant_reason:
        return _insufficient(constant_reason, len(df))
    r, p = scipy_stats.pearsonr(df["x"], df["y"])
    return {"status": "ok", "r": round(float(r), 4), "p_value": round(float(p), 4), "n": len(df)}


def spearman_corr(x: pd.Series, y: pd.Series, min_n: int = 30) -> dict:
    df = _align(x, y)
    if len(df) < min_n:
        return _insufficient("Not enough aligned, valid observation pairs", len(df), min_n)
    constant_reason = _constant_input_reason(df)
    if constant_reason:
        return _insufficient(constant_reason, len(df))
    r, p = scipy_stats.spearmanr(df["x"], df["y"])
    return {"status": "ok", "r": round(float(r), 4), "p_value": round(float(p), 4), "n": len(df)}


def rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    df = _align(x, y)
    return df["x"].rolling(window, min_periods=window).corr(df["y"])


def lag_correlogram(x: pd.Series, y: pd.Series, max_lag: int, min_n: int = 30) -> dict:
    """Correlation between x_t and y_{t+lag} for lag in [-max_lag, max_lag].
    A positive best_lag means y leads x by that many observations; negative
    means x leads y.
    """
    df = _align(x, y)
    if len(df) < min_n:
        return _insufficient("Not enough aligned, valid observation pairs", len(df), min_n)

    lags, correlations = [], []
    for lag in range(-max_lag, max_lag + 1):
        shifted = pd.concat([df["x"], df["y"].shift(-lag)], axis=1).dropna()
        if len(shifted) < min_n:
            continue
        r = shifted.iloc[:, 0].corr(shifted.iloc[:, 1])
        lags.append(lag)
        correlations.append(None if pd.isna(r) else round(float(r), 4))

    if not lags:
        return _insufficient("No lag offset retained enough aligned pairs", len(df), min_n)

    valid = [(l, c) for l, c in zip(lags, correlations) if c is not None]
    best_lag, best_corr = max(valid, key=lambda pair: abs(pair[1])) if valid else (None, None)
    return {
        "status": "ok",
        "lags": lags,
        "correlations": correlations,
        "best_lag": best_lag,
        "best_corr": best_corr,
    }


def log_log_elasticity(x: pd.Series, y: pd.Series, min_n: int = 30) -> dict:
    """OLS slope of ln(y) on ln(x): the %-change in y associated with a 1%
    change in x. Rows with non-positive values are dropped (logs undefined)."""
    df = _align(x, y)
    df = df[(df["x"] > 0) & (df["y"] > 0)]
    if len(df) < min_n:
        return _insufficient("Not enough positive-valued aligned pairs for a log-log regression", len(df), min_n)
    log_x, log_y = np.log(df["x"].to_numpy()), np.log(df["y"].to_numpy())
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = slope * log_x + intercept
    ss_res = np.sum((log_y - fitted) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "status": "ok",
        "elasticity": round(float(slope), 4),
        "r_squared": round(float(r_squared), 3),
        "n": len(df),
    }


def seasonal_index(series: pd.Series, min_years: int = 2) -> dict:
    """Classic ratio-to-centered-moving-average seasonal index. Returns a
    factor per calendar month where 100 = neutral (no seasonal effect),
    normalized so the 12 factors average to 100. Any month with fewer than
    `min_years` contributing years is reported but flagged individually.
    """
    monthly = series.dropna().resample("MS").mean()
    if len(monthly) < 24:
        return _insufficient("Need at least 24 months of history to compute a seasonal index", len(monthly), 24)

    trend = monthly.rolling(12, center=True, min_periods=12).mean()
    ratio = (monthly / trend).dropna()
    if ratio.empty:
        return _insufficient("No valid trend-relative ratios could be computed", len(monthly))

    by_month = ratio.groupby(ratio.index.month)
    raw_factor = by_month.mean()
    years_used = by_month.apply(lambda s: s.index.year.nunique())

    normalization = 100.0 / raw_factor.mean()
    index_by_month = {}
    insufficient_months = []
    for month in range(1, 13):
        if month not in raw_factor.index:
            insufficient_months.append(month)
            continue
        n_years = int(years_used.loc[month])
        entry = {"index": round(float(raw_factor.loc[month] * normalization), 1), "years_used": n_years}
        if n_years < min_years:
            insufficient_months.append(month)
        index_by_month[month] = entry

    return {
        "status": "ok",
        "index_by_month": index_by_month,
        "months_with_insufficient_evidence": insufficient_months,
    }
