"""Small shared helpers for the forecasting engine."""

import numpy as np
import pandas as pd


def nearest_value(series: pd.Series, target_date, tolerance_days: int = 3):
    """The observation closest to `target_date` within `tolerance_days`, or
    None. Used both to look up backtest "actuals" and by Seasonal Naive to
    find "the same date last cycle" when that exact date has a gap."""
    target_date = pd.Timestamp(target_date)
    mask = (series.index >= target_date - pd.Timedelta(days=tolerance_days)) & (
        series.index <= target_date + pd.Timedelta(days=tolerance_days)
    )
    window = series[mask].dropna()
    if window.empty:
        return None
    deltas = np.abs((window.index - target_date).days)
    return float(window.iloc[int(np.argmin(deltas))])


def rmse(errors) -> float:
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors) -> float:
    return float(np.mean(np.abs(np.asarray(errors, dtype=float))))


def mape(actuals, preds) -> float:
    actuals, preds = np.asarray(actuals, dtype=float), np.asarray(preds, dtype=float)
    return float(np.mean(np.abs((actuals - preds) / actuals)) * 100)


def smape(actuals, preds) -> float:
    actuals, preds = np.asarray(actuals, dtype=float), np.asarray(preds, dtype=float)
    denom = (np.abs(actuals) + np.abs(preds)) / 2
    return float(np.mean(np.abs(actuals - preds) / denom) * 100)
