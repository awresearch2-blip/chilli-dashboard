"""Holt-Winters / ETS via statsmodels. The daily series is resampled to
weekly means internally (fitting an additive-seasonal ETS with a 365-day
season directly on ~2600 daily points is slow and numerically fragile;
52-week seasonality on the resampled series is the standard, stable way to
capture the same yearly pattern). horizon_days is converted to whole weeks
(rounded) -- a documented approximation, not a hidden one.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

WEEKLY_SEASONAL_PERIODS = 52
MIN_WEEKLY_OBS = 2 * WEEKLY_SEASONAL_PERIODS


def fit(train_series: pd.Series):
    weekly = train_series.dropna().resample("W").mean().dropna()
    if len(weekly) < MIN_WEEKLY_OBS:
        return None
    try:
        model = ExponentialSmoothing(
            weekly, trend="add", seasonal="add", seasonal_periods=WEEKLY_SEASONAL_PERIODS, initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)
    except Exception:
        return None
    return {"fitted": fitted, "last_date": weekly.index[-1]}


def predict(fitted_state, origin_date, horizon_days: int):
    if fitted_state is None:
        return None
    steps_ahead = max(1, round(horizon_days / 7))
    try:
        forecast = fitted_state["fitted"].forecast(steps_ahead)
        point = float(forecast.iloc[-1])
    except Exception:
        return None

    lower, upper = None, None
    try:
        sims = fitted_state["fitted"].simulate(nsimulations=steps_ahead, repetitions=200, error="add")
        final_step = sims.iloc[-1]
        lower, upper = float(final_step.quantile(0.05)), float(final_step.quantile(0.95))
    except Exception:
        pass

    return {"point": point, "lower": lower, "upper": upper}
