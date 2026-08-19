"""Seasonal Naive: forecast = the observed value one seasonal cycle (1 year)
before the target date. The trivial baseline every other model must beat."""

import pandas as pd

from forecasting.utils import nearest_value

SEASONAL_PERIOD_DAYS = 365


def fit(train_series: pd.Series):
    series = train_series.dropna()
    if series.empty:
        return None
    return {"series": series}


def predict(fitted_state, origin_date, horizon_days: int):
    if fitted_state is None:
        return None
    target_date = pd.Timestamp(origin_date) + pd.Timedelta(days=horizon_days)
    lookup_date = target_date - pd.Timedelta(days=SEASONAL_PERIOD_DAYS)
    point = nearest_value(fitted_state["series"], lookup_date, tolerance_days=3)
    if point is None:
        return None
    return {"point": point, "lower": None, "upper": None}
