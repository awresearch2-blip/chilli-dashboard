"""SARIMA via statsmodels SARIMAX, fit directly on the daily series.

Order is chosen by a small self-implemented grid search over a handful of
documented (p,d,q)(P,D,Q,s) candidates, selected by AIC on the training
data -- not pmdarima/auto_arima (avoids a dependency with a history of
Windows/numpy version friction, especially on a very new Python). Seasonal
period is 7 (weekly) for tractable fitting speed on a long daily series;
yearly seasonality is instead captured by Holt-Winters and the ML model's
calendar features, so SARIMA isn't asked to do that job alone.

The series is reindexed to an explicit daily frequency (`asfreq("D")`)
before fitting: SARIMAX needs a fixed-frequency index, and statsmodels'
Kalman filter handles the resulting few NaN gap-days as missing
observations, not as invented values.
"""

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

CANDIDATE_ORDERS = [
    ((1, 1, 1), (0, 1, 1, 7)),
    ((2, 1, 0), (1, 1, 0, 7)),
    ((1, 1, 0), (0, 1, 1, 7)),
    ((0, 1, 1), (1, 1, 0, 7)),
]
MIN_TRAIN_OBS = 200
MAX_TRAIN_OBS = 1500  # cap for fitting speed; still several years of daily history


def fit(train_series: pd.Series):
    series = train_series.dropna()
    if len(series) < MIN_TRAIN_OBS:
        return None
    series = series.tail(MAX_TRAIN_OBS).asfreq("D")

    best = None
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
            if best is None or result.aic < best["aic"]:
                best = {"result": result, "aic": float(result.aic), "order": order, "seasonal_order": seasonal_order, "last_date": series.index[-1]}
        except Exception:
            continue
    return best


def predict(fitted_state, origin_date, horizon_days: int):
    if fitted_state is None:
        return None
    try:
        forecast_result = fitted_state["result"].get_forecast(steps=horizon_days)
        point = float(forecast_result.predicted_mean.iloc[-1])
        ci = forecast_result.conf_int(alpha=0.10)
        lower, upper = float(ci.iloc[-1, 0]), float(ci.iloc[-1, 1])
    except Exception:
        return None
    return {"point": point, "lower": lower, "upper": upper}
