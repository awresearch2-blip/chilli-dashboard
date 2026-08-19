"""Expanding-window walk-forward backtesting.

A shared set of origin dates spans the later portion of history (enough
trailing data at the earliest origin for the classical models to stabilize,
enough trailing future data after the latest origin to evaluate a 180-day
forecast). At each origin, every candidate model forecasts every horizon;
only the actual h-step-ahead point is compared to truth -- this is what
"accuracy at 90 days" is supposed to mean, not an average over the whole
forecast path.
"""

import pandas as pd

from forecasting.models import holt_winters, sarima, seasonal_naive, xgboost_direct
from forecasting.feature_engineering import build_feature_frame
from forecasting.utils import mae, mape, nearest_value, rmse, smape

HORIZONS = [7, 14, 30, 60, 90, 120, 180]
MIN_TRAIN_YEARS = 3
N_ORIGINS = 6

CLASSICAL_MODELS = {
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters,
    "sarima": sarima,
}


def select_origins(price_series: pd.Series) -> list:
    valid = price_series.dropna()
    if valid.empty:
        return []
    first, last = valid.index[0], valid.index[-1]
    earliest_origin = first + pd.DateOffset(years=MIN_TRAIN_YEARS)
    latest_origin = last - pd.Timedelta(days=max(HORIZONS) + 5)
    if earliest_origin >= latest_origin:
        return []
    return list(pd.date_range(earliest_origin, latest_origin, periods=N_ORIGINS))


def run_backtest(price_series: pd.Series, arrivals_series: pd.Series, fx_series: pd.Series) -> dict:
    origins = select_origins(price_series)
    if not origins:
        return {"status": "insufficient_evidence", "reason": "Not enough history to establish backtest origins"}

    records = []
    for origin in origins:
        train_price = price_series.loc[:origin]
        train_arrivals = arrivals_series.loc[:origin] if arrivals_series is not None else None
        train_fx = fx_series.loc[:origin] if fx_series is not None else None

        fitted_classical = {name: mod.fit(train_price) for name, mod in CLASSICAL_MODELS.items()}
        features_df = (
            build_feature_frame(train_price, train_arrivals, train_fx)
            if train_arrivals is not None and train_fx is not None
            else None
        )

        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon)
            actual = nearest_value(price_series, target_date, tolerance_days=2)
            if actual is None:
                continue

            for name, fitted in fitted_classical.items():
                pred = CLASSICAL_MODELS[name].predict(fitted, origin, horizon)
                if pred and pred["point"] is not None:
                    records.append({"model": name, "horizon": horizon, "origin": str(origin.date()), "actual": actual, "pred": pred["point"]})

            if features_df is not None:
                xgb_fitted = xgboost_direct.fit(features_df, train_price, horizon)
                pred = xgboost_direct.predict(xgb_fitted, origin)
                if pred and pred["point"] is not None:
                    records.append({"model": "xgboost", "horizon": horizon, "origin": str(origin.date()), "actual": actual, "pred": pred["point"]})

    if not records:
        return {"status": "insufficient_evidence", "reason": "No backtest predictions could be produced"}

    df = pd.DataFrame(records)
    metrics_by_horizon = {}
    for (model, horizon), group in df.groupby(["model", "horizon"]):
        errors = (group["actual"] - group["pred"]).to_numpy()
        metrics_by_horizon.setdefault(str(horizon), {})[model] = {
            "rmse": rmse(errors),
            "mae": mae(errors),
            "mape": mape(group["actual"], group["pred"]),
            "smape": smape(group["actual"], group["pred"]),
            "n_folds": len(group),
        }

    return {
        "status": "ok",
        "origins_used": [str(o.date()) for o in origins],
        "metrics_by_horizon": metrics_by_horizon,
        "raw_records": records,
    }
