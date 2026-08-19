"""Orchestrates: backtest -> select best model per horizon -> refit each
selected model on the FULL available history -> produce a point forecast +
confidence interval + explanation for each requested horizon.

Any model whose own `predict()` doesn't return a native confidence interval
(Seasonal Naive, XGBoost) gets an empirical one attached here, uniformly,
from that specific (model, horizon)'s out-of-sample backtest residuals --
not duplicated per model, and not presented with false precision. The same
residuals feed the bullish/bearish probability (probability.py).
"""

import pandas as pd

from analytics.arrival_module import SHEET_NAME as ARRIVAL_SHEET
from analytics.arrival_module import field_series
from analytics.data_access import AnalyticalResultNotFound, date_indexed_series, load_analytical_result, load_clean_sheet
from analytics.fx_module import FX_SHEET
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series
from forecasting import explainability
from forecasting.backtest import run_backtest
from forecasting.feature_engineering import build_feature_frame
from forecasting.model_selection import select_best_models
from forecasting.models import holt_winters, sarima, seasonal_naive, xgboost_direct
from forecasting.probability import bullish_bearish_probability
from forecasting.similar_periods import find_similar_periods

CLASSICAL_MODELS = {"seasonal_naive": seasonal_naive, "holt_winters": holt_winters, "sarima": sarima}


def _empirical_ci(backtest_result: dict, model_name: str, horizon: int, point: float):
    records = backtest_result.get("raw_records", [])
    residuals = [r["actual"] - r["pred"] for r in records if r["model"] == model_name and r["horizon"] == horizon]
    if len(residuals) < 3:
        return None, None
    residuals_series = pd.Series(residuals)
    return point + float(residuals_series.quantile(0.05)), point + float(residuals_series.quantile(0.95))


def forecast_variety(
    variety: str,
    price_series: pd.Series,
    arrivals_series: pd.Series,
    fx_series: pd.Series,
    composite_result_for_variety: dict,
) -> dict:
    if price_series.dropna().empty:
        return {"status": "insufficient_evidence", "reason": f"No price data for '{variety}'"}

    backtest_result = run_backtest(price_series, arrivals_series, fx_series)
    selection = select_best_models(backtest_result)
    if selection.get("status") != "ok":
        return {
            "status": "insufficient_evidence",
            "reason": "Backtesting could not establish a model selection",
            "backtest_status": backtest_result.get("status"),
            "backtest_reason": backtest_result.get("reason"),
        }

    origin_date = price_series.dropna().index[-1]
    current_price = float(price_series.dropna().iloc[-1])
    fitted_classical = {name: mod.fit(price_series) for name, mod in CLASSICAL_MODELS.items()}
    features_df = build_feature_frame(price_series, arrivals_series, fx_series)
    fitted_xgb_by_horizon = {}

    forecasts = {}
    for horizon_str, choice in selection["selection"].items():
        horizon = int(horizon_str)
        model_name = choice["model"]

        if model_name == "xgboost":
            fitted = xgboost_direct.fit(features_df, price_series, horizon)
            fitted_xgb_by_horizon[horizon] = fitted
            pred = xgboost_direct.predict(fitted, origin_date)
        else:
            pred = CLASSICAL_MODELS[model_name].predict(fitted_classical[model_name], origin_date, horizon)

        if pred is None or pred.get("point") is None:
            forecasts[horizon_str] = {"status": "insufficient_evidence", "reason": f"Selected model '{model_name}' could not produce a forecast for this horizon"}
            continue

        lower, upper, ci_is_empirical = pred.get("lower"), pred.get("upper"), False
        if lower is None or upper is None:
            emp_lower, emp_upper = _empirical_ci(backtest_result, model_name, horizon, pred["point"])
            lower = lower if lower is not None else emp_lower
            upper = upper if upper is not None else emp_upper
            ci_is_empirical = True

        fitted_registry = {"sarima": fitted_classical["sarima"], "holt_winters": fitted_classical["holt_winters"], "xgboost": fitted_xgb_by_horizon}

        forecasts[horizon_str] = {
            "status": "ok",
            "target_date": str((origin_date + pd.Timedelta(days=horizon)).date()),
            "point_forecast": round(pred["point"], 2),
            "lower_ci": round(lower, 2) if lower is not None else None,
            "upper_ci": round(upper, 2) if upper is not None else None,
            "ci_is_empirical_from_backtest_residuals": ci_is_empirical,
            "model_used": model_name,
            "backtest_accuracy": choice["backtest_accuracy"],
            "model_explanation": explainability.explain_model(model_name, horizon, fitted_registry),
            "probability": bullish_bearish_probability(backtest_result, model_name, horizon, pred["point"], current_price),
            "similar_historical_periods": find_similar_periods(features_df, price_series, origin_date, horizon),
            "risks_and_limitations": explainability.risks_and_limitations(model_name, horizon, choice["backtest_accuracy"]),
        }

    return {
        "status": "ok",
        "as_of": str(origin_date.date()),
        "current_price": current_price,
        "key_drivers": explainability.key_drivers(composite_result_for_variety),
        "forecasts_by_horizon_days": forecasts,
        "backtest_origins_used": backtest_result.get("origins_used"),
    }


def run(price_df: pd.DataFrame = None, arrivals_df: pd.DataFrame = None, fx_df: pd.DataFrame = None, composite_result: dict = None) -> dict:
    if price_df is None:
        price_df = load_clean_sheet(PRICE_SHEET)
    if arrivals_df is None:
        arrivals_df = load_clean_sheet(ARRIVAL_SHEET)
    if fx_df is None:
        fx_df = load_clean_sheet(FX_SHEET)
    if composite_result is None:
        try:
            composite_result = load_analytical_result("composite_indices")
        except AnalyticalResultNotFound:
            composite_result = {}

    per_variety_composite = composite_result.get("per_variety", {}) if composite_result else {}
    arrivals_series = field_series(arrivals_df, "arrivals_bags")
    fx_series = date_indexed_series(fx_df, "usd_inr_rate")

    return {
        variety: forecast_variety(
            variety,
            variety_series(price_df, variety, "avg_price"),
            arrivals_series,
            fx_series,
            per_variety_composite.get(variety, {}),
        )
        for variety in VARIETIES
    }
