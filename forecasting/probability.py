"""Bullish/bearish probability for a forecast, derived from that exact
(model, horizon) combination's own out-of-sample backtest residuals -- the
same residuals already used for the empirical confidence interval in
forecast_engine.py, not a new assumption and not a different (parametric)
method for some models vs others.
"""

import numpy as np

MIN_RESIDUALS = 5


def bullish_bearish_probability(backtest_result: dict, model_name: str, horizon: int, point_forecast: float, current_price: float) -> dict:
    records = backtest_result.get("raw_records", [])
    residuals = [r["actual"] - r["pred"] for r in records if r["model"] == model_name and r["horizon"] == horizon]
    if len(residuals) < MIN_RESIDUALS:
        return {
            "status": "insufficient_evidence",
            "reason": "Not enough backtest residuals for this model/horizon to build a probability distribution",
            "available_n": len(residuals),
        }

    simulated_outcomes = np.array(residuals) + point_forecast
    bullish_probability = float(np.mean(simulated_outcomes > current_price))
    return {
        "status": "ok",
        "bullish_probability_pct": round(bullish_probability * 100, 1),
        "bearish_probability_pct": round((1 - bullish_probability) * 100, 1),
        "method": "empirical: point_forecast + this exact model/horizon's real backtest residuals, compared to the current price",
        "n_residuals_used": len(residuals),
    }
