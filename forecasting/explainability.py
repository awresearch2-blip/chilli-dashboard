"""Explains a forecast in the selected model's own real terms (never a
generic canned sentence forced onto a model that doesn't have that
concept), pulls "why" from Phase 2c's already-computed composite signals,
and states risks as a deterministic checklist triggered by real properties
of that specific forecast.
"""

from forecasting.feature_engineering import FEATURE_COLUMNS


def explain_model(model_name: str, horizon: int, fitted_registry: dict) -> dict:
    if model_name == "xgboost":
        fitted = (fitted_registry.get("xgboost") or {}).get(horizon)
        if not fitted or fitted.get("model") is None:
            return {"type": "feature_importance", "status": "insufficient_evidence"}
        importances = fitted["model"].feature_importances_
        pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda p: p[1], reverse=True)
        top = [{"feature": f, "importance_pct": round(float(v) * 100, 1)} for f, v in pairs[:6] if v > 0]
        return {
            "type": "feature_importance",
            "status": "ok",
            "top_features": top,
            "description": "Tree-ensemble model; percentages are each feature's real share of total split gain in the fitted model.",
        }

    if model_name == "sarima":
        fitted = fitted_registry.get("sarima")
        if not fitted:
            return {"type": "model_structure", "status": "insufficient_evidence"}
        return {
            "type": "model_structure",
            "status": "ok",
            "order_pdq": list(fitted["order"]),
            "seasonal_order_PDQs": list(fitted["seasonal_order"]),
            "aic": round(fitted["aic"], 1),
            "description": "Statistical time-series model: the forecast is driven by the series' own recent autocorrelation and a weekly seasonal adjustment, not by external variables like arrivals or FX.",
        }

    if model_name == "holt_winters":
        fitted = fitted_registry.get("holt_winters")
        if not fitted:
            return {"type": "model_structure", "status": "insufficient_evidence"}
        params = dict(fitted["fitted"].params)
        return {
            "type": "model_structure",
            "status": "ok",
            "smoothing_level": round(float(params.get("smoothing_level", 0) or 0), 3),
            "smoothing_trend": round(float(params.get("smoothing_trend", 0) or 0), 3),
            "smoothing_seasonal": round(float(params.get("smoothing_seasonal", 0) or 0), 3),
            "description": "Exponential-smoothing model: blends the series' current level, trend, and a 52-week seasonal pattern; each smoothing weight above is how much that component reacts to recent observations (higher = more reactive).",
        }

    if model_name == "seasonal_naive":
        return {
            "type": "lookup",
            "status": "ok",
            "description": "Forecast equals the observed price from approximately 365 days before the target date -- the simplest possible baseline, used here because it out-performed every other candidate for this specific horizon in backtesting.",
        }

    return {"type": "unknown", "status": "insufficient_evidence"}


def key_drivers(composite_result_for_variety: dict) -> dict:
    """Presentation layer over Phase 2c's already-computed, already-
    transparent signals -- adds no new analysis."""
    if not composite_result_for_variety:
        return {"status": "insufficient_evidence", "reason": "No composite analytics available for this variety"}

    bullish_bearish = composite_result_for_variety.get("bullish_bearish", {})
    market_strength = composite_result_for_variety.get("market_strength_index", {})

    statements = []
    if bullish_bearish.get("status") == "ok":
        for signal_name, direction in bullish_bearish.get("signals", {}).items():
            if direction != "neutral":
                statements.append(f"{signal_name.replace('_', ' ')}: {direction}")
    if market_strength.get("status") == "ok":
        for name, component in market_strength.get("components", {}).items():
            if component.get("raw") is not None:
                statements.append(f"{name.replace('_', ' ')} = {component['raw']}")

    if not statements:
        return {"status": "insufficient_evidence", "reason": "No underlying signals were available"}
    return {
        "status": "ok",
        "statements": statements,
        "bullish_score": bullish_bearish.get("bullish_score"),
        "bearish_score": bullish_bearish.get("bearish_score"),
    }


def risks_and_limitations(model_name: str, horizon: int, backtest_accuracy: dict) -> list:
    risks = []
    if horizon >= 90:
        risks.append(f"{horizon}-day-ahead commodity price forecasts carry substantial inherent uncertainty -- treat the confidence interval, not the point value, as the honest answer.")
    if backtest_accuracy:
        if backtest_accuracy.get("n_folds", 0) < 6:
            risks.append(f"Only {backtest_accuracy['n_folds']} backtest folds were available for this model/horizon -- the accuracy estimate itself is based on a small sample.")
        if backtest_accuracy.get("mape", 0) > 10:
            risks.append(f"Backtested MAPE for this model/horizon is {backtest_accuracy['mape']:.1f}% -- historically, forecasts at this horizon have missed actual prices by a meaningful margin.")

    if model_name == "seasonal_naive":
        risks.append("Seasonal Naive assumes this year repeats last year's pattern exactly -- it has no mechanism to react to a genuinely new supply or demand shock.")
    elif model_name == "holt_winters":
        risks.append("This horizon was rounded to the nearest whole week for Holt-Winters -- the modeled target date may be off by up to 3 days from the exact requested horizon.")
    elif model_name == "xgboost":
        risks.append("Tree-based forecast; feature importance reflects patterns in past data and does not by itself imply causation.")
    elif model_name == "sarima":
        risks.append("SARIMA here only models weekly seasonality directly -- yearly seasonal effects are not captured by this specific model.")

    return risks
