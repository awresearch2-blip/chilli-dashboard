"""Selects the best-backtested model independently per horizon. Never one
hardcoded model -- it's expected and fine for different models to win at
different horizons."""

SELECTION_METRIC = "rmse"


def select_best_models(backtest_result: dict, metric: str = SELECTION_METRIC) -> dict:
    if backtest_result.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": backtest_result.get("reason", "Backtest unavailable")}

    selection = {}
    for horizon_str, models in backtest_result["metrics_by_horizon"].items():
        best_model_name, best_metrics = min(models.items(), key=lambda kv: kv[1][metric])
        selection[horizon_str] = {"model": best_model_name, "backtest_accuracy": best_metrics}

    return {"status": "ok", "selection_metric": metric, "selection": selection}
