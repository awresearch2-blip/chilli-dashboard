"""Tracks REAL forecast accuracy over time -- distinct from backtesting,
which retrospectively simulates history. Every time a forecast is actually
produced, it's appended to a permanent log with its target date. Once that
target date has passed and a real actual becomes available, it can be
scored. On a freshly-started system there is nothing to score yet -- every
current forecast's target date is still in the future -- and that's the
honest state to report, not something to fake.
"""

import json

import pandas as pd

from forecasting.utils import mae, mape, nearest_value, rmse
from utils.paths import FORECASTS_DIR

FORECAST_LOG_PATH = FORECASTS_DIR / "forecast_log.jsonl"
LOOKUP_TOLERANCE_DAYS = 2


def log_forecasts(refresh_id: str, variety: str, as_of: str, forecasts_by_horizon: dict) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FORECAST_LOG_PATH, "a", encoding="utf-8") as f:
        for horizon_str, forecast in forecasts_by_horizon.items():
            if forecast.get("status") != "ok":
                continue
            entry = {
                "refresh_id": refresh_id,
                "variety": variety,
                "as_of": as_of,
                "horizon_days": int(horizon_str),
                "target_date": forecast["target_date"],
                "point_forecast": forecast["point_forecast"],
                "model_used": forecast["model_used"],
            }
            f.write(json.dumps(entry) + "\n")


def _load_log() -> list:
    if not FORECAST_LOG_PATH.exists():
        return []
    with open(FORECAST_LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_realized_accuracy(price_series_by_variety: dict) -> dict:
    """price_series_by_variety: {variety: date-indexed price Series}, used to
    look up what actually happened at each logged forecast's target date."""
    entries = _load_log()
    if not entries:
        return {
            "status": "insufficient_evidence",
            "reason": "No forecasts have been logged yet -- this is a freshly-started system. Check back after forecasts have been running for a while and their target dates have passed.",
        }

    scored = []
    for entry in entries:
        series = price_series_by_variety.get(entry["variety"])
        if series is None:
            continue
        actual = nearest_value(series, pd.Timestamp(entry["target_date"]), tolerance_days=LOOKUP_TOLERANCE_DAYS)
        if actual is None:
            continue  # target date hasn't happened yet (or still a data gap) -- not scoreable
        scored.append({**entry, "actual": actual, "error": actual - entry["point_forecast"]})

    if not scored:
        return {
            "status": "insufficient_evidence",
            "reason": "No logged forecasts have reached their target date yet -- nothing to score",
            "forecasts_pending": len(entries),
        }

    df = pd.DataFrame(scored)
    realized_accuracy = {}
    for (variety, model, horizon), group in df.groupby(["variety", "model_used", "horizon_days"]):
        errors = group["error"].to_numpy()
        realized_accuracy.setdefault(variety, {}).setdefault(str(horizon), {})[model] = {
            "rmse": rmse(errors),
            "mae": mae(errors),
            "mape": mape(group["actual"], group["point_forecast"]),
            "n": len(group),
        }

    return {
        "status": "ok",
        "realized_accuracy": realized_accuracy,
        "forecasts_scored": len(scored),
        "forecasts_pending": len(entries) - len(scored),
    }
