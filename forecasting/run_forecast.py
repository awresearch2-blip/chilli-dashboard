"""Gated forecasting entrypoint: only re-runs backtesting/selection/refitting
when the clean data's latest price date has advanced since the last run.
Model training isn't free, and --watch can fire on every workbook save --
retraining on a save that added no new price rows would be pure waste.

Realized-accuracy scoring runs on every call regardless of that gate -- it's
just reading the forecast log and looking up actuals, not retraining, and
its result can change even between retrains as previously-future target
dates arrive.
"""

import json

import pandas as pd

from analytics.arrival_module import SHEET_NAME as ARRIVAL_SHEET
from analytics.data_access import load_clean_sheet
from analytics.fx_module import FX_SHEET
from analytics.price_module import SHEET_NAME as PRICE_SHEET
from analytics.price_module import VARIETIES, variety_series
from forecasting import accuracy_tracker, forecast_engine
from utils.logging_config import get_logger
from utils.paths import FORECASTS_DIR, ensure_directories

logger = get_logger("forecasting")

STATE_PATH = FORECASTS_DIR / "_last_trained_through.json"
REALIZED_ACCURACY_PATH = FORECASTS_DIR / "realized_accuracy.json"


def _latest_price_date(price_df: pd.DataFrame) -> str:
    return str(pd.to_datetime(price_df["date"]).max().date())


def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run(refresh_id: str, force: bool = False) -> dict:
    ensure_directories()
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_clean_sheet(PRICE_SHEET)
    arrivals_df = load_clean_sheet(ARRIVAL_SHEET)
    fx_df = load_clean_sheet(FX_SHEET)

    price_series_by_variety = {variety: variety_series(price_df, variety, "avg_price") for variety in VARIETIES}
    accuracy_result = accuracy_tracker.compute_realized_accuracy(price_series_by_variety)
    with open(REALIZED_ACCURACY_PATH, "w", encoding="utf-8") as f:
        json.dump({"refresh_id": refresh_id, "result": accuracy_result}, f, indent=2, default=str)

    latest_date = _latest_price_date(price_df)
    state = _load_state()
    if not force and state.get("last_trained_through") == latest_date:
        logger.info("Forecasting skipped -- no new price data since last training (through %s)", latest_date)
        return {"status": "skipped_no_new_data", "last_trained_through": latest_date, "realized_accuracy": accuracy_result.get("status")}

    result = forecast_engine.run(price_df, arrivals_df, fx_df)

    for variety, variety_result in result.items():
        path = FORECASTS_DIR / f"{str(variety).lower()}_forecast.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"refresh_id": refresh_id, "variety": variety, "result": variety_result}, f, indent=2, default=str)

        if variety_result.get("status") == "ok":
            accuracy_tracker.log_forecasts(refresh_id, variety, variety_result["as_of"], variety_result["forecasts_by_horizon_days"])

    _save_state({"last_trained_through": latest_date, "refresh_id": refresh_id})
    logger.info("Forecasting complete, trained through %s", latest_date)
    return {"status": "ok", "trained_through": latest_date, "varieties": list(result.keys()), "realized_accuracy": accuracy_result.get("status")}
