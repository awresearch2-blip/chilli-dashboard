"""Single refresh entrypoint: discover -> parse -> validate -> clean -> persist -> report.

Used identically by the CLI's --once, the watchdog callback for --watch, and
(later) the dashboard's manual Refresh button. Never raises -- a failed
refresh is reported as a status dict so a long-running --watch process keeps
going instead of dying on one bad read.
"""

import datetime as dt
import json
from pathlib import Path

from analytics.run_analytics import run_all as run_analytics
from cleaning.cleaners import clean_workbook
from forecasting.run_forecast import run as run_forecast
from ingestion.sheet_parser import parse_workbook
from ingestion.workbook_reader import (
    detect_stale_formula_cache,
    find_unconfigured_sheets,
    load_sheet_config,
    open_workbook,
)
from utils.logging_config import get_logger
from utils.paths import (
    CLEAN_LATEST_DIR,
    CLEANING_LOG_PATH,
    RAW_LATEST_DIR,
    WORKBOOK_PATH,
    ensure_directories,
    slugify,
)
from validation.quality_report import build_report, save_report
from validation.rules import validate_workbook

logger = get_logger("refresh")


def _persist_raw(dfs: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.pkl"):
        existing.unlink()
    for sheet_name, df in dfs.items():
        df.to_pickle(directory / f"{slugify(sheet_name)}.pkl")


def _persist_clean(dfs: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.parquet"):
        existing.unlink()
    for sheet_name, df in dfs.items():
        path = directory / f"{slugify(sheet_name)}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # pyarrow can't infer a type for a genuinely mixed-type column
            logger.warning(
                "Parquet write failed for '%s' (%s) -- stringifying object columns as a fallback",
                sheet_name, exc,
            )
            safe_df = df.copy()
            for col in safe_df.columns:
                if safe_df[col].dtype == object:
                    safe_df[col] = safe_df[col].map(lambda v: v if v is None else str(v))
            safe_df.to_parquet(path, index=False)


def _append_cleaning_log(log_entries: list, refresh_id: str):
    if not log_entries:
        return
    ensure_directories()
    with open(CLEANING_LOG_PATH, "a", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps({"refresh_id": refresh_id, **entry}, default=str) + "\n")


def run_refresh(workbook_path: Path = WORKBOOK_PATH) -> dict:
    refresh_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    logger.info("=== Refresh %s starting ===", refresh_id)

    try:
        sheet_config = load_sheet_config()
        wb_values, wb_formulas = open_workbook(workbook_path)
    except Exception as exc:
        logger.error("Refresh %s aborted -- could not open workbook: %s", refresh_id, exc, exc_info=True)
        return {"refresh_id": refresh_id, "status": "failed", "error": str(exc)}

    unconfigured = find_unconfigured_sheets(wb_values, sheet_config)
    if unconfigured:
        logger.warning("Unconfigured sheets detected (not parsed): %s", unconfigured)

    parsed, skipped = parse_workbook(wb_values, sheet_config)

    stale_formula_cache = {}
    for sheet_name in parsed:
        try:
            count = detect_stale_formula_cache(wb_values[sheet_name], wb_formulas[sheet_name])
            if count > 0:
                stale_formula_cache[sheet_name] = count
        except Exception:
            logger.warning("Could not check stale formula cache for '%s'", sheet_name)

    validation_reports = validate_workbook(parsed, sheet_config)
    cleaned, log_entries = clean_workbook(parsed, sheet_config)

    _persist_raw(parsed, RAW_LATEST_DIR)
    _persist_clean(cleaned, CLEAN_LATEST_DIR)
    _append_cleaning_log(log_entries, refresh_id)

    report = build_report(refresh_id, validation_reports, unconfigured, skipped, stale_formula_cache)
    report_path = save_report(report)

    try:
        analytics_results = run_analytics(refresh_id)
    except Exception as exc:
        logger.error("Analytics run failed entirely for refresh %s: %s", refresh_id, exc, exc_info=True)
        analytics_results = {"error": str(exc)}

    try:
        forecast_results = run_forecast(refresh_id)
    except Exception as exc:
        logger.error("Forecasting run failed entirely for refresh %s: %s", refresh_id, exc, exc_info=True)
        forecast_results = {"status": "failed", "error": str(exc)}

    logger.info(
        "=== Refresh %s complete: %d sheets parsed, %d skipped, %d cleaning actions, analytics=%s, forecasting=%s ===",
        refresh_id, len(parsed), len(skipped), len(log_entries), analytics_results, forecast_results,
    )

    return {
        "refresh_id": refresh_id,
        "status": "ok",
        "sheets_parsed": len(parsed),
        "sheets_skipped": len(skipped),
        "cleaning_actions": len(log_entries),
        "quality_report_path": str(report_path),
        "analytics": analytics_results,
        "forecasting": forecast_results,
    }
