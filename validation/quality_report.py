"""Builds and persists the Data Quality Report for a single refresh."""

import json
from pathlib import Path

from utils.logging_config import get_logger
from utils.paths import DATA_QUALITY_DIR, DATA_QUALITY_LATEST_PATH, ensure_directories

logger = get_logger("quality_report")


def build_report(
    refresh_id: str,
    validation_reports: dict,
    unconfigured_sheets: list,
    skipped_sheets: dict,
    stale_formula_cache: dict,
) -> dict:
    total_issues = 0
    for report in validation_reports.values():
        if "error" in report:
            total_issues += 1
            continue
        total_issues += report.get("invalid_dates", 0) + report.get("duplicate_rows", 0)
        for col_report in report.get("columns", {}).values():
            total_issues += (
                col_report.get("missing", 0)
                + len(col_report.get("text_contamination", []))
                + len(col_report.get("negative_values", []))
            )

    return {
        "refresh_id": refresh_id,
        "unconfigured_sheets": unconfigured_sheets,
        "skipped_sheets": skipped_sheets,
        "stale_formula_cache": stale_formula_cache,
        "sheets": validation_reports,
        "summary": {
            "sheets_validated": len(validation_reports),
            "sheets_skipped": len(skipped_sheets),
            "unconfigured_sheets_found": len(unconfigured_sheets),
            "total_issues_flagged": total_issues,
        },
    }


def save_report(report: dict) -> Path:
    ensure_directories()
    path = DATA_QUALITY_DIR / f"{report['refresh_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(DATA_QUALITY_LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(
        "Data Quality Report saved: %s (%d issues flagged across %d sheets)",
        path, report["summary"]["total_issues_flagged"], report["summary"]["sheets_validated"],
    )
    return path
