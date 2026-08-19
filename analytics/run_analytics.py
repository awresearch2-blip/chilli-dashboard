"""Orchestrates the analytics modules and persists their output.

Ten "base" modules read only from data/clean/latest/ (never the workbook
directly) and run independently. The composite-index module runs last and
composes their already-computed results in memory (no disk round-trip) --
see composite_index_module.py for why. One module's failure never blocks
the others or the refresh -- same log-and-continue pattern as
ingestion/validation/cleaning.
"""

import json
from pathlib import Path

from analytics import (
    arrival_module,
    balance_sheet_module,
    cold_storage_module,
    composite_index_module,
    export_module,
    fx_module,
    market_comparison_module,
    price_arrival_module,
    price_module,
    seasonality_module,
    variety_module,
)
from analytics.data_access import CleanSheetNotFound
from utils.logging_config import get_logger
from utils.paths import ANALYTICAL_DIR, ensure_directories

logger = get_logger("analytics")

BASE_MODULES = {
    "price_analysis": price_module.run,
    "arrival_analysis": arrival_module.run,
    "price_vs_arrivals": price_arrival_module.run,
    "seasonality": seasonality_module.run,
    "variety_analysis": variety_module.run,
    "market_comparison": market_comparison_module.run,
    "export_analysis": export_module.run,
    "balance_sheet": balance_sheet_module.run,
    "fx_analysis": fx_module.run,
    "cold_storage": cold_storage_module.run,
}


def _write(module_name: str, refresh_id: str, result) -> Path:
    path = ANALYTICAL_DIR / f"{module_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"module": module_name, "refresh_id": refresh_id, "result": result}, f, indent=2, default=str)
    return path


def run_all(refresh_id: str) -> dict:
    ensure_directories()
    ANALYTICAL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    computed = {}

    for module_name, module_fn in BASE_MODULES.items():
        try:
            computed[module_name] = module_fn()
            path = _write(module_name, refresh_id, computed[module_name])
            results[module_name] = "ok"
            logger.info("Analytics module '%s' complete -> %s", module_name, path)
        except CleanSheetNotFound as exc:
            results[module_name] = f"skipped: {exc}"
            logger.warning("Analytics module '%s' skipped: %s", module_name, exc)
        except Exception as exc:  # noqa: BLE001 - one bad module must not block the others
            results[module_name] = f"failed: {exc}"
            logger.error("Analytics module '%s' failed: %s", module_name, exc, exc_info=True)

    try:
        composite_result = composite_index_module.run(computed)
        path = _write("composite_indices", refresh_id, composite_result)
        results["composite_indices"] = "ok"
        logger.info("Analytics module 'composite_indices' complete -> %s", path)
    except Exception as exc:  # noqa: BLE001
        results["composite_indices"] = f"failed: {exc}"
        logger.error("Analytics module 'composite_indices' failed: %s", exc, exc_info=True)

    return results
