"""Central path resolution so every module agrees on where things live."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKBOOK_PATH = PROJECT_ROOT / "Chilli mastersheet for dashboard.xlsx"
SHEETS_CONFIG_PATH = PROJECT_ROOT / "config" / "sheets.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_LATEST_DIR = DATA_DIR / "raw" / "latest"
CLEAN_LATEST_DIR = DATA_DIR / "clean" / "latest"
ANALYTICAL_DIR = DATA_DIR / "analytical"
FORECASTS_DIR = DATA_DIR / "forecasts"

LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"
CLEANING_LOG_PATH = LOGS_DIR / "cleaning_log.jsonl"
DATA_QUALITY_DIR = LOGS_DIR / "data_quality"
DATA_QUALITY_LATEST_PATH = DATA_QUALITY_DIR / "latest.json"


def ensure_directories() -> None:
    for d in (RAW_LATEST_DIR, CLEAN_LATEST_DIR, ANALYTICAL_DIR, FORECASTS_DIR, LOGS_DIR, DATA_QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    """Turn a sheet name into the filename stem used for its persisted data
    (data/raw/latest, data/clean/latest). Shared by the pipeline (writer) and
    analytics (reader) so both agree on the same filenames."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
