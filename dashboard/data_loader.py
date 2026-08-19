"""Cached readers for the dashboard. Every loader's cache key includes the
underlying file's modification time, so a refresh (from any process --
the dashboard's own "Refresh Now" button or a separate `--watch` process)
automatically invalidates the cache without a manual clear.

These functions are plain and Streamlit-free except for the `st.cache_data`
decorators themselves, so they're directly unit-testable.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.data_access import load_clean_sheet as _load_clean_sheet_raw
from utils.paths import ANALYTICAL_DIR, CLEAN_LATEST_DIR, DATA_QUALITY_LATEST_PATH, FORECASTS_DIR, slugify


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_analytical_cached(module_name: str, mtime: float) -> dict:
    path = ANALYTICAL_DIR / f"{module_name}.json"
    if not path.exists():
        return {"status": "insufficient_evidence", "reason": f"No analytics output yet for '{module_name}' -- run a refresh first."}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["result"]


def load_analytical(module_name: str) -> dict:
    path = ANALYTICAL_DIR / f"{module_name}.json"
    return _load_analytical_cached(module_name, _mtime(path))


@st.cache_data(show_spinner=False)
def _load_forecast_cached(variety: str, mtime: float) -> dict:
    path = FORECASTS_DIR / f"{variety.lower()}_forecast.json"
    if not path.exists():
        return {"status": "insufficient_evidence", "reason": f"No forecast yet for '{variety}' -- run a refresh first."}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["result"]


def load_forecast(variety: str) -> dict:
    path = FORECASTS_DIR / f"{variety.lower()}_forecast.json"
    return _load_forecast_cached(variety, _mtime(path))


@st.cache_data(show_spinner=False)
def _load_realized_accuracy_cached(mtime: float) -> dict:
    path = FORECASTS_DIR / "realized_accuracy.json"
    if not path.exists():
        return {"status": "insufficient_evidence", "reason": "No realized-accuracy data yet -- run a refresh first."}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["result"]


def load_realized_accuracy() -> dict:
    path = FORECASTS_DIR / "realized_accuracy.json"
    return _load_realized_accuracy_cached(_mtime(path))


@st.cache_data(show_spinner=False)
def _load_clean_sheet_cached(sheet_name: str, mtime: float) -> pd.DataFrame:
    return _load_clean_sheet_raw(sheet_name)


def load_clean_sheet(sheet_name: str) -> pd.DataFrame:
    path = CLEAN_LATEST_DIR / f"{slugify(sheet_name)}.parquet"
    return _load_clean_sheet_cached(sheet_name, _mtime(path))


def last_refresh_id() -> str:
    if not DATA_QUALITY_LATEST_PATH.exists():
        return "never"
    with open(DATA_QUALITY_LATEST_PATH, encoding="utf-8") as f:
        return json.load(f).get("refresh_id", "unknown")
