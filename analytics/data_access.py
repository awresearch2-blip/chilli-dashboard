"""Read-only access to the clean data tier for analytics modules.

Analytics never touches the workbook or the raw tier directly -- it only
ever reads data/clean/latest/*.parquet, produced by pipeline/refresh.py
after validation and deterministic cleaning.
"""

import json

import pandas as pd

from utils.paths import ANALYTICAL_DIR, CLEAN_LATEST_DIR, slugify


class CleanSheetNotFound(Exception):
    pass


class AnalyticalResultNotFound(Exception):
    pass


def load_clean_sheet(sheet_name: str) -> pd.DataFrame:
    path = CLEAN_LATEST_DIR / f"{slugify(sheet_name)}.parquet"
    if not path.exists():
        raise CleanSheetNotFound(
            f"No clean data for '{sheet_name}' at {path} -- has a refresh run yet?"
        )
    return pd.read_parquet(path)


def load_analytical_result(module_name: str) -> dict:
    """Reads a finished analytics module's output (e.g. "composite_indices")
    from data/analytical/ -- lets other packages (forecasting) consume an
    analytics module's result as a decoupled artifact rather than needing
    an in-memory handoff across the package boundary."""
    path = ANALYTICAL_DIR / f"{module_name}.json"
    if not path.exists():
        raise AnalyticalResultNotFound(
            f"No analytical output for '{module_name}' at {path} -- has a refresh run yet?"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["result"]


def date_indexed_series(df: pd.DataFrame, column: str, filters: dict = None) -> pd.Series:
    """Extract a numeric, date-indexed, sorted, deduplicated Series from a
    clean tidy DataFrame -- optionally filtered first (e.g. to one variety
    and one metric in a melted sheet)."""
    if filters:
        mask = pd.Series(True, index=df.index)
        for key, value in filters.items():
            mask &= df[key] == value
        df = df[mask]
    s = df.set_index("date")[column]
    s = pd.to_numeric(s, errors="coerce")
    return s[~s.index.duplicated(keep="last")].sort_index()
