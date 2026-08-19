"""Data-quality checks over the raw parsed (pre-cleaning) DataFrames.

These checks only ever *observe and report* -- they never modify data. What
they find feeds both the Data Quality Report and the cleaning module (which
decides what, if anything, is safe to do about a finding).
"""

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("validation")

NON_NEGATIVE_ROLES = {
    "price", "arrival", "area", "production", "storage_stock",
    "export_volume", "fx_rate", "yield", "index", "computed",
}


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scan_value_column(df: pd.DataFrame, column: str, role: str, invalid_tokens: set, label: str = None) -> dict:
    """Core scan applied to one value column: missing / text-contamination / negatives.

    `column` is the DataFrame column to read values from; `label` (defaults to
    `column`) is what gets recorded in the report -- lets callers report a
    semantic name (e.g. a metric name) when the underlying column is a
    generic melted "value" column.
    """
    label = label or column
    if column not in df.columns:
        return {"missing": 0, "text_contamination": [], "negative_values": []}

    missing = 0
    text_contamination = []
    negative_values = []

    for _, row in df.iterrows():
        value = row[column]
        row_ref = row.get("_row")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            missing += 1
            continue
        if isinstance(value, str):
            token = value.strip().lower()
            if token in invalid_tokens or token == "":
                missing += 1
            else:
                text_contamination.append({"row": row_ref, "column": label, "value": value})
            continue
        if _is_numeric(value) and role in NON_NEGATIVE_ROLES and value < 0:
            negative_values.append({"row": row_ref, "column": label, "value": value})

    return {
        "missing": missing,
        "text_contamination": text_contamination,
        "negative_values": negative_values,
    }


def _date_range(df: pd.DataFrame, date_column: str):
    if date_column not in df.columns:
        return None
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None
    return {"min": str(valid.min().date()), "max": str(valid.max().date())}


def _invalid_dates(df: pd.DataFrame, date_column: str) -> int:
    if date_column not in df.columns:
        return 0
    original_missing = df[date_column].isna().sum()
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    return int(parsed.isna().sum() - original_missing)


def _duplicate_count(df: pd.DataFrame, key_columns: list) -> int:
    present = [c for c in key_columns if c in df.columns]
    if not present or df.empty:
        return 0
    return int(df.duplicated(subset=present, keep=False).sum())


def validate_long(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set) -> dict:
    id_name = spec["id_column"].get("rename") or spec["id_column"]["role"]
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    value_specs = spec.get("columns", {})

    report = {
        "row_count": len(df),
        "date_range": _date_range(df, id_name) if spec["id_column"]["role"] == "date" else None,
        "invalid_dates": _invalid_dates(df, id_name) if spec["id_column"]["role"] == "date" else 0,
        "duplicate_rows": _duplicate_count(df, [id_name]),
        "columns": {},
    }
    for info in value_specs.values():
        col = info["rename"]
        report["columns"][col] = _scan_value_column(df, col, info.get("role", "unknown"), invalid_tokens)
    return report


def validate_wide_pivot_year_month(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set) -> dict:
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    value_name = spec.get("value_name", "value")
    role = spec.get("value_role", "unknown")
    report = {
        "row_count": len(df),
        "date_range": _date_range(df, "date") if "date" in df.columns else None,
        "duplicate_rows": _duplicate_count(df, [spec["id_column"]["role"], "period_label"]),
        "columns": {value_name: _scan_value_column(df, value_name, role, invalid_tokens)},
    }
    return report


def validate_wide_pivot_year_cols(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set) -> dict:
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    value_name = spec.get("value_name", "value")
    id_name = spec["id_column"].get("rename") or spec["id_column"]["role"]
    report = {
        "row_count": len(df),
        "date_range": None,
        "duplicate_rows": _duplicate_count(df, [id_name, "year"]),
        "columns": {value_name: _scan_value_column(df, value_name, spec.get("value_role", "unknown"), invalid_tokens)},
        "estimate_rows": int(df["is_estimate"].sum()) if "is_estimate" in df.columns else 0,
    }
    return report


def validate_wide_pivot_merged_header(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set) -> dict:
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    id_name = spec["id_column"].get("rename") or spec["id_column"]["role"]
    report = {
        "row_count": len(df),
        "date_range": _date_range(df, id_name) if spec["id_column"]["role"] == "date" else None,
        "duplicate_rows": _duplicate_count(df, [id_name, "metric", spec.get("group_role", "group")]),
        "columns": {},
        "estimate_rows": int(df["is_estimate"].sum()) if "is_estimate" in df.columns else 0,
    }
    if df.empty:
        return report
    for metric in sorted(df["metric"].dropna().unique()):
        subset = df[df["metric"] == metric]
        role = subset["_role"].iloc[0] if "_role" in subset.columns and len(subset) else "unknown"
        scan = _scan_value_column(subset, "value", role, invalid_tokens, label=metric)
        report["columns"][metric] = scan
    return report


def validate_sparse_tracker(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set) -> dict:
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    col_specs = spec.get("columns", {})
    report = {
        "row_count": len(df),
        "date_range": _date_range(df, "date") if "date" in df.columns else None,
        "duplicate_rows": _duplicate_count(df, ["date"]),
        "columns": {},
        "sparse_flag": spec.get("sparse_flag", False),
    }
    for info in col_specs.values():
        col = info["rename"]
        report["columns"][col] = _scan_value_column(df, col, info.get("role", "unknown"), invalid_tokens)
    return report


VALIDATORS = {
    "long": validate_long,
    "wide_pivot_year_month": validate_wide_pivot_year_month,
    "wide_pivot_year_cols": validate_wide_pivot_year_cols,
    "wide_pivot_merged_header": validate_wide_pivot_merged_header,
    "sparse_tracker": validate_sparse_tracker,
}


def validate_workbook(parsed: dict, sheet_config: dict) -> dict:
    """Validate every successfully-parsed sheet. Returns {sheet_name: report}."""
    global_invalid_tokens = {t.lower() for t in sheet_config.get("global_invalid_tokens", [])}
    reports = {}
    for sheet_name, df in parsed.items():
        spec = sheet_config["sheets"][sheet_name]
        validator = VALIDATORS.get(spec["layout"])
        if validator is None:
            continue
        try:
            reports[sheet_name] = validator(sheet_name, df, spec, global_invalid_tokens)
        except Exception as exc:  # noqa: BLE001 - one bad sheet must not block the whole report
            logger.error("Validation failed for sheet '%s': %s", sheet_name, exc, exc_info=True)
            reports[sheet_name] = {"error": str(exc)}
    return reports
