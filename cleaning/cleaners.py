"""Deterministic-only cleaning of the raw parsed DataFrames.

Every allowed operation (trim whitespace, convert a KNOWN invalid token to
missing, convert a numeric-looking string to a real number, standardize
dates, drop an exact duplicate row, drop a fully blank row) is logged with
the original value before it's changed. Nothing is ever imputed, estimated,
or smoothed -- text that doesn't match a known invalid token is left exactly
as-is (still flagged by validation) rather than guessed at.
"""

import re

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("cleaning")

_NUMERIC_STRING_PATTERN = re.compile(r"^-?[0-9][0-9,]*\.?[0-9]*$")


def _try_parse_numeric_string(stripped: str):
    if not _NUMERIC_STRING_PATTERN.match(stripped):
        return None
    normalized = stripped.replace(",", "")
    try:
        return float(normalized) if "." in normalized else int(normalized)
    except ValueError:
        return None


def _clean_value(value, sheet_name: str, column: str, row_ref, invalid_tokens: set, log_entries: list):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "" or stripped.lower() in invalid_tokens:
        log_entries.append(
            {"sheet": sheet_name, "row": row_ref, "column": column, "action": "token_to_nan", "original_value": value}
        )
        return None

    numeric = _try_parse_numeric_string(stripped)
    if numeric is not None:
        log_entries.append(
            {
                "sheet": sheet_name, "row": row_ref, "column": column,
                "action": "numeric_string_converted", "original_value": value, "new_value": numeric,
            }
        )
        return numeric

    if stripped != value:
        log_entries.append(
            {
                "sheet": sheet_name, "row": row_ref, "column": column,
                "action": "trimmed_whitespace", "original_value": value, "new_value": stripped,
            }
        )
    return stripped


def _value_columns_for_spec(spec: dict) -> list:
    layout = spec["layout"]
    if layout in ("long", "sparse_tracker"):
        return [info["rename"] for info in spec.get("columns", {}).values()]
    if layout in ("wide_pivot_year_month", "wide_pivot_year_cols"):
        return [spec.get("value_name", "value")]
    if layout == "wide_pivot_merged_header":
        return ["value"]
    return []


def _date_column_for_spec(spec: dict) -> str | None:
    layout = spec["layout"]
    if layout in ("long", "wide_pivot_merged_header") and spec["id_column"]["role"] == "date":
        return spec["id_column"].get("rename") or "date"
    if layout in ("wide_pivot_year_month", "sparse_tracker"):
        return "date"
    return None


def clean_sheet(sheet_name: str, df: pd.DataFrame, spec: dict, global_invalid_tokens: set, log_entries: list) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    invalid_tokens = global_invalid_tokens | {t.lower() for t in spec.get("invalid_tokens", [])}
    value_columns = [c for c in _value_columns_for_spec(spec) if c in df.columns]

    for col in value_columns:
        df[col] = [
            _clean_value(v, sheet_name, col, row_ref, invalid_tokens, log_entries)
            for v, row_ref in zip(df[col], df.get("_row", range(len(df))))
        ]

    date_col = _date_column_for_spec(spec)
    if date_col and date_col in df.columns:
        before_missing = df[date_col].isna().sum()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        coerced_to_nat = int(df[date_col].isna().sum() - before_missing)
        if coerced_to_nat > 0:
            log_entries.append(
                {"sheet": sheet_name, "action": "date_standardized", "column": date_col, "coerced_to_nat_count": coerced_to_nat}
            )

    # Only meaningful for layouts where one row aggregates several fields for
    # one date/entity (long, sparse_tracker) -- there, all-fields-blank really
    # does mean an empty row. In melted single-value layouts (each row is one
    # metric observation), a NaN value is a legitimate missing observation
    # and must be kept, not deleted as if it were a blank row.
    if value_columns and spec["layout"] in ("long", "sparse_tracker"):
        fully_blank_mask = df[value_columns].isna().all(axis=1)
        if fully_blank_mask.any():
            for _, row in df[fully_blank_mask].iterrows():
                log_entries.append(
                    {"sheet": sheet_name, "row": row.get("_row"), "action": "blank_row_dropped"}
                )
            df = df[~fully_blank_mask].reset_index(drop=True)

    compare_cols = [c for c in df.columns if c != "_row"]
    if compare_cols:
        dup_mask = df.duplicated(subset=compare_cols, keep="first")
        if dup_mask.any():
            for _, row in df[dup_mask].iterrows():
                log_entries.append(
                    {
                        "sheet": sheet_name, "row": row.get("_row"), "action": "duplicate_row_dropped",
                        "original_value": row.drop(labels=["_row"], errors="ignore").to_dict(),
                    }
                )
            df = df[~dup_mask].reset_index(drop=True)

    return df


def clean_workbook(parsed: dict, sheet_config: dict) -> tuple[dict, list]:
    """Clean every parsed sheet. Returns (cleaned: {sheet_name: DataFrame}, log_entries: list)."""
    global_invalid_tokens = {t.lower() for t in sheet_config.get("global_invalid_tokens", [])}
    log_entries = []
    cleaned = {}
    for sheet_name, df in parsed.items():
        spec = sheet_config["sheets"][sheet_name]
        try:
            cleaned[sheet_name] = clean_sheet(sheet_name, df, spec, global_invalid_tokens, log_entries)
        except Exception as exc:  # noqa: BLE001 - one bad sheet must not block the whole refresh
            logger.error("Cleaning failed for sheet '%s': %s", sheet_name, exc, exc_info=True)
    logger.info("Cleaning complete: %d actions logged across %d sheets", len(log_entries), len(cleaned))
    return cleaned, log_entries
