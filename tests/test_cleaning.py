import pandas as pd

from cleaning.cleaners import clean_sheet
from ingestion.sheet_parser import parse_long, parse_wide_pivot_merged_header


def test_clean_sheet_converts_known_token_and_numeric_string_but_not_unknown_text(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["LongSheet"]
    spec = fixture_sheet_config["sheets"]["LongSheet"]
    df = parse_long(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    log_entries = []
    cleaned = clean_sheet("LongSheet", df, spec, global_tokens, log_entries)

    price_by_date = cleaned.set_index("date")["price"]
    # Known invalid token -> converted to missing, and logged.
    assert pd.isna(price_by_date[pd.Timestamp(2024, 1, 2)])
    assert any(e["action"] == "token_to_nan" and e["original_value"] == "Closed" for e in log_entries)

    # Numeric-string -> converted to a real number, and logged.
    assert price_by_date[pd.Timestamp(2024, 1, 4)] == 110
    assert any(e["action"] == "numeric_string_converted" and e["new_value"] == 110 for e in log_entries)

    # Unrecognized text ("TBD") is NOT a known invalid token -- must survive
    # untouched. Converting it would be exactly the kind of guess the brief
    # forbids ("never invent data").
    assert price_by_date[pd.Timestamp(2024, 1, 5)] == "TBD"
    assert not any(e.get("original_value") == "TBD" for e in log_entries)


def test_clean_sheet_drops_exact_duplicate_row_and_logs_it(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["LongSheet"]
    spec = fixture_sheet_config["sheets"]["LongSheet"]
    df = parse_long(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    log_entries = []
    cleaned = clean_sheet("LongSheet", df, spec, global_tokens, log_entries)

    matches = cleaned[cleaned["date"] == pd.Timestamp(2024, 1, 1)]
    assert len(matches) == 1  # the exact duplicate was dropped
    assert any(e["action"] == "duplicate_row_dropped" for e in log_entries)


def test_clean_sheet_never_touches_a_real_negative_value(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["LongSheet"]
    spec = fixture_sheet_config["sheets"]["LongSheet"]
    df = parse_long(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    log_entries = []
    cleaned = clean_sheet("LongSheet", df, spec, global_tokens, log_entries)

    arrivals_by_date = cleaned.set_index("date")["arrivals"]
    # -20 is real (if suspicious) data -- cleaning must never impute or drop it,
    # only validation flags it for human review.
    assert arrivals_by_date[pd.Timestamp(2024, 1, 3)] == -20


def test_clean_sheet_converts_configured_error_token_in_merged_header_sheet(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["MergedSheet"]
    spec = fixture_sheet_config["sheets"]["MergedSheet"]
    df = parse_wide_pivot_merged_header(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    log_entries = []
    cleaned = clean_sheet("MergedSheet", df, spec, global_tokens, log_entries)

    row = cleaned[
        (cleaned["date"] == pd.Timestamp(2024, 1, 2))
        & (cleaned["variety"] == "VarietyA")
        & (cleaned["metric"] == "low_price")
    ]
    assert pd.isna(row["value"].iloc[0])
