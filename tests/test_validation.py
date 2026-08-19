from ingestion.sheet_parser import parse_long, parse_wide_pivot_merged_header
from ingestion.workbook_reader import find_unconfigured_sheets
from validation.rules import validate_long, validate_wide_pivot_merged_header


def test_validate_long_flags_missing_negative_and_text_contamination(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["LongSheet"]
    spec = fixture_sheet_config["sheets"]["LongSheet"]
    df = parse_long(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    report = validate_long("LongSheet", df, spec, global_tokens)

    # "Closed" is a known invalid token -> counted as missing, not contamination.
    assert report["columns"]["price"]["missing"] == 1
    # "110" (numeric-as-text) and "TBD" (unrecognized) are both still strings
    # at the validation stage -- validation only observes, it never coerces
    # or guesses, so both are flagged as text contamination.
    contaminated_values = {c["value"] for c in report["columns"]["price"]["text_contamination"]}
    assert contaminated_values == {"110", "TBD"}
    # -20 arrivals is a real negative value that must be flagged, never silently dropped.
    assert len(report["columns"]["arrivals"]["negative_values"]) == 1
    assert report["columns"]["arrivals"]["negative_values"][0]["value"] == -20


def test_validate_merged_header_labels_columns_by_metric_not_generic_value(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["MergedSheet"]
    spec = fixture_sheet_config["sheets"]["MergedSheet"]
    df = parse_wide_pivot_merged_header(ws, spec)
    global_tokens = {t.lower() for t in fixture_sheet_config["global_invalid_tokens"]}

    report = validate_wide_pivot_merged_header("MergedSheet", df, spec, global_tokens)

    # "#DIV/0!" is a configured invalid token for this sheet -> missing under low_price,
    # and any reported location must reference the real metric name, not "value".
    assert report["columns"]["low_price"]["missing"] == 1


def test_unconfigured_sheet_is_detected_not_silently_parsed(fixture_wb, fixture_sheet_config):
    unconfigured = find_unconfigured_sheets(fixture_wb, fixture_sheet_config)
    assert "UnconfiguredSheet" in unconfigured
    # Every sheet actually declared in the test config must NOT show up as unconfigured.
    for configured_name in fixture_sheet_config["sheets"]:
        assert configured_name not in unconfigured
