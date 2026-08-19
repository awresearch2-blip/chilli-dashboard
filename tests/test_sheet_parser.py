import pandas as pd

from ingestion.sheet_parser import (
    parse_long,
    parse_wide_pivot_merged_header,
    parse_wide_pivot_year_month,
)


def test_parse_long_keeps_raw_values_and_trims_trailing_blank(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["LongSheet"]
    spec = fixture_sheet_config["sheets"]["LongSheet"]
    df = parse_long(ws, spec)

    # Trailing blank rows must be gone (6 real data rows remain, including an
    # unrecognized-text row and an exact-duplicate row -- parsing doesn't
    # touch either of those, that's cleaning's job).
    assert len(df) == 6
    # Values are carried through exactly as read -- no coercion, no token cleanup here.
    assert df.loc[df["date"] == pd.Timestamp(2024, 1, 2), "price"].iloc[0] == "Closed"
    assert df.loc[df["date"] == pd.Timestamp(2024, 1, 4), "price"].iloc[0] == "110"
    assert df.loc[df["date"] == pd.Timestamp(2024, 1, 3), "arrivals"].iloc[0] == -20


def test_parse_wide_pivot_merged_header_reconstructs_groups(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["MergedSheet"]
    spec = fixture_sheet_config["sheets"]["MergedSheet"]
    df = parse_wide_pivot_merged_header(ws, spec)

    # 2 dates x 2 varieties x 2 metrics
    assert len(df) == 8
    row = df[(df["date"] == pd.Timestamp(2024, 1, 1)) & (df["variety"] == "VarietyB") & (df["metric"] == "high_price")]
    assert row["value"].iloc[0] == 40

    # The error-token cell must be preserved raw (cleaning, not parsing, converts it).
    row = df[(df["date"] == pd.Timestamp(2024, 1, 2)) & (df["variety"] == "VarietyA") & (df["metric"] == "low_price")]
    assert row["value"].iloc[0] == "#DIV/0!"


def test_parse_wide_pivot_year_month_excludes_non_year_legend_row(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["PivotSheet"]
    spec = fixture_sheet_config["sheets"]["PivotSheet"]
    df = parse_wide_pivot_year_month(ws, spec)

    # 2 years x 2 months = 4 rows; the legend row must never appear.
    assert len(df) == 4
    assert set(df["year"].unique()) == {2024, 2025}
    assert not df["year"].astype(str).str.contains("legend").any()


def test_parse_wide_pivot_year_month_stops_before_second_stacked_table(fixture_wb, fixture_sheet_config):
    ws = fixture_wb["StackedTablesSheet"]
    spec = fixture_sheet_config["sheets"]["StackedTablesSheet"]
    df = parse_wide_pivot_year_month(ws, spec)

    # Only the first table's 2 months x 2 years = 4 rows; the second
    # (unit-converted) table below the blank separator must be excluded.
    assert len(df) == 4
    assert 450 not in df["arrivals_bags"].values
    assert 495 not in df["arrivals_bags"].values
