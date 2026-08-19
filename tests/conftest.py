import openpyxl
import pytest

from tests.fixtures.build_fixture_workbook import build_fixture_workbook


@pytest.fixture
def fixture_workbook_path(tmp_path):
    path = tmp_path / "test_fixture.xlsx"
    build_fixture_workbook(path)
    return path


@pytest.fixture
def fixture_wb(fixture_workbook_path):
    return openpyxl.load_workbook(fixture_workbook_path, data_only=True)


@pytest.fixture
def fixture_sheet_config():
    return {
        "global_invalid_tokens": ["closed", "na", "n/a", "-", "", "#div/0!"],
        "sheets": {
            "LongSheet": {
                "layout": "long",
                "id_column": {"index": 1, "name": "Date", "role": "date"},
                "header_row": 1,
                "data_start_row": 2,
                "trim_trailing_blank": True,
                "columns": {
                    "Price": {"rename": "price", "role": "price"},
                    "Arrivals": {"rename": "arrivals", "role": "arrival"},
                },
                "invalid_tokens": [],
            },
            "MergedSheet": {
                "layout": "wide_pivot_merged_header",
                "id_column": {"index": 1, "name": "Date", "role": "date"},
                "header_rows": [1, 2],
                "data_start_row": 3,
                "trim_trailing_blank": True,
                "group_role": "variety",
                "metric_roles": {
                    "low": {"rename": "low_price", "role": "price"},
                    "high": {"rename": "high_price", "role": "price"},
                },
                "invalid_tokens": ["#div/0!"],
            },
            "PivotSheet": {
                "layout": "wide_pivot_year_month",
                "id_column": {"index": 1, "name": "Year", "role": "year"},
                "header_row": 1,
                "data_start_row": 2,
                "data_row_filter": "id_is_4digit_year",
                "value_name": "index_value",
                "value_role": "index",
                "invalid_tokens": [],
            },
            "StackedTablesSheet": {
                "layout": "wide_pivot_year_month",
                "id_column": {"index": 1, "name": "Months", "role": "month_name"},
                "header_row": 1,
                "data_start_row": 2,
                "data_row_filter": ["stop_at_blank_id"],
                "value_name": "arrivals_bags",
                "value_role": "arrival",
                "invalid_tokens": [],
            },
        },
    }
