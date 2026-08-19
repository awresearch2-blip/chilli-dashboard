"""Lightweight wiring tests for the Phase 2a modules: correct column names
and shapes end-to-end against small synthetic clean-shaped DataFrames (not
the real workbook -- the real numbers were already reviewed manually
against the actual data during Phase 2a verification).
"""

import numpy as np
import pandas as pd

from analytics import arrival_module, price_arrival_module, price_module, seasonality_module


def _synthetic_price_df():
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    rows = []
    for variety, base in [("Teja", 10000), ("334", 7000)]:
        trend = np.linspace(base, base * 1.3, len(dates))
        avg = trend + np.random.RandomState(0).normal(0, 50, len(dates))
        for metric, values in [("avg_price", avg), ("low_price", avg - 200), ("high_price", avg + 200)]:
            rows.append(pd.DataFrame({"date": dates, "variety": variety, "metric": metric, "value": values}))
    return pd.concat(rows, ignore_index=True)


def _synthetic_arrivals_df():
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    rng = np.random.RandomState(1)
    return pd.DataFrame(
        {
            "date": dates,
            "arrivals_bags": rng.normal(500000, 50000, len(dates)).clip(min=0),
            "offtake_bags": rng.normal(400000, 40000, len(dates)).clip(min=0),
        }
    )


def _synthetic_seasonality_sheet_df():
    rows = []
    for year in range(2022, 2025):
        for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
            rows.append({"year": year, "period_label": month, "seasonality_index": 10000.0})
    return pd.DataFrame(rows)


def test_price_module_runs_for_both_priority_varieties():
    result = price_module.run(_synthetic_price_df())
    assert set(result.keys()) == {"Teja", "334"}
    for variety_result in result.values():
        assert variety_result["status"] == "ok"
        assert "cagr" in variety_result
        assert "moving_averages" in variety_result


def test_arrival_module_runs_for_both_fields():
    result = arrival_module.run(_synthetic_arrivals_df())
    assert set(result.keys()) == {"arrivals_bags", "offtake_bags"}
    for field_result in result.values():
        assert field_result["status"] == "ok"
        assert "seasonal_peak_lean" in field_result


def test_price_arrival_module_includes_the_documented_caveat():
    result = price_arrival_module.run(_synthetic_price_df(), _synthetic_arrivals_df())
    for variety_result in result.values():
        assert variety_result["status"] == "ok"
        assert price_arrival_module.CAVEAT in variety_result["data_quality_caveats"]


def test_seasonality_module_flags_workbook_sheet_as_reference_only():
    result = seasonality_module.run(
        _synthetic_price_df(), _synthetic_arrivals_df(), _synthetic_seasonality_sheet_df()
    )
    teja = result["price_seasonality"]["Teja"]
    assert "workbook_reported_monthly_avg_by_month" in teja
    assert seasonality_module.WORKBOOK_SHEET_CAVEAT in teja["data_quality_caveats"]
    assert "harvest_vs_offseason" in result
