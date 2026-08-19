"""Wiring tests for Phase 2b modules (Variety, Market Comparison, Export,
Balance Sheet, FX, Cold Storage) against small synthetic clean-shaped
DataFrames -- same spirit as tests/test_analytics_modules.py.
"""

import numpy as np
import pandas as pd

from analytics import (
    balance_sheet_module,
    cold_storage_module,
    export_module,
    fx_module,
    market_comparison_module,
    variety_module,
)

DATES = pd.date_range("2022-01-01", "2024-06-30", freq="D")


def _price_df(varieties_and_base):
    rows = []
    for variety, base in varieties_and_base:
        trend = np.linspace(base, base * 1.2, len(DATES))
        avg = trend + np.random.RandomState(hash(variety) % 1000).normal(0, 30, len(DATES))
        for metric, values in [("avg_price", avg), ("low_price", avg - 100), ("high_price", avg + 100)]:
            rows.append(pd.DataFrame({"date": DATES, "variety": variety, "metric": metric, "value": values}))
    return pd.concat(rows, ignore_index=True)


def _long_price_df(base, column="avg_price"):
    trend = np.linspace(base, base * 1.15, len(DATES))
    values = trend + np.random.RandomState(2).normal(0, 40, len(DATES))
    return pd.DataFrame({"date": DATES, column: values})


def _khammam_df():
    rows = []
    for metric, base in [("avg_price_non_cold_storage", 8000), ("avg_price_cold_storage", 8500)]:
        values = base + np.random.RandomState(3).normal(0, 50, len(DATES))
        rows.append(pd.DataFrame({"date": DATES, "metric": metric, "value": values}))
    return pd.concat(rows, ignore_index=True)


def _balance_sheet_df():
    rows = []
    for category, base in [("Total Supply", 10.0), ("Total Demand/Usage", 9.0), ("Production", 8.0), ("Import", 0.0)]:
        for i, year in enumerate(range(2017, 2026)):
            rows.append({"category": category, "year": str(year), "value_lakh_tons": base + i * 0.2, "is_estimate": False})
        rows.append({"category": category, "year": "2026(exp)", "value_lakh_tons": base + 2.0, "is_estimate": True})
    return pd.DataFrame(rows)


def _cold_storage_df():
    return pd.DataFrame(
        {
            "year": [2025, 2026],
            "month": ["Apr", "Mar"],
            "date": [pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-01")],
            "ap_stock_bags": [1000000.0, None],
            "telangana_stock_bags": [None, 900000.0],
            "karnataka_stock_bags": [None, None],
            "guntur_stock_bags": [500000.0, None],
            "warangal_stock_bags": [None, None],
            "khammam_stock_bags": [None, 400000.0],
        }
    )


def test_variety_module_recomputes_correlation_matrix():
    df = _price_df([("Teja", 10000), ("334", 7000), ("Byadgi", 9000)])
    result = variety_module.run(df)
    assert result["benchmark"] == "Teja"
    assert set(result["per_variety_vs_benchmark"].keys()) == {"334", "Byadgi"}
    assert result["per_variety_vs_benchmark"]["334"]["status"] == "ok"
    assert "334 vs Byadgi" in result["pairwise_correlation_matrix"] or "Byadgi vs 334" not in result["pairwise_correlation_matrix"]


def test_market_comparison_module_flags_documented_caveats():
    guntur_df = _price_df([("Teja", 10000)])
    warangal_df = _long_price_df(9500)
    khammam_df = _khammam_df()
    result = market_comparison_module.run(guntur_df, warangal_df, khammam_df)
    assert set(result["pairs"].keys()) == {
        "guntur_vs_warangal", "guntur_vs_khammam_noncold", "warangal_vs_khammam_noncold", "guntur_vs_khammam_cold",
    }
    assert market_comparison_module.ANOMALY_CAVEAT in result["data_quality_caveats"]


def test_export_module_computes_seasonal_index_and_price_correlation():
    months = pd.date_range("2015-01-01", "2026-06-01", freq="MS")
    exports_df = pd.DataFrame({"date": months, "export_volume": np.linspace(20000, 40000, len(months))})
    price_df = _price_df([("Teja", 10000), ("334", 7000)])
    result = export_module.run(exports_df, price_df)
    assert result["status"] == "ok"
    assert result["seasonal_export_index"]["status"] == "ok"
    assert "Teja" in result["vs_price"] and "334" in result["vs_price"]


def test_balance_sheet_module_excludes_estimate_and_uses_lower_min_n():
    price_df = _price_df([("Teja", 10000), ("334", 7000)])
    result = balance_sheet_module.run(_balance_sheet_df(), price_df)
    total_supply = result["per_category"]["Total Supply"]
    assert total_supply["estimate_2026"] is not None
    assert "2026(exp)" not in total_supply["annual_values_lakh_tons"]
    # Import is constant 0.0 every year -- must be reported as insufficient
    # evidence (undefined correlation), never a fabricated r value.
    import_corr = result["per_category"]["Import"]["correlation_with_price"]["Teja"]
    assert import_corr["status"] == "insufficient_evidence"


def test_fx_module_includes_trend_correlation_caveat():
    fx_df = _long_price_df(70, column="usd_inr_rate")
    price_df = _price_df([("Teja", 10000), ("334", 7000)])
    result = fx_module.run(fx_df, price_df)
    assert result["status"] == "ok"
    assert fx_module.TREND_CORRELATION_CAVEAT in result["vs_price"]["Teja"]["data_quality_caveats"]


def test_cold_storage_module_reports_insufficient_evidence_not_fabricated_trend():
    result = cold_storage_module.run(_cold_storage_df())
    assert result["per_region"]["Andhra Pradesh"]["status"] == "insufficient_evidence_for_trend"
    assert result["per_region"]["Andhra Pradesh"]["latest_known_value"] == 1000000.0
    assert result["per_region"]["Karnataka"]["status"] == "insufficient_evidence"
    assert result["impact_on_price"]["status"] == "insufficient_evidence"
