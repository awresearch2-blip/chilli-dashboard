import numpy as np
import pandas as pd

from analytics import timeseries_stats as ts


def test_cagr_matches_hand_computed_doubling():
    idx = pd.date_range("2022-01-01", "2024-01-01", freq="D")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    result = ts.cagr(s)
    assert result["status"] == "ok"
    # Doubling over exactly 2 years -> CAGR = sqrt(2) - 1 ~= 41.42%
    assert abs(result["cagr_pct"] - 41.42) < 1.0


def test_cagr_insufficient_evidence_below_min_n():
    short = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3))
    result = ts.cagr(short, min_n=30)
    assert result["status"] == "insufficient_evidence"
    assert result["available_n"] == 3


def test_sma_matches_manual_rolling_mean():
    s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5))
    result = ts.sma(s, 3).tolist()
    assert result[:2] == [None, None] or all(pd.isna(v) for v in result[:2])
    assert result[2:] == [2.0, 3.0, 4.0]


def test_drawdown_finds_correct_peak_trough_and_recovery():
    vals = [100, 90, 80, 70, 60, 50, 60, 70, 90, 100, 110]
    s = pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals)))
    result = ts.drawdown(s)
    assert result["max_drawdown_pct"] == -50.0
    assert result["peak_date"] == "2024-01-01"
    assert result["trough_date"] == "2024-01-06"
    assert result["recovered"] is True
    assert result["days_to_recover"] == 4


def test_seasonal_index_recovers_known_injected_pattern():
    dates = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    trend = np.linspace(100, 150, len(dates))
    factor = np.array([0.8 if d.month == 1 else (1.2 if d.month == 7 else 1.0) for d in dates])
    s = pd.Series(trend * factor, index=dates)

    result = ts.seasonal_index(s)
    assert result["status"] == "ok"
    assert abs(result["index_by_month"][1]["index"] - 80.0) < 2.0
    assert abs(result["index_by_month"][7]["index"] - 120.0) < 2.0
    assert abs(result["index_by_month"][4]["index"] - 100.0) < 2.0


def test_pearson_corr_detects_known_relationship_and_gates_on_min_n():
    x = pd.Series(np.arange(50), index=pd.date_range("2024-01-01", periods=50))
    y = x * 2 + 1  # perfectly correlated
    result = ts.pearson_corr(x, y, min_n=30)
    assert result["status"] == "ok"
    assert abs(result["r"] - 1.0) < 1e-6

    short_x, short_y = x.head(5), y.head(5)
    result_short = ts.pearson_corr(short_x, short_y, min_n=30)
    assert result_short["status"] == "insufficient_evidence"


def test_lag_correlogram_recovers_known_lead_lag_relationship():
    idx = pd.date_range("2024-01-01", periods=200)
    x = pd.Series(np.sin(np.linspace(0, 20, 200)), index=idx)
    y = x.shift(5)  # y is x shifted 5 observations later -> x leads y by 5
    result = ts.lag_correlogram(x, y, max_lag=10, min_n=30)
    assert result["status"] == "ok"
    assert result["best_lag"] in (-5, 5)  # sign convention -- either way, magnitude 5 must win


def test_rolling_zscore_matches_hand_computed_value():
    s = pd.Series([10, 20, 30, 40], index=pd.date_range("2024-01-01", periods=4))
    result = ts.rolling_zscore(s, window=4)
    assert pd.isna(result.iloc[2])  # window not yet full
    expected = (40 - s.mean()) / s.std()
    assert abs(result.iloc[3] - expected) < 1e-9

    # A value far outside its own trailing window's spread must be the largest
    # |z| in the series (the window includes the point itself, so the z-score
    # is damped by the spike inflating its own std -- it's still the max).
    spiky = pd.Series([10, 11, 9, 10, 100], index=pd.date_range("2024-02-01", periods=5))
    spiky_z = ts.rolling_zscore(spiky, window=4).abs()
    assert spiky_z.idxmax() == spiky.index[4]


def test_log_log_elasticity_recovers_known_slope():
    x = pd.Series(np.linspace(1, 100, 100), index=pd.date_range("2024-01-01", periods=100))
    y = x ** 0.5  # elasticity should be ~0.5
    result = ts.log_log_elasticity(x, y, min_n=30)
    assert result["status"] == "ok"
    assert abs(result["elasticity"] - 0.5) < 0.01
