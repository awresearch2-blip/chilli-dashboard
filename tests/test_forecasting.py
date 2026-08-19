import numpy as np
import pandas as pd

from forecasting import backtest, model_selection
from forecasting.feature_engineering import FEATURE_COLUMNS, add_target, build_feature_frame
from forecasting.models import holt_winters, seasonal_naive


def _synthetic_seasonal_series(years=6, yearly_amplitude=1000, base=10000, trend_per_year=200):
    dates = pd.date_range("2018-01-01", periods=365 * years, freq="D")
    day_of_year = dates.dayofyear
    seasonal = yearly_amplitude * np.sin(2 * np.pi * day_of_year / 365)
    trend = trend_per_year * (dates.year - dates.year[0])
    values = base + seasonal + trend
    return pd.Series(values, index=dates)


def test_feature_engineering_uses_only_past_information_no_leakage():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    # A series that jumps sharply on one specific future date -- if any
    # feature at an earlier row picks this up, that's leakage.
    values = np.arange(200, dtype=float)
    series = pd.Series(values, index=dates)
    spike_date = dates[150]
    series.loc[spike_date] += 1_000_000

    features = build_feature_frame(series, None, None)
    row_before_spike = features.loc[dates[149]]
    # None of row 149's features may reflect the spike planted at row 150.
    assert row_before_spike["price_lag_0"] < 1000
    assert row_before_spike["price_rollmean_7"] < 1000
    assert row_before_spike["price_rollmean_30"] < 1000


def test_add_target_uses_exact_calendar_date_not_row_position():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    series = pd.Series(np.arange(100, dtype=float), index=dates)
    # Remove one specific date to create a real gap.
    gap_date = dates[50]
    series = series.drop(gap_date)

    features = build_feature_frame(series, None, None)
    labeled = add_target(features, series, horizon_days=10)

    origin_that_lands_on_gap = gap_date - pd.Timedelta(days=10)
    assert pd.isna(labeled.loc[origin_that_lands_on_gap, "target"])

    origin_with_clean_target = dates[10]
    expected_value = series.loc[dates[10] + pd.Timedelta(days=10)]
    assert labeled.loc[origin_with_clean_target, "target"] == expected_value


def test_seasonal_naive_recovers_the_value_one_year_earlier():
    series = _synthetic_seasonal_series()
    fitted = seasonal_naive.fit(series)
    origin = series.index[-1] - pd.Timedelta(days=30)
    pred = seasonal_naive.predict(fitted, origin, horizon_days=30)
    target_date = origin + pd.Timedelta(days=30)
    # The prediction should exactly match the actual value closest to ~1 year before the target date.
    lookup_date = target_date - pd.Timedelta(days=365)
    window = series[(series.index >= lookup_date - pd.Timedelta(days=3)) & (series.index <= lookup_date + pd.Timedelta(days=3))]
    deltas = np.abs((window.index - lookup_date).days)
    nearby_actual = window.iloc[int(np.argmin(deltas))]
    assert abs(pred["point"] - nearby_actual) < 1e-6


def test_holt_winters_produces_a_plausible_forecast_with_ci():
    series = _synthetic_seasonal_series()
    fitted = holt_winters.fit(series)
    assert fitted is not None
    pred = holt_winters.predict(fitted, series.index[-1], horizon_days=30)
    assert pred is not None
    # Forecast should be in the right ballpark of the series' overall level, not wildly off.
    assert 5000 < pred["point"] < 20000
    if pred["lower"] is not None:
        assert pred["lower"] < pred["point"] < pred["upper"]


def test_backtest_origins_respect_min_train_years_and_horizon_buffer():
    series = _synthetic_seasonal_series(years=8)
    origins = backtest.select_origins(series)
    assert len(origins) == backtest.N_ORIGINS
    first, last = series.dropna().index[0], series.dropna().index[-1]
    for origin in origins:
        assert (origin - first).days >= 365 * backtest.MIN_TRAIN_YEARS - 1
        assert (last - origin).days >= max(backtest.HORIZONS)


def test_backtest_returns_insufficient_evidence_for_short_series():
    short_series = pd.Series(np.arange(50, dtype=float), index=pd.date_range("2024-01-01", periods=50))
    result = backtest.run_backtest(short_series, None, None)
    assert result["status"] == "insufficient_evidence"


def test_model_selection_picks_lowest_rmse_per_horizon():
    fake_backtest = {
        "status": "ok",
        "metrics_by_horizon": {
            "7": {"seasonal_naive": {"rmse": 500, "mae": 1, "mape": 1, "smape": 1, "n_folds": 6}, "sarima": {"rmse": 200, "mae": 1, "mape": 1, "smape": 1, "n_folds": 6}},
            "90": {"seasonal_naive": {"rmse": 300, "mae": 1, "mape": 1, "smape": 1, "n_folds": 6}, "sarima": {"rmse": 900, "mae": 1, "mape": 1, "smape": 1, "n_folds": 6}},
        },
    }
    selection = model_selection.select_best_models(fake_backtest)
    assert selection["selection"]["7"]["model"] == "sarima"
    assert selection["selection"]["90"]["model"] == "seasonal_naive"


def test_feature_columns_all_present_in_built_frame():
    series = _synthetic_seasonal_series(years=2)
    features = build_feature_frame(series, series, series)
    for col in FEATURE_COLUMNS:
        assert col in features.columns
