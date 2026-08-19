import json

import numpy as np
import pandas as pd
import pytest

from forecasting import accuracy_tracker, explainability
from forecasting.probability import bullish_bearish_probability
from forecasting.similar_periods import find_similar_periods


# ---------------------------------------------------------------------------
# explainability.explain_model
# ---------------------------------------------------------------------------

class _FakeXGBModel:
    feature_importances_ = np.array([0.5, 0.3, 0.0, 0.2] + [0.0] * (19 - 4))


def test_explain_model_xgboost_returns_real_feature_importances():
    from forecasting.feature_engineering import FEATURE_COLUMNS

    registry = {"xgboost": {7: {"model": _FakeXGBModel()}}}
    result = explainability.explain_model("xgboost", 7, registry)
    assert result["status"] == "ok"
    assert result["top_features"][0]["feature"] == FEATURE_COLUMNS[0]
    assert result["top_features"][0]["importance_pct"] == 50.0
    # A zero-importance feature must never appear in the report.
    assert all(f["importance_pct"] > 0 for f in result["top_features"])


def test_explain_model_sarima_reports_real_fitted_order():
    registry = {"sarima": {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 7), "aic": 12345.6}}
    result = explainability.explain_model("sarima", 30, registry)
    assert result["status"] == "ok"
    assert result["order_pdq"] == [1, 1, 1]
    assert result["seasonal_order_PDQs"] == [0, 1, 1, 7]


def test_explain_model_holt_winters_reports_smoothing_params():
    class _FakeHWResult:
        params = {"smoothing_level": 0.8, "smoothing_trend": 0.1, "smoothing_seasonal": 0.05}

    registry = {"holt_winters": {"fitted": _FakeHWResult()}}
    result = explainability.explain_model("holt_winters", 30, registry)
    assert result["status"] == "ok"
    assert result["smoothing_level"] == 0.8


def test_explain_model_seasonal_naive_is_a_plain_lookup_description():
    result = explainability.explain_model("seasonal_naive", 30, {})
    assert result["status"] == "ok"
    assert result["type"] == "lookup"


def test_explain_model_missing_fitted_state_is_insufficient_not_a_crash():
    assert explainability.explain_model("sarima", 30, {"sarima": None})["status"] == "insufficient_evidence"
    assert explainability.explain_model("xgboost", 30, {"xgboost": {}})["status"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# explainability.key_drivers / risks_and_limitations
# ---------------------------------------------------------------------------

def test_key_drivers_surfaces_only_non_neutral_signals():
    composite = {
        "bullish_bearish": {"status": "ok", "bullish_score": 75.0, "bearish_score": 0.0, "signals": {"trend_regime": "bullish", "momentum_roc_30": "neutral"}},
        "market_strength_index": {"status": "ok", "components": {"trend_regime": {"raw": "uptrend", "score": 100}}},
    }
    result = explainability.key_drivers(composite)
    assert result["status"] == "ok"
    assert any("bullish" in s for s in result["statements"])
    assert not any("neutral" in s for s in result["statements"])


def test_key_drivers_insufficient_evidence_when_empty():
    assert explainability.key_drivers({})["status"] == "insufficient_evidence"


def test_risks_flags_long_horizon_and_small_fold_count():
    risks = explainability.risks_and_limitations("sarima", 120, {"n_folds": 3, "mape": 2.0})
    joined = " ".join(risks)
    assert "120-day" in joined
    assert "3 backtest folds" in joined


# ---------------------------------------------------------------------------
# probability.bullish_bearish_probability
# ---------------------------------------------------------------------------

def test_bullish_bearish_probability_reflects_known_residual_bias():
    # Residuals mostly positive (actual > pred) -> forecast understates reality
    # -> most simulated outcomes exceed current_price -> should read bullish.
    records = [{"model": "sarima", "horizon": 30, "actual": 100 + i, "pred": 100} for i in range(1, 11)]
    backtest_result = {"raw_records": records}
    result = bullish_bearish_probability(backtest_result, "sarima", 30, point_forecast=100, current_price=100)
    assert result["status"] == "ok"
    assert result["bullish_probability_pct"] > 50


def test_bullish_bearish_probability_insufficient_evidence_with_few_residuals():
    records = [{"model": "sarima", "horizon": 30, "actual": 101, "pred": 100}]
    result = bullish_bearish_probability({"raw_records": records}, "sarima", 30, 100, 100)
    assert result["status"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# similar_periods.find_similar_periods
# ---------------------------------------------------------------------------

def test_find_similar_periods_excludes_recent_past_and_reports_real_outcome():
    from forecasting.feature_engineering import build_feature_frame

    # Real callers always pass real arrivals/fx series (both are permanent
    # workbook sheets) -- build_feature_frame's None-handling degrades to an
    # all-NaN column, not a dropped one, which would fail every row's dropna.
    dates = pd.date_range("2018-01-01", periods=365 * 6, freq="D")
    day_of_year = dates.dayofyear
    values = 10000 + 500 * np.sin(2 * np.pi * day_of_year / 365)
    series = pd.Series(values, index=dates)
    arrivals = pd.Series(500000 + 10000 * np.sin(2 * np.pi * day_of_year / 365), index=dates)
    fx = pd.Series(np.linspace(65, 90, len(dates)), index=dates)
    features = build_feature_frame(series, arrivals, fx)

    origin = dates[-1]
    result = find_similar_periods(features, series, origin, horizon_days=30)
    assert result["status"] == "ok"
    for period in result["similar_periods"]:
        date = pd.Timestamp(period["date"])
        assert (origin - date).days >= 120  # recent-past exclusion window respected
        # realized_change_pct must be a real, internally-consistent computation.
        expected_pct = round((period["price_after_horizon"] - period["price_then"]) / period["price_then"] * 100, 2)
        assert period["realized_change_pct"] == expected_pct


def test_find_similar_periods_falls_back_when_origin_itself_has_a_gap():
    from forecasting.feature_engineering import build_feature_frame

    dates = pd.date_range("2018-01-01", periods=365 * 6, freq="D")
    day_of_year = dates.dayofyear
    values = 10000 + 500 * np.sin(2 * np.pi * day_of_year / 365)
    series = pd.Series(values, index=dates)
    arrivals = pd.Series(500000 + 10000 * np.sin(2 * np.pi * day_of_year / 365), index=dates)
    fx = pd.Series(np.linspace(65, 90, len(dates)), index=dates)

    # Knock out the single day immediately before the origin -- this poisons
    # price_lag_1 at the origin itself (a real, single-day lookup has no
    # partial-window fallback), the exact scenario found against real data.
    # NaN the value (a real "Closed"-token row still has a date), not drop
    # the row entirely -- dropping would shift lag-by-position instead of
    # by calendar day and not reproduce the real scenario.
    origin = dates[-1]
    series = series.copy()
    series.loc[origin - pd.Timedelta(days=1)] = float("nan")

    features = build_feature_frame(series, arrivals, fx)
    result = find_similar_periods(features, series, origin, horizon_days=30)
    assert result["status"] == "ok"
    # Must disclose that a fallback date was used, not silently substitute it.
    # The NaN'd row itself also fails (its own price_lag_0 is now unknown), so
    # the fallback lands 2 days back, not 1 -- both skips are correct.
    assert "note" in result
    assert "day(s) earlier" in result["note"]


def test_find_similar_periods_insufficient_evidence_on_short_series():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    series = pd.Series(np.arange(40, dtype=float), index=dates)
    arrivals = pd.Series(np.arange(40, dtype=float) + 500, index=dates)
    fx = pd.Series(np.linspace(70, 75, 40), index=dates)
    from forecasting.feature_engineering import build_feature_frame

    features = build_feature_frame(series, arrivals, fx)
    result = find_similar_periods(features, series, dates[-1], horizon_days=7)
    assert result["status"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# accuracy_tracker
# ---------------------------------------------------------------------------

def test_accuracy_tracker_reports_insufficient_evidence_on_empty_log(tmp_path, monkeypatch):
    fake_log = tmp_path / "forecast_log.jsonl"
    monkeypatch.setattr(accuracy_tracker, "FORECAST_LOG_PATH", fake_log)
    result = accuracy_tracker.compute_realized_accuracy({"Teja": pd.Series([1.0], index=[pd.Timestamp("2024-01-01")])})
    assert result["status"] == "insufficient_evidence"
    assert "freshly-started" in result["reason"]


def test_accuracy_tracker_scores_a_logged_forecast_whose_target_date_has_passed(tmp_path, monkeypatch):
    fake_log = tmp_path / "forecast_log.jsonl"
    monkeypatch.setattr(accuracy_tracker, "FORECAST_LOG_PATH", fake_log)

    entry = {
        "refresh_id": "test", "variety": "Teja", "as_of": "2024-01-01",
        "horizon_days": 30, "target_date": "2024-01-31", "point_forecast": 10000.0, "model_used": "sarima",
    }
    fake_log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    price_series = pd.Series([10500.0], index=[pd.Timestamp("2024-01-31")])
    result = accuracy_tracker.compute_realized_accuracy({"Teja": price_series})
    assert result["status"] == "ok"
    assert result["forecasts_scored"] == 1
    accuracy = result["realized_accuracy"]["Teja"]["30"]["sarima"]
    assert accuracy["mae"] == pytest.approx(500.0)


def test_accuracy_tracker_pending_forecasts_whose_target_date_is_still_future(tmp_path, monkeypatch):
    fake_log = tmp_path / "forecast_log.jsonl"
    monkeypatch.setattr(accuracy_tracker, "FORECAST_LOG_PATH", fake_log)

    entry = {
        "refresh_id": "test", "variety": "Teja", "as_of": "2026-07-20",
        "horizon_days": 180, "target_date": "2027-01-16", "point_forecast": 21000.0, "model_used": "holt_winters",
    }
    fake_log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    price_series = pd.Series([20750.0], index=[pd.Timestamp("2026-07-20")])
    result = accuracy_tracker.compute_realized_accuracy({"Teja": price_series})
    assert result["status"] == "insufficient_evidence"
    assert result["forecasts_pending"] == 1
