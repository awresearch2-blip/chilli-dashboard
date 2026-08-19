"""Tests for Phase 2c's composite scoring, against small hand-crafted
synthetic result dicts (this module needs no DataFrames -- it composes
other modules' already-computed outputs in memory).
"""

from analytics import composite_index_module as cim


def _price_result(regime="uptrend", roc_30=5.0, percentile=90.0, cv=0.03, drawdown_pct=-10.0, n=500, trend_status="ok", percentile_status="ok", drawdown_status="ok"):
    return {
        "Teja": {
            "status": "ok",
            "volatility_cv_30d": cv,
            "drawdown": {"status": drawdown_status, "max_drawdown_pct": drawdown_pct},
            "momentum_roc_pct": {"roc_30": roc_30},
            "trend_strength_90obs": {"status": trend_status, "regime": regime},
            "percentile_distribution": {"trailing_1y": {"status": percentile_status, "percentile": percentile}},
            "observations_used": n,
        }
    }


def _seasonality_result(deviation_pct=10.0):
    return {
        "price_seasonality": {
            "Teja": {
                "yoy_seasonal_comparison": {
                    "status": "ok",
                    "monthly_vs_historical": {
                        "2026-06-01": {"deviation_pct": deviation_pct - 5},
                        "2026-07-01": {"deviation_pct": deviation_pct},
                    },
                }
            }
        }
    }


def _arrival_result(arrivals_yoy=-20.0, offtake_yoy=-10.0, offtake_mom=-5.0):
    return {
        "arrivals_bags": {
            "status": "ok",
            "yoy_growth_pct_latest": arrivals_yoy,
            "as_of": "2026-07-20",
            "rolling_30d_mean": 40000,
            "rolling_90d_mean": 80000,
            "seasonal_peak_lean": {"status": "ok", "peak_months": [3], "lean_months": [6], "monthly_avg_arrivals_bags": {7: 1000000}},
        },
        "offtake_bags": {"status": "ok", "yoy_growth_pct_latest": offtake_yoy, "mom_growth_pct_latest": offtake_mom},
    }


def test_clearly_bullish_scenario_scores_high_bullish_low_bearish():
    base = {
        "price_analysis": _price_result(regime="uptrend", roc_30=5.0),
        "seasonality": _seasonality_result(deviation_pct=15.0),
        "arrival_analysis": _arrival_result(arrivals_yoy=-20.0),
        "balance_sheet": {},
    }
    result = cim.run(base)
    bb = result["per_variety"]["Teja"]["bullish_bearish"]
    assert bb["bullish_score"] == 100.0
    assert bb["bearish_score"] == 0.0
    assert bb["signals_evaluated"] == 4


def test_clearly_bearish_scenario_is_the_mirror_image():
    base = {
        "price_analysis": _price_result(regime="downtrend", roc_30=-5.0),
        "seasonality": _seasonality_result(deviation_pct=-15.0),
        "arrival_analysis": _arrival_result(arrivals_yoy=20.0),
        "balance_sheet": {},
    }
    result = cim.run(base)
    bb = result["per_variety"]["Teja"]["bullish_bearish"]
    assert bb["bearish_score"] == 100.0
    assert bb["bullish_score"] == 0.0


def test_insufficient_evidence_signals_are_excluded_not_counted_neutral():
    base = {
        "price_analysis": _price_result(trend_status="insufficient_evidence", roc_30=None),
        "seasonality": _seasonality_result(deviation_pct=15.0),
        "arrival_analysis": _arrival_result(arrivals_yoy=-20.0),
        "balance_sheet": {},
    }
    result = cim.run(base)
    bb = result["per_variety"]["Teja"]["bullish_bearish"]
    # Only 2 signals are actually available (seasonal deviation, arrival yoy) --
    # both bullish. If the missing ones were wrongly counted as neutral this
    # would be 50.0 instead of 100.0.
    assert bb["signals_evaluated"] == 2
    assert bb["bullish_score"] == 100.0


def test_confidence_score_drops_when_inputs_are_missing():
    full = cim.run({
        "price_analysis": _price_result(),
        "seasonality": _seasonality_result(),
        "arrival_analysis": _arrival_result(),
        "balance_sheet": {},
    })
    thin = cim.run({
        "price_analysis": _price_result(trend_status="insufficient_evidence", percentile_status="insufficient_evidence", drawdown_status="insufficient_evidence", n=None),
        "seasonality": _seasonality_result(),
        "arrival_analysis": _arrival_result(),
        "balance_sheet": {},
    })
    full_conf = full["per_variety"]["Teja"]["confidence_score"]["score"]
    thin_conf = thin["per_variety"]["Teja"]["confidence_score"]["score"]
    assert thin_conf < full_conf


def test_market_strength_requires_at_least_two_of_three_components():
    base = _price_result(trend_status="insufficient_evidence", roc_30=None, percentile_status="insufficient_evidence")
    result = cim.market_strength_index(base, "Teja")
    assert result["status"] == "insufficient_evidence"


def test_supply_pressure_index_reflects_falling_arrivals_as_low_pressure():
    arrival_result = _arrival_result(arrivals_yoy=-40.0)
    result = cim.supply_pressure_index(arrival_result, {})
    assert result["status"] == "ok"
    assert result["index"] < 50  # sharply falling arrivals -> low supply pressure


def test_composite_commodity_index_renormalizes_over_available_components():
    ms = {"status": "ok", "index": 80}
    supply = {"status": "insufficient_evidence"}  # missing on purpose
    demand = {"status": "ok", "index": 60}
    stability = {"status": "ok", "index": 90}
    result = cim.composite_commodity_index(ms, supply, demand, stability)
    assert result["status"] == "ok"
    assert result["weights_renormalized"] is True
    assert "supply_pressure_inverted" not in result["components_used"]


def test_missing_variety_returns_insufficient_evidence_not_a_crash():
    base = {"price_analysis": _price_result(), "seasonality": _seasonality_result(), "arrival_analysis": _arrival_result(), "balance_sheet": {}}
    del base["price_analysis"]["Teja"]
    result = cim.run(base)
    assert result["per_variety"]["Teja"]["market_strength_index"]["status"] == "insufficient_evidence"
    assert result["per_variety"]["334"]["market_strength_index"]["status"] == "insufficient_evidence"
