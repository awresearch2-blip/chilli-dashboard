"""Phase 2c -- composite indices and scores.

Every score here is a documented, inspectable function of metrics already
computed by Phases 2a/2b -- never a new read of raw data, never a new
assumption about the market. Each output includes a `components` (or
`signals`) breakdown so "why is this score X?" is always answerable from
the JSON itself. These are transparent heuristics for a research terminal,
not predictions and not financial advice.

Formulas (bounds are named constants below, each with a one-line reason):

Per-variety (Teja, "334"):
- market_strength_index = avg of {trend regime score, momentum score,
  trailing-1y percentile}, needs >=2 of 3.
- price_stability_index = 100 - scale(volatility_cv_30d).
- risk_score = avg of {scale(volatility_cv_30d), scale(|max_drawdown_pct|)}
  -- market risk only; data-quality concerns live in confidence_score
  instead, so the two concerns are never conflated into one number.
- bullish_score / bearish_score = share of exactly 4 signals (trend regime,
  momentum sign, seasonal deviation sign, market-wide arrival YoY sign)
  that vote bullish/bearish; a signal that's itself insufficient_evidence
  is dropped from the denominator, never counted as neutral.
- confidence_score = avg of (fraction of underlying pieces that are "ok",
  sample-size adequacy) -- an evidentiary-strength meta-score, not a market
  call.
- composite_commodity_index = weighted blend of the above four market-wide
  and per-variety indices (weights below), renormalized over whichever
  components are actually available.

Market-wide (arrivals aren't variety-specific):
- supply_pressure_index = avg of {scale(arrival YoY growth), scale(rolling
  arrivals vs. this month's seasonal norm), scale(balance-sheet surplus)}.
- arrival_pressure_index = avg of {seasonal position score (peak/lean
  month), scale(rolling-30d / rolling-90d arrivals ratio)}.
- demand_index = avg of {scale(offtake YoY growth), scale(offtake MoM
  growth)} -- offtake is the demand-side proxy already computed in
  arrival_analysis.
"""

VARIETIES = ["Teja", "334"]

MOMENTUM_ROC_BOUNDS = (-20, 20)          # % over 30 observations; wider swings are rare in this series (Phase 2a real data)
VOLATILITY_CV_BOUNDS = (0, 0.10)         # observed real CVs were ~2-4%; 10% is a generous ceiling, not a tight one
DRAWDOWN_BOUNDS = (0, 80)                # % magnitude; treated as fully "extreme" at 80% loss from peak
ARRIVAL_YOY_BOUNDS = (-50, 50)           # % YoY arrival growth
OFFTAKE_YOY_BOUNDS = (-50, 50)
OFFTAKE_MOM_BOUNDS = (-20, 20)
ARRIVAL_RATIO_BOUNDS = (0.5, 1.5)        # rolling-30d / rolling-90d arrivals ratio
BALANCE_SHEET_SURPLUS_BOUNDS = (-5, 5)   # Lakh Tons; matches the real magnitude seen in Module 8's output

REGIME_SCORES = {"uptrend": 100, "range_bound": 50, "downtrend": 0}

COMPOSITE_WEIGHTS = {"market_strength": 0.4, "supply_pressure_inverted": 0.3, "demand": 0.2, "stability": 0.1}

BULLISH_MOMENTUM_THRESHOLD = 2.0     # % roc_30
BEARISH_MOMENTUM_THRESHOLD = -2.0
BULLISH_SEASONAL_DEVIATION_THRESHOLD = 5.0    # % above seasonal norm
BEARISH_SEASONAL_DEVIATION_THRESHOLD = -5.0

METHODOLOGY_NOTE = (
    "All scores are transparent, documented formulas over already-computed "
    "Phase 2a/2b metrics -- see this module's docstring for exact formulas. "
    "These are research-terminal heuristics, not predictions or financial advice."
)


def _scale(value, lo, hi):
    if value is None:
        return None
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo) * 100


def _avg(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _get(d, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def market_strength_index(price_result: dict, variety: str) -> dict:
    v = price_result.get(variety, {})
    if v.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": f"No price analysis available for '{variety}'"}

    regime = _get(v, "trend_strength_90obs", "regime") if _get(v, "trend_strength_90obs", "status") == "ok" else None
    regime_score = REGIME_SCORES.get(regime) if regime else None

    roc_30 = _get(v, "momentum_roc_pct", "roc_30")
    momentum_score = _scale(roc_30, *MOMENTUM_ROC_BOUNDS)

    percentile = (
        _get(v, "percentile_distribution", "trailing_1y", "percentile")
        if _get(v, "percentile_distribution", "trailing_1y", "status") == "ok"
        else None
    )

    components = {
        "trend_regime": {"raw": regime, "score": regime_score},
        "momentum_roc_30": {"raw": roc_30, "score": momentum_score},
        "percentile_trailing_1y": {"raw": percentile, "score": percentile},
    }
    available = [c["score"] for c in components.values() if c["score"] is not None]
    if len(available) < 2:
        return {"status": "insufficient_evidence", "reason": "Fewer than 2 of 3 components available", "components": components}
    return {"status": "ok", "index": round(_avg(available), 1), "components_used": len(available), "components": components}


def price_stability_index(price_result: dict, variety: str) -> dict:
    v = price_result.get(variety, {})
    if v.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": f"No price analysis available for '{variety}'"}
    cv = v.get("volatility_cv_30d")
    if cv is None:
        return {"status": "insufficient_evidence", "reason": "No 30-day volatility available"}
    instability = _scale(cv, *VOLATILITY_CV_BOUNDS)
    return {"status": "ok", "index": round(100 - instability, 1), "components": {"volatility_cv_30d": {"raw": cv, "instability_score": round(instability, 1)}}}


def risk_score(price_result: dict, variety: str) -> dict:
    v = price_result.get(variety, {})
    if v.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": f"No price analysis available for '{variety}'"}

    cv = v.get("volatility_cv_30d")
    dd = _get(v, "drawdown", "max_drawdown_pct") if _get(v, "drawdown", "status") == "ok" else None
    vol_score = _scale(cv, *VOLATILITY_CV_BOUNDS) if cv is not None else None
    dd_score = _scale(abs(dd), *DRAWDOWN_BOUNDS) if dd is not None else None

    components = {"volatility_cv_30d": {"raw": cv, "score": vol_score}, "max_drawdown_pct": {"raw": dd, "score": dd_score}}
    available = [c["score"] for c in components.values() if c["score"] is not None]
    if not available:
        return {"status": "insufficient_evidence", "reason": "No volatility or drawdown data available", "components": components}
    return {"status": "ok", "score": round(_avg(available), 1), "components_used": len(available), "components": components}


def _latest_seasonal_deviation_pct(seasonality_result: dict, variety: str):
    monthly = _get(seasonality_result, "price_seasonality", variety, "yoy_seasonal_comparison", "monthly_vs_historical")
    if not monthly:
        return None
    latest_key = max(monthly.keys())  # ISO "YYYY-MM-DD" strings sort chronologically
    return monthly[latest_key].get("deviation_pct")


def bullish_bearish_scores(price_result: dict, seasonality_result: dict, arrival_result: dict, variety: str) -> dict:
    v = price_result.get(variety, {})
    if v.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": f"No price analysis available for '{variety}'"}

    signals = {}

    regime = _get(v, "trend_strength_90obs", "regime") if _get(v, "trend_strength_90obs", "status") == "ok" else None
    if regime is not None:
        signals["trend_regime"] = "bullish" if regime == "uptrend" else ("bearish" if regime == "downtrend" else "neutral")

    roc_30 = _get(v, "momentum_roc_pct", "roc_30")
    if roc_30 is not None:
        signals["momentum_roc_30"] = (
            "bullish" if roc_30 > BULLISH_MOMENTUM_THRESHOLD
            else ("bearish" if roc_30 < BEARISH_MOMENTUM_THRESHOLD else "neutral")
        )

    deviation = _latest_seasonal_deviation_pct(seasonality_result, variety)
    if deviation is not None:
        signals["seasonal_deviation"] = (
            "bullish" if deviation > BULLISH_SEASONAL_DEVIATION_THRESHOLD
            else ("bearish" if deviation < BEARISH_SEASONAL_DEVIATION_THRESHOLD else "neutral")
        )

    arrivals_yoy = (
        _get(arrival_result, "arrivals_bags", "yoy_growth_pct_latest")
        if _get(arrival_result, "arrivals_bags", "status") == "ok"
        else None
    )
    if arrivals_yoy is not None:
        # Falling arrivals (negative YoY) -> supply squeeze -> bullish for price. Rising -> bearish.
        signals["arrival_yoy_growth"] = "bullish" if arrivals_yoy < 0 else ("bearish" if arrivals_yoy > 0 else "neutral")

    if not signals:
        return {"status": "insufficient_evidence", "reason": "No signals available"}

    n = len(signals)
    bullish_votes = sum(1 for s in signals.values() if s == "bullish")
    bearish_votes = sum(1 for s in signals.values() if s == "bearish")
    return {
        "status": "ok",
        "bullish_score": round(bullish_votes / n * 100, 1),
        "bearish_score": round(bearish_votes / n * 100, 1),
        "signals_evaluated": n,
        "signals": signals,
    }


def confidence_score(price_result: dict, variety: str) -> dict:
    v = price_result.get(variety, {})
    if v.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": f"No price analysis available for '{variety}'"}

    checks = [
        _get(v, "trend_strength_90obs", "status"),
        _get(v, "percentile_distribution", "trailing_1y", "status"),
        _get(v, "drawdown", "status"),
    ]
    ok_fraction = sum(1 for c in checks if c == "ok") / len(checks) if checks else 0.0

    ns = [v.get("observations_used")]
    ns = [n for n in ns if n is not None]
    sample_adequacy = _avg([min(n / 100, 1) * 100 for n in ns]) if ns else 0.0

    combined = _avg([ok_fraction * 100, sample_adequacy])
    return {
        "status": "ok",
        "score": round(combined, 1) if combined is not None else None,
        "components": {"component_availability_pct": round(ok_fraction * 100, 1), "sample_size_adequacy_pct": round(sample_adequacy, 1) if sample_adequacy is not None else None},
    }


def supply_pressure_index(arrival_result: dict, balance_sheet_result: dict) -> dict:
    arr = arrival_result.get("arrivals_bags", {})
    if arr.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": "No arrivals data"}

    yoy = arr.get("yoy_growth_pct_latest")
    yoy_score = _scale(yoy, *ARRIVAL_YOY_BOUNDS) if yoy is not None else None

    seasonal_ratio_score = None
    seasonal_ratio = None
    if _get(arr, "seasonal_peak_lean", "status") == "ok" and arr.get("as_of") and arr.get("rolling_30d_mean"):
        current_month = int(arr["as_of"][5:7])
        monthly_avg = _get(arr, "seasonal_peak_lean", "monthly_avg_arrivals_bags", default={}).get(current_month)
        if monthly_avg:
            monthly_equivalent = arr["rolling_30d_mean"] * 30
            seasonal_ratio = monthly_equivalent / monthly_avg
            seasonal_ratio_score = _scale(seasonal_ratio, *ARRIVAL_RATIO_BOUNDS)

    surplus = (
        _get(balance_sheet_result, "surplus_deficit_total_supply_minus_demand", "latest")
        if _get(balance_sheet_result, "surplus_deficit_total_supply_minus_demand", "status") == "ok"
        else None
    )
    surplus_score = _scale(surplus, *BALANCE_SHEET_SURPLUS_BOUNDS) if surplus is not None else None

    components = {
        "arrival_yoy_growth": {"raw": yoy, "score": yoy_score},
        "rolling_arrivals_vs_seasonal_norm": {"raw": seasonal_ratio, "score": seasonal_ratio_score},
        "balance_sheet_surplus_lakh_tons": {"raw": surplus, "score": surplus_score},
    }
    available = [c["score"] for c in components.values() if c["score"] is not None]
    if not available:
        return {"status": "insufficient_evidence", "reason": "No supply-pressure components available", "components": components}
    return {"status": "ok", "index": round(_avg(available), 1), "components_used": len(available), "components": components}


def arrival_pressure_index(arrival_result: dict) -> dict:
    arr = arrival_result.get("arrivals_bags", {})
    if arr.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": "No arrivals data"}

    seasonal_position_score = None
    current_month = int(arr["as_of"][5:7]) if arr.get("as_of") else None
    if current_month is not None and _get(arr, "seasonal_peak_lean", "status") == "ok":
        peak_months = _get(arr, "seasonal_peak_lean", "peak_months") or []
        lean_months = _get(arr, "seasonal_peak_lean", "lean_months") or []
        if current_month in peak_months:
            seasonal_position_score = 80
        elif current_month in lean_months:
            seasonal_position_score = 20
        else:
            seasonal_position_score = 50

    r30, r90 = arr.get("rolling_30d_mean"), arr.get("rolling_90d_mean")
    ratio, ratio_score = None, None
    if r30 is not None and r90:
        ratio = r30 / r90
        ratio_score = _scale(ratio, *ARRIVAL_RATIO_BOUNDS)

    components = {
        "seasonal_position": {"raw": current_month, "score": seasonal_position_score},
        "rolling_30_vs_90_ratio": {"raw": ratio, "score": ratio_score},
    }
    available = [c["score"] for c in components.values() if c["score"] is not None]
    if not available:
        return {"status": "insufficient_evidence", "reason": "No seasonal position or rolling ratio available", "components": components}
    return {"status": "ok", "index": round(_avg(available), 1), "components": components}


def demand_index(arrival_result: dict) -> dict:
    off = arrival_result.get("offtake_bags", {})
    if off.get("status") != "ok":
        return {"status": "insufficient_evidence", "reason": "No offtake data"}

    yoy, mom = off.get("yoy_growth_pct_latest"), off.get("mom_growth_pct_latest")
    yoy_score = _scale(yoy, *OFFTAKE_YOY_BOUNDS) if yoy is not None else None
    mom_score = _scale(mom, *OFFTAKE_MOM_BOUNDS) if mom is not None else None

    components = {"offtake_yoy_growth": {"raw": yoy, "score": yoy_score}, "offtake_mom_growth": {"raw": mom, "score": mom_score}}
    available = [c["score"] for c in components.values() if c["score"] is not None]
    if not available:
        return {"status": "insufficient_evidence", "reason": "No offtake growth data available", "components": components}
    return {"status": "ok", "index": round(_avg(available), 1), "components": components}


def composite_commodity_index(market_strength: dict, supply_pressure: dict, demand: dict, stability: dict) -> dict:
    parts = {
        "market_strength": (market_strength.get("index") if market_strength.get("status") == "ok" else None, COMPOSITE_WEIGHTS["market_strength"]),
        "supply_pressure_inverted": (
            (100 - supply_pressure["index"]) if supply_pressure.get("status") == "ok" else None,
            COMPOSITE_WEIGHTS["supply_pressure_inverted"],
        ),
        "demand": (demand.get("index") if demand.get("status") == "ok" else None, COMPOSITE_WEIGHTS["demand"]),
        "stability": (stability.get("index") if stability.get("status") == "ok" else None, COMPOSITE_WEIGHTS["stability"]),
    }
    available = {k: (v, w) for k, (v, w) in parts.items() if v is not None}
    if not available:
        return {"status": "insufficient_evidence", "reason": "No component indices available"}

    total_weight = sum(w for _, w in available.values())
    score = sum(v * w for v, w in available.values()) / total_weight
    return {
        "status": "ok",
        "index": round(score, 1),
        "components_used": list(available.keys()),
        "weights_renormalized": len(available) < len(parts),
    }


def run(base_results: dict) -> dict:
    price_result = base_results.get("price_analysis", {}) or {}
    arrival_result = base_results.get("arrival_analysis", {}) or {}
    seasonality_result = base_results.get("seasonality", {}) or {}
    balance_sheet_result = base_results.get("balance_sheet", {}) or {}

    market_wide = {
        "supply_pressure_index": supply_pressure_index(arrival_result, balance_sheet_result),
        "arrival_pressure_index": arrival_pressure_index(arrival_result),
        "demand_index": demand_index(arrival_result),
    }

    per_variety = {}
    for variety in VARIETIES:
        ms = market_strength_index(price_result, variety)
        stability = price_stability_index(price_result, variety)
        per_variety[variety] = {
            "market_strength_index": ms,
            "price_stability_index": stability,
            "risk_score": risk_score(price_result, variety),
            "bullish_bearish": bullish_bearish_scores(price_result, seasonality_result, arrival_result, variety),
            "confidence_score": confidence_score(price_result, variety),
            "composite_commodity_index": composite_commodity_index(ms, market_wide["supply_pressure_index"], market_wide["demand_index"], stability),
        }

    return {"per_variety": per_variety, "market_wide": market_wide, "methodology_note": METHODOLOGY_NOTE}
