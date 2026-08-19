"""Module 5 -- Variety Analysis, Guntur market.

Benchmark = Teja (the most widely referenced variety throughout the brief).
Every other variety present in the sheet is analyzed against it, enumerated
dynamically from the data rather than a hardcoded list so a newly added
variety is picked up automatically on the next refresh.

Also recomputes the full pairwise correlation matrix across all varieties --
this is what replaces the workbook's own "Guntur Variety correlation" sheet,
which Phase 1 deliberately left unread (`derived_skip`) precisely so this
module could recompute it from the real daily price series instead of
trusting a static precomputed snapshot.
"""

import pandas as pd

from analytics import timeseries_stats as ts
from analytics.data_access import date_indexed_series, load_clean_sheet
from analytics.price_module import SHEET_NAME

BENCHMARK = "Teja"

CAVEAT = (
    "Recomputed directly from daily avg_price series -- replaces the "
    "workbook's own precomputed 'Guntur Variety correlation' sheet, which "
    "Phase 1 intentionally did not ingest as a raw source (it's a derived "
    "snapshot, not new data)."
)


def _latest(series: pd.Series, ndigits: int = 4):
    s = series.dropna()
    return None if s.empty else round(float(s.iloc[-1]), ndigits)


def _price_series(df: pd.DataFrame, variety: str) -> pd.Series:
    return date_indexed_series(df, "value", {"variety": variety, "metric": "avg_price"})


def analyze_variety_vs_benchmark(variety_series: pd.Series, benchmark_series: pd.Series) -> dict:
    aligned = pd.concat([variety_series.rename("variety"), benchmark_series.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 30:
        return {"status": "insufficient_evidence", "reason": "Not enough overlapping valid observations with benchmark", "available_n": len(aligned), "required_n": 30}

    ratio = aligned["variety"] / aligned["benchmark"]
    spread = aligned["variety"] - aligned["benchmark"]
    premium_pct = spread / aligned["benchmark"] * 100

    return {
        "status": "ok",
        "latest_ratio_to_benchmark": _latest(ratio),
        "latest_premium_discount_pct": _latest(premium_pct, 2),
        "latest_spread": _latest(spread, 2),
        "spread_std_30d": _latest(spread.rolling(30, min_periods=30).std(), 2),
        "ratio_volatility_cv_30d": _latest(ts.rolling_cv(ratio, 30)),
        "relative_momentum_roc_pct": {
            "roc_30": _latest(ts.momentum_roc(ratio, 30) * 100, 2),
            "roc_90": _latest(ts.momentum_roc(ratio, 90) * 100, 2),
        },
        "correlation_to_benchmark": ts.pearson_corr(aligned["variety"], aligned["benchmark"]),
        "ratio_percentile_distribution": ts.percentile_rank(ratio),
        "observations_used": len(aligned),
    }


def correlation_matrix(series_by_variety: dict) -> dict:
    varieties = list(series_by_variety.keys())
    matrix = {}
    for i, a in enumerate(varieties):
        for b in varieties[i + 1:]:
            sa, sb = series_by_variety[a], series_by_variety[b]
            if sa.dropna().empty or sb.dropna().empty:
                matrix[f"{a} vs {b}"] = {"status": "insufficient_evidence", "reason": "missing series"}
                continue
            matrix[f"{a} vs {b}"] = ts.pearson_corr(sa, sb)
    return matrix


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_clean_sheet(SHEET_NAME)

    all_varieties = sorted(v for v in df["variety"].dropna().unique())
    if BENCHMARK not in all_varieties:
        return {"status": "insufficient_evidence", "reason": f"Benchmark variety '{BENCHMARK}' not found in sheet"}

    series_by_variety = {v: _price_series(df, v) for v in all_varieties}
    benchmark_series = series_by_variety[BENCHMARK]

    per_variety = {
        v: analyze_variety_vs_benchmark(series_by_variety[v], benchmark_series)
        for v in all_varieties
        if v != BENCHMARK
    }

    return {
        "benchmark": BENCHMARK,
        "per_variety_vs_benchmark": per_variety,
        "pairwise_correlation_matrix": correlation_matrix(series_by_variety),
        "data_quality_caveats": [CAVEAT],
    }
