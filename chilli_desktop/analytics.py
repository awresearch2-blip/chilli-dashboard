"""The statistical engine: every test, decomposition and relationship measure.

Each public function returns a :class:`~chilli_desktop.utils.Result`. When the
workbook cannot support a calculation -- too few observations, no overlapping
history, a singular design matrix -- the Result explains why instead of
returning a number that would look authoritative and be meaningless.

Conventions
-----------
*   ``freq`` is a pandas offset alias: ``D`` daily, ``W`` weekly, ``SME``
    fortnightly (semi-month end), ``ME`` monthly.
*   "Lead" is expressed in *periods of the analysis frequency*, and every
    lead-lag result also reports the equivalent in days so a trader can read
    it directly.
*   Positive lag *k* in :func:`cross_correlation` means the first series leads
    the second by *k* periods.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import (
    acf,
    adfuller,
    coint,
    grangercausalitytests,
    kpss,
    pacf,
)
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from . import settings
from .utils import (
    LOG,
    Result,
    describe_strength,
    periods_per_year,
    safe_analysis,
)

CFG = settings.ANALYTICS


# ==========================================================================
# Descriptive statistics
# ==========================================================================


@safe_analysis()
def descriptive_stats(series: pd.Series, source: str = "") -> Result[pd.Series]:
    """Full descriptive profile of a series, including shape and tails."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        return Result.unavailable(
            f"Only {len(clean)} observation(s) available; at least 3 are "
            "needed for descriptive statistics.",
            source,
        )
    values = clean.to_numpy(dtype=float)
    stats = {
        "Observations": float(len(values)),
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "Std deviation": float(np.std(values, ddof=1)),
        "Coefficient of variation": (
            float(np.std(values, ddof=1) / np.mean(values)) if np.mean(values) else np.nan
        ),
        "Minimum": float(np.min(values)),
        "25th percentile": float(np.percentile(values, 25)),
        "75th percentile": float(np.percentile(values, 75)),
        "Maximum": float(np.max(values)),
        "Range": float(np.max(values) - np.min(values)),
        "Interquartile range": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "Skewness": float(sp_stats.skew(values)) if len(values) > 2 else np.nan,
        "Kurtosis (excess)": float(sp_stats.kurtosis(values)) if len(values) > 3 else np.nan,
    }
    if isinstance(clean.index, pd.DatetimeIndex):
        stats["First observation"] = float(clean.index.min().value)
        stats["Last observation"] = float(clean.index.max().value)
    out = pd.Series(stats, name=str(series.name or "series"))
    return Result.of(out, source)


@safe_analysis()
def rolling_statistics(
    series: pd.Series, window: int, source: str = ""
) -> Result[pd.DataFrame]:
    """Rolling mean, standard deviation, volatility, and simple/exponential MAs.

    Volatility is the annualised standard deviation of log returns over the
    same window, scaled by the series' own frequency.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < window + 1:
        return Result.unavailable(
            f"A {window}-period window needs at least {window + 1} "
            f"observations; only {len(clean)} are available.",
            source,
        )
    inferred = _infer_freq_alias(clean)
    ppy = periods_per_year(inferred)
    returns = np.log(clean / clean.shift(1)).replace([np.inf, -np.inf], np.nan)

    frame = pd.DataFrame(
        {
            "Value": clean,
            f"Rolling mean ({window})": clean.rolling(window).mean(),
            f"Rolling std ({window})": clean.rolling(window).std(ddof=1),
            f"Moving average ({window})": clean.rolling(window).mean(),
            f"EMA ({window})": clean.ewm(span=window, adjust=False).mean(),
            f"Rolling volatility ({window}, annualised)": returns.rolling(window).std(ddof=1)
            * np.sqrt(ppy),
        }
    )
    upper = frame[f"Rolling mean ({window})"] + 2 * frame[f"Rolling std ({window})"]
    lower = frame[f"Rolling mean ({window})"] - 2 * frame[f"Rolling std ({window})"]
    frame["Upper band (+2 sd)"] = upper
    frame["Lower band (-2 sd)"] = lower
    return Result.of(
        frame,
        source,
        [f"Volatility annualised using {ppy} periods per year for '{inferred}' data."],
    )


@safe_analysis()
def change_summary(series: pd.Series, source: str = "") -> Result[pd.DataFrame]:
    """Period-on-period, week-on-week, month-on-month and year-on-year change.

    Each horizon is computed by *date offset*, not by row count, so an
    irregular trading calendar does not distort the comparison. The reference
    observation is the last one at or before the target date; when no
    observation exists within a tolerance window, the change is reported as
    unavailable rather than approximated.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return Result.unavailable("The series contains no observations.", source)
    if not isinstance(clean.index, pd.DatetimeIndex):
        return Result.unavailable(
            "Change-over-time requires a date-indexed series.", source
        )

    latest_date = clean.index[-1]
    latest_value = float(clean.iloc[-1])

    horizons: list[tuple[str, pd.Timestamp, int]] = [
        ("Previous observation", clean.index[-2] if len(clean) > 1 else latest_date, 0),
        ("Week on week (WoW)", latest_date - pd.Timedelta(days=7), 5),
        ("Fortnight on fortnight", latest_date - pd.Timedelta(days=14), 6),
        ("Month on month (MoM)", latest_date - pd.DateOffset(months=1), 10),
        ("Quarter on quarter (QoQ)", latest_date - pd.DateOffset(months=3), 15),
        ("Year on year (YoY)", latest_date - pd.DateOffset(years=1), 20),
    ]

    rows: list[dict[str, Any]] = []
    for label, target, tolerance in horizons:
        target = pd.Timestamp(target)
        eligible = clean[clean.index <= target]
        if eligible.empty:
            rows.append(
                {
                    "Horizon": label,
                    "Reference date": pd.NaT,
                    "Reference value": np.nan,
                    "Change": np.nan,
                    "Change %": np.nan,
                    "Note": "No observation on or before the comparison date.",
                }
            )
            continue
        ref_date = eligible.index[-1]
        drift = (target - ref_date).days
        if tolerance and drift > tolerance:
            note = (
                f"Nearest prior observation is {drift} days before the target "
                f"date (tolerance {tolerance}); treat with care."
            )
        else:
            note = ""
        ref_value = float(eligible.iloc[-1])
        rows.append(
            {
                "Horizon": label,
                "Reference date": ref_date,
                "Reference value": ref_value,
                "Change": latest_value - ref_value,
                "Change %": (latest_value - ref_value) / ref_value if ref_value else np.nan,
                "Note": note,
            }
        )

    frame = pd.DataFrame(rows)
    return Result.of(
        frame,
        source,
        [
            f"Latest observation: {latest_date:%d %b %Y} at "
            f"{latest_value:,.0f}. Comparisons use the last observation at or "
            "before each target date.",
        ],
    )


@safe_analysis()
def zscore_and_outliers(
    series: pd.Series, window: int | None = None, source: str = ""
) -> Result[pd.DataFrame]:
    """Z-scores plus outlier flags from both the z-rule and the IQR rule."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 10:
        return Result.unavailable(
            f"Only {len(clean)} observation(s); at least 10 are needed for "
            "outlier detection.",
            source,
        )
    frame = pd.DataFrame({"Value": clean})
    mean, std = clean.mean(), clean.std(ddof=1)
    frame["Z-score (full sample)"] = (clean - mean) / std if std else np.nan

    if window and len(clean) > window:
        roll_mean = clean.rolling(window).mean()
        roll_std = clean.rolling(window).std(ddof=1)
        frame[f"Z-score (rolling {window})"] = (clean - roll_mean) / roll_std

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    frame["IQR outlier"] = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
    frame["Z outlier"] = frame["Z-score (full sample)"].abs() > CFG.outlier_z_threshold

    n_z = int(frame["Z outlier"].sum())
    n_iqr = int(frame["IQR outlier"].sum())
    return Result.of(
        frame,
        source,
        [
            f"{n_z} observation(s) exceed |z| > {CFG.outlier_z_threshold}; "
            f"{n_iqr} fall outside the 1.5x IQR fence. Outliers are flagged "
            "for review only -- none are removed from any calculation.",
        ],
    )


# ==========================================================================
# Decomposition
# ==========================================================================


def _infer_freq_alias(series: pd.Series) -> str:
    """Guess the analysis frequency of a date-indexed series from its spacing."""
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 3:
        return "D"
    median_gap = float(pd.Series(series.index).diff().dt.days.median())
    if median_gap <= 3:
        return "D"
    if median_gap <= 10:
        return "W"
    if median_gap <= 20:
        return "SME"
    return "ME"


@safe_analysis()
def decompose(
    series: pd.Series,
    freq: str | None = None,
    method: str = "STL",
    source: str = "",
) -> Result[pd.DataFrame]:
    """Split a series into trend, seasonal and residual components.

    STL is preferred (it tolerates changing seasonal amplitude, which chilli
    prices show). Classical decomposition is used as a fallback. The series is
    regularised onto its own frequency grid first, because both routines need
    evenly spaced observations; the interpolation used for that is confined to
    the decomposition and never leaks into any other statistic.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    alias = freq or _infer_freq_alias(clean)
    period = periods_per_year(alias)

    if len(clean) < 2 * period + 1:
        return Result.unavailable(
            f"Seasonal decomposition at {alias} frequency needs at least "
            f"{2 * period + 1} observations (two full cycles of {period}); "
            f"only {len(clean)} are available.",
            source,
        )

    grid = clean.resample(alias).mean()
    filled = grid.interpolate(method="time", limit_direction="both")
    n_interpolated = int(grid.isna().sum())

    if method.upper() == "STL":
        fitted = STL(filled, period=period, robust=True).fit()
        frame = pd.DataFrame(
            {
                "Observed": filled,
                "Trend": fitted.trend,
                "Seasonal": fitted.seasonal,
                "Residual": fitted.resid,
            }
        )
        used = "STL (robust, LOESS-based)"
    else:
        fitted = seasonal_decompose(filled, model="additive", period=period)
        frame = pd.DataFrame(
            {
                "Observed": filled,
                "Trend": fitted.trend,
                "Seasonal": fitted.seasonal,
                "Residual": fitted.resid,
            }
        )
        used = "Classical additive decomposition"

    variance = frame[["Trend", "Seasonal", "Residual"]].var()
    total = float(variance.sum())
    shares = (variance / total).to_dict() if total else {}

    notes = [
        f"Method: {used}, seasonal period {period} at {alias} frequency.",
        (
            f"{n_interpolated} empty period(s) on the regular {alias} grid were "
            "time-interpolated so the decomposition could run; this affects "
            "the decomposition only."
        )
        if n_interpolated
        else "No interpolation was required.",
    ]
    if shares:
        notes.append(
            "Variance share -- trend "
            f"{shares.get('Trend', 0):.0%}, seasonal {shares.get('Seasonal', 0):.0%}, "
            f"residual {shares.get('Residual', 0):.0%}."
        )
    result = Result.of(frame, source, notes)
    result.value.attrs["variance_shares"] = shares
    result.value.attrs["period"] = period
    return result


@safe_analysis()
def seasonal_indices(
    series: pd.Series, source: str = ""
) -> Result[pd.DataFrame]:
    """Monthly seasonal indices computed as month mean / overall mean.

    This mirrors the arithmetic the workbook itself uses on its seasonality
    sheet, so the two can be compared like for like.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(clean.index, pd.DatetimeIndex):
        return Result.unavailable("A date-indexed series is required.", source)
    if len(clean) < 24:
        return Result.unavailable(
            f"Monthly seasonal indices need at least 24 observations; "
            f"{len(clean)} are available.",
            source,
        )

    monthly = clean.resample("ME").mean().dropna()
    by_month = monthly.groupby(monthly.index.month)
    overall = float(monthly.mean())
    table = pd.DataFrame(
        {
            "Mean": by_month.mean(),
            "Median": by_month.median(),
            "Std deviation": by_month.std(ddof=1),
            "Years observed": by_month.count(),
            "Minimum": by_month.min(),
            "Maximum": by_month.max(),
        }
    )
    table["Seasonal index"] = table["Mean"] / overall if overall else np.nan
    table["Deviation from average"] = table["Seasonal index"] - 1.0
    table.index = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in table.index]
    table.index.name = "Month"
    return Result.of(
        table,
        source,
        [
            f"Index = month mean / overall mean ({overall:,.0f}). "
            "Values above 1.00 mark seasonally firm months.",
        ],
    )


@safe_analysis()
def weekday_seasonality(series: pd.Series, source: str = "") -> Result[pd.DataFrame]:
    """Average level and return by day of week."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(clean.index, pd.DatetimeIndex) or len(clean) < 30:
        return Result.unavailable(
            "Weekly seasonality needs a date-indexed series with at least 30 "
            "observations.",
            source,
        )
    returns = clean.pct_change().dropna()
    grouped = clean.groupby(clean.index.dayofweek)
    ret_grouped = returns.groupby(returns.index.dayofweek)
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    table = pd.DataFrame(
        {
            "Mean level": grouped.mean(),
            "Observations": grouped.count(),
            "Mean change %": ret_grouped.mean() * 100,
            "Std of change %": ret_grouped.std(ddof=1) * 100,
        }
    )
    table.index = [names[i] for i in table.index]
    table.index.name = "Day"
    overall = float(clean.mean())
    table["Index vs overall"] = table["Mean level"] / overall if overall else np.nan
    return Result.of(
        table,
        source,
        [
            "Day-of-week effects on a mandi series largely reflect which days "
            "the market trades, not a behavioural pattern; read alongside the "
            "observation count.",
        ],
    )


# ==========================================================================
# Correlation
# ==========================================================================


@safe_analysis()
def correlation_matrix(
    frame: pd.DataFrame, method: str = "pearson", source: str = ""
) -> Result[pd.DataFrame]:
    """Pairwise correlation with an accompanying count of overlapping points."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2:
        return Result.unavailable(
            "At least two series are needed for a correlation matrix.", source
        )
    counts = numeric.notna().astype(int).T.dot(numeric.notna().astype(int))
    matrix = numeric.corr(method=method, min_periods=CFG.min_obs_correlation)

    thin = int((counts < CFG.min_obs_correlation).sum().sum() - (counts.shape[0]))
    notes = [
        f"{method.title()} correlation on overlapping observations only; "
        f"pairs with fewer than {CFG.min_obs_correlation} shared points are "
        "left blank."
    ]
    if thin > 0:
        notes.append(f"{thin // 2} pair(s) fell below that threshold.")
    result = Result.of(matrix, source, notes)
    result.value.attrs["counts"] = counts
    return result


@safe_analysis()
def correlation_pair(
    a: pd.Series, b: pd.Series, source: str = ""
) -> Result[pd.Series]:
    """Pearson and Spearman correlation for one pair, with significance."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < CFG.min_obs_correlation:
        return Result.unavailable(
            f"Only {len(joined)} overlapping observation(s); at least "
            f"{CFG.min_obs_correlation} are required.",
            source,
        )
    pear_r, pear_p = sp_stats.pearsonr(joined["a"], joined["b"])
    spear_r, spear_p = sp_stats.spearmanr(joined["a"], joined["b"])
    out = pd.Series(
        {
            "Overlapping observations": float(len(joined)),
            "Pearson r": float(pear_r),
            "Pearson p-value": float(pear_p),
            "Spearman rho": float(spear_r),
            "Spearman p-value": float(spear_p),
            "R-squared (Pearson)": float(pear_r**2),
        }
    )
    return Result.of(
        out,
        source,
        [
            f"Relationship is {describe_strength(pear_r)} and "
            f"{'statistically significant' if pear_p < CFG.alpha else 'not statistically significant'} "
            f"at the {CFG.alpha:.0%} level.",
        ],
    )


@safe_analysis()
def rolling_correlation(
    a: pd.Series, b: pd.Series, window: int, source: str = ""
) -> Result[pd.Series]:
    """Correlation of two series through time."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < window + 5:
        return Result.unavailable(
            f"A {window}-period rolling correlation needs at least "
            f"{window + 5} overlapping observations; {len(joined)} available.",
            source,
        )
    rolled = joined["a"].rolling(window).corr(joined["b"]).dropna()
    if rolled.empty:
        return Result.unavailable("The rolling correlation is undefined (zero variance).", source)
    rolled.name = f"Rolling correlation ({window})"
    return Result.of(
        rolled,
        source,
        [
            f"Range over the sample: {rolled.min():.2f} to {rolled.max():.2f}; "
            f"latest {rolled.iloc[-1]:.2f}. A widening range signals an "
            "unstable relationship.",
        ],
    )


@safe_analysis()
def cross_correlation(
    a: pd.Series,
    b: pd.Series,
    max_lag: int | None = None,
    source: str = "",
    differenced: bool = True,
) -> Result[pd.DataFrame]:
    """Cross-correlation across leads and lags.

    A positive lag *k* means series ``a`` leads series ``b`` by *k* periods:
    ``corr(a[t], b[t + k])``.

    By default both series are first-differenced. Correlating two trending
    price levels produces a broad, flat, near-1.0 cross-correlation curve
    whose peak is meaningless; differencing removes the shared trend so the
    peak reflects genuine lead-lag timing.
    """
    max_lag = max_lag or CFG.max_lag
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if differenced:
        joined = joined.diff().dropna()
    if len(joined) < max(3 * max_lag, CFG.min_obs_correlation):
        max_lag = max(1, min(max_lag, len(joined) // 4))
    if len(joined) < CFG.min_obs_correlation:
        return Result.unavailable(
            f"Only {len(joined)} overlapping observation(s) after alignment"
            f"{' and differencing' if differenced else ''}; at least "
            f"{CFG.min_obs_correlation} are required.",
            source,
        )

    rows: list[dict[str, float]] = []
    n = len(joined)
    for lag in range(-max_lag, max_lag + 1):
        shifted = joined["b"].shift(-lag)
        pair = pd.concat([joined["a"], shifted], axis=1).dropna()
        if len(pair) < max(10, CFG.min_obs_correlation // 3):
            continue
        r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if not np.isfinite(r):
            continue
        # Bartlett's approximate standard error for a sample correlation.
        se = 1.0 / np.sqrt(len(pair))
        rows.append(
            {
                "lag": lag,
                "correlation": r,
                "n": len(pair),
                "upper_95": 1.96 * se,
                "lower_95": -1.96 * se,
                "significant": abs(r) > 1.96 * se,
            }
        )

    if not rows:
        return Result.unavailable("No lag produced enough overlapping data.", source)

    frame = pd.DataFrame(rows).set_index("lag")
    # One "lag" is one row of the *aligned* index, which is not one calendar
    # period when the two markets trade on different days. Measuring the
    # actual spacing keeps the day-equivalent honest.
    spacing = float(pd.Series(joined.index).diff().dt.days.median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0
    frame.attrs["median_spacing_days"] = spacing

    notes = [
        "Positive lag = the first series leads the second.",
        "Both series first-differenced before correlating, so the peak "
        "reflects timing rather than a shared trend."
        if differenced
        else "Computed on levels; a shared trend may dominate.",
        f"Dashed band is the approximate 95% significance level (+/-1.96/sqrt(n)) for n≈{n}.",
        f"One lag step equals {spacing:.1f} calendar day(s), measured from the "
        "median spacing of the two series' common trading dates.",
    ]
    return Result.of(frame, source, notes)


@dataclass
class LeadLag:
    """The headline reading of a cross-correlation curve."""

    leader: str
    follower: str
    lag_periods: int
    lag_days: float
    correlation: float
    contemporaneous: float
    significant: bool
    frequency: str
    n_obs: int

    def sentence(self) -> str:
        if self.lag_periods == 0:
            return (
                f"{self.leader} and {self.follower} move together with no "
                f"measurable lead (peak correlation {self.correlation:.2f} at "
                "lag 0)."
            )
        direction = "leads" if self.lag_periods > 0 else "lags"
        return (
            f"{self.leader} {direction} {self.follower} by "
            f"{abs(self.lag_periods)} {self.frequency} period(s) "
            f"(~{abs(self.lag_days):.0f} days), peak correlation "
            f"{self.correlation:.2f} versus {self.contemporaneous:.2f} "
            "at zero lag."
        )


@safe_analysis()
def lead_lag(
    a: pd.Series,
    b: pd.Series,
    name_a: str,
    name_b: str,
    freq: str,
    max_lag: int | None = None,
    source: str = "",
) -> Result[LeadLag]:
    """Identify which of two series leads, and by how much."""
    ccf = cross_correlation(a, b, max_lag=max_lag, source=source, differenced=True)
    if not ccf:
        return Result.unavailable(ccf.reason, source)
    frame = ccf.unwrap()

    peak_lag = int(frame["correlation"].abs().idxmax())
    peak = frame.loc[peak_lag]
    zero = float(frame.loc[0, "correlation"]) if 0 in frame.index else np.nan
    # Prefer the spacing measured from the aligned index; fall back to the
    # nominal length of the analysis frequency.
    days_per_period = float(
        frame.attrs.get("median_spacing_days")
        or {"D": 1, "W": 7, "SME": 15.2, "ME": 30.4}.get(freq, 1)
    )

    leader, follower = (name_a, name_b) if peak_lag >= 0 else (name_b, name_a)
    reading = LeadLag(
        leader=leader,
        follower=follower,
        lag_periods=peak_lag,
        lag_days=peak_lag * days_per_period,
        correlation=float(peak["correlation"]),
        contemporaneous=zero,
        significant=bool(peak["significant"]),
        frequency=freq,
        n_obs=int(peak["n"]),
    )
    notes = list(ccf.notes)
    if not reading.significant:
        notes.append(
            "The peak does not clear the 95% significance band, so the "
            "lead should be treated as indicative only."
        )
    if peak_lag != 0 and np.isfinite(zero) and abs(peak["correlation"]) - abs(zero) < 0.02:
        notes.append(
            "The peak is barely above the zero-lag value; the two series are "
            "effectively contemporaneous."
        )
    result = Result.of(reading, source, notes)
    return result


@safe_analysis()
def lead_lag_matrix(
    frame: pd.DataFrame, freq: str, max_lag: int | None = None, source: str = ""
) -> Result[pd.DataFrame]:
    """Pairwise lead-lag readings for every column pair in a panel."""
    columns = list(frame.columns)
    if len(columns) < 2:
        return Result.unavailable(
            "At least two series are needed for a lead-lag matrix.", source
        )
    rows: list[dict[str, Any]] = []
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            reading = lead_lag(
                frame[first], frame[second], str(first), str(second), freq,
                max_lag=max_lag, source=source,
            )
            if not reading:
                rows.append(
                    {
                        "Series A": first,
                        "Series B": second,
                        "Leader": "—",
                        "Lead (periods)": np.nan,
                        "Lead (days)": np.nan,
                        "Peak correlation": np.nan,
                        "Zero-lag correlation": np.nan,
                        "Significant": False,
                        "Observations": 0,
                        "Note": reading.reason,
                    }
                )
                continue
            r = reading.unwrap()
            rows.append(
                {
                    "Series A": first,
                    "Series B": second,
                    "Leader": r.leader if r.lag_periods != 0 else "Contemporaneous",
                    "Lead (periods)": abs(r.lag_periods),
                    "Lead (days)": abs(r.lag_days),
                    "Peak correlation": r.correlation,
                    "Zero-lag correlation": r.contemporaneous,
                    "Significant": r.significant,
                    "Observations": r.n_obs,
                    "Note": "",
                }
            )
    return Result.of(pd.DataFrame(rows), source)


# ==========================================================================
# Stationarity, cointegration, causality
# ==========================================================================


@safe_analysis()
def stationarity_tests(series: pd.Series, source: str = "") -> Result[pd.DataFrame]:
    """ADF and KPSS, with a combined verdict.

    The two tests have opposite null hypotheses, so reading them together is
    more informative than either alone:

    =================  =================  ======================================
    ADF                KPSS               Verdict
    =================  =================  ======================================
    rejects            fails to reject    Stationary
    fails to reject    rejects            Non-stationary (unit root)
    rejects            rejects            Difference-stationary around a trend
    fails to reject    fails to reject    Inconclusive; too little information
    =================  =================  ======================================
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < CFG.min_obs_stationarity:
        return Result.unavailable(
            f"Stationarity testing needs at least {CFG.min_obs_stationarity} "
            f"observations; {len(clean)} are available.",
            source,
        )

    rows: list[dict[str, Any]] = []
    adf_stationary: bool | None = None
    kpss_stationary: bool | None = None

    try:
        adf_stat, adf_p, adf_lags, adf_n, adf_crit, _ = adfuller(clean, autolag="AIC")
        adf_stationary = adf_p < CFG.alpha
        rows.append(
            {
                "Test": "Augmented Dickey-Fuller",
                "Null hypothesis": "Series has a unit root (non-stationary)",
                "Statistic": float(adf_stat),
                "p-value": float(adf_p),
                "Critical 5%": float(adf_crit.get("5%", np.nan)),
                "Lags used": int(adf_lags),
                "Observations": int(adf_n),
                "Conclusion": "Stationary" if adf_stationary else "Non-stationary",
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "Test": "Augmented Dickey-Fuller",
                "Null hypothesis": "Series has a unit root (non-stationary)",
                "Statistic": np.nan,
                "p-value": np.nan,
                "Critical 5%": np.nan,
                "Lags used": np.nan,
                "Observations": len(clean),
                "Conclusion": f"Could not run: {exc}",
            }
        )

    try:
        import warnings as _warnings

        with _warnings.catch_warnings():
            # KPSS clamps its p-value at the lookup-table edges; the warning is
            # expected and the clamped value is still interpretable.
            _warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
                clean, regression="c", nlags="auto"
            )
        kpss_stationary = kpss_p >= CFG.alpha
        rows.append(
            {
                "Test": "KPSS",
                "Null hypothesis": "Series is stationary around a constant",
                "Statistic": float(kpss_stat),
                "p-value": float(kpss_p),
                "Critical 5%": float(kpss_crit.get("5%", np.nan)),
                "Lags used": int(kpss_lags),
                "Observations": len(clean),
                "Conclusion": "Stationary" if kpss_stationary else "Non-stationary",
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "Test": "KPSS",
                "Null hypothesis": "Series is stationary around a constant",
                "Statistic": np.nan,
                "p-value": np.nan,
                "Critical 5%": np.nan,
                "Lags used": np.nan,
                "Observations": len(clean),
                "Conclusion": f"Could not run: {exc}",
            }
        )

    if adf_stationary is None or kpss_stationary is None:
        verdict = "Incomplete: one of the two tests could not be run."
    elif adf_stationary and kpss_stationary:
        verdict = (
            "Both tests agree the series is stationary; it can be modelled in "
            "levels."
        )
    elif not adf_stationary and not kpss_stationary:
        verdict = (
            "Both tests point to a unit root; the series should be "
            "differenced before modelling."
        )
    elif adf_stationary and not kpss_stationary:
        verdict = (
            "The tests disagree, which typically indicates trend-stationarity: "
            "stationary once a deterministic trend is removed."
        )
    else:
        verdict = (
            "The tests disagree, which typically indicates difference-"
            "stationarity: take one difference before modelling."
        )

    # How many differences until both tests are satisfied?
    order = 0
    probe = clean.copy()
    for candidate in range(1, 3):
        try:
            p = adfuller(probe, autolag="AIC")[1]
        except Exception:  # noqa: BLE001
            break
        if p < CFG.alpha:
            order = candidate - 1
            break
        probe = probe.diff().dropna()
        order = candidate
        if len(probe) < CFG.min_obs_stationarity:
            break

    frame = pd.DataFrame(rows)
    frame.attrs["integration_order"] = order
    return Result.of(
        frame,
        source,
        [verdict, f"Suggested order of differencing (d): {order}."],
    )


@safe_analysis()
def autocorrelation(
    series: pd.Series, nlags: int = 40, source: str = ""
) -> Result[pd.DataFrame]:
    """ACF and PACF with 95% confidence bands."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 20:
        return Result.unavailable(
            f"Autocorrelation needs at least 20 observations; {len(clean)} "
            "are available.",
            source,
        )
    nlags = int(min(nlags, max(1, len(clean) // 2 - 1)))
    acf_values = acf(clean, nlags=nlags, fft=True)
    try:
        pacf_values = pacf(clean, nlags=min(nlags, len(clean) // 2 - 1), method="ywm")
    except Exception:  # noqa: BLE001
        pacf_values = np.full(nlags + 1, np.nan)

    length = min(len(acf_values), len(pacf_values))
    band = 1.96 / np.sqrt(len(clean))
    frame = pd.DataFrame(
        {
            "ACF": acf_values[:length],
            "PACF": pacf_values[:length],
            "Upper 95%": band,
            "Lower 95%": -band,
        },
        index=pd.RangeIndex(length, name="lag"),
    )
    significant = [
        int(lag)
        for lag in frame.index[1:]
        if abs(frame.loc[lag, "ACF"]) > band
    ]
    notes = [
        f"95% band is +/-{band:.3f} for n={len(clean)}.",
        (
            f"ACF is significant out to lag {max(significant)}; slow decay of "
            "this kind is the signature of a trending, non-stationary series."
            if significant and max(significant) > 10
            else (
                f"Significant ACF at lag(s) {significant[:8]}."
                if significant
                else "No lag shows significant autocorrelation."
            )
        ),
    ]
    return Result.of(frame, source, notes)


@safe_analysis()
def granger_causality(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int | None = None,
    source: str = "",
    difference: bool = True,
) -> Result[pd.DataFrame]:
    """Test whether past values of ``cause`` help predict ``effect``.

    Series are differenced by default: Granger causality assumes stationary
    inputs, and running it on trending price levels reliably produces
    spurious significance.
    """
    max_lag = max_lag or CFG.granger_max_lag
    joined = pd.concat([effect.rename("effect"), cause.rename("cause")], axis=1).dropna()
    if difference:
        joined = joined.diff().dropna()
    needed = 3 * max_lag + 10
    if len(joined) < needed:
        max_lag = max(1, (len(joined) - 10) // 3)
    if len(joined) < 20 or max_lag < 1:
        return Result.unavailable(
            f"Only {len(joined)} overlapping observation(s) after alignment"
            f"{' and differencing' if difference else ''}; Granger causality "
            "needs a substantially longer common history.",
            source,
        )

    import warnings as _warnings

    # statsmodels prints a full test table per lag straight to stdout and
    # offers no switch to stop it, so the call is run inside a stdout trap.
    with _warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        _warnings.simplefilter("ignore")
        raw = grangercausalitytests(joined[["effect", "cause"]], maxlag=max_lag)

    rows: list[dict[str, Any]] = []
    for lag, (tests, _) in raw.items():
        f_stat, p_value = tests["ssr_ftest"][0], tests["ssr_ftest"][1]
        rows.append(
            {
                "Lag": int(lag),
                "F statistic": float(f_stat),
                "p-value": float(p_value),
                "Significant": bool(p_value < CFG.alpha),
            }
        )
    frame = pd.DataFrame(rows).set_index("Lag")

    significant = frame.index[frame["Significant"]].tolist()
    cause_name = cause.name or "the driver"
    effect_name = effect.name or "the target"
    if significant:
        best = int(frame.loc[significant, "p-value"].idxmin())
        verdict = (
            f"Past values of {cause_name} improve the prediction of "
            f"{effect_name} at lag(s) {significant} (strongest at lag {best}, "
            f"p={frame.loc[best, 'p-value']:.4f})."
        )
    else:
        verdict = (
            f"No lag of {cause_name} significantly improves the prediction of "
            f"{effect_name} at the {CFG.alpha:.0%} level."
        )
    return Result.of(
        frame,
        source,
        [
            verdict,
            "Granger causality is predictive precedence, not proof of a "
            "causal mechanism.",
            "Series were differenced to satisfy the stationarity assumption."
            if difference
            else "Computed on levels.",
        ],
    )


@safe_analysis()
def cointegration(
    frame: pd.DataFrame, source: str = "", det_order: int = 0, k_ar_diff: int = 1
) -> Result[dict[str, Any]]:
    """Engle-Granger (pairwise) and Johansen (system) cointegration tests.

    Cointegration is the formal test of market integration: two mandis whose
    prices are cointegrated cannot drift apart indefinitely, which is exactly
    the question the brief asks about Guntur, Warangal and Khammam.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 2:
        return Result.unavailable("At least two series are required.", source)
    if len(numeric) < CFG.min_obs_cointegration:
        return Result.unavailable(
            f"Cointegration testing needs at least {CFG.min_obs_cointegration} "
            f"overlapping observations; only {len(numeric)} are available "
            "across all selected markets simultaneously.",
            source,
        )

    columns = list(numeric.columns)
    pairwise: list[dict[str, Any]] = []
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            try:
                stat, p_value, crit = coint(numeric[first], numeric[second])
                pairwise.append(
                    {
                        "Series A": first,
                        "Series B": second,
                        "Statistic": float(stat),
                        "p-value": float(p_value),
                        "Critical 5%": float(crit[1]),
                        "Cointegrated": bool(p_value < CFG.alpha),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                pairwise.append(
                    {
                        "Series A": first,
                        "Series B": second,
                        "Statistic": np.nan,
                        "p-value": np.nan,
                        "Critical 5%": np.nan,
                        "Cointegrated": False,
                        "Note": str(exc),
                    }
                )
    pairwise_frame = pd.DataFrame(pairwise)

    johansen_frame = pd.DataFrame()
    rank = 0
    johansen_note = ""
    try:
        jo = coint_johansen(numeric.values, det_order=det_order, k_ar_diff=k_ar_diff)
        rows: list[dict[str, Any]] = []
        for i in range(len(jo.lr1)):
            trace_stat = float(jo.lr1[i])
            crit_95 = float(jo.cvt[i, 1])
            reject = trace_stat > crit_95
            rows.append(
                {
                    "Null hypothesis": f"rank <= {i}",
                    "Trace statistic": trace_stat,
                    "Critical 90%": float(jo.cvt[i, 0]),
                    "Critical 95%": crit_95,
                    "Critical 99%": float(jo.cvt[i, 2]),
                    "Reject at 5%": reject,
                    "Max-eigen statistic": float(jo.lr2[i]),
                    "Max-eigen critical 95%": float(jo.cvm[i, 1]),
                }
            )
            if reject:
                rank = i + 1
        johansen_frame = pd.DataFrame(rows)
        johansen_note = (
            f"Johansen trace test finds {rank} cointegrating relationship(s) "
            f"among {len(columns)} series."
        )
    except Exception as exc:  # noqa: BLE001
        johansen_note = f"Johansen test could not be run: {exc}"

    n_pairs = int(pairwise_frame["Cointegrated"].sum()) if not pairwise_frame.empty else 0
    total_pairs = len(pairwise_frame)
    if rank > 0:
        verdict = (
            f"The markets are cointegrated (rank {rank}): they share a "
            "long-run equilibrium and cannot drift apart permanently. "
            "Deviations are temporary and mean-reverting, which is the "
            "statistical definition of an integrated market."
        )
    elif n_pairs > 0:
        verdict = (
            f"{n_pairs} of {total_pairs} pairs are cointegrated bilaterally, "
            "but the system as a whole shows no common trend."
        )
    else:
        verdict = (
            "No cointegrating relationship is detected: over this sample the "
            "series can drift apart without a force pulling them back."
        )

    payload = {
        "pairwise": pairwise_frame,
        "johansen": johansen_frame,
        "rank": rank,
        "n_obs": len(numeric),
        "columns": columns,
        "span": (numeric.index.min(), numeric.index.max()),
    }
    return Result.of(
        payload,
        source,
        [
            verdict,
            johansen_note,
            f"Tested on {len(numeric)} overlapping observations from "
            f"{numeric.index.min():%b %Y} to {numeric.index.max():%b %Y}.",
        ],
    )


@safe_analysis()
def variance_inflation(frame: pd.DataFrame, source: str = "") -> Result[pd.DataFrame]:
    """Variance Inflation Factors for a set of candidate drivers.

    High VIF means two drivers carry the same information, so a regression
    cannot attribute an effect between them. Reported before any multi-driver
    model is interpreted.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 2:
        return Result.unavailable(
            "VIF needs at least two explanatory variables.", source
        )
    if len(numeric) <= numeric.shape[1] + 1:
        return Result.unavailable(
            f"Only {len(numeric)} complete observation(s) for "
            f"{numeric.shape[1]} variables; too few to compute VIF.",
            source,
        )
    # Constant columns make the design matrix singular.
    varying = numeric.loc[:, numeric.std(ddof=0) > 0]
    dropped = [c for c in numeric.columns if c not in varying.columns]
    if varying.shape[1] < 2:
        return Result.unavailable(
            "Fewer than two variables show any variation over the sample.",
            source,
        )

    design = add_constant(varying, has_constant="add")
    rows = []
    for i, name in enumerate(design.columns):
        if name == "const":
            continue
        try:
            value = float(variance_inflation_factor(design.values, i))
        except Exception:  # noqa: BLE001
            value = float("nan")
        rows.append(
            {
                "Variable": name,
                "VIF": value,
                "Interpretation": (
                    "Severe collinearity"
                    if value > 10
                    else "Moderate collinearity"
                    if value > 5
                    else "Acceptable"
                ),
            }
        )
    notes = [
        "VIF above 5 indicates moderate, above 10 severe collinearity: the "
        "affected drivers cannot be separately attributed.",
        f"Computed on {len(numeric)} complete observations.",
    ]
    if dropped:
        notes.append(f"Dropped constant column(s): {', '.join(map(str, dropped))}.")
    return Result.of(pd.DataFrame(rows), source, notes)


# ==========================================================================
# Price-arrival relationship
# ==========================================================================


@safe_analysis()
def elasticity(
    price: pd.Series,
    quantity: pd.Series,
    source: str = "",
    lags: Sequence[int] = (0, 1, 2, 3, 4),
) -> Result[pd.DataFrame]:
    """Price elasticity with respect to arrivals, at several lags.

    Estimated in logs, so the slope reads directly as "a 1% rise in arrivals
    is associated with an X% change in price". Both series are
    log-differenced: the level relationship between two trending series is
    dominated by their common trend, and would overstate the true
    responsiveness.
    """
    joined = pd.concat(
        [price.rename("price"), quantity.rename("quantity")], axis=1
    ).dropna()
    joined = joined[(joined["price"] > 0) & (joined["quantity"] > 0)]
    if len(joined) < CFG.min_obs_correlation:
        return Result.unavailable(
            f"Only {len(joined)} paired positive observation(s); at least "
            f"{CFG.min_obs_correlation} are needed to estimate elasticity.",
            source,
        )

    log_price = np.log(joined["price"])
    log_qty = np.log(joined["quantity"])
    d_price = log_price.diff()
    d_qty = log_qty.diff()

    rows: list[dict[str, Any]] = []
    for lag in lags:
        pair = pd.concat(
            [d_price.rename("dp"), d_qty.shift(lag).rename("dq")], axis=1
        ).dropna()
        if len(pair) < 20:
            continue
        model = OLS(pair["dp"], add_constant(pair["dq"])).fit()
        slope = float(model.params.iloc[1])
        rows.append(
            {
                "Arrivals lag (periods)": lag,
                "Elasticity": slope,
                "Std error": float(model.bse.iloc[1]),
                "t statistic": float(model.tvalues.iloc[1]),
                "p-value": float(model.pvalues.iloc[1]),
                "R-squared": float(model.rsquared),
                "Observations": int(model.nobs),
                "Significant": bool(model.pvalues.iloc[1] < CFG.alpha),
            }
        )

    if not rows:
        return Result.unavailable(
            "No lag had enough paired observations to fit a regression.", source
        )

    frame = pd.DataFrame(rows).set_index("Arrivals lag (periods)")
    significant = frame[frame["Significant"]]
    if not significant.empty:
        best = int(significant["p-value"].idxmin())
        value = float(significant.loc[best, "Elasticity"])
        direction = "lower" if value < 0 else "higher"
        verdict = (
            f"Strongest response at lag {best}: a 1% rise in arrivals is "
            f"associated with a {abs(value):.2f}% {direction} price "
            f"(p={significant.loc[best, 'p-value']:.4f})."
        )
    else:
        verdict = (
            "No lag shows a statistically significant arrivals-price "
            f"elasticity at the {CFG.alpha:.0%} level over this sample."
        )
    return Result.of(
        frame,
        source,
        [
            verdict,
            "Estimated on log differences, so coefficients are percentage "
            "responses and the shared trend is removed.",
        ],
    )


@safe_analysis()
def threshold_effects(
    price: pd.Series,
    quantity: pd.Series,
    buckets: int | None = None,
    source: str = "",
) -> Result[pd.DataFrame]:
    """Average price behaviour by arrivals bucket.

    Answers "above what arrival level do prices start to come under
    pressure?" by splitting arrivals into quantiles and reporting the
    contemporaneous and next-period price change in each.
    """
    buckets = buckets or CFG.threshold_buckets
    joined = pd.concat(
        [price.rename("price"), quantity.rename("quantity")], axis=1
    ).dropna()
    if len(joined) < buckets * 10:
        return Result.unavailable(
            f"Only {len(joined)} paired observation(s); at least "
            f"{buckets * 10} are needed for {buckets} arrival buckets.",
            source,
        )

    joined["price_change"] = joined["price"].pct_change()
    joined["next_change"] = joined["price"].pct_change().shift(-1)
    try:
        joined["bucket"] = pd.qcut(joined["quantity"], buckets, duplicates="drop")
    except ValueError:
        return Result.unavailable(
            "Arrivals show too little variation to form distinct buckets.",
            source,
        )

    grouped = joined.groupby("bucket", observed=True)
    table = pd.DataFrame(
        {
            "Arrivals from": grouped["quantity"].min(),
            "Arrivals to": grouped["quantity"].max(),
            "Mean arrivals": grouped["quantity"].mean(),
            "Mean price": grouped["price"].mean(),
            "Median price": grouped["price"].median(),
            "Mean same-period change %": grouped["price_change"].mean() * 100,
            "Mean next-period change %": grouped["next_change"].mean() * 100,
            "Share of periods with a fall": grouped["price_change"].apply(
                lambda s: float((s < 0).mean()) * 100 if s.notna().any() else np.nan
            ),
            "Observations": grouped["price"].count(),
        }
    )
    table.index = [f"Q{i + 1}" for i in range(len(table))]
    table.index.name = "Arrivals bucket"

    notes: list[str] = []
    top = table.iloc[-1]
    bottom = table.iloc[0]
    if np.isfinite(top["Mean next-period change %"]) and np.isfinite(
        bottom["Mean next-period change %"]
    ):
        notes.append(
            f"In the heaviest arrivals bucket (above {top['Arrivals from']:,.0f}), "
            f"the average next-period price change is "
            f"{top['Mean next-period change %']:+.2f}%, against "
            f"{bottom['Mean next-period change %']:+.2f}% in the lightest "
            f"bucket (below {bottom['Arrivals to']:,.0f})."
        )
    # Test whether the top and bottom buckets genuinely differ.
    high = joined[joined["bucket"] == joined["bucket"].cat.categories[-1]]["next_change"].dropna()
    low = joined[joined["bucket"] == joined["bucket"].cat.categories[0]]["next_change"].dropna()
    if len(high) > 5 and len(low) > 5:
        t_stat, p_value = sp_stats.ttest_ind(high, low, equal_var=False)
        notes.append(
            "Difference between the heaviest and lightest buckets: "
            f"t={t_stat:.2f}, p={p_value:.4f} "
            f"({'statistically significant' if p_value < CFG.alpha else 'not statistically significant'})."
        )
        table.attrs["threshold_pvalue"] = float(p_value)
        table.attrs["threshold_level"] = float(top["Arrivals from"])
    return Result.of(table, source, notes)


@safe_analysis()
def lagged_impact(
    price: pd.Series,
    quantity: pd.Series,
    max_lag: int = 8,
    source: str = "",
) -> Result[pd.DataFrame]:
    """Correlation between arrivals and price change at successive lags."""
    joined = pd.concat(
        [price.rename("price"), quantity.rename("quantity")], axis=1
    ).dropna()
    if len(joined) < max(30, max_lag * 4):
        return Result.unavailable(
            f"Only {len(joined)} paired observation(s); at least "
            f"{max(30, max_lag * 4)} are needed to trace {max_lag} lags.",
            source,
        )
    d_price = joined["price"].pct_change()
    d_qty = joined["quantity"].pct_change()

    rows: list[dict[str, Any]] = []
    for lag in range(0, max_lag + 1):
        pair = pd.concat([d_price, d_qty.shift(lag)], axis=1).dropna()
        if len(pair) < 20:
            continue
        r, p = sp_stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append(
            {
                "Lag (periods)": lag,
                "Correlation with price change": float(r),
                "p-value": float(p),
                "Observations": len(pair),
                "Significant": bool(p < CFG.alpha),
            }
        )
    if not rows:
        return Result.unavailable("No lag had enough paired observations.", source)
    frame = pd.DataFrame(rows).set_index("Lag (periods)")
    significant = frame[frame["Significant"]]
    if not significant.empty:
        strongest = int(significant["Correlation with price change"].abs().idxmax())
        note = (
            f"The arrivals effect is strongest {strongest} period(s) after the "
            f"arrival ("
            f"r={frame.loc[strongest, 'Correlation with price change']:.2f})."
        )
    else:
        note = "No lag shows a significant arrivals-to-price-change relationship."
    return Result.of(frame, source, [note])


# ==========================================================================
# Price leadership
# ==========================================================================


@safe_analysis()
def directional_influence(
    frame: pd.DataFrame,
    freq: str,
    source: str = "",
    max_lag: int | None = None,
) -> Result[pd.DataFrame]:
    """Pairwise directional verdicts combining Granger tests in both directions.

    For every unordered pair this runs Granger causality *both ways* and
    classifies the relationship:

    ``A -> B``
        Only A's past helps predict B. A influences B.
    ``B -> A``
        The mirror case.
    ``Feedback``
        Each helps predict the other -- a jointly determined, well-arbitraged
        pair.
    ``Independent``
        Neither direction is significant.

    The one-directional cases are the only ones that constitute evidence of
    leadership, which is what stops a symmetric, tightly-arbitraged market
    from producing a spurious "leader".
    """
    columns = [c for c in frame.columns if frame[c].notna().sum() >= CFG.min_obs_correlation]
    if len(columns) < 2:
        return Result.unavailable(
            "At least two series with sufficient history are needed.", source
        )

    rows: list[dict[str, Any]] = []
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            forward = granger_causality(
                frame[first].rename(str(first)),
                frame[second].rename(str(second)),
                source=source,
            )
            backward = granger_causality(
                frame[second].rename(str(second)),
                frame[first].rename(str(first)),
                source=source,
            )
            if not forward or not backward:
                rows.append(
                    {
                        "Series A": first,
                        "Series B": second,
                        "A -> B significant": False,
                        "B -> A significant": False,
                        "A -> B best p": np.nan,
                        "B -> A best p": np.nan,
                        "Verdict": "Not testable",
                        "Direction": "—",
                        "Peak lag (periods)": np.nan,
                        "Peak lag (days)": np.nan,
                        "Peak correlation": np.nan,
                        "Note": (forward.reason or backward.reason)[:160],
                    }
                )
                continue

            f_frame, b_frame = forward.unwrap(), backward.unwrap()
            f_sig, b_sig = bool(f_frame["Significant"].any()), bool(b_frame["Significant"].any())
            f_p, b_p = float(f_frame["p-value"].min()), float(b_frame["p-value"].min())

            if f_sig and not b_sig:
                verdict, direction = "One-way", f"{first} -> {second}"
            elif b_sig and not f_sig:
                verdict, direction = "One-way", f"{second} -> {first}"
            elif f_sig and b_sig:
                verdict = "Feedback (both directions)"
                stronger = first if f_p < b_p else second
                other = second if f_p < b_p else first
                direction = f"{stronger} dominant -> {other}"
            else:
                verdict, direction = "Independent", "—"

            reading = lead_lag(
                frame[first], frame[second], str(first), str(second), freq,
                max_lag=max_lag, source=source,
            )
            if reading:
                r = reading.unwrap()
                peak_lag, peak_days, peak_corr = r.lag_periods, r.lag_days, r.correlation
            else:
                peak_lag = peak_days = peak_corr = np.nan

            rows.append(
                {
                    "Series A": first,
                    "Series B": second,
                    "A -> B significant": f_sig,
                    "B -> A significant": b_sig,
                    "A -> B best p": f_p,
                    "B -> A best p": b_p,
                    "Verdict": verdict,
                    "Direction": direction,
                    "Peak lag (periods)": peak_lag,
                    "Peak lag (days)": peak_days,
                    "Peak correlation": peak_corr,
                    "Note": "",
                }
            )

    table = pd.DataFrame(rows)
    one_way = int((table["Verdict"] == "One-way").sum())
    feedback = int((table["Verdict"].str.startswith("Feedback")).sum())
    independent = int((table["Verdict"] == "Independent").sum())
    return Result.of(
        table,
        source,
        [
            f"{one_way} one-way relationship(s), {feedback} two-way feedback "
            f"pair(s) and {independent} independent pair(s) out of "
            f"{len(table)} tested.",
            "Only one-way relationships are evidence of price leadership; "
            "feedback pairs are jointly determined.",
        ],
    )


@dataclass
class LeadershipScore:
    """Aggregate evidence that one series leads a group of others."""

    name: str
    times_leader: int
    comparisons: int
    mean_lead_days: float
    mean_peak_correlation: float
    #: Pairs this series influences one-way.
    influences: int
    #: Pairs where this series is the one being influenced one-way.
    influenced_by: int
    #: Pairs where this series is the dominant side of a feedback loop.
    dominant_in_feedback: int
    granger_wins: int
    granger_tests: int

    @property
    def lead_share(self) -> float:
        return self.times_leader / self.comparisons if self.comparisons else 0.0

    @property
    def granger_share(self) -> float:
        return self.granger_wins / self.granger_tests if self.granger_tests else 0.0

    @property
    def influence_share(self) -> float:
        """Net one-way influence, rescaled from [-1, 1] to [0, 1]."""
        if not self.comparisons:
            return 0.0
        net = (self.influences - self.influenced_by) / self.comparisons
        return (net + 1.0) / 2.0

    @property
    def dominance_share(self) -> float:
        return self.dominant_in_feedback / self.comparisons if self.comparisons else 0.0

    @property
    def composite(self) -> float:
        """Blend of timing, net influence and predictive evidence, 0-1."""
        return (
            0.35 * self.influence_share
            + 0.25 * self.dominance_share
            + 0.25 * self.granger_share
            + 0.15 * self.lead_share
        )


@safe_analysis()
def leadership_ranking(
    frame: pd.DataFrame,
    freq: str,
    source: str = "",
    max_lag: int | None = None,
) -> Result[pd.DataFrame]:
    """Rank series by how consistently they lead the others.

    Four independent lines of evidence are combined:

    1. **Net one-way influence** -- how often the series Granger-causes
       another *without* being Granger-caused back. This is the strongest
       single piece of evidence and carries the largest weight.
    2. **Dominance in feedback pairs** -- where causality runs both ways,
       which side has the stronger (lower p-value) evidence.
    3. **Predictive power** -- the raw share of Granger tests passed.
    4. **Timing** -- how often the series is the leader in a pairwise
       cross-correlation.

    Weighting influence above raw timing matters on this data: mandi prices
    move largely same-day, so timing alone cannot separate the markets, while
    the direction of predictive power still can.
    """
    columns = [c for c in frame.columns if frame[c].notna().sum() >= CFG.min_obs_correlation]
    if len(columns) < 2:
        return Result.unavailable(
            "At least two series with sufficient history are needed to rank "
            "price leadership.",
            source,
        )

    influence = directional_influence(frame, freq, source, max_lag)
    if not influence:
        return Result.unavailable(influence.reason, source)
    pairs = influence.unwrap()

    scores: dict[str, dict[str, Any]] = {
        str(c): {
            "times_leader": 0,
            "comparisons": 0,
            "leads": [],
            "peaks": [],
            "granger_wins": 0,
            "granger_tests": 0,
            "influences": 0,
            "influenced_by": 0,
            "dominant": 0,
        }
        for c in columns
    }

    for _, row in pairs.iterrows():
        first, second = str(row["Series A"]), str(row["Series B"])
        if first not in scores or second not in scores:
            continue
        scores[first]["comparisons"] += 1
        scores[second]["comparisons"] += 1

        for name, flag in ((first, "A -> B significant"), (second, "B -> A significant")):
            scores[name]["granger_tests"] += 1
            if bool(row[flag]):
                scores[name]["granger_wins"] += 1

        if row["Verdict"] == "One-way":
            source_name, target_name = [s.strip() for s in str(row["Direction"]).split("->")]
            if source_name in scores:
                scores[source_name]["influences"] += 1
            if target_name in scores:
                scores[target_name]["influenced_by"] += 1
        elif str(row["Verdict"]).startswith("Feedback"):
            dominant = str(row["Direction"]).split("dominant")[0].strip()
            if dominant in scores:
                scores[dominant]["dominant"] += 1

        peak_lag = row["Peak lag (periods)"]
        peak_corr = row["Peak correlation"]
        if pd.notna(peak_corr):
            scores[first]["peaks"].append(abs(float(peak_corr)))
            scores[second]["peaks"].append(abs(float(peak_corr)))
        if pd.notna(peak_lag) and int(peak_lag) != 0:
            leader = first if int(peak_lag) > 0 else second
            scores[leader]["times_leader"] += 1
            scores[leader]["leads"].append(abs(float(row["Peak lag (days)"])))

    rows: list[dict[str, Any]] = []
    for name, data in scores.items():
        score = LeadershipScore(
            name=name,
            times_leader=data["times_leader"],
            comparisons=data["comparisons"],
            mean_lead_days=float(np.mean(data["leads"])) if data["leads"] else 0.0,
            mean_peak_correlation=float(np.mean(data["peaks"])) if data["peaks"] else np.nan,
            influences=data["influences"],
            influenced_by=data["influenced_by"],
            dominant_in_feedback=data["dominant"],
            granger_wins=data["granger_wins"],
            granger_tests=data["granger_tests"],
        )
        rows.append(
            {
                "Series": score.name,
                "Influences (one-way)": score.influences,
                "Influenced by (one-way)": score.influenced_by,
                "Dominant in feedback": score.dominant_in_feedback,
                "Times leader (timing)": score.times_leader,
                "Mean lead (days)": score.mean_lead_days,
                "Granger-causes": score.granger_wins,
                "Granger tests": score.granger_tests,
                "Granger win rate": score.granger_share,
                "Mean peak correlation": score.mean_peak_correlation,
                "Comparisons": score.comparisons,
                "Leadership score": score.composite,
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["Leadership score", "Granger win rate"], ascending=False
    )
    table = table.reset_index(drop=True)
    table.index = table.index + 1
    table.index.name = "Rank"

    notes: list[str] = []
    any_timing = bool(table["Times leader (timing)"].sum())
    any_oneway = bool(table["Influences (one-way)"].sum())

    if table.empty:
        notes.append("No series could be scored.")
    else:
        top = table.iloc[0]
        runner = table.iloc[1] if len(table) > 1 else None
        evidence: list[str] = []
        if top["Influences (one-way)"]:
            evidence.append(
                f"one-way influence over {int(top['Influences (one-way)'])} "
                f"of {int(top['Comparisons'])} counterparts"
            )
        if top["Dominant in feedback"]:
            evidence.append(
                f"the dominant side in {int(top['Dominant in feedback'])} "
                "feedback pair(s)"
            )
        if top["Times leader (timing)"]:
            evidence.append(
                f"an average timing lead of {top['Mean lead (days)']:.0f} days"
            )
        if top["Granger win rate"]:
            evidence.append(
                f"a {top['Granger win rate']:.0%} Granger pass rate"
            )
        notes.append(
            f"{top['Series']} ranks first (score {top['Leadership score']:.2f}) on "
            + (", ".join(evidence) if evidence else "the combined criteria")
            + "."
        )
        if runner is not None and top["Leadership score"] - runner["Leadership score"] < 0.05:
            notes.append(
                f"The margin over {runner['Series']} is very narrow "
                f"({top['Leadership score']:.2f} vs "
                f"{runner['Leadership score']:.2f}); treat the top of the "
                "ranking as a tie rather than a decisive result."
            )

    if not any_timing:
        notes.append(
            "No pair shows a cross-correlation peak away from zero lag: on "
            "this data the series move contemporaneously, so there is no "
            "exploitable timing lead. The ranking therefore rests on the "
            "direction of predictive power rather than on timing."
        )
    if not any_oneway:
        notes.append(
            "No relationship is purely one-way; every significant pair shows "
            "feedback in both directions, which is the signature of a "
            "well-arbitraged market rather than a leader-follower structure."
        )
    notes.append(
        "Score = 0.35 x net one-way influence + 0.25 x feedback dominance "
        "+ 0.25 x Granger pass rate + 0.15 x timing lead share."
    )
    result = Result.of(table, source, notes)
    result.value.attrs["pairs"] = pairs
    return result


# ==========================================================================
# Regression-based driver attribution
# ==========================================================================


@safe_analysis()
def driver_regression(
    target: pd.Series,
    drivers: pd.DataFrame,
    source: str = "",
    difference: bool = True,
) -> Result[dict[str, Any]]:
    """Regress a price series on the workbook's candidate drivers.

    Run on differences by default so the coefficients describe genuine
    co-movement rather than shared trend. Reported alongside VIF so that
    collinear drivers are not over-interpreted.
    """
    joined = pd.concat([target.rename("target"), drivers], axis=1).dropna()
    if difference:
        joined = joined.diff().dropna()
    n_vars = joined.shape[1] - 1
    if n_vars < 1:
        return Result.unavailable("No usable driver series overlap the target.", source)
    if len(joined) < max(30, 5 * n_vars):
        return Result.unavailable(
            f"Only {len(joined)} complete observation(s) across the target and "
            f"{n_vars} driver(s); at least {max(30, 5 * n_vars)} are needed.",
            source,
        )

    y = joined["target"]
    X = joined.drop(columns="target")
    X = X.loc[:, X.std(ddof=0) > 0]
    if X.empty:
        return Result.unavailable(
            "All candidate drivers are constant over the overlapping window.",
            source,
        )

    model = OLS(y, add_constant(X)).fit()
    coefficients = pd.DataFrame(
        {
            "Coefficient": model.params,
            "Std error": model.bse,
            "t statistic": model.tvalues,
            "p-value": model.pvalues,
            "Significant": model.pvalues < CFG.alpha,
        }
    )
    # Standardised betas make drivers on different scales comparable.
    stds = X.std(ddof=1)
    y_std = float(y.std(ddof=1))
    betas = {
        name: float(model.params[name] * stds[name] / y_std) if y_std else np.nan
        for name in X.columns
    }
    coefficients["Standardised beta"] = pd.Series(betas)

    vif = variance_inflation(X, source=source)

    payload = {
        "coefficients": coefficients,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue),
        "n_obs": int(model.nobs),
        "vif": vif,
        "residuals": pd.Series(model.resid, index=y.index),
        "fitted": pd.Series(model.fittedvalues, index=y.index),
        "differenced": difference,
    }

    significant = [
        str(name)
        for name in X.columns
        if model.pvalues.get(name, 1.0) < CFG.alpha
    ]
    if significant:
        ranked = sorted(significant, key=lambda n: abs(betas.get(n, 0)), reverse=True)
        verdict = (
            f"{', '.join(ranked)} show a statistically significant "
            f"relationship with the target; the model explains "
            f"{model.rsquared:.0%} of the variation in "
            f"{'period-on-period changes' if difference else 'levels'}."
        )
    else:
        verdict = (
            "None of the available drivers shows a statistically significant "
            f"relationship (model R-squared {model.rsquared:.0%})."
        )
    return Result.of(
        payload,
        source,
        [
            verdict,
            "Estimated on first differences to avoid spurious regression."
            if difference
            else "Estimated on levels.",
            "Standardised betas allow drivers measured on different scales to "
            "be ranked against each other.",
        ],
    )


# ==========================================================================
# Convenience: full statistical profile for one series
# ==========================================================================


def full_profile(
    series: pd.Series, freq: str, source: str = "", window: int | None = None
) -> dict[str, Result[Any]]:
    """Run the standard battery on a single series.

    Used by the Price Analysis page and by the forecast explainability panel.
    """
    window = window or CFG.default_rolling_window
    return {
        "descriptive": descriptive_stats(series, source),
        "rolling": rolling_statistics(series, window, source),
        "changes": change_summary(series, source),
        "outliers": zscore_and_outliers(series, window, source),
        "stationarity": stationarity_tests(series, source),
        "autocorrelation": autocorrelation(series, source=source),
        "decomposition": decompose(series, freq, source=source),
        "seasonal_index": seasonal_indices(series, source),
    }
