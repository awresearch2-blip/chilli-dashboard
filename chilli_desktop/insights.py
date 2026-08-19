"""Automated, evidence-backed insight generation.

Every insight carries the statistic that produced it, the workbook sheet it
came from, and a strength rating. Nothing is emitted on a hunch: a generator
that cannot find statistical support returns either nothing or an explicit
"cannot be determined from this workbook" note.

That last category matters as much as the positive findings. The brief asks
about festival effects and cold-storage inventory behaviour; the workbook
cannot answer either, and saying so plainly is more useful than a
confident-sounding sentence with nothing behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import analytics, settings
from .preprocessing import DataService, FilterState
from .utils import (
    LOG,
    Result,
    describe_strength,
    fmt_number,
    fmt_pct,
    humanise_list,
    safe_analysis,
)

CFG = settings.ANALYTICS


# ==========================================================================
# Containers
# ==========================================================================

STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "informational": 3, "data gap": 4}


@dataclass
class Insight:
    """One automatically generated finding."""

    category: str
    headline: str
    detail: str = ""
    source: str = ""
    #: strong / moderate / weak / informational / data gap
    strength: str = "informational"
    #: bullish / bearish / neutral / n/a
    direction: str = "n/a"
    evidence: list[str] = field(default_factory=list)

    @property
    def sort_key(self) -> tuple[int, str]:
        return (STRENGTH_ORDER.get(self.strength, 9), self.category)


@dataclass
class SentimentComponent:
    """One input to the composite sentiment score."""

    name: str
    score: float
    explanation: str
    source: str = ""


@dataclass
class MarketSentiment:
    """Composite bullish/bearish reading with its components exposed."""

    score: float
    label: str
    components: list[SentimentComponent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        """Score rescaled from [-1, 1] to [0, 100] for gauge display."""
        return (self.score + 1.0) / 2.0 * 100.0


@dataclass
class Snapshot:
    """The headline numbers for one series, as shown on summary cards."""

    name: str
    latest: float
    latest_date: pd.Timestamp | None
    changes: pd.DataFrame
    source: str
    notes: list[str] = field(default_factory=list)

    def change_pct(self, horizon: str) -> float:
        if self.changes is None or self.changes.empty:
            return float("nan")
        row = self.changes[self.changes["Horizon"].str.contains(horizon, case=False, na=False)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["Change %"])


# ==========================================================================
# Snapshots
# ==========================================================================


def build_snapshot(
    service: DataService, series: pd.Series, name: str, source: str
) -> Snapshot:
    """Latest level plus multi-horizon change for a single series."""
    changes = analytics.change_summary(series, source)
    latest = float(series.iloc[-1]) if not series.empty else float("nan")
    latest_date = series.index[-1] if not series.empty else None
    return Snapshot(
        name=name,
        latest=latest,
        latest_date=latest_date,
        changes=changes.unwrap() if changes else pd.DataFrame(),
        source=source,
        notes=list(changes.notes) if changes else [changes.reason],
    )


def variety_snapshots(service: DataService, filters: FilterState) -> list[Snapshot]:
    """Snapshots for every variety the workbook quotes."""
    out: list[Snapshot] = []
    for variety in service.varieties():
        result = service.variety_series(variety)
        if not result:
            continue
        series = service.apply_filters(result.unwrap(), filters, kind="price")
        if series.empty:
            continue
        out.append(build_snapshot(service, series, variety, result.source))
    return out


def market_snapshots(service: DataService, filters: FilterState) -> list[Snapshot]:
    """Snapshots for every market the workbook covers."""
    out: list[Snapshot] = []
    for market in service.markets():
        result = service.market_price(market)
        if not result:
            continue
        series = service.apply_filters(result.unwrap(), filters, kind="price")
        if series.empty:
            continue
        out.append(build_snapshot(service, series, market, result.source))
    return out


# ==========================================================================
# Sentiment
# ==========================================================================


def compute_sentiment(
    service: DataService, variety: str, filters: FilterState
) -> Result[MarketSentiment]:
    """Build a composite bullish/bearish score from workbook data only.

    Five components, each scored in [-1, +1] and equally weighted:

    1. **Trend** -- latest price versus its own long moving average.
    2. **Momentum** -- month-on-month change, scaled by the series' own
       typical monthly move so the scale adapts to this commodity.
    3. **Range position** -- where the price sits in its trailing one-year
       range.
    4. **Arrivals pressure** -- recent arrivals against their seasonal norm;
       heavy arrivals score bearish.
    5. **Seasonal setup** -- the seasonal index of the month ahead.

    A component that the workbook cannot support is dropped and the remaining
    ones are re-averaged, with the omission recorded in the notes.
    """
    price_result = service.variety_series(variety)
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)

    daily = service.apply_filters(price_result.unwrap(), filters, kind="price")
    if len(daily) < 60:
        return Result.unavailable(
            f"Only {len(daily)} observation(s) of {variety} after filtering; "
            "at least 60 are needed for a sentiment reading.",
            price_result.source,
        )

    components: list[SentimentComponent] = []
    notes: list[str] = []
    latest = float(daily.iloc[-1])

    # 1. Trend against a long moving average.
    window = min(180, max(30, len(daily) // 3))
    long_ma = float(daily.rolling(window).mean().iloc[-1])
    if np.isfinite(long_ma) and long_ma:
        deviation = (latest - long_ma) / long_ma
        score = float(np.clip(deviation / 0.15, -1, 1))
        components.append(
            SentimentComponent(
                "Trend",
                score,
                f"Price {fmt_number(latest, 0)} is {fmt_pct(deviation, 1, signed=True)} "
                f"versus its {window}-day average of {fmt_number(long_ma, 0)}.",
                price_result.source,
            )
        )
    else:
        notes.append("Trend component omitted: the moving average is undefined.")

    # 2. Momentum, scaled by this series' own typical monthly move.
    monthly = service.series_at(daily, "ME")
    if len(monthly) >= 13:
        month_change = monthly.pct_change().dropna()
        recent = float(month_change.iloc[-1])
        typical = float(month_change.abs().median())
        if typical > 0:
            score = float(np.clip(recent / (3 * typical), -1, 1))
            components.append(
                SentimentComponent(
                    "Momentum",
                    score,
                    f"Latest monthly move {fmt_pct(recent, 1, signed=True)} against a "
                    f"typical monthly move of {fmt_pct(typical, 1)}.",
                    price_result.source,
                )
            )
        else:
            notes.append("Momentum component omitted: the series shows no monthly variation.")
    else:
        notes.append(
            f"Momentum component omitted: only {len(monthly)} monthly "
            "observation(s) available."
        )

    # 3. Position within the trailing one-year range.
    trailing = daily[daily.index >= daily.index[-1] - pd.DateOffset(years=1)]
    if len(trailing) >= 20:
        low, high = float(trailing.min()), float(trailing.max())
        if high > low:
            position = (latest - low) / (high - low)
            score = float(np.clip((position - 0.5) * 2, -1, 1))
            components.append(
                SentimentComponent(
                    "Range position",
                    score,
                    f"Price sits at the {position:.0%} mark of its trailing "
                    f"one-year range ({fmt_number(low, 0)} to {fmt_number(high, 0)}).",
                    price_result.source,
                )
            )
        else:
            notes.append("Range component omitted: the trailing year shows no range.")
    else:
        notes.append("Range component omitted: less than a year of recent data.")

    # 4. Arrivals pressure versus the seasonal norm.
    arrivals_result = service.guntur_arrivals()
    if arrivals_result:
        arrivals = service.apply_filters(arrivals_result.unwrap(), filters, kind="arrivals")
        monthly_arrivals = service.series_at(arrivals, "ME", "sum")
        if len(monthly_arrivals) >= 24:
            current_month = int(monthly_arrivals.index[-1].month)
            same_month = monthly_arrivals[monthly_arrivals.index.month == current_month]
            history = same_month.iloc[:-1]
            if len(history) >= 3:
                norm = float(history.mean())
                recent = float(same_month.iloc[-1])
                if norm > 0:
                    excess = (recent - norm) / norm
                    # Heavy arrivals weigh on price, hence the sign flip.
                    score = float(np.clip(-excess / 0.4, -1, 1))
                    components.append(
                        SentimentComponent(
                            "Arrivals pressure",
                            score,
                            f"Arrivals in the latest month are "
                            f"{fmt_pct(excess, 0, signed=True)} versus the "
                            f"{len(history)}-year average for the same month; "
                            "heavier arrivals weigh on prices.",
                            arrivals_result.source,
                        )
                    )
            else:
                notes.append(
                    "Arrivals component omitted: too few prior years for this "
                    "calendar month."
                )
        else:
            notes.append("Arrivals component omitted: fewer than 24 monthly observations.")
    else:
        notes.append(f"Arrivals component omitted: {arrivals_result.reason}")

    # 5. Seasonal setup for the month ahead.
    seasonal = analytics.seasonal_indices(daily, price_result.source)
    if seasonal:
        table = seasonal.unwrap()
        next_month = int((daily.index[-1] + pd.DateOffset(months=1)).month)
        label = settings.MONTH_ABBREVIATIONS[next_month - 1].title()
        if label in table.index:
            index_value = float(table.loc[label, "Seasonal index"])
            score = float(np.clip((index_value - 1.0) / 0.10, -1, 1))
            components.append(
                SentimentComponent(
                    "Seasonal setup",
                    score,
                    f"{label} carries a seasonal index of {index_value:.3f} "
                    f"({fmt_pct(index_value - 1, 1, signed=True)} versus the "
                    "all-month average) across this series' history.",
                    seasonal.source,
                )
            )
    else:
        notes.append(f"Seasonal component omitted: {seasonal.reason}")

    if not components:
        return Result.unavailable(
            "None of the five sentiment components could be computed from the "
            "workbook for this selection.",
            price_result.source,
        )

    score = float(np.mean([c.score for c in components]))
    if score >= 0.45:
        label = "Strongly bullish"
    elif score >= 0.15:
        label = "Mildly bullish"
    elif score > -0.15:
        label = "Neutral"
    elif score > -0.45:
        label = "Mildly bearish"
    else:
        label = "Strongly bearish"

    notes.insert(
        0,
        f"Composite of {len(components)} equally weighted component(s), each "
        "scored between -1 (bearish) and +1 (bullish) from workbook data only.",
    )
    if len(components) < 5:
        notes.append(
            f"{5 - len(components)} of the 5 intended components could not be "
            "computed and were excluded from the average rather than assumed "
            "neutral."
        )

    return Result.of(
        MarketSentiment(score=score, label=label, components=components, notes=notes),
        price_result.source,
        notes,
    )


# ==========================================================================
# Insight generators
# ==========================================================================


@safe_analysis()
def price_leadership_insights(
    service: DataService, filters: FilterState, freq: str = "D"
) -> Result[list[Insight]]:
    """Which variety leads price discovery, and which follow."""
    panel = service.variety_panel(freq, filters)
    if not panel:
        return Result.unavailable(panel.reason, panel.source)
    frame = panel.unwrap()
    if frame.shape[1] < 2:
        return Result.unavailable(
            "At least two varieties are needed to assess price leadership; "
            f"the current filters leave {frame.shape[1]}.",
            panel.source,
        )

    ranking = analytics.leadership_ranking(frame, freq, panel.source)
    if not ranking:
        return Result.unavailable(ranking.reason, panel.source)
    table = ranking.unwrap()
    out: list[Insight] = []

    leader = table.iloc[0]
    followers = table.tail(min(2, max(0, len(table) - 1)))
    strength = (
        "strong"
        if float(leader["Leadership score"]) >= 0.6
        else "moderate"
        if float(leader["Leadership score"]) >= 0.45
        else "weak"
    )
    detail_parts = [
        f"{leader['Series']} scores {float(leader['Leadership score']):.2f} on the "
        f"combined leadership measure across {int(leader['Comparisons'])} "
        "pairwise comparisons."
    ]
    if int(leader["Influences (one-way)"]):
        detail_parts.append(
            f"It Granger-causes {int(leader['Influences (one-way)'])} other "
            "variety without being caused back."
        )
    if int(leader["Dominant in feedback"]):
        detail_parts.append(
            f"It is the stronger side in {int(leader['Dominant in feedback'])} "
            "two-way feedback pair(s)."
        )
    out.append(
        Insight(
            category="Price leadership",
            headline=f"{leader['Series']} leads price discovery among Guntur varieties.",
            detail=" ".join(detail_parts),
            source=panel.source,
            strength=strength,
            evidence=list(ranking.notes),
        )
    )

    if not followers.empty:
        names = [str(r["Series"]) for _, r in followers.iterrows()]
        out.append(
            Insight(
                category="Price leadership",
                headline=f"{humanise_list(names)} follow rather than lead.",
                detail=(
                    "These varieties rank lowest on net influence: their moves "
                    "are better explained by other varieties than the reverse. "
                    "A move that shows up only here is unlikely to spread."
                ),
                source=panel.source,
                strength="moderate",
            )
        )

    if not table["Times leader (timing)"].sum():
        out.append(
            Insight(
                category="Price leadership",
                headline="No variety offers an exploitable timing lead.",
                detail=(
                    "Every pairwise cross-correlation peaks at zero lag, so "
                    "varieties re-price on the same day. Leadership here means "
                    "carrying information, not moving first — there is no lag "
                    "window to trade."
                ),
                source=panel.source,
                strength="strong",
            )
        )

    # Sentiment driver: the variety most correlated with the rest of the complex.
    correlation = analytics.correlation_matrix(frame, "pearson", panel.source)
    if correlation:
        matrix = correlation.unwrap()
        mean_correlation = (matrix.sum() - 1) / max(1, matrix.shape[1] - 1)
        if mean_correlation.notna().any():
            driver = str(mean_correlation.idxmax())
            out.append(
                Insight(
                    category="Price leadership",
                    headline=f"{driver} best represents overall market sentiment.",
                    detail=(
                        f"It shows the highest average correlation "
                        f"({mean_correlation.max():.2f}) with the other "
                        "varieties, so it is the single best proxy for the "
                        "complex as a whole."
                    ),
                    source=panel.source,
                    strength="moderate",
                )
            )
    return Result.of(out, panel.source)


@safe_analysis()
def market_integration_insights(
    service: DataService, filters: FilterState, freq: str = "D"
) -> Result[list[Insight]]:
    """Whether Guntur drives Warangal and Khammam, and how integrated they are."""
    panel = service.price_panel(freq, filters)
    if not panel:
        return Result.unavailable(panel.reason, panel.source)
    frame = panel.unwrap()
    if frame.shape[1] < 2:
        return Result.unavailable(
            "At least two markets are needed to assess integration.", panel.source
        )

    out: list[Insight] = []

    influence = analytics.directional_influence(frame, freq, panel.source)
    if influence:
        pairs = influence.unwrap()
        guntur_rows = pairs[
            (pairs["Series A"] == "Guntur") | (pairs["Series B"] == "Guntur")
        ]
        # (market name, relationship description) pairs -- keep the names
        # intact, since "Khammam (cold storage)" and "Khammam (non cold
        # storage)" are different markets that must not collapse to "Khammam".
        drives: list[tuple[str, str]] = []
        for _, row in guntur_rows.iterrows():
            other = str(row["Series B"] if row["Series A"] == "Guntur" else row["Series A"])
            direction = str(row["Direction"])
            if row["Verdict"] == "One-way" and direction.startswith("Guntur"):
                drives.append((other, "one-way"))
            elif str(row["Verdict"]).startswith("Feedback") and direction.startswith("Guntur"):
                drives.append((other, "dominant side of a two-way relationship"))
        if drives:
            out.append(
                Insight(
                    category="Market leadership",
                    headline=(
                        "Guntur influences "
                        + humanise_list([name for name, _ in drives])
                        + "."
                    ),
                    detail=(
                        "Direction of influence: "
                        + "; ".join(f"{name} — {kind}" for name, kind in drives)
                        + ". Direction is established by running Granger "
                        "causality both ways and comparing the strength of "
                        "evidence."
                    ),
                    source=panel.source,
                    strength=(
                        "strong" if any(kind == "one-way" for _, kind in drives) else "moderate"
                    ),
                    evidence=list(influence.notes),
                )
            )
        else:
            out.append(
                Insight(
                    category="Market leadership",
                    headline="Guntur does not measurably lead the other markets.",
                    detail=(
                        "No one-way or dominant relationship from Guntur to "
                        "Warangal or Khammam is detectable over this sample."
                    ),
                    source=panel.source,
                    strength="moderate",
                    evidence=list(influence.notes),
                )
            )

        for _, row in pairs[pairs["Verdict"] == "Independent"].iterrows():
            out.append(
                Insight(
                    category="Market leadership",
                    headline=(
                        f"{row['Series A']} and {row['Series B']} move independently."
                    ),
                    detail=(
                        "Neither market's past helps predict the other "
                        f"(best p-values {row['A -> B best p']:.3f} and "
                        f"{row['B -> A best p']:.3f}). Treat them as separate "
                        "price pools rather than one market."
                    ),
                    source=panel.source,
                    strength="moderate",
                )
            )

    cointegration = analytics.cointegration(frame.dropna(), panel.source)
    if cointegration:
        payload = cointegration.unwrap()
        rank = int(payload["rank"])
        n_obs = int(payload["n_obs"])
        if rank > 0:
            out.append(
                Insight(
                    category="Market integration",
                    headline=(
                        f"The markets are cointegrated (rank {rank}): spreads "
                        "mean-revert."
                    ),
                    detail=(
                        f"Tested on {n_obs} periods where all "
                        f"{len(payload['columns'])} markets traded "
                        "simultaneously. A cointegrated system cannot drift "
                        "apart permanently, so an unusually wide spread between "
                        "two of these mandis is a temporary dislocation rather "
                        "than a new level."
                    ),
                    source=panel.source,
                    strength="strong",
                    evidence=list(cointegration.notes),
                )
            )
        else:
            out.append(
                Insight(
                    category="Market integration",
                    headline="No cointegrating relationship is detectable.",
                    detail=(
                        f"Across {n_obs} common periods the markets show no "
                        "shared long-run trend, so spread positions have no "
                        "statistical anchor to revert to."
                    ),
                    source=panel.source,
                    strength="moderate",
                    evidence=list(cointegration.notes),
                )
            )
    else:
        out.append(
            Insight(
                category="Market integration",
                headline="Cointegration could not be tested.",
                detail=cointegration.reason,
                source=panel.source,
                strength="data gap",
            )
        )
    return Result.of(out, panel.source)


@safe_analysis()
def arrival_impact_insights(
    service: DataService, filters: FilterState, variety: str = "Teja", freq: str = "W"
) -> Result[list[Insight]]:
    """How arrivals move prices: elasticity, thresholds and timing."""
    price_result = service.variety_series(variety)
    arrivals_result = service.guntur_arrivals()
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)
    if not arrivals_result:
        return Result.unavailable(arrivals_result.reason, arrivals_result.source)

    source = f"{price_result.source}; {arrivals_result.source}"
    price = service.series_at(
        service.apply_filters(price_result.unwrap(), filters, kind="price"), freq
    )
    arrivals = service.series_at(
        service.apply_filters(arrivals_result.unwrap(), filters, kind="arrivals"),
        freq,
        "sum",
    )
    out: list[Insight] = []

    elasticity = analytics.elasticity(price, arrivals, source)
    if elasticity:
        table = elasticity.unwrap()
        significant = table[table["Significant"]]
        if not significant.empty:
            best = int(significant["p-value"].idxmin())
            value = float(significant.loc[best, "Elasticity"])
            out.append(
                Insight(
                    category="Arrivals impact",
                    headline=(
                        f"A 1% rise in arrivals moves {variety} "
                        f"{'down' if value < 0 else 'up'} {abs(value):.3f}% "
                        f"{'in the same period' if best == 0 else f'{best} period(s) later'}."
                    ),
                    detail=(
                        f"Estimated on log differences over "
                        f"{int(significant.loc[best, 'Observations'])} periods "
                        f"(p={significant.loc[best, 'p-value']:.4f}, "
                        f"R²={significant.loc[best, 'R-squared']:.3f}). The "
                        "elasticity is small in absolute terms, so arrivals "
                        "alone explain little of the price move."
                    ),
                    source=source,
                    strength="moderate" if abs(value) > 0.05 else "weak",
                    direction="bearish" if value < 0 else "bullish",
                    evidence=list(elasticity.notes),
                )
            )
        else:
            out.append(
                Insight(
                    category="Arrivals impact",
                    headline=(
                        f"Arrivals show no statistically significant elasticity "
                        f"on {variety} prices at this frequency."
                    ),
                    detail=(
                        "No lag from 0 to 4 periods produces a significant "
                        "log-log relationship. On this workbook, arrivals are "
                        "not a reliable standalone price signal."
                    ),
                    source=source,
                    strength="moderate",
                    evidence=list(elasticity.notes),
                )
            )

    thresholds = analytics.threshold_effects(price, arrivals, source=source)
    if thresholds:
        table = thresholds.unwrap()
        level = table.attrs.get("threshold_level")
        pvalue = table.attrs.get("threshold_pvalue")
        heaviest = table.iloc[-1]
        lightest = table.iloc[0]
        significant = pvalue is not None and np.isfinite(pvalue) and pvalue < CFG.alpha
        if level is not None and np.isfinite(level):
            out.append(
                Insight(
                    category="Arrivals impact",
                    headline=(
                        f"Arrivals above {fmt_number(level, 0)} bags coincide "
                        f"with average price changes of "
                        f"{heaviest['Mean next-period change %']:+.2f}% next period."
                    ),
                    detail=(
                        f"In the lightest arrivals quintile (below "
                        f"{fmt_number(lightest['Arrivals to'], 0)} bags) the "
                        f"average next-period change is "
                        f"{lightest['Mean next-period change %']:+.2f}%, and prices "
                        f"fell in {lightest['Share of periods with a fall']:.0f}% of "
                        f"periods against "
                        f"{heaviest['Share of periods with a fall']:.0f}% in the "
                        "heaviest quintile. "
                        + (
                            f"The difference is statistically significant (p={pvalue:.3f})."
                            if significant
                            else f"The difference is not statistically significant "
                            f"(p={pvalue:.3f}), so treat the threshold as a "
                            "descriptive tendency rather than a rule."
                            if pvalue is not None and np.isfinite(pvalue)
                            else ""
                        )
                    ),
                    source=source,
                    strength="moderate" if significant else "weak",
                    direction="bearish",
                    evidence=list(thresholds.notes),
                )
            )

    lagged = analytics.lagged_impact(price, arrivals, source=source)
    if lagged:
        table = lagged.unwrap()
        significant = table[table["Significant"]]
        if not significant.empty:
            best = int(significant["Correlation with price change"].abs().idxmax())
            out.append(
                Insight(
                    category="Arrivals impact",
                    headline=(
                        "The arrivals effect on prices is "
                        + (
                            "immediate (same period)."
                            if best == 0
                            else f"delayed by {best} period(s)."
                        )
                    ),
                    detail=(
                        f"Correlation between the change in arrivals and the "
                        f"change in price peaks at lag {best} "
                        f"(r={table.loc[best, 'Correlation with price change']:.3f}, "
                        f"p={table.loc[best, 'p-value']:.4f})."
                    ),
                    source=source,
                    strength="weak",
                    evidence=list(lagged.notes),
                )
            )
    return Result.of(out, source)


@safe_analysis()
def seasonality_insights(
    service: DataService, filters: FilterState, variety: str = "Teja"
) -> Result[list[Insight]]:
    """Seasonal strength, firm and soft months, harvest and lean periods."""
    price_result = service.variety_series(variety)
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)
    series = service.apply_filters(price_result.unwrap(), filters, kind="price")
    out: list[Insight] = []

    indices = analytics.seasonal_indices(series, price_result.source)
    if indices:
        table = indices.unwrap()
        firm = table.nlargest(3, "Seasonal index")
        soft = table.nsmallest(3, "Seasonal index")
        spread = float(table["Seasonal index"].max() - table["Seasonal index"].min())
        out.append(
            Insight(
                category="Seasonality",
                headline=(
                    f"{variety} is seasonally firmest in "
                    f"{humanise_list(list(firm.index))} and softest in "
                    f"{humanise_list(list(soft.index))}."
                ),
                detail=(
                    f"Peak seasonal index {firm['Seasonal index'].iloc[0]:.3f} "
                    f"({firm.index[0]}) against trough "
                    f"{soft['Seasonal index'].iloc[0]:.3f} ({soft.index[0]}) — a "
                    f"spread of {spread * 100:.1f}% of the average price level. "
                    + (
                        "That is a wide enough seasonal swing to time purchases around."
                        if spread > 0.15
                        else "That is a modest seasonal swing; the calendar is a "
                        "secondary consideration next to the trend."
                    )
                ),
                source=indices.source,
                strength="strong" if spread > 0.15 else "moderate",
                evidence=list(indices.notes),
            )
        )

    decomposition = analytics.decompose(series, "ME", source=price_result.source)
    if decomposition:
        shares = decomposition.value.attrs.get("variance_shares", {})
        seasonal_share = shares.get("Seasonal", float("nan"))
        if np.isfinite(seasonal_share):
            out.append(
                Insight(
                    category="Seasonality",
                    headline=(
                        f"Seasonality explains {seasonal_share:.0%} of the "
                        f"variation in {variety} prices."
                    ),
                    detail=(
                        f"Trend accounts for {shares.get('Trend', 0):.0%} and "
                        f"irregular movement {shares.get('Residual', 0):.0%}. "
                        + (
                            "With trend dominating this strongly, seasonal "
                            "timing is a refinement, not the main driver."
                            if shares.get("Trend", 0) > 0.6
                            else "Seasonality is a material component and worth "
                            "trading around."
                        )
                    ),
                    source=decomposition.source,
                    strength="moderate",
                    evidence=list(decomposition.notes),
                )
            )

    season = service.season_profile()
    if season:
        table = season.unwrap()
        peak = table[table["season"] == "Peak arrivals"]["month_name"].tolist()
        lean = table[table["season"] == "Lean arrivals"]["month_name"].tolist()
        if peak and lean:
            out.append(
                Insight(
                    category="Seasonality",
                    headline=(
                        f"Peak arrivals fall in {humanise_list(peak)}; "
                        f"{humanise_list(lean)} are the lean months."
                    ),
                    detail=(
                        "Derived by ranking calendar months on mean arrivals "
                        "across every year in the workbook — the top third is "
                        "labelled peak, the bottom third lean. This is the "
                        "workbook's own harvest signature, not an external "
                        "crop calendar."
                    ),
                    source=season.source,
                    strength="strong",
                    evidence=list(season.notes),
                )
            )

    # The brief asks about festival effects; the workbook has no festival dates.
    out.append(
        Insight(
            category="Seasonality",
            headline="Festival effects cannot be measured from this workbook.",
            detail=(
                f"{settings.DATA_UNAVAILABLE_MESSAGE} Isolating a festival "
                "effect requires a festival calendar — Diwali, Sankranti and "
                "similar dates move through the Gregorian calendar year to "
                "year. The workbook contains no such dates, so any monthly "
                "pattern shown here blends festival demand with harvest "
                "timing and cannot separate the two."
            ),
            source="—",
            strength="data gap",
        )
    )
    return Result.of(out, price_result.source)


@safe_analysis()
def export_insights(
    service: DataService, filters: FilterState, variety: str = "Teja"
) -> Result[list[Insight]]:
    """Export volume against price: level, timing and stability."""
    exports_result = service.exports_monthly()
    price_result = service.variety_series(variety)
    if not exports_result:
        return Result.unavailable(exports_result.reason, exports_result.source)
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)

    source = f"{exports_result.source}; {price_result.source}"
    exports = exports_result.unwrap()  # already month-end indexed
    price = service.series_at(
        service.apply_filters(price_result.unwrap(), filters, kind="price"), "ME"
    )
    out: list[Insight] = []

    pair = analytics.correlation_pair(price, exports, source)
    if pair:
        stats = pair.unwrap()
        r = float(stats["Pearson r"])
        p = float(stats["Pearson p-value"])
        out.append(
            Insight(
                category="Exports",
                headline=(
                    f"Exports show a {describe_strength(r)} "
                    f"{'positive' if r > 0 else 'negative'} relationship with "
                    f"{variety} prices (r={r:.2f})."
                ),
                detail=(
                    f"Measured on {int(stats['Overlapping observations'])} "
                    f"overlapping months; p={p:.4f}, so the relationship is "
                    + ("statistically significant. " if p < CFG.alpha else "not statistically significant. ")
                    + "Correlation on monthly levels does not establish "
                    "direction — see the lead-lag reading below."
                ),
                source=source,
                strength="moderate" if p < CFG.alpha else "weak",
                direction="bullish" if r > 0.3 and p < CFG.alpha else "n/a",
                evidence=list(pair.notes),
            )
        )

    reading = analytics.lead_lag(exports, price, "Exports", f"{variety} price", "ME", 12, source)
    if reading:
        value = reading.unwrap()
        out.append(
            Insight(
                category="Exports",
                headline=value.sentence(),
                detail=(
                    "Computed on month-on-month changes so a shared trend "
                    "cannot manufacture a lead. "
                    + (
                        "The peak clears the 95% band, so the timing is "
                        "statistically supported."
                        if value.significant
                        else "The peak does not clear the 95% significance "
                        "band, so treat the timing as indicative only."
                    )
                ),
                source=source,
                strength="moderate" if value.significant else "weak",
                evidence=list(reading.notes),
            )
        )

    rolling = analytics.rolling_correlation(price, exports, 24, source)
    if rolling:
        values = rolling.unwrap()
        out.append(
            Insight(
                category="Exports",
                headline=(
                    "The export-price relationship is "
                    + (
                        "unstable over time."
                        if float(values.max() - values.min()) > 0.8
                        else "reasonably stable over time."
                    )
                ),
                detail=(
                    f"A 24-month rolling correlation ranges from "
                    f"{values.min():.2f} to {values.max():.2f}, currently "
                    f"{values.iloc[-1]:.2f}. A relationship that flips sign "
                    "cannot be relied on for positioning."
                ),
                source=source,
                strength="moderate",
                evidence=list(rolling.notes),
            )
        )
    return Result.of(out, source)


@safe_analysis()
def currency_insights(
    service: DataService, filters: FilterState, variety: str = "Teja"
) -> Result[list[Insight]]:
    """USD/INR against prices, plus the workbook's coverage limitation."""
    fx_result = service.usd_inr()
    price_result = service.variety_series(variety)
    if not fx_result:
        return Result.unavailable(fx_result.reason, fx_result.source)
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)

    source = f"{fx_result.source}; {price_result.source}"
    fx = fx_result.unwrap()
    out: list[Insight] = []

    gaps = service.coverage_gaps(fx)
    if not gaps.empty:
        worst = gaps.iloc[0]
        out.append(
            Insight(
                category="Currency",
                headline=(
                    f"The USD/INR series has a {int(worst['days'])}-day hole "
                    f"({worst['from']} to {worst['to']})."
                ),
                detail=(
                    f"{len(gaps)} gap(s) longer than 45 days are present. Any "
                    "correlation or regression that spans this break is "
                    "effectively joining two disconnected periods, and the "
                    "exchange-rate results below should be read with that in "
                    "mind."
                ),
                source=fx_result.source,
                strength="data gap",
            )
        )

    monthly_fx = service.series_at(fx, "ME")
    price = service.series_at(
        service.apply_filters(price_result.unwrap(), filters, kind="price"), "ME"
    )
    pair = analytics.correlation_pair(price, monthly_fx, source)
    if pair:
        stats = pair.unwrap()
        r = float(stats["Pearson r"])
        p = float(stats["Pearson p-value"])
        out.append(
            Insight(
                category="Currency",
                headline=(
                    f"USD/INR and {variety} prices show a {describe_strength(r)} "
                    f"{'positive' if r > 0 else 'negative'} co-movement (r={r:.2f})."
                ),
                detail=(
                    f"On {int(stats['Overlapping observations'])} overlapping "
                    f"months, p={p:.4f}. A weaker rupee raises the rupee value "
                    "of an export sale, which supports domestic prices — but "
                    "both series also trend upward over this period, so part "
                    "of this correlation is shared trend rather than a "
                    "mechanism."
                ),
                source=source,
                strength="moderate" if p < CFG.alpha else "weak",
                direction="bullish" if r > 0.3 and p < CFG.alpha else "n/a",
                evidence=list(pair.notes),
            )
        )

    granger = analytics.granger_causality(
        monthly_fx.rename("USD/INR"), price.rename(f"{variety} price"), 6, source
    )
    if granger:
        table = granger.unwrap()
        if bool(table["Significant"].any()):
            best = int(table["p-value"].idxmin())
            out.append(
                Insight(
                    category="Currency",
                    headline=(
                        f"Past USD/INR moves help predict {variety} prices "
                        f"({best} month(s) ahead)."
                    ),
                    detail=(
                        f"Granger test on differenced monthly series, "
                        f"p={table.loc[best, 'p-value']:.4f}. This is "
                        "predictive precedence rather than proof of causation, "
                        "but it does mean the exchange rate carries usable "
                        "information."
                    ),
                    source=source,
                    strength="moderate",
                    evidence=list(granger.notes),
                )
            )
        else:
            out.append(
                Insight(
                    category="Currency",
                    headline=f"USD/INR does not predict {variety} prices.",
                    detail=(
                        "No lag up to 6 months improves the forecast of price "
                        "changes. The exchange rate co-moves with prices but "
                        "does not lead them in this sample."
                    ),
                    source=source,
                    strength="moderate",
                    evidence=list(granger.notes),
                )
            )
    return Result.of(out, source)


@safe_analysis()
def balance_sheet_insights(service: DataService) -> Result[list[Insight]]:
    """Supply, demand and stocks against the price record."""
    table_result = service.balance_sheet()
    if not table_result:
        return Result.unavailable(table_result.reason, table_result.source)
    table = table_result.unwrap()
    out: list[Insight] = []

    stock_use = service.balance_sheet_row("Stock to Use")
    production = service.balance_sheet_row("Production")
    ending = service.balance_sheet_row("Ending Stock")

    if stock_use:
        values = stock_use.unwrap()
        latest_year = values.index[-1]
        latest = float(values.iloc[-1])
        historical = values.iloc[:-1]
        mean = float(historical.mean()) if not historical.empty else float("nan")
        out.append(
            Insight(
                category="Balance sheet",
                headline=(
                    f"The stock-to-use ratio for {latest_year} is {latest:.1f}%, "
                    + (
                        "below"
                        if np.isfinite(mean) and latest < mean
                        else "above"
                    )
                    + f" the {int(len(historical))}-year average of {mean:.1f}%."
                ),
                detail=(
                    "A low stock-to-use ratio means little buffer between "
                    "supply and demand, which historically coincides with "
                    "firmer and more volatile prices. "
                    + (
                        f"{latest_year} is flagged as expected in the workbook, "
                        "so this is the workbook's own projection rather than "
                        "realised data."
                        if latest_year in service.data.datasets["balance_sheet"].meta.get(
                            "projected_years", []
                        )
                        else ""
                    )
                ),
                source=table_result.source,
                strength="moderate",
                direction="bullish" if np.isfinite(mean) and latest < mean else "bearish",
                evidence=list(table_result.notes),
            )
        )

    # Relate the annual balance sheet to annual average price.
    price_result = service.variety_series("Teja")
    if price_result and stock_use:
        annual_price = service.series_at(price_result.unwrap(), "YE")
        # Re-key by calendar year so it joins the year-indexed balance sheet.
        annual_price = pd.Series(
            annual_price.to_numpy(),
            index=pd.Index(annual_price.index.year, name="year"),
        )
        ratio = stock_use.unwrap()
        joined = pd.concat(
            [annual_price.rename("price"), ratio.rename("stock_to_use")], axis=1
        ).dropna()
        if len(joined) >= 5:
            r = float(joined["price"].corr(joined["stock_to_use"]))
            # Supply-buffer logic predicts a *negative* relationship: more
            # carry-out, softer prices. Say plainly when the data disagrees.
            if r < -0.2:
                reading = (
                    "The negative sign is what supply-buffer logic predicts: "
                    "more carry-out, softer prices."
                )
            elif r > 0.2:
                reading = (
                    "The positive sign runs against supply-buffer logic, which "
                    "would predict softer prices when carry-out is larger."
                )
            else:
                reading = (
                    "The relationship is effectively flat — on this sample the "
                    "stock-to-use ratio carries no information about the annual "
                    "average price."
                )
            out.append(
                Insight(
                    category="Balance sheet",
                    headline=(
                        f"Stock-to-use and the annual average price show a "
                        f"{describe_strength(r)} "
                        f"{'inverse' if r < 0 else 'positive'} relationship "
                        f"(r={r:.2f})."
                    ),
                    detail=(
                        f"Based on only {len(joined)} paired years, which is far "
                        "too few for a significance test or a regression. "
                        + reading
                        + " With this sample size it is a directional "
                        "observation, not a finding."
                    ),
                    source=f"{table_result.source}; {price_result.source}",
                    strength="weak",
                )
            )

    if production:
        values = production.unwrap()
        if len(values) >= 3:
            change = (float(values.iloc[-1]) - float(values.iloc[-2])) / float(values.iloc[-2])
            out.append(
                Insight(
                    category="Balance sheet",
                    headline=(
                        f"Production for {values.index[-1]} is "
                        f"{fmt_pct(change, 1, signed=True)} versus "
                        f"{values.index[-2]}."
                    ),
                    detail=(
                        f"{fmt_number(values.iloc[-1], 2)} against "
                        f"{fmt_number(values.iloc[-2], 2)} "
                        f"{service.data.datasets['balance_sheet'].meta.get('unit_note', '')}. "
                        + (
                            "A contraction of this size tightens the balance "
                            "and is price-supportive."
                            if change < -0.05
                            else "An expansion of this size loosens the balance "
                            "and weighs on prices."
                            if change > 0.05
                            else "A broadly flat year."
                        )
                    ),
                    source=table_result.source,
                    strength="moderate",
                    direction="bullish" if change < -0.05 else "bearish" if change > 0.05 else "neutral",
                )
            )
    return Result.of(out, table_result.source)


@safe_analysis()
def cold_storage_insights(service: DataService) -> Result[list[Insight]]:
    """Cold storage stock: what the workbook can and cannot support."""
    result = service.cold_storage()
    if not result:
        return Result.of(
            [
                Insight(
                    category="Cold storage",
                    headline="Cold storage inventory analysis cannot be performed.",
                    detail=f"{settings.DATA_UNAVAILABLE_MESSAGE} {result.reason}",
                    source="—",
                    strength="data gap",
                )
            ]
        )

    frame = result.unwrap()
    dataset = service.data.datasets["cold_storage_stock"]
    coverage = dataset.meta.get("observations_per_location", {})
    densest = max(coverage.values()) if coverage else 0
    out: list[Insight] = [
        Insight(
            category="Cold storage",
            headline=(
                "Cold storage stock is too sparsely reported for inventory-price "
                "analysis."
            ),
            detail=(
                f"{settings.DATA_UNAVAILABLE_MESSAGE} The sheet holds "
                f"{len(frame)} reporting month(s) across "
                f"{frame.shape[1]} location(s), and the best-covered location "
                f"has {densest} observation(s) — against the "
                f"{CFG.min_obs_correlation} minimum this application requires "
                "before reporting any correlation. Inventory-versus-price, "
                "delayed effects and seasonal storage behaviour therefore "
                "cannot be measured. The levels themselves are shown on the "
                "Balance Sheet page for reference."
            ),
            source=result.source,
            strength="data gap",
            evidence=list(dataset.warnings),
        )
    ]

    # The Khammam sheets do split cold from non-cold, which *is* analysable.
    cold = service.market_price("Khammam (cold storage)")
    fresh = service.market_price("Khammam (non cold storage)")
    if cold and fresh:
        joined = pd.concat(
            [cold.unwrap().rename("cold"), fresh.unwrap().rename("fresh")], axis=1
        ).dropna()
        if len(joined) >= CFG.min_obs_correlation:
            premium = joined["cold"] - joined["fresh"]
            share = float((premium > 0).mean())
            out.append(
                Insight(
                    category="Cold storage",
                    headline=(
                        f"Khammam cold-storage lots trade at a median premium of "
                        f"{fmt_number(premium.median(), 0)} over fresh lots."
                    ),
                    detail=(
                        f"Across {len(joined)} days when both were quoted, "
                        f"cold-storage material was dearer on {share:.0%} of "
                        f"days. Premium range {fmt_number(premium.min(), 0)} to "
                        f"{fmt_number(premium.max(), 0)}. This is the one "
                        "storage-related relationship the workbook does support, "
                        "because it comes from the two Khammam price sheets "
                        "rather than the sparse stock sheet."
                    ),
                    source=f"{cold.source}; {fresh.source}",
                    strength="strong",
                    evidence=[
                        f"Median premium {fmt_number(premium.median(), 0)}, "
                        f"mean {fmt_number(premium.mean(), 0)}."
                    ],
                )
            )
    return Result.of(out, result.source)


@safe_analysis()
def volatility_insights(
    service: DataService, filters: FilterState, variety: str = "Teja", freq: str = "W"
) -> Result[list[Insight]]:
    """Current volatility and outlier behaviour versus history."""
    price_result = service.variety_series(variety)
    if not price_result:
        return Result.unavailable(price_result.reason, price_result.source)
    series = service.series_at(
        service.apply_filters(price_result.unwrap(), filters, kind="price"), freq
    )
    out: list[Insight] = []

    rolling = analytics.rolling_statistics(series, CFG.default_rolling_window, price_result.source)
    if rolling:
        frame = rolling.unwrap()
        column = next(
            (c for c in frame.columns if c.startswith("Rolling volatility")), None
        )
        if column:
            values = frame[column].dropna()
            if len(values) > 10:
                latest = float(values.iloc[-1])
                median = float(values.median())
                ratio = latest / median if median else float("nan")
                out.append(
                    Insight(
                        category="Volatility",
                        headline=(
                            f"{variety} volatility is running at "
                            f"{latest * 100:.0f}% annualised, "
                            + (
                                f"{ratio:.1f}x its historical median."
                                if np.isfinite(ratio) and ratio >= 1.15
                                else f"{1 / ratio:.1f}x below its historical median."
                                if np.isfinite(ratio) and ratio <= 0.87
                                else "close to its historical median."
                            )
                        ),
                        detail=(
                            f"Median annualised volatility over the sample is "
                            f"{median * 100:.0f}%. Elevated volatility widens "
                            "every forecast interval and raises the value of "
                            "optionality in a purchase programme."
                        ),
                        source=rolling.source,
                        strength="moderate",
                        evidence=list(rolling.notes),
                    )
                )

    outliers = analytics.zscore_and_outliers(series, CFG.default_rolling_window, price_result.source)
    if outliers:
        frame = outliers.unwrap()
        recent = frame.tail(min(26, len(frame)))
        flagged = int(recent["Z outlier"].sum() + recent["IQR outlier"].sum())
        if flagged:
            out.append(
                Insight(
                    category="Volatility",
                    headline=f"{flagged} outlier observation(s) in the most recent window.",
                    detail=(
                        "Outliers are flagged for review and are never removed "
                        "from any calculation in this application. Check them "
                        "against the source sheet before treating them as real "
                        "price action."
                    ),
                    source=outliers.source,
                    strength="informational",
                    evidence=list(outliers.notes),
                )
            )
    return Result.of(out, price_result.source)


@safe_analysis()
def data_quality_insights(service: DataService) -> Result[list[Insight]]:
    """Surface coverage holes and thin sheets as first-class findings."""
    notes = service.data_quality_notes()
    out = [
        Insight(
            category="Data quality",
            headline=note.split(":")[0].strip() + " has a coverage limitation."
            if ":" in note
            else "Coverage limitation",
            detail=note,
            source=service.data.path.name,
            strength="data gap",
        )
        for note in notes
    ]
    if service.data.warnings:
        out.append(
            Insight(
                category="Data quality",
                headline=f"{len(service.data.warnings)} parse warning(s) on load.",
                detail=" | ".join(service.data.warnings[:6]),
                source=service.data.path.name,
                strength="data gap",
            )
        )
    return Result.of(out, service.data.path.name)


# ==========================================================================
# Aggregation
# ==========================================================================

#: The generators run for the Automated Insights page, in narrative order.
GENERATORS: tuple[tuple[str, Any], ...] = (
    ("Price leadership", price_leadership_insights),
    ("Market integration", market_integration_insights),
    ("Arrivals impact", arrival_impact_insights),
    ("Seasonality", seasonality_insights),
    ("Exports", export_insights),
    ("Currency", currency_insights),
    ("Balance sheet", balance_sheet_insights),
    ("Cold storage", cold_storage_insights),
    ("Volatility", volatility_insights),
    ("Data quality", data_quality_insights),
)


def generate_all(
    service: DataService,
    filters: FilterState,
    variety: str = "Teja",
    progress: Any = None,
) -> tuple[list[Insight], list[str]]:
    """Run every generator, returning the insights and any failures.

    Generators are independent: one that fails or finds nothing does not stop
    the others, and its reason is collected for display.
    """
    insights: list[Insight] = []
    failures: list[str] = []
    total = len(GENERATORS)

    for index, (name, generator) in enumerate(GENERATORS):
        if progress:
            try:
                progress(f"Analysing {name.lower()}…", int(index / total * 100))
            except Exception:  # noqa: BLE001
                pass
        try:
            if generator in (balance_sheet_insights, cold_storage_insights, data_quality_insights):
                result = generator(service)
            elif generator in (price_leadership_insights, market_integration_insights):
                result = generator(service, filters)
            else:
                result = generator(service, filters, variety)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Insight generator %s failed", name)
            failures.append(f"{name}: {exc}")
            continue

        if result and result.value:
            insights.extend(result.unwrap())
        elif not result:
            failures.append(f"{name}: {result.reason}")

    insights.sort(key=lambda i: i.sort_key)
    if progress:
        try:
            progress("Insights complete", 100)
        except Exception:  # noqa: BLE001
            pass
    return insights, failures


def executive_highlights(insights: Sequence[Insight], limit: int = 6) -> list[Insight]:
    """The strongest findings, for the Executive Summary page."""
    ranked = [i for i in insights if i.strength in ("strong", "moderate")]
    ranked.sort(key=lambda i: i.sort_key)
    return ranked[:limit]
