"""Turn parsed sheets into the canonical series the analytics layer consumes.

:class:`DataService` is the single access point every page uses. It owns a
memo cache so that repeated filtering and resampling never re-derive the same
series, which is what keeps the UI responsive on a twelve-year daily panel.

No series produced here is gap-filled, extended or smoothed unless the caller
explicitly asks for it. Where a market simply has no data, the service returns
an unavailable :class:`~chilli_desktop.utils.Result` carrying the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from . import settings
from .data_loader import Dataset, WorkbookData
from .utils import (
    LOG,
    Result,
    clean_series,
    fmt_date,
    fortnight_end_index,
    normalise_text,
    resample_frame,
    resample_series,
    squash,
)
from .settings import FORTNIGHT_FREQ

Measure = Literal["average", "low", "high", "difference"]


# --------------------------------------------------------------------------
# Filter state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterState:
    """The global filter selection shared by every page.

    ``None`` on any field means "no restriction". Filters never create data;
    they only narrow what is shown, and the active narrowing is echoed in each
    panel's assumptions caption so a reader always knows the sample.
    """

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    varieties: tuple[str, ...] = ()
    market: str = ""
    price_min: float | None = None
    price_max: float | None = None
    arrival_min: float | None = None
    arrival_max: float | None = None
    #: Calendar months (1-12) to keep; empty means all.
    months: tuple[int, ...] = ()
    frequency: str = "W"
    horizon: int = 0

    def describe(self) -> str:
        """One-line, human-readable summary for the assumptions caption."""
        bits: list[str] = []
        if self.start is not None or self.end is not None:
            bits.append(
                f"dates {fmt_date(self.start) if self.start is not None else 'start'}"
                f" to {fmt_date(self.end) if self.end is not None else 'latest'}"
            )
        if self.varieties:
            bits.append("varieties " + ", ".join(self.varieties))
        if self.market:
            bits.append(f"market {self.market}")
        if self.price_min is not None or self.price_max is not None:
            bits.append(
                f"price {self.price_min if self.price_min is not None else '-inf'}"
                f"..{self.price_max if self.price_max is not None else 'inf'}"
            )
        if self.arrival_min is not None or self.arrival_max is not None:
            bits.append(
                f"arrivals {self.arrival_min if self.arrival_min is not None else '-inf'}"
                f"..{self.arrival_max if self.arrival_max is not None else 'inf'}"
            )
        if self.months:
            names = ", ".join(
                settings.MONTH_ABBREVIATIONS[m - 1].title() for m in sorted(self.months)
            )
            bits.append(f"months {names}")
        return "No filters applied (full workbook history)." if not bits else "Filtered on " + "; ".join(bits) + "."

    def key(self) -> tuple:
        """Hashable identity used as part of the memo-cache key."""
        return (
            None if self.start is None else self.start.value,
            None if self.end is None else self.end.value,
            self.varieties,
            self.market,
            self.price_min,
            self.price_max,
            self.arrival_min,
            self.arrival_max,
            self.months,
        )


# --------------------------------------------------------------------------
# Market registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSource:
    """Where a market's price and arrival series come from."""

    label: str
    dataset_key: str
    #: Column holding the average price, or the variety name for Guntur.
    price_field: str
    arrival_dataset_key: str = ""
    arrival_field: str = ""
    #: Set when the price must be pulled from the multi-variety wide frame.
    from_variety_sheet: bool = False
    note: str = ""


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------


class DataService:
    """Canonical, cached access to every series the dashboard needs."""

    def __init__(self, data: WorkbookData) -> None:
        self.data = data
        self._cache: dict[tuple, Any] = {}
        self._markets = self._build_market_registry()

    # -- infrastructure ---------------------------------------------------

    def _memo(self, key: tuple, factory):
        """Return a cached value, computing it on first request."""
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def _memo_frame(self, key: tuple, factory) -> Result[pd.DataFrame]:
        """Memoise a frame-valued Result, handing back a copy each time.

        Same reasoning as :meth:`series_at`: pages must not be able to mutate
        each other's data through the cache.
        """
        cached: Result[pd.DataFrame] = self._memo(key, factory)
        if not cached:
            return cached
        return Result(
            value=cached.value.copy(),
            reason=cached.reason,
            source=cached.source,
            notes=list(cached.notes),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- market registry --------------------------------------------------

    def _build_market_registry(self) -> dict[str, MarketSource]:
        """Discover which of the configured markets the workbook supports.

        Guntur's Teja price lives in the multi-variety sheet; the other
        markets have dedicated price/arrival sheets. Only markets whose
        underlying sheet actually parsed are registered.
        """
        registry: dict[str, MarketSource] = {}

        teja = self.resolve_variety("Teja")
        if self.data.get("guntur_variety_prices") and teja:
            registry["Guntur"] = MarketSource(
                label="Guntur",
                dataset_key="guntur_variety_prices",
                price_field=teja,
                arrival_dataset_key="guntur_daily_arrivals",
                arrival_field="Arrivals",
                from_variety_sheet=True,
                note=(
                    f"Price = '{teja}' average from the Guntur variety sheet, "
                    "so that all three markets are compared on the same variety."
                ),
            )

        for label, key in (
            ("Warangal", "warangal"),
            ("Khammam (cold storage)", "khammam_cold"),
            ("Khammam (non cold storage)", "khammam_non_cold"),
        ):
            ds = self.data.get(key)
            if ds is None:
                continue
            price_field = "average" if "average" in ds.frame.columns else ""
            if not price_field:
                continue
            registry[label] = MarketSource(
                label=label,
                dataset_key=key,
                price_field=price_field,
                arrival_dataset_key=key,
                arrival_field="arrivals" if "arrivals" in ds.frame.columns else "",
            )
        return registry

    def markets(self) -> list[str]:
        return list(self._markets)

    def market_note(self, market: str) -> str:
        source = self._markets.get(market)
        return source.note if source else ""

    # -- varieties --------------------------------------------------------

    def varieties(self) -> list[str]:
        ds = self.data.get("guntur_variety_prices")
        if ds is None:
            return []
        return list(ds.meta.get("varieties", []))

    def resolve_variety(self, wanted: str) -> str:
        """Match a requested variety to the workbook's own spelling.

        Returns the workbook label (``'LCA 334'``) or ``''`` when the variety
        is not present. Matching is by squashed substring so that the brief's
        "Guntur LCA 334" finds the sheet's "LCA 334".
        """
        available = self.varieties()
        if not available:
            return ""
        if wanted in available:
            return wanted

        keywords = settings.FOCUS_VARIETY_KEYWORDS.get(wanted)
        squashed_target = squash(wanted)
        for name in available:
            s = squash(name)
            if s == squashed_target or squashed_target in s or s in squashed_target:
                return name
        if keywords:
            for name in available:
                s = squash(name)
                if all(squash(k) in s for k in keywords):
                    return name
        return ""

    def focus_varieties(self) -> dict[str, str]:
        """Map the brief's two headline varieties to workbook labels."""
        out: dict[str, str] = {}
        for wanted in settings.FOCUS_VARIETY_KEYWORDS:
            resolved = self.resolve_variety(wanted)
            if resolved:
                out[wanted] = resolved
        return out

    def variety_matrix(self, measure: Measure = "average") -> Result[pd.DataFrame]:
        """Wide frame of daily prices: one column per variety."""

        def build() -> Result[pd.DataFrame]:
            ds = self.data.get("guntur_variety_prices")
            if ds is None:
                return Result.unavailable(
                    "The Guntur variety-wise daily price sheet is not present "
                    "in the workbook.",
                )
            frame = ds.meta.get(f"wide_{'avg' if measure == 'average' else measure}")
            if frame is None or frame.empty:
                return Result.unavailable(
                    f"The variety sheet contains no '{measure}' price columns.",
                    ds.sheet_name,
                )
            return Result.of(frame, ds.sheet_name)

        return self._memo_frame(("variety_matrix", measure), build)

    def variety_series(self, variety: str, measure: Measure = "average") -> Result[pd.Series]:
        """Daily price series for one variety."""

        def build() -> Result[pd.Series]:
            matrix = self.variety_matrix(measure)
            if not matrix:
                return Result.unavailable(matrix.reason, matrix.source)
            resolved = self.resolve_variety(variety)
            frame = matrix.unwrap()
            if not resolved or resolved not in frame.columns:
                return Result.unavailable(
                    f"'{variety}' is not one of the varieties quoted in the "
                    f"workbook. Available: {', '.join(frame.columns)}.",
                    matrix.source,
                )
            series = clean_series(frame[resolved], resolved)
            if series.empty:
                return Result.unavailable(
                    f"'{resolved}' has no usable {measure} prices.", matrix.source
                )
            return Result.of(series, matrix.source)

        return self._memo(("variety_series", variety, measure), build)

    # -- market series ----------------------------------------------------

    def market_price(self, market: str) -> Result[pd.Series]:
        """Daily average price for a market (Teja basis throughout)."""

        def build() -> Result[pd.Series]:
            source = self._markets.get(market)
            if source is None:
                return Result.unavailable(
                    f"'{market}' has no price sheet in the workbook. "
                    f"Available markets: {', '.join(self.markets()) or 'none'}."
                )
            if source.from_variety_sheet:
                return self.variety_series(source.price_field, "average")
            ds = self.data.get(source.dataset_key)
            if ds is None:
                return Result.unavailable(
                    f"The sheet backing '{market}' is not present.",
                )
            series = clean_series(ds.frame[source.price_field], market)
            if series.empty:
                return Result.unavailable(
                    f"'{market}' has no usable average prices.", ds.sheet_name
                )
            return Result.of(series, ds.sheet_name)

        return self._memo(("market_price", market), build)

    def market_arrivals(self, market: str) -> Result[pd.Series]:
        """Daily arrivals for a market, in the workbook's own unit (bags)."""

        def build() -> Result[pd.Series]:
            source = self._markets.get(market)
            if source is None or not source.arrival_dataset_key or not source.arrival_field:
                return Result.unavailable(
                    f"No arrivals series is recorded for '{market}'."
                )
            ds = self.data.get(source.arrival_dataset_key)
            if ds is None or source.arrival_field not in ds.frame.columns:
                return Result.unavailable(
                    f"No arrivals column was found for '{market}'."
                )
            series = clean_series(ds.frame[source.arrival_field], f"{market} arrivals")
            if series.empty:
                return Result.unavailable(
                    f"'{market}' has no usable arrivals observations.", ds.sheet_name
                )
            return Result.of(series, ds.sheet_name)

        return self._memo(("market_arrivals", market), build)

    def market_bag_weight(self, market: str) -> float | None:
        """Kilograms per bag for a market, as stated in its sheet header."""
        source = self._markets.get(market)
        if source is None or not source.arrival_dataset_key:
            return None
        ds = self.data.get(source.arrival_dataset_key)
        if ds is None:
            return None
        weights = ds.meta.get("bag_weights_kg", {})
        return weights.get(source.arrival_field)

    def market_arrivals_tonnes(self, market: str) -> Result[pd.Series]:
        """Arrivals converted to tonnes using the sheet's stated bag weight.

        Returns unavailable (rather than guessing) when the sheet does not
        state a conversion.
        """
        bags = self.market_arrivals(market)
        if not bags:
            return bags
        weight = self.market_bag_weight(market)
        if not weight:
            return Result.unavailable(
                f"The '{market}' sheet does not state a kilograms-per-bag "
                "conversion, so arrivals cannot be expressed in tonnes.",
                bags.source,
            )
        series = bags.unwrap() * weight / 1000.0
        series.name = f"{market} arrivals (t)"
        return Result.of(
            series,
            bags.source,
            [f"Converted at {weight:g} kg/bag as stated in the sheet header."],
        )

    # -- Guntur specifics -------------------------------------------------

    def guntur_arrivals(self) -> Result[pd.Series]:
        return self.market_arrivals("Guntur")

    def guntur_offtake(self) -> Result[pd.Series]:
        def build() -> Result[pd.Series]:
            ds = self.data.get("guntur_daily_arrivals")
            if ds is None or "Offtake" not in ds.frame.columns:
                return Result.unavailable(
                    "The Guntur daily arrivals sheet has no offtake column."
                )
            series = clean_series(ds.frame["Offtake"], "Guntur offtake")
            if series.empty:
                return Result.unavailable("No usable offtake observations.", ds.sheet_name)
            return Result.of(series, ds.sheet_name)

        return self._memo(("guntur_offtake",), build)

    def guntur_monthly_arrivals(self) -> Result[pd.Series]:
        """Month-start series from the dedicated monthly arrivals sheet."""

        def build() -> Result[pd.Series]:
            ds = self.data.get("guntur_monthly_arrivals")
            if ds is None:
                return Result.unavailable(
                    "The Guntur monthly arrivals sheet is not present."
                )
            series = ds.meta.get("monthly_series")
            if series is None or series.empty:
                return Result.unavailable("No monthly arrivals were parsed.", ds.sheet_name)
            out = series.copy()
            out.name = "Guntur monthly arrivals"
            return Result.of(out, ds.sheet_name, [ds.meta.get("primary_unit", "")])

        return self._memo(("guntur_monthly_arrivals",), build)

    # -- macro / trade ----------------------------------------------------

    def usd_inr(self) -> Result[pd.Series]:
        def build() -> Result[pd.Series]:
            ds = self.data.get("usd_inr")
            if ds is None or ds.frame.empty:
                return Result.unavailable("The USD/INR sheet is not present.")
            column = ds.frame.columns[0]
            series = clean_series(ds.frame[column], "USD/INR")
            if series.empty:
                return Result.unavailable("No usable exchange rates.", ds.sheet_name)
            return Result.of(series, ds.sheet_name)

        return self._memo(("usd_inr",), build)

    def exports_monthly(self) -> Result[pd.Series]:
        """Monthly exports, indexed on month *end*.

        The parser builds a month-start index; it is shifted to month end here,
        once, so that it aligns with every other monthly series in the
        application (all of which are period-end labelled). Callers must not
        re-shift it.
        """

        def build() -> Result[pd.Series]:
            ds = self.data.get("exports")
            if ds is None:
                return Result.unavailable("The exports sheet is not present.")
            series = ds.meta.get("monthly_series")
            if series is None or series.empty:
                return Result.unavailable("No monthly export values were parsed.", ds.sheet_name)
            out = series.copy()
            out.index = pd.DatetimeIndex(out.index) + pd.offsets.MonthEnd(0)
            out.name = "Exports"
            return Result.of(
                out,
                ds.sheet_name,
                [
                    ds.meta.get("unit_note", ""),
                    "Indexed on month end to align with the other monthly series.",
                ],
            )

        return self._memo(("exports_monthly",), build)

    def exports_matrix(self) -> Result[pd.DataFrame]:
        ds = self.data.get("exports")
        if ds is None:
            return Result.unavailable("The exports sheet is not present.")
        return Result.of(ds.frame, ds.sheet_name)

    def balance_sheet(self) -> Result[pd.DataFrame]:
        ds = self.data.get("balance_sheet")
        if ds is None:
            return Result.unavailable("The balance sheet is not present.")
        return Result.of(
            ds.frame,
            ds.sheet_name,
            [
                ds.meta.get("unit_note", ""),
                "Years marked (exp) in the workbook are the workbook's own "
                "expectations, not this application's forecasts.",
            ],
        )

    def balance_sheet_row(self, keyword: str) -> Result[pd.Series]:
        """Pull one balance-sheet line item by fuzzy label match."""
        table = self.balance_sheet()
        if not table:
            return Result.unavailable(table.reason, table.source)
        frame = table.unwrap()
        target = squash(keyword)
        for label in frame.index:
            if target in squash(label):
                series = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
                series.name = str(label).strip()
                return Result.of(series, table.source, table.notes)
        return Result.unavailable(
            f"No balance-sheet line matching '{keyword}'. Available: "
            + ", ".join(str(i).strip() for i in frame.index),
            table.source,
        )

    def apy(self) -> Result[pd.DataFrame]:
        ds = self.data.get("apy")
        if ds is None:
            return Result.unavailable("The APY sheet is not present.")
        return Result.of(ds.frame, ds.sheet_name)

    def cold_storage(self) -> Result[pd.DataFrame]:
        ds = self.data.get("cold_storage_stock")
        if ds is None:
            return Result.unavailable("The cold storage sheet is not present.")
        return Result.of(ds.frame, ds.sheet_name, ds.warnings)

    def workbook_seasonality(self) -> Result[Dataset]:
        ds = self.data.get("seasonality_teja")
        if ds is None:
            return Result.unavailable("The seasonality sheet is not present.")
        return Result.of(ds, ds.sheet_name)

    def workbook_variety_correlation(self) -> Result[pd.DataFrame]:
        ds = self.data.get("variety_correlation_workbook")
        if ds is None:
            return Result.unavailable("The variety correlation sheet is not present.")
        return Result.of(ds.frame, ds.sheet_name, [ds.meta.get("note", "")])

    # -- filtering and resampling ----------------------------------------

    @staticmethod
    def apply_filters(
        series: pd.Series,
        filters: FilterState,
        *,
        kind: Literal["price", "arrivals", "other"] = "other",
    ) -> pd.Series:
        """Narrow a series by the global filter selection.

        ``kind`` decides whether the price-range or arrival-range bounds
        apply, so that a price filter never silently truncates an arrivals
        chart.
        """
        if series is None or series.empty:
            return series
        out = series
        if filters.start is not None:
            out = out[out.index >= filters.start]
        if filters.end is not None:
            out = out[out.index <= filters.end]
        if filters.months:
            out = out[out.index.month.isin(filters.months)]
        if kind == "price":
            if filters.price_min is not None:
                out = out[out >= filters.price_min]
            if filters.price_max is not None:
                out = out[out <= filters.price_max]
        elif kind == "arrivals":
            if filters.arrival_min is not None:
                out = out[out >= filters.arrival_min]
            if filters.arrival_max is not None:
                out = out[out <= filters.arrival_max]
        return out

    @staticmethod
    def apply_frame_filters(frame: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
        """Date/month filtering for wide frames (value bounds do not apply)."""
        if frame is None or frame.empty:
            return frame
        out = frame
        if filters.start is not None:
            out = out[out.index >= filters.start]
        if filters.end is not None:
            out = out[out.index <= filters.end]
        if filters.months:
            out = out[out.index.month.isin(filters.months)]
        return out

    def series_at(
        self,
        series: pd.Series,
        freq: str,
        how: str = "mean",
    ) -> pd.Series:
        """Resample with caching keyed on the series identity and frequency.

        A *copy* is handed back on every call. The cache would otherwise share
        one object between every page, and a caller that reindexes or renames
        it in place would silently corrupt the value every other page sees.
        Copying a few thousand floats is far cheaper than the resample itself.
        """
        if series is None or series.empty:
            return series
        key = (
            "resample",
            series.name,
            freq,
            how,
            len(series),
            series.index[0].value,
            series.index[-1].value,
            float(series.iloc[-1]),
        )
        return self._memo(key, lambda: resample_series(series, freq, how)).copy()

    # -- exogenous variables ---------------------------------------------

    def exogenous_candidates(self, freq: str) -> dict[str, Result[pd.Series]]:
        """Every exogenous driver the workbook can supply at ``freq``.

        Used by SARIMAX and by the driver-attribution panel. Each entry is a
        Result so that unusable drivers explain themselves rather than being
        silently omitted.
        """
        out: dict[str, Result[pd.Series]] = {}

        arrivals = self.guntur_arrivals()
        if arrivals:
            out["Guntur arrivals"] = Result.of(
                self.series_at(arrivals.unwrap(), freq, "mean"), arrivals.source
            )
        else:
            out["Guntur arrivals"] = arrivals

        offtake = self.guntur_offtake()
        if offtake:
            out["Guntur offtake"] = Result.of(
                self.series_at(offtake.unwrap(), freq, "mean"), offtake.source
            )
        else:
            out["Guntur offtake"] = offtake

        fx = self.usd_inr()
        if fx:
            out["USD/INR"] = Result.of(self.series_at(fx.unwrap(), freq, "mean"), fx.source)
        else:
            out["USD/INR"] = fx

        exports = self.exports_monthly()
        if exports:
            # Monthly exports cannot be disaggregated to a weekly series
            # without inventing observations, so they are only offered at
            # monthly frequency.
            if freq == "ME":
                # Already month-end indexed by exports_monthly().
                out["Exports"] = Result.of(exports.unwrap(), exports.source)
            else:
                out["Exports"] = Result.unavailable(
                    "Exports are recorded monthly. Using them at "
                    f"{settings.FORECAST.frequency_labels.get(freq, freq).lower()} "
                    "frequency would require inventing intra-month values, so "
                    "they are offered only on the monthly view.",
                    exports.source,
                )
        else:
            out["Exports"] = exports

        return out

    def exogenous_matrix(self, freq: str, names: Iterable[str] | None = None) -> Result[pd.DataFrame]:
        """Assemble the usable exogenous drivers into one aligned frame."""
        candidates = self.exogenous_candidates(freq)
        wanted = list(names) if names is not None else list(candidates)
        usable: dict[str, pd.Series] = {}
        sources: list[str] = []
        for name in wanted:
            res = candidates.get(name)
            if res is None or not res:
                continue
            series = res.unwrap()
            if series is None or series.empty:
                continue
            usable[name] = series
            if res.source and res.source not in sources:
                sources.append(res.source)
        if not usable:
            return Result.unavailable(
                "None of the workbook's exogenous drivers are usable at this "
                "frequency."
            )
        frame = pd.concat(usable, axis=1).dropna(how="all")
        return Result.of(frame, "; ".join(sources))

    # -- derived views ----------------------------------------------------

    def price_panel(self, freq: str, filters: FilterState) -> Result[pd.DataFrame]:
        """Aligned Teja price across every available market."""

        def build() -> Result[pd.DataFrame]:
            series: dict[str, pd.Series] = {}
            sources: list[str] = []
            for market in self.markets():
                res = self.market_price(market)
                if not res:
                    continue
                filtered = self.apply_filters(res.unwrap(), filters, kind="price")
                sampled = self.series_at(filtered, freq, "mean")
                if sampled is None or sampled.empty:
                    continue
                series[market] = sampled
                if res.source not in sources:
                    sources.append(res.source)
            if not series:
                return Result.unavailable(
                    "No market price series survived the current filters."
                )
            frame = pd.concat(series, axis=1)
            return Result.of(frame, "; ".join(sources))

        return self._memo_frame(("price_panel", freq, filters.key()), build)

    def arrivals_panel(self, freq: str, filters: FilterState) -> Result[pd.DataFrame]:
        """Aligned arrivals across every market that reports them."""

        def build() -> Result[pd.DataFrame]:
            series: dict[str, pd.Series] = {}
            sources: list[str] = []
            for market in self.markets():
                res = self.market_arrivals(market)
                if not res:
                    continue
                filtered = self.apply_filters(res.unwrap(), filters, kind="arrivals")
                sampled = self.series_at(filtered, freq, "sum" if freq != "D" else "mean")
                if sampled is None or sampled.empty:
                    continue
                series[market] = sampled
                if res.source not in sources:
                    sources.append(res.source)
            if not series:
                return Result.unavailable(
                    "No market arrivals series survived the current filters."
                )
            return Result.of(pd.concat(series, axis=1), "; ".join(sources))

        return self._memo_frame(("arrivals_panel", freq, filters.key()), build)

    def variety_panel(self, freq: str, filters: FilterState) -> Result[pd.DataFrame]:
        """Aligned prices for the selected varieties (all when unspecified)."""

        def build() -> Result[pd.DataFrame]:
            matrix = self.variety_matrix("average")
            if not matrix:
                return Result.unavailable(matrix.reason, matrix.source)
            frame = self.apply_frame_filters(matrix.unwrap(), filters)
            if filters.varieties:
                keep = [
                    self.resolve_variety(v)
                    for v in filters.varieties
                    if self.resolve_variety(v)
                ]
                keep = [c for c in keep if c in frame.columns]
                if keep:
                    frame = frame[keep]
            if frame.empty:
                return Result.unavailable(
                    "No variety prices survived the current filters.", matrix.source
                )
            sampled = resample_frame(frame, freq, "mean") if freq != "D" else frame
            if sampled is None or sampled.empty:
                return Result.unavailable(
                    "Resampling left no observations.", matrix.source
                )
            return Result.of(sampled, matrix.source)

        return self._memo_frame(("variety_panel", freq, filters.key()), build)

    # -- data-driven season definition ------------------------------------

    def season_profile(self) -> Result[pd.DataFrame]:
        """Classify calendar months into harvest / lean from arrivals data.

        The seasons are *derived*, not asserted: months are ranked by their
        mean Guntur arrivals across all available years, and the top and
        bottom terciles are labelled peak-arrival (harvest) and lean.
        """

        def build() -> Result[pd.DataFrame]:
            monthly = self.guntur_monthly_arrivals()
            source = monthly.source
            if monthly:
                series = monthly.unwrap()
            else:
                daily = self.guntur_arrivals()
                if not daily:
                    return Result.unavailable(
                        "Neither monthly nor daily Guntur arrivals are "
                        "available, so seasons cannot be derived.",
                        daily.source,
                    )
                series = resample_series(daily.unwrap(), "MS", "sum")
                source = daily.source

            by_month = series.groupby(series.index.month)
            table = pd.DataFrame(
                {
                    "mean_arrivals": by_month.mean(),
                    "median_arrivals": by_month.median(),
                    "years_observed": by_month.count(),
                }
            )
            table.index.name = "month"
            overall = table["mean_arrivals"].mean()
            table["index_vs_mean"] = table["mean_arrivals"] / overall
            ranked = table["mean_arrivals"].rank(ascending=False)
            n = len(table)
            table["season"] = np.where(
                ranked <= n / 3,
                "Peak arrivals",
                np.where(ranked > 2 * n / 3, "Lean arrivals", "Shoulder"),
            )
            table["month_name"] = [
                settings.MONTH_ABBREVIATIONS[m - 1].title() for m in table.index
            ]
            return Result.of(
                table,
                source,
                [
                    "Seasons are derived by ranking calendar months on mean "
                    "arrivals across all years in the workbook; the top third "
                    "is labelled peak, the bottom third lean.",
                ],
            )

        return self._memo(("season_profile",), build)

    # -- coverage summary -------------------------------------------------

    def coverage_table(self) -> pd.DataFrame:
        """Per-dataset coverage, used by the status bar and data dictionary."""
        rows: list[dict[str, Any]] = []
        for spec in settings.SHEET_SPECS:
            ds = self.data.datasets.get(spec.key)
            if ds is None or ds.empty:
                rows.append(
                    {
                        "Dataset": spec.key,
                        "Sheet": ds.sheet_name if ds else "(not found)",
                        "Rows": 0,
                        "From": "—",
                        "To": "—",
                        "Status": settings.DATA_UNAVAILABLE_MESSAGE,
                    }
                )
                continue
            lo, hi = ds.span()
            rows.append(
                {
                    "Dataset": spec.key,
                    "Sheet": ds.sheet_name,
                    "Rows": ds.n_rows,
                    "From": fmt_date(lo) if lo is not None else "—",
                    "To": fmt_date(hi) if hi is not None else "—",
                    "Status": "; ".join(ds.warnings) if ds.warnings else "OK",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def partial_last_period(raw: pd.Series, freq: str) -> str:
        """Warn when the newest resampled bucket covers only part of a period.

        The workbook stops mid-month, so a monthly average for the final month
        is computed from a handful of days and is not comparable with the
        complete months before it. Callers surface this string so nobody reads
        a part-month average as a full one.
        """
        if raw is None or raw.empty or freq in ("D", ""):
            return ""
        last_obs = pd.Timestamp(raw.index.max())
        try:
            if freq == FORTNIGHT_FREQ:
                period_end = pd.Timestamp(
                    fortnight_end_index(pd.DatetimeIndex([last_obs]))[0]
                )
            else:
                # A real tail of the series: a single point does not give
                # pandas enough range to place the bin boundaries correctly.
                period_end = pd.Timestamp(raw.tail(120).resample(freq).last().index[-1])
        except (ValueError, TypeError, IndexError):
            return ""
        if period_end <= last_obs:
            return ""
        try:
            period_start = (
                pd.date_range(end=period_end, periods=2, freq=freq)[0]
                + pd.Timedelta(days=1)
            )
        except (ValueError, TypeError):
            return ""
        covered = int((last_obs - period_start).days) + 1
        total = int((pd.Timestamp(period_end) - period_start).days) + 1
        if total <= 1 or covered >= total:
            return ""
        label = settings.FORECAST.frequency_labels.get(freq, freq).lower()
        return (
            f"The final {label} period ending {pd.Timestamp(period_end):%d %b %Y} "
            f"is incomplete: the workbook's last observation is "
            f"{last_obs:%d %b %Y}, so that bucket averages only {covered} of "
            f"{total} days and is not directly comparable with the complete "
            "periods before it."
        )

    @staticmethod
    def coverage_gaps(series: pd.Series, min_days: int = 45) -> pd.DataFrame:
        """Find breaks longer than ``min_days`` in a date-indexed series.

        Long holes matter: a regression or rolling correlation run straight
        across a multi-year gap silently joins two disconnected regimes. Every
        panel that consumes a gappy series prints the gaps beside it.
        """
        empty = pd.DataFrame(columns=["from", "to", "days"])
        if series is None or series.empty or len(series) < 2:
            return empty
        idx = pd.DatetimeIndex(series.index).sort_values()
        deltas = idx.to_series().diff().dt.days
        flagged = deltas[deltas > min_days]
        if flagged.empty:
            return empty
        rows = [
            {
                "from": (stamp - pd.Timedelta(days=int(days))).date(),
                "to": stamp.date(),
                "days": int(days),
            }
            for stamp, days in flagged.items()
        ]
        return pd.DataFrame(rows).sort_values("days", ascending=False).reset_index(drop=True)

    def data_quality_notes(self) -> list[str]:
        """Plain-language warnings about coverage holes and thin datasets.

        Surfaced in the status bar, the Data Dictionary page and anywhere the
        affected series is charted.
        """
        notes: list[str] = []
        checks: list[tuple[str, Result[pd.Series]]] = [
            ("USD/INR", self.usd_inr()),
            ("Guntur arrivals", self.guntur_arrivals()),
            ("Guntur offtake", self.guntur_offtake()),
        ]
        for market in self.markets():
            checks.append((f"{market} price", self.market_price(market)))

        for label, res in checks:
            if not res:
                continue
            gaps = self.coverage_gaps(res.unwrap())
            if gaps.empty:
                continue
            worst = gaps.iloc[0]
            missing_years = sorted(
                {
                    y
                    for y in range(
                        pd.Timestamp(worst["from"]).year, pd.Timestamp(worst["to"]).year + 1
                    )
                }
            )
            notes.append(
                f"{label}: {len(gaps)} coverage gap(s) over 45 days; the "
                f"largest spans {int(worst['days'])} days "
                f"({worst['from']} to {worst['to']}, affecting "
                f"{', '.join(str(y) for y in missing_years)}). Statistics "
                "spanning this break join two disconnected periods."
            )

        for ds in self.data.datasets.values():
            for warning in ds.warnings:
                notes.append(f"{ds.sheet_name}: {warning}")
        return notes

    def latest_observation_date(self) -> pd.Timestamp | None:
        """The most recent date anywhere in the workbook's daily sheets."""
        stamps: list[pd.Timestamp] = []
        for key in (
            "guntur_variety_prices",
            "guntur_daily_arrivals",
            "warangal",
            "khammam_cold",
            "khammam_non_cold",
            "usd_inr",
        ):
            ds = self.data.get(key)
            if ds is None:
                continue
            _, hi = ds.span()
            if hi is not None:
                stamps.append(hi)
        return max(stamps) if stamps else None

    def full_date_span(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Earliest and latest dates across all date-indexed datasets."""
        lows: list[pd.Timestamp] = []
        highs: list[pd.Timestamp] = []
        for ds in self.data.datasets.values():
            lo, hi = ds.span()
            if lo is not None:
                lows.append(lo)
            if hi is not None:
                highs.append(hi)
        return (min(lows) if lows else None, max(highs) if highs else None)


def default_filters(service: DataService) -> FilterState:
    """A sensible opening filter state: the workbook's full history."""
    start, end = service.full_date_span()
    return FilterState(start=start, end=end, frequency="W")

