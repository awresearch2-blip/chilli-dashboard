"""Workbook ingestion: read the master Excel file once and parse every sheet.

Design notes
------------
*   The workbook is read a single time into raw, header-less frames and then
    parsed by layout-specific routines. The resulting :class:`WorkbookData`
    object is cached process-wide, satisfying the performance requirement to
    "load the workbook only once".
*   Nothing is imputed. Blank cells, ``"Closed"`` markers and short series stay
    exactly as short and as gappy as the workbook makes them.
*   Unit conversions (bags to kilograms, the second Guntur arrivals block) are
    *read or measured from the workbook*, never assumed.
*   Every parsed dataset records the sheet it came from so the UI can print
    provenance on each chart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import settings
from .utils import (
    LOG,
    WorkbookError,
    cell_text,
    classify_column,
    is_blank,
    is_projection_label,
    normalise_text,
    parse_bag_weight,
    parse_month,
    parse_year,
    squash,
    strip_unit_clause,
    to_datetime,
    to_number,
)


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------


@dataclass
class ColumnDoc:
    """One row of the auto-generated data dictionary."""

    name: str
    role: str
    dtype: str
    non_null: int
    total: int
    minimum: str
    maximum: str
    unit: str = ""
    note: str = ""

    @property
    def coverage(self) -> float:
        return self.non_null / self.total if self.total else 0.0


@dataclass
class Dataset:
    """A parsed sheet: tidy data plus everything needed to document it."""

    key: str
    sheet_name: str
    layout: str
    description: str
    frame: pd.DataFrame
    #: Layout-specific extras (units, variety lists, workbook-supplied
    #: aggregate rows, observed conversion factors, ...).
    meta: dict[str, Any] = field(default_factory=dict)
    columns: list[ColumnDoc] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.frame is None or self.frame.empty

    @property
    def n_rows(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))

    def span(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Date span of the dataset when it is date-indexed."""
        if self.frame is None or self.frame.empty:
            return None, None
        if isinstance(self.frame.index, pd.DatetimeIndex):
            return self.frame.index.min(), self.frame.index.max()
        for col in ("date", "Date"):
            if col in self.frame.columns:
                series = pd.to_datetime(self.frame[col], errors="coerce").dropna()
                if not series.empty:
                    return series.min(), series.max()
        return None, None


@dataclass
class WorkbookData:
    """Everything parsed out of the master workbook."""

    path: Path
    loaded_at: pd.Timestamp
    load_seconds: float
    datasets: dict[str, Dataset] = field(default_factory=dict)
    #: Sheets present in the file but not matched by any spec.
    unmapped_sheets: list[str] = field(default_factory=list)
    #: Spec keys expected but not found.
    missing_sheets: list[str] = field(default_factory=list)
    #: Non-fatal problems worth surfacing in the status bar.
    warnings: list[str] = field(default_factory=list)
    #: Raw header-less frames, retained for the data dictionary's raw preview.
    raw_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    def get(self, key: str) -> Dataset | None:
        ds = self.datasets.get(key)
        return ds if ds is not None and not ds.empty else None

    def sheet_name_for(self, key: str) -> str:
        ds = self.datasets.get(key)
        return ds.sheet_name if ds else "(sheet not found)"

    def has(self, *keys: str) -> bool:
        return all(self.get(k) is not None for k in keys)


# --------------------------------------------------------------------------
# Raw reading helpers
# --------------------------------------------------------------------------


def resolve_workbook_path(explicit: str | Path | None = None) -> Path:
    """Locate the workbook, raising :class:`WorkbookError` if nothing is found."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise WorkbookError(f"Workbook not found at the supplied path: {candidate}")

    for candidate in settings.workbook_search_paths():
        if candidate.is_file():
            return candidate

    tried = "\n  ".join(str(p) for p in settings.workbook_search_paths())
    raise WorkbookError(
        "Could not locate '"
        + settings.WORKBOOK_FILENAME
        + "'. Paths tried:\n  "
        + tried
        + f"\n\nSet the {settings.WORKBOOK_ENV_VAR} environment variable to the "
        "workbook's full path, or pass --workbook on the command line."
    )


def _trim(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty trailing rows and columns from a raw sheet."""
    if raw.empty:
        return raw
    mask_rows = raw.notna().any(axis=1)
    mask_cols = raw.notna().any(axis=0)
    if not mask_rows.any() or not mask_cols.any():
        return raw.iloc[:0, :0]
    last_row = int(np.flatnonzero(mask_rows.to_numpy())[-1])
    last_col = int(np.flatnonzero(mask_cols.to_numpy())[-1])
    return raw.iloc[: last_row + 1, : last_col + 1]


def _match_sheet(spec: settings.SheetSpec, available: Iterable[str]) -> str | None:
    """Find the sheet whose name contains every keyword of ``spec``.

    Matching is done on the squashed name (letters and digits only) so that
    ``'KhammamTejacoldstorage'`` and ``'Khammam Teja cold storage'`` both
    match. When several sheets qualify, the most specific (longest keyword
    coverage relative to name length) wins -- this is what keeps the
    ``khammam_cold`` spec from stealing the ``khammam_non_cold`` sheet.
    """
    hits: list[tuple[float, str]] = []
    for name in available:
        squashed = squash(name)
        if all(squash(k) in squashed for k in spec.keywords):
            covered = sum(len(squash(k)) for k in spec.keywords)
            hits.append((covered / max(len(squashed), 1), name))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0], reverse=True)
    return hits[0][1]


def _assign_sheets(available: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
    """Map spec keys to sheet names, ensuring no sheet is claimed twice.

    Specs are resolved most-specific-first (by keyword count) so that
    ``khammam_non_cold`` ("khammam", "non", "cold") consumes the fresh-lot
    sheet before ``khammam_cold`` ("khammam", "cold") looks for one.
    """
    assignment: dict[str, str] = {}
    taken: set[str] = set()
    missing: list[str] = []

    ordered = sorted(settings.SHEET_SPECS, key=lambda s: len(s.keywords), reverse=True)
    for spec in ordered:
        pool = [n for n in available if n not in taken]
        name = _match_sheet(spec, pool)
        if name is None:
            missing.append(spec.key)
            continue
        assignment[spec.key] = name
        taken.add(name)

    unmapped = [n for n in available if n not in taken]
    return assignment, unmapped, missing


# --------------------------------------------------------------------------
# Column documentation
# --------------------------------------------------------------------------


def _document_columns(frame: pd.DataFrame, units: dict[str, str] | None = None) -> list[ColumnDoc]:
    """Build data-dictionary rows for a tidy frame."""
    docs: list[ColumnDoc] = []
    units = units or {}
    total = len(frame)

    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.name:
        idx = frame.index
        docs.append(
            ColumnDoc(
                name=str(frame.index.name),
                role="date (index)",
                dtype="datetime64",
                non_null=int(idx.notna().sum()),
                total=total,
                minimum=str(idx.min().date()) if len(idx) else "—",
                maximum=str(idx.max().date()) if len(idx) else "—",
            )
        )

    for col in frame.columns:
        series = frame[col]
        non_null = int(series.notna().sum())
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            lo = f"{values.min():,.4g}" if not values.empty else "—"
            hi = f"{values.max():,.4g}" if not values.empty else "—"
        elif pd.api.types.is_datetime64_any_dtype(series):
            values = pd.to_datetime(series, errors="coerce").dropna()
            lo = str(values.min().date()) if not values.empty else "—"
            hi = str(values.max().date()) if not values.empty else "—"
        else:
            values = series.dropna().astype(str)
            uniques = sorted(values.unique().tolist())
            lo = uniques[0] if uniques else "—"
            hi = uniques[-1] if uniques else "—"
        docs.append(
            ColumnDoc(
                name=str(col),
                role=classify_column(col),
                dtype=str(series.dtype),
                non_null=non_null,
                total=total,
                minimum=lo,
                maximum=hi,
                unit=units.get(str(col), ""),
            )
        )
    return docs


# --------------------------------------------------------------------------
# Layout parsers
# --------------------------------------------------------------------------


def _parse_daily_multivariety(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Guntur variety-wise daily prices: two header rows, four fields per variety.

    Row 0 carries the variety name once per four-column block; row 1 carries
    the measure (Low / High / Avg / Difference). The result is a long frame
    indexed by date with ``variety``/``measure``/``value`` columns, plus a
    convenience wide frame of average prices in ``meta['wide_avg']``.
    """
    raw = _trim(raw)
    if raw.shape[0] < 3:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    variety_row = raw.iloc[0].tolist()
    measure_row = raw.iloc[1].tolist()

    # Forward-fill the merged variety name across its four sub-columns. The
    # date column heads the row, so it is excluded from the fill.
    varieties: list[str] = []
    current = ""
    for idx, cell in enumerate(variety_row):
        text = cell_text(cell)
        if idx == 0:
            varieties.append("")  # the Date column belongs to no variety
            continue
        if text:
            current = text
        varieties.append(current)

    records: list[tuple[pd.Timestamp, str, str, float]] = []
    ordered_varieties: list[str] = []
    measures_seen: set[str] = set()

    body = raw.iloc[2:]
    for _, row in body.iterrows():
        values = row.tolist()
        stamp = to_datetime(values[0])
        if stamp is None:
            continue  # trailing blank / stray rows
        for col_idx in range(1, len(values)):
            variety = varieties[col_idx] if col_idx < len(varieties) else ""
            if not variety:
                continue
            measure = classify_column(measure_row[col_idx] if col_idx < len(measure_row) else "")
            if measure == "unknown":
                continue
            number = to_number(values[col_idx])
            if np.isnan(number):
                continue
            records.append((stamp, variety, measure, number))
            measures_seen.add(measure)
            if variety not in ordered_varieties:
                ordered_varieties.append(variety)

    if not records:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    long = pd.DataFrame(records, columns=["date", "variety", "measure", "value"])
    long = long.drop_duplicates(subset=["date", "variety", "measure"], keep="last")
    long = long.set_index("date").sort_index()

    wide_avg = (
        long[long["measure"] == "average"]
        .pivot_table(index="date", columns="variety", values="value", aggfunc="mean")
        .sort_index()
    )
    # Preserve the workbook's own left-to-right variety order.
    wide_avg = wide_avg[[v for v in ordered_varieties if v in wide_avg.columns]]

    wide_low = (
        long[long["measure"] == "low"]
        .pivot_table(index="date", columns="variety", values="value", aggfunc="mean")
        .sort_index()
    )
    wide_high = (
        long[long["measure"] == "high"]
        .pivot_table(index="date", columns="variety", values="value", aggfunc="mean")
        .sort_index()
    )

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=long,
        meta={
            "varieties": ordered_varieties,
            "measures": sorted(measures_seen),
            "wide_avg": wide_avg,
            "wide_low": wide_low,
            "wide_high": wide_high,
            "price_unit": "INR per quintal (as recorded in the workbook)",
        },
    )
    ds.columns = _document_columns(wide_avg)
    for doc in ds.columns:
        doc.role = "variety average price"
        doc.unit = "INR/quintal"
    return ds


def _parse_daily_ohlc_arrivals(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Warangal / Khammam sheets: Date, low, high, average, arrivals."""
    raw = _trim(raw)
    if raw.shape[0] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    headers = raw.iloc[0].tolist()
    roles = [classify_column(h) for h in headers]
    units: dict[str, str] = {}
    bag_weights: dict[str, float] = {}

    columns: dict[int, str] = {}
    for idx, (header, role) in enumerate(zip(headers, roles)):
        if idx == 0 or role == "date":
            continue
        label = role if role != "unknown" else normalise_text(header) or f"col{idx}"
        # Disambiguate repeated roles.
        base, n = label, 2
        while label in columns.values():
            label = f"{base}_{n}"
            n += 1
        columns[idx] = label
        weight = parse_bag_weight(header)
        if weight:
            bag_weights[label] = weight
            units[label] = f"bags (1 bag = {weight:g} kg, per sheet header)"
        elif role in ("low", "high", "average"):
            units[label] = "INR/quintal"
        elif role == "arrivals":
            units[label] = "bags (unit not stated in sheet header)"

    records: list[dict[str, Any]] = []
    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        stamp = to_datetime(values[0])
        if stamp is None:
            continue
        record: dict[str, Any] = {"date": stamp}
        any_value = False
        for idx, label in columns.items():
            number = to_number(values[idx]) if idx < len(values) else float("nan")
            record[label] = number
            any_value = any_value or not np.isnan(number)
        if any_value:
            records.append(record)

    if not records:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    frame = pd.DataFrame(records).drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()
    frame.index.name = "date"

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=frame,
        meta={
            "bag_weights_kg": bag_weights,
            "raw_headers": [str(h) for h in headers],
            "units": units,
        },
    )
    ds.columns = _document_columns(frame, units)
    return ds


def _parse_daily_series(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """A date column followed by one or more numeric columns."""
    raw = _trim(raw)
    if raw.shape[0] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    headers = raw.iloc[0].tolist()
    units: dict[str, str] = {}
    bag_weights: dict[str, float] = {}
    columns: dict[int, str] = {}

    for idx, header in enumerate(headers):
        if idx == 0:
            continue
        role = classify_column(header)
        # Keep the workbook's own wording, minus any embedded unit annotation.
        label = strip_unit_clause(header)
        if not label:
            label = role if role != "unknown" else f"Col{idx}"
        base, n = label, 2
        while label in columns.values():
            label = f"{base} {n}"
            n += 1
        columns[idx] = label
        weight = parse_bag_weight(header)
        if weight:
            bag_weights[label] = weight
            units[label] = f"bags (1 bag = {weight:g} kg, per sheet header)"
        elif role == "rate":
            units[label] = "INR per USD"

    records: list[dict[str, Any]] = []
    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        stamp = to_datetime(values[0])
        if stamp is None:
            continue
        record: dict[str, Any] = {"date": stamp}
        any_value = False
        for idx, label in columns.items():
            number = to_number(values[idx]) if idx < len(values) else float("nan")
            record[label] = number
            any_value = any_value or not np.isnan(number)
        if any_value:
            records.append(record)

    if not records:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    frame = pd.DataFrame(records).drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()
    frame.index.name = "date"

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=frame,
        meta={"bag_weights_kg": bag_weights, "units": units,
              "raw_headers": [str(h) for h in headers]},
    )
    ds.columns = _document_columns(frame, units)
    return ds


def _year_month_to_series(
    table: pd.DataFrame, year_axis: str = "index"
) -> pd.Series:
    """Convert a year x month table into a month-start indexed series."""
    records: list[tuple[pd.Timestamp, float]] = []
    for year_label, row in table.iterrows():
        year = parse_year(year_label)
        if year is None:
            continue
        for month_label, cell in row.items():
            month = parse_month(month_label)
            if month is None:
                continue
            value = to_number(cell)
            if np.isnan(value):
                continue
            records.append((pd.Timestamp(year=year, month=month, day=1), value))
    if not records:
        return pd.Series(dtype="float64")
    series = pd.Series(dict(records)).sort_index()
    series.index.name = "date"
    return series


def _parse_year_month_matrix(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Seasonality sheet: rows are years, columns are months.

    Rows whose first cell is not a year (``'10 yr Average'``,
    ``'Seasonality Index'``) are workbook-supplied aggregates. They are kept
    separately in ``meta`` and clearly labelled as the workbook's own numbers
    rather than mixed into the observation set.
    """
    raw = _trim(raw)
    if raw.shape[0] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    header = raw.iloc[0].tolist()
    month_cols: dict[int, int] = {}
    for idx, cell in enumerate(header):
        if idx == 0:
            continue
        month = parse_month(cell)
        if month is not None:
            month_cols[idx] = month

    observations: dict[int, dict[int, float]] = {}
    supplied: dict[str, dict[int, float]] = {}
    legend: list[str] = []

    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        label = values[0]
        year = parse_year(label)
        row_values = {m: to_number(values[i]) for i, m in month_cols.items() if i < len(values)}
        row_values = {m: v for m, v in row_values.items() if not np.isnan(v)}
        label_text = cell_text(label)
        if year is not None:
            if row_values:
                observations[year] = row_values
        elif label_text and row_values:
            supplied[label_text] = row_values
        else:
            # Unlabelled rows carrying stray text: the sheet's colour legend
            # describing the low / average / high demand bands.
            for cell in values[1:]:
                text = cell_text(cell)
                if text and np.isnan(to_number(cell)):
                    legend.append(text)

    if not observations:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    table = pd.DataFrame(observations).T.sort_index()
    table.columns = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in table.columns]
    table.index.name = "year"

    series = _year_month_to_series(table)

    supplied_table = pd.DataFrame()
    if supplied:
        supplied_table = pd.DataFrame(supplied).T
        supplied_table.columns = [
            settings.MONTH_ABBREVIATIONS[m - 1].title() for m in supplied_table.columns
        ]

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=table,
        meta={
            "monthly_series": series,
            "workbook_supplied_rows": supplied_table,
            "legend_text": sorted(set(legend)),
        },
    )
    ds.columns = _document_columns(table)
    for doc in ds.columns:
        doc.role = "monthly average price"
        doc.unit = "INR/quintal"
    return ds


def _parse_month_year_matrix(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Exports sheet: rows are months, columns are years."""
    raw = _trim(raw)
    if raw.shape[0] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    header = raw.iloc[0].tolist()
    year_cols: dict[int, int] = {}
    projected_years: set[int] = set()
    for idx, cell in enumerate(header):
        if idx == 0:
            continue
        year = parse_year(cell)
        if year is not None:
            year_cols[idx] = year
            if is_projection_label(cell):
                projected_years.add(year)

    grid: dict[int, dict[int, float]] = {}
    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        month = parse_month(values[0])
        if month is None:
            continue
        for idx, year in year_cols.items():
            if idx >= len(values):
                continue
            value = to_number(values[idx])
            if np.isnan(value):
                continue
            grid.setdefault(year, {})[month] = value

    if not grid:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    table = pd.DataFrame(grid).T.sort_index()
    table = table[sorted(table.columns)]
    table.columns = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in table.columns]
    table.index.name = "year"
    series = _year_month_to_series(table)

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=table,
        meta={
            "monthly_series": series,
            "projected_years": sorted(projected_years),
            "unit_note": (
                "Units are not stated on the sheet; values are used as "
                "supplied and compared only in relative terms."
            ),
        },
    )
    ds.columns = _document_columns(table)
    for doc in ds.columns:
        doc.role = "monthly exports"
        doc.unit = "as supplied (unit not stated on sheet)"
    return ds


def _parse_stacked_month_year_matrix(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Guntur monthly arrivals: two stacked month x year blocks in different units.

    Blocks are located by scanning for rows whose first cell reads "Months".
    The unit of the first block is read from the sheet's own annotation
    (``1 bag = 45 Kg``); the second block carries no label, so the conversion
    factor between the blocks is *measured* from overlapping cells and
    reported as an observation rather than assumed.
    """
    raw = _trim(raw)
    block_starts = [
        i for i, cell in enumerate(raw.iloc[:, 0].tolist())
        if normalise_text(cell).startswith("month")
    ]
    if not block_starts:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    blocks: list[dict[str, Any]] = []
    for order, start in enumerate(block_starts):
        end = block_starts[order + 1] if order + 1 < len(block_starts) else len(raw)
        header = raw.iloc[start].tolist()
        year_cols = {i: parse_year(c) for i, c in enumerate(header) if i > 0}
        year_cols = {i: y for i, y in year_cols.items() if y is not None}

        # Annotation cells to the right of the year headers carry the unit.
        annotation = " ".join(
            cell_text(c)
            for i, c in enumerate(header)
            if i > 0 and i not in year_cols and cell_text(c)
        )

        grid: dict[int, dict[int, float]] = {}
        totals: dict[int, float] = {}
        for _, row in raw.iloc[start + 1 : end].iterrows():
            values = row.tolist()
            label = normalise_text(values[0])
            month = parse_month(values[0])
            is_total = label in settings.AGGREGATE_ROW_LABELS
            if month is None and not is_total:
                continue
            for idx, year in year_cols.items():
                if idx >= len(values):
                    continue
                value = to_number(values[idx])
                if np.isnan(value):
                    continue
                if is_total:
                    totals[year] = value
                else:
                    grid.setdefault(year, {})[month] = value

        if not grid:
            continue
        table = pd.DataFrame(grid).T.sort_index()
        table = table[sorted(table.columns)]
        table.columns = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in table.columns]
        table.index.name = "year"
        bag_weight = parse_bag_weight(annotation)
        blocks.append(
            {
                "index": order,
                "table": table,
                "series": _year_month_to_series(table),
                "totals": pd.Series(totals).sort_index() if totals else pd.Series(dtype="float64"),
                "annotation": annotation.strip(),
                "bag_weight_kg": bag_weight,
            }
        )

    if not blocks:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    primary = blocks[0]
    meta: dict[str, Any] = {
        "blocks": blocks,
        "monthly_series": primary["series"],
        "annual_totals": primary["totals"],
        "primary_unit": (
            f"bags (1 bag = {primary['bag_weight_kg']:g} kg, per sheet annotation)"
            if primary["bag_weight_kg"]
            else "bags (weight not stated on sheet)"
        ),
    }

    # Measure the relationship between the two blocks instead of assuming it.
    if len(blocks) > 1:
        a, b = blocks[0]["series"], blocks[1]["series"]
        joined = pd.concat([a.rename("block1"), b.rename("block2")], axis=1).dropna()
        joined = joined[joined["block1"] != 0]
        if not joined.empty:
            ratios = joined["block2"] / joined["block1"]
            factor = float(ratios.median())
            consistent = bool(np.isclose(ratios, factor, rtol=1e-6).all())
            meta["block_conversion_factor"] = factor
            meta["block_conversion_consistent"] = consistent
            meta["secondary_unit"] = (
                f"second block = first block x {factor:g} "
                f"({'constant across all cells' if consistent else 'factor varies by cell'}); "
                f"sheet annotation: {blocks[1]['annotation'] or 'none'}"
            )

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=primary["table"],
        meta=meta,
    )
    ds.columns = _document_columns(primary["table"])
    for doc in ds.columns:
        doc.role = "monthly arrivals"
        doc.unit = meta["primary_unit"]
    return ds


def _parse_particulars_by_year(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Balance sheet: line items down, years across, with a unit note above."""
    raw = _trim(raw)
    header_row = None
    for i, row in raw.iterrows():
        years = [parse_year(c) for c in row.tolist()[1:]]
        if sum(1 for y in years if y is not None) >= 3:
            header_row = i
            break
    if header_row is None:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    unit_note = ""
    for i in range(0, header_row):
        text = cell_text(raw.iloc[i, 0])
        if text:
            unit_note = text
            break

    header = raw.iloc[header_row].tolist()
    year_cols: dict[int, int] = {}
    projected: set[int] = set()
    for idx, cell in enumerate(header):
        if idx == 0:
            continue
        year = parse_year(cell)
        if year is None:
            continue
        year_cols[idx] = year
        if is_projection_label(cell):
            projected.add(year)

    rows: dict[str, dict[int, float]] = {}
    order: list[str] = []
    for _, row in raw.iloc[header_row + 1 :].iterrows():
        values = row.tolist()
        label = cell_text(values[0])
        if not label:
            continue
        entry = {}
        for idx, year in year_cols.items():
            if idx >= len(values):
                continue
            value = to_number(values[idx])
            if not np.isnan(value):
                entry[year] = value
        if entry:
            rows[label] = entry
            order.append(label)

    if not rows:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    table = pd.DataFrame(rows).T.reindex(order)
    table = table[sorted(table.columns)]
    table.index.name = "particular"

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=table,
        meta={
            "unit_note": unit_note,
            "projected_years": sorted(projected),
            "particulars": order,
            "header_labels": {parse_year(c): str(c) for c in header[1:] if parse_year(c)},
        },
    )
    ds.columns = [
        ColumnDoc(
            name=str(year),
            role="projection" if year in projected else "historical",
            dtype="float64",
            non_null=int(table[year].notna().sum()),
            total=int(len(table)),
            minimum=f"{table[year].min():,.4g}" if table[year].notna().any() else "—",
            maximum=f"{table[year].max():,.4g}" if table[year].notna().any() else "—",
            unit=unit_note,
            note="Marked (exp) in the workbook" if year in projected else "",
        )
        for year in table.columns
    ]
    return ds


def _parse_state_year_metric_matrix(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """APY sheet: states down; years across, each spanning Area/Production/Yield."""
    raw = _trim(raw)
    if raw.shape[0] < 3:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    year_row = raw.iloc[0].tolist()
    metric_row = raw.iloc[1].tolist()

    years: list[int | None] = []
    projected: set[int] = set()
    current: int | None = None
    for cell in year_row:
        year = parse_year(cell)
        if year is not None:
            current = year
            if is_projection_label(cell):
                projected.add(year)
        years.append(current)

    def metric_of(cell: Any) -> str | None:
        text = normalise_text(cell)
        if "area" in text:
            return "Area (Ha)"
        if "production" in text:
            return "Production (MT)"
        if "yield" in text:
            return "Yield (t/Ha)"
        return None

    records: list[dict[str, Any]] = []
    for _, row in raw.iloc[2:].iterrows():
        values = row.tolist()
        state = cell_text(values[0])
        if not state:
            continue
        for idx in range(1, len(values)):
            year = years[idx] if idx < len(years) else None
            metric = metric_of(metric_row[idx]) if idx < len(metric_row) else None
            if year is None or metric is None:
                continue
            value = to_number(values[idx])
            if np.isnan(value):
                continue
            records.append({"state": state, "year": year, "metric": metric, "value": value})

    if not records:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    long = pd.DataFrame(records)
    wide = long.pivot_table(
        index=["state", "year"], columns="metric", values="value", aggfunc="mean"
    ).reset_index()

    # Which state row represents the national aggregate?
    states = long["state"].unique().tolist()
    national = next((s for s in states if normalise_text(s) in ("india", "all india", "total")), None)

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=wide,
        meta={
            "long": long,
            "states": states,
            "national_row": national,
            "projected_years": sorted(projected),
            "metrics": sorted(long["metric"].unique().tolist()),
        },
    )
    ds.columns = _document_columns(wide)
    return ds


def _parse_square_matrix(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """A labelled numeric matrix (the workbook's own variety correlations)."""
    raw = _trim(raw)
    if raw.shape[0] < 2 or raw.shape[1] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    col_labels = [cell_text(c) for c in raw.iloc[0].tolist()[1:]]
    rows: dict[str, list[float]] = {}
    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        label = cell_text(values[0])
        if not label:
            continue
        rows[label] = [to_number(v) for v in values[1 : len(col_labels) + 1]]

    if not rows:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    matrix = pd.DataFrame(rows).T
    matrix.columns = col_labels[: matrix.shape[1]]
    matrix.index.name = "variety"

    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=matrix,
        meta={
            "row_labels": list(matrix.index),
            "col_labels": list(matrix.columns),
            "square": list(matrix.index) == list(matrix.columns),
            "note": (
                "Supplied by the workbook. The Correlation Studio recomputes "
                "correlations independently from the daily price sheet; the "
                "two are shown side by side."
            ),
        },
    )
    ds.columns = _document_columns(matrix)
    for doc in ds.columns:
        doc.role = "correlation coefficient"
        doc.unit = "dimensionless (-1 to +1)"
    return ds


def _parse_sparse_records(raw: pd.DataFrame, spec, sheet: str) -> Dataset:
    """Cold storage stock: Year, Month, then one column per state/market."""
    raw = _trim(raw)
    if raw.shape[0] < 2:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    headers = [cell_text(h) for h in raw.iloc[0].tolist()]
    lowered = [normalise_text(h) for h in headers]
    try:
        year_idx = next(i for i, h in enumerate(lowered) if h.startswith("year"))
        month_idx = next(i for i, h in enumerate(lowered) if h.startswith("month"))
    except StopIteration:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    value_cols = {
        i: headers[i]
        for i in range(len(headers))
        if i not in (year_idx, month_idx) and headers[i]
    }

    records: list[dict[str, Any]] = []
    for _, row in raw.iloc[1:].iterrows():
        values = row.tolist()
        year = parse_year(values[year_idx]) if year_idx < len(values) else None
        month = parse_month(values[month_idx]) if month_idx < len(values) else None
        if year is None or month is None:
            continue
        stamp = pd.Timestamp(year=year, month=month, day=1)
        for idx, label in value_cols.items():
            if idx >= len(values):
                continue
            value = to_number(values[idx])
            if np.isnan(value):
                continue
            records.append({"date": stamp, "location": label, "stock": value})

    if not records:
        return Dataset(spec.key, sheet, spec.layout, spec.description, pd.DataFrame())

    long = pd.DataFrame(records).sort_values(["date", "location"]).reset_index(drop=True)
    wide = long.pivot_table(index="date", columns="location", values="stock", aggfunc="mean")

    coverage = {str(c): int(wide[c].notna().sum()) for c in wide.columns}
    ds = Dataset(
        key=spec.key,
        sheet_name=sheet,
        layout=spec.layout,
        description=spec.description,
        frame=wide,
        meta={
            "long": long,
            "observations_per_location": coverage,
            "unit": "bags (per column headers)",
        },
        warnings=[
            f"Only {len(wide)} reporting month(s) are present; the densest "
            f"location has {max(coverage.values()) if coverage else 0} observation(s). "
            "This is far below what any time-series or correlation analysis "
            "requires."
        ],
    )
    ds.columns = _document_columns(wide, {c: "bags" for c in map(str, wide.columns)})
    return ds


_PARSERS = {
    "daily_multivariety": _parse_daily_multivariety,
    "daily_ohlc_arrivals": _parse_daily_ohlc_arrivals,
    "daily_series": _parse_daily_series,
    "year_month_matrix": _parse_year_month_matrix,
    "month_year_matrix": _parse_month_year_matrix,
    "stacked_month_year_matrix": _parse_stacked_month_year_matrix,
    "particulars_by_year": _parse_particulars_by_year,
    "state_year_metric_matrix": _parse_state_year_metric_matrix,
    "square_matrix": _parse_square_matrix,
    "sparse_records": _parse_sparse_records,
}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

_CACHE: dict[str, WorkbookData] = {}


def load_workbook(
    path: str | Path | None = None, *, force_reload: bool = False
) -> WorkbookData:
    """Read and parse the master workbook, caching the result.

    Parameters
    ----------
    path:
        Optional explicit workbook path. When omitted the standard search
        order in :func:`settings.workbook_search_paths` applies.
    force_reload:
        Bypass the cache (used by the UI's Reload action).
    """
    resolved = resolve_workbook_path(path)
    cache_key = str(resolved.resolve())
    if not force_reload and cache_key in _CACHE:
        LOG.debug("Workbook served from cache: %s", cache_key)
        return _CACHE[cache_key]

    start = time.perf_counter()
    LOG.info("Reading workbook %s", resolved)
    try:
        raw_sheets: dict[str, pd.DataFrame] = pd.read_excel(
            resolved, sheet_name=None, header=None, engine="openpyxl"
        )
    except Exception as exc:  # noqa: BLE001
        raise WorkbookError(f"Could not read '{resolved.name}': {exc}") from exc

    if not raw_sheets:
        raise WorkbookError(f"'{resolved.name}' contains no worksheets.")

    available = list(raw_sheets.keys())
    assignment, unmapped, missing = _assign_sheets(available)

    datasets: dict[str, Dataset] = {}
    warnings: list[str] = []

    spec_by_key = {s.key: s for s in settings.SHEET_SPECS}
    for key, sheet_name in assignment.items():
        spec = spec_by_key[key]
        parser = _PARSERS.get(spec.layout)
        if parser is None:  # pragma: no cover - guards a typo in the specs
            warnings.append(f"No parser registered for layout '{spec.layout}'.")
            continue
        try:
            dataset = parser(raw_sheets[sheet_name], spec, sheet_name)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Failed to parse sheet '%s'", sheet_name)
            warnings.append(f"Sheet '{sheet_name}' could not be parsed: {exc}")
            continue
        datasets[key] = dataset
        if dataset.empty:
            warnings.append(f"Sheet '{sheet_name}' parsed to zero usable rows.")
        warnings.extend(f"{sheet_name}: {w}" for w in dataset.warnings)

    for key in missing:
        spec = spec_by_key[key]
        if spec.required:
            warnings.append(
                f"Expected sheet for '{spec.key}' "
                f"(keywords: {', '.join(spec.keywords)}) was not found."
            )

    elapsed = time.perf_counter() - start
    data = WorkbookData(
        path=resolved,
        loaded_at=pd.Timestamp.now(),
        load_seconds=elapsed,
        datasets=datasets,
        unmapped_sheets=unmapped,
        missing_sheets=missing,
        warnings=warnings,
        raw_shapes={name: frame.shape for name, frame in raw_sheets.items()},
    )
    _CACHE[cache_key] = data
    LOG.info(
        "Parsed %d/%d sheets in %.2fs (%d warning(s))",
        len(datasets), len(available), elapsed, len(warnings),
    )
    return data


def clear_cache() -> None:
    """Drop the cached workbook (used before a forced reload)."""
    _CACHE.clear()


# --------------------------------------------------------------------------
# Data dictionary
# --------------------------------------------------------------------------


def build_data_dictionary(data: WorkbookData) -> pd.DataFrame:
    """Assemble the auto-generated data dictionary across all parsed sheets."""
    rows: list[dict[str, Any]] = []
    for spec in settings.SHEET_SPECS:
        dataset = data.datasets.get(spec.key)
        if dataset is None:
            rows.append(
                {
                    "Sheet": "(not found)",
                    "Dataset": spec.key,
                    "Field": "—",
                    "Role": "—",
                    "Type": "—",
                    "Rows": 0,
                    "Populated": "0%",
                    "Minimum": "—",
                    "Maximum": "—",
                    "Unit": "—",
                    "Notes": settings.DATA_UNAVAILABLE_MESSAGE,
                }
            )
            continue
        if not dataset.columns:
            rows.append(
                {
                    "Sheet": dataset.sheet_name,
                    "Dataset": spec.key,
                    "Field": "—",
                    "Role": "—",
                    "Type": "—",
                    "Rows": dataset.n_rows,
                    "Populated": "0%",
                    "Minimum": "—",
                    "Maximum": "—",
                    "Unit": "—",
                    "Notes": "Sheet found but produced no usable columns.",
                }
            )
            continue
        for doc in dataset.columns:
            rows.append(
                {
                    "Sheet": dataset.sheet_name,
                    "Dataset": spec.key,
                    "Field": doc.name,
                    "Role": doc.role,
                    "Type": doc.dtype,
                    "Rows": doc.total,
                    "Populated": f"{doc.coverage * 100:.0f}%",
                    "Minimum": doc.minimum,
                    "Maximum": doc.maximum,
                    "Unit": doc.unit or "—",
                    "Notes": doc.note or dataset.description,
                }
            )

    for name in data.unmapped_sheets:
        shape = data.raw_shapes.get(name, (0, 0))
        rows.append(
            {
                "Sheet": name,
                "Dataset": "(unmapped)",
                "Field": "—",
                "Role": "—",
                "Type": "—",
                "Rows": shape[0],
                "Populated": "—",
                "Minimum": "—",
                "Maximum": "—",
                "Unit": "—",
                "Notes": "Present in the workbook but not used by any analysis.",
            }
        )

    return pd.DataFrame(rows)


def data_dictionary_markdown(data: WorkbookData) -> str:
    """Render the data dictionary as Markdown for the docs deliverable."""
    frame = build_data_dictionary(data)
    lines = [
        "# Data Dictionary",
        "",
        f"Auto-generated from `{data.path.name}` on "
        f"{data.loaded_at:%Y-%m-%d %H:%M:%S}.",
        "",
        f"- Worksheets in file: **{len(data.raw_shapes)}**",
        f"- Worksheets mapped to analyses: **{len(data.datasets)}**",
        f"- Worksheets unmapped: **{len(data.unmapped_sheets)}**",
        f"- Parse time: **{data.load_seconds:.2f}s**",
        "",
    ]
    for sheet, group in frame.groupby("Sheet", sort=False):
        lines.append(f"## {sheet}")
        lines.append("")
        dataset_key = group["Dataset"].iloc[0]
        spec = next((s for s in settings.SHEET_SPECS if s.key == dataset_key), None)
        if spec:
            lines.extend([spec.description, ""])
        header = ["Field", "Role", "Type", "Rows", "Populated", "Minimum", "Maximum", "Unit"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for _, row in group.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in header) + " |")
        lines.append("")

    if data.warnings:
        lines.extend(["## Parse warnings", ""])
        lines.extend(f"- {w}" for w in data.warnings)
        lines.append("")

    return "\n".join(lines)
