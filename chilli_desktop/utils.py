"""Shared helpers: logging, result wrappers, numeric coercion and formatting.

The single most important idea in this module is :class:`Result`. Every
analytical function in the application returns one. It carries either a value
or a machine-readable reason why the analysis could not be performed, which
lets the UI honour strict data rule 6 ("Data not available in uploaded
workbook.") without any function ever inventing a fallback number.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from . import settings
from .settings import FORTNIGHT_FREQ

T = TypeVar("T")

_LOGGER_NAME = "chilli_desktop"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the application logger once and return it.

    Logs go to both stderr and a rotating-by-run file under ``logs/``. Failure
    to create the log directory is non-fatal -- the console handler still
    works, and a dashboard that cannot write logs should still start.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s.%(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(fmt)
    console.setLevel(level)
    logger.addHandler(console)

    try:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(
            settings.LOG_DIR / f"chilli_desktop_{stamp}.log", encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover - environment dependent
        logger.warning("File logging disabled (%s)", exc)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Return a child of the application logger."""
    base = logging.getLogger(_LOGGER_NAME)
    return base.getChild(name) if name else base


LOG = get_logger()


# --------------------------------------------------------------------------
# Result / availability plumbing
# --------------------------------------------------------------------------


@dataclass
class Result(Generic[T]):
    """An analysis outcome that may legitimately be unavailable.

    Parameters
    ----------
    value:
        The computed object when :attr:`ok` is ``True``; ``None`` otherwise.
    reason:
        Plain-language explanation of *why* the analysis is unavailable. Shown
        beneath the standard unavailable message so the user understands
        whether the gap is missing data, too few observations, or a failed
        model fit.
    source:
        Workbook sheet name(s) the result was derived from. Rendered as the
        provenance caption on every chart and table (strict data rule 8).
    notes:
        Assumptions and caveats to display separately (strict data rule 9).
    """

    value: T | None = None
    reason: str = ""
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    def __bool__(self) -> bool:
        return self.ok

    def unwrap(self) -> T:
        """Return the value, raising if unavailable. Use only after ``if res``."""
        if self.value is None:
            raise DataUnavailableError(self.reason or settings.DATA_UNAVAILABLE_MESSAGE)
        return self.value

    def message(self) -> str:
        """The full user-facing message for an unavailable result."""
        if self.ok:
            return ""
        base = settings.DATA_UNAVAILABLE_MESSAGE
        return f"{base}\n\n{self.reason}" if self.reason else base

    @classmethod
    def unavailable(cls, reason: str, source: str = "") -> "Result[T]":
        return cls(value=None, reason=reason, source=source)

    @classmethod
    def of(cls, value: T, source: str = "", notes: Sequence[str] | None = None) -> "Result[T]":
        return cls(value=value, source=source, notes=list(notes or []))


class DataUnavailableError(RuntimeError):
    """Raised when a value is unwrapped from an unavailable :class:`Result`."""


class WorkbookError(RuntimeError):
    """Raised when the workbook is missing or structurally unreadable."""


def safe_analysis(source: str = ""):
    """Decorator that converts an unexpected exception into an unavailable Result.

    Statistical routines fail in a wide variety of ways on real data --
    singular matrices, non-convergence, all-NaN slices. Rather than crash the
    dashboard, we log the traceback and degrade that one panel.
    """

    def decorator(fn: Callable[..., Result[Any]]) -> Callable[..., Result[Any]]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Result[Any]:
            try:
                return fn(*args, **kwargs)
            except DataUnavailableError as exc:
                return Result.unavailable(str(exc), source)
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all
                LOG.exception("Analysis %s failed", fn.__name__)
                return Result.unavailable(
                    f"The calculation could not be completed: "
                    f"{type(exc).__name__}: {exc}",
                    source,
                )

        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Text and numeric coercion
# --------------------------------------------------------------------------


def is_blank(value: Any) -> bool:
    """True for cells that carry no content.

    Guards the trap that ``str(float('nan'))`` is the *non-empty* string
    ``'nan'``: a naive truthiness test on a stringified cell silently turns
    every empty spreadsheet cell into a real label.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def cell_text(value: Any) -> str:
    """Original-case text of a cell, or ``''`` when the cell is blank.

    Integral floats render without the ``.0`` tail so that a numeric header
    such as ``5531`` does not become ``'5531.0'``.
    """
    if is_blank(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).replace("\xa0", " ").strip()


def strip_unit_clause(header: Any) -> str:
    """Remove a trailing ``1 bag = 45 Kg`` style annotation from a header."""
    text = cell_text(header)
    if not text:
        return ""
    text = re.sub(
        r"[\(,]?\s*(?:1|one)?\s*bags?\s*=\s*[\d.]+\s*(?:kgs?|kilograms?)\.?\)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" ,-()")


def normalise_text(value: Any) -> str:
    """Lower-case, collapse whitespace and strip punctuation-ish noise."""
    if is_blank(value):
        return ""
    text = str(value).replace("\xa0", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def squash(value: Any) -> str:
    """Aggressively normalise a label for fuzzy matching: keep only a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", normalise_text(value))


def to_number(value: Any) -> float:
    """Coerce a workbook cell to a float, or NaN.

    Non-numeric market tokens such as ``"Closed"`` become NaN rather than
    zero. This matters: a closed market is a *missing* observation, and
    treating it as zero would silently fabricate a price collapse.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else float("nan")

    text = normalise_text(value)
    if text in settings.NON_NUMERIC_MISSING_TOKENS:
        return float("nan")

    # Strip thousands separators, currency marks and stray unit suffixes.
    cleaned = re.sub(r"[,\s₹$]", "", text)
    cleaned = re.sub(r"(rs\.?|inr|kg|mt|tons?|tonnes?|bags?|lakh)$", "", cleaned)
    if cleaned in ("", "-", "."):
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def to_datetime(value: Any) -> pd.Timestamp | None:
    """Coerce a workbook cell to a Timestamp, or ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        try:
            return pd.Timestamp(value).normalize()
        except (ValueError, OverflowError):
            return None
    text = str(value).strip()
    if not text or normalise_text(text) in settings.NON_NUMERIC_MISSING_TOKENS:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    except (ValueError, TypeError):
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def parse_year(value: Any) -> int | None:
    """Extract a four-digit year from a header cell such as ``'2026(exp)'``."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and 1900 <= int(value) <= 2200:
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        candidate = int(value)
        if 1900 <= candidate <= 2200:
            return candidate
    match = re.search(r"(19|20|21)\d{2}", str(value))
    return int(match.group(0)) if match else None


def parse_month(value: Any) -> int | None:
    """Map a month label (``'Jan'``, ``'january'``, ``3``) to 1-12."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and 1 <= int(value) <= 12:
        return int(value)
    token = normalise_text(value)[:3]
    if token in settings.MONTH_ABBREVIATIONS:
        return settings.MONTH_ABBREVIATIONS.index(token) + 1
    return None


def is_projection_label(value: Any) -> bool:
    """True when a header marks projected rather than realised data.

    The workbook flags forward-looking columns with ``(exp)``; those must be
    visually distinguished from history (strict data rule 10).
    """
    text = normalise_text(value)
    return any(tok in text for tok in ("exp", "proj", "est", "forecast", "f)"))


def parse_bag_weight(header: Any) -> float | None:
    """Read a ``1 bag = N kg`` conversion out of a column header.

    Returns ``None`` when the header carries no conversion, in which case the
    application reports quantities in their native unit rather than assuming
    a weight.
    """
    match = re.search(settings.BAG_WEIGHT_PATTERN, normalise_text(header))
    return float(match.group(1)) if match else None


def classify_column(header: Any) -> str:
    """Return the semantic role of a column header (see COLUMN_ROLE_KEYWORDS)."""
    text = normalise_text(header)
    if not text:
        return "unknown"
    for role, tokens in settings.COLUMN_ROLE_KEYWORDS:
        if any(token in text for token in tokens):
            return role
    return "unknown"


# --------------------------------------------------------------------------
# Series helpers
# --------------------------------------------------------------------------


def clean_series(series: pd.Series, name: str | None = None) -> pd.Series:
    """Return a numeric, date-indexed, de-duplicated, sorted series with no NaNs.

    Duplicated dates are collapsed by mean. No gap filling is performed --
    absent trading days stay absent (strict data rules 4 and 5).
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64", name=name)
    out = pd.to_numeric(series, errors="coerce")
    out = out[~out.index.isna()]
    out = out.dropna()
    if not out.empty:
        out = out.groupby(level=0).mean().sort_index()
    if name:
        out.name = name
    return out


def fortnight_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each date to the end of its half-month period (the 15th or month-end).

    Used instead of pandas' ``SME`` resample rule, which labels each bucket by
    its *start* boundary while ``W`` and ``ME`` label by the end. Mixing the
    two conventions would shift every fortnightly chart and forecast by one
    period relative to the weekly and monthly views.
    """
    idx = pd.DatetimeIndex(index).normalize()
    first_half = idx.day <= 15
    fifteenth = idx + pd.to_timedelta(15 - idx.day, unit="D")
    month_end = idx + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(np.where(first_half, fifteenth, month_end))


def resample_series(series: pd.Series, freq: str, how: str = "mean") -> pd.Series:
    """Resample a daily series to ``freq`` using ``how``, dropping empty periods.

    Empty periods are dropped rather than interpolated: a fortnight with no
    trading produces no observation, not a fabricated one.

    Every frequency is labelled by the *end* of its period, so the three views
    (weekly, fortnightly, monthly) are directly comparable.
    """
    if series.empty:
        return series
    if freq in ("D", ""):
        return series
    if freq == FORTNIGHT_FREQ:
        keys = fortnight_end_index(series.index)
        out = getattr(series.groupby(keys), how)()
        out.index.name = series.index.name or "date"
        return out.dropna().sort_index()
    grouped = series.resample(freq)
    out = getattr(grouped, how)()
    return out.dropna()


def resample_frame(frame: pd.DataFrame, freq: str, how: str = "mean") -> pd.DataFrame:
    """Frame equivalent of :func:`resample_series`, with the same conventions."""
    if frame is None or frame.empty or freq in ("D", ""):
        return frame
    if freq == FORTNIGHT_FREQ:
        keys = fortnight_end_index(frame.index)
        out = getattr(frame.groupby(keys), how)()
        out.index.name = frame.index.name or "date"
        return out.dropna(how="all").sort_index()
    return getattr(frame.resample(freq), how)().dropna(how="all")


def align_frames(*series: pd.Series, how: str = "inner") -> pd.DataFrame:
    """Join several named series on their date index."""
    named = [s for s in series if s is not None and not s.empty]
    if not named:
        return pd.DataFrame()
    frame = pd.concat(named, axis=1, join=how)
    return frame.dropna(how="all")


def common_span(*series: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the overlapping date span of the supplied series."""
    spans = [(s.index.min(), s.index.max()) for s in series if s is not None and not s.empty]
    if not spans:
        return None, None
    start = max(s for s, _ in spans)
    end = min(e for _, e in spans)
    return (start, end) if start <= end else (None, None)


def periods_per_year(freq: str) -> int:
    """Approximate number of observations per year at a given frequency."""
    return settings.ANALYTICS.seasonal_periods.get(freq, 12)


def frequency_label(freq: str) -> str:
    return settings.FORECAST.frequency_labels.get(freq, freq)


# --------------------------------------------------------------------------
# Formatting for display
# --------------------------------------------------------------------------


def fmt_number(value: Any, decimals: int = 2, dash: str = "—") -> str:
    """Format a number with thousands separators, or a dash when missing."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(num):
        return dash
    return f"{num:,.{decimals}f}"


def fmt_int(value: Any, dash: str = "—") -> str:
    return fmt_number(value, decimals=0, dash=dash)


def fmt_pct(value: Any, decimals: int = 2, dash: str = "—", signed: bool = False) -> str:
    """Format a fraction (0.043) as a percentage string (``+4.30%``)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(num):
        return dash
    sign = "+" if signed and num > 0 else ""
    return f"{sign}{num * 100:,.{decimals}f}%"


def fmt_signed(value: Any, decimals: int = 2, dash: str = "—") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(num):
        return dash
    sign = "+" if num > 0 else ""
    return f"{sign}{num:,.{decimals}f}"


def fmt_date(value: Any, dash: str = "—") -> str:
    ts = to_datetime(value)
    return ts.strftime("%d %b %Y") if ts is not None else dash


def fmt_pvalue(value: Any) -> str:
    """Format a p-value, collapsing very small values to ``<0.001``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    return "<0.001" if num < 0.001 else f"{num:.3f}"


def significance_stars(pvalue: Any) -> str:
    try:
        p = float(pvalue)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def describe_strength(correlation: Any) -> str:
    """Translate a correlation coefficient into trading-desk language."""
    try:
        r = abs(float(correlation))
    except (TypeError, ValueError):
        return "undetermined"
    if not math.isfinite(r):
        return "undetermined"
    if r >= 0.90:
        return "very strong"
    if r >= 0.70:
        return "strong"
    if r >= 0.50:
        return "moderate"
    if r >= 0.30:
        return "weak"
    return "negligible"


def humanise_list(items: Iterable[str], conjunction: str = "and") -> str:
    """``['a','b','c'] -> "a, b and c"``."""
    values = [str(i) for i in items if str(i)]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} {conjunction} {values[-1]}"


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
