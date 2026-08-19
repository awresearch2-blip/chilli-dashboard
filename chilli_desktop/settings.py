"""Central configuration for the Chilli Intelligence Desktop dashboard.

Everything in this module is *structural* configuration: where the workbook
lives, how sheets are recognised, which statistical parameters are used, and
how the UI is themed.

Deliberately absent from this module: any market data value. Prices, arrivals,
seasonal indices, correlations and balance-sheet figures are read from the
workbook at runtime and never appear as literals anywhere in the codebase
(strict data rule 12: "Do not hardcode any values").

The only literals here that touch the data are *recognition keywords* -- the
words used to identify which sheet is which and which column plays which role.
Those are metadata about the workbook's layout, not observations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Application identity
# --------------------------------------------------------------------------

APP_NAME: Final[str] = "Chilli Intelligence Desktop"
APP_SHORT_NAME: Final[str] = "Chilli Intelligence"
APP_VERSION: Final[str] = "1.0.0"
ORG_NAME: Final[str] = "AgriWatch"

#: Message shown verbatim wherever the workbook cannot support an analysis.
#: Mandated by strict data rule 6.
DATA_UNAVAILABLE_MESSAGE: Final[str] = "Data not available in uploaded workbook."


# --------------------------------------------------------------------------
# Workbook location
# --------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent

#: Filename of the master workbook. The loader searches the candidate
#: directories below, in order, for the first file that exists.
WORKBOOK_FILENAME: Final[str] = "Chilli mastersheet for dashboard.xlsx"

#: Environment variable that, when set, overrides all search paths.
WORKBOOK_ENV_VAR: Final[str] = "CHILLI_WORKBOOK"


def workbook_search_paths() -> list[Path]:
    """Return candidate workbook locations, highest priority first.

    Order of precedence:
      1. ``CHILLI_WORKBOOK`` environment variable (absolute path to a file).
      2. The user's source folder (where the analyst maintains the master).
      3. The project root (a working copy kept alongside the code).
      4. The package directory.
    """
    candidates: list[Path] = []

    override = os.environ.get(WORKBOOK_ENV_VAR, "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home
            / "OneDrive"
            / "Desktop"
            / "Desktop"
            / "Shahaniya"
            / "AgriWatch"
            / "chilli dashboard"
            / WORKBOOK_FILENAME,
            PROJECT_ROOT / WORKBOOK_FILENAME,
            PACKAGE_ROOT / WORKBOOK_FILENAME,
        ]
    )
    return candidates


#: Directory for exported charts, tables and generated documentation.
EXPORT_DIR: Final[Path] = PROJECT_ROOT / "exports"
DOCS_DIR: Final[Path] = PACKAGE_ROOT / "docs"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"


# --------------------------------------------------------------------------
# Sheet recognition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SheetSpec:
    """Describes how to recognise and interpret one workbook sheet.

    Attributes
    ----------
    key:
        Stable internal identifier used throughout the application.
    keywords:
        Lower-cased tokens that must *all* appear in the sheet name (after
        normalisation) for the sheet to match. Matching is whitespace- and
        case-insensitive, which lets the loader cope with the workbook's
        inconsistent naming (``'Khammam Teja Price& Arrivals'`` vs
        ``'KhammamTejacoldstorage'``).
    layout:
        Parser strategy. One of ``daily_multivariety``, ``daily_ohlc_arrivals``,
        ``daily_series``, ``year_month_matrix``, ``month_year_matrix``,
        ``particulars_by_year``, ``stacked_month_year_matrix``,
        ``state_year_metric_matrix``, ``square_matrix`` or ``sparse_records``.
    description:
        Human-readable role, surfaced in the auto-generated data dictionary.
    required:
        When ``True`` the application warns loudly if the sheet is absent.
    """

    key: str
    keywords: tuple[str, ...]
    layout: str
    description: str
    required: bool = True


#: The workbook's expected sheets. A sheet that is present but unlisted is
#: still catalogued in the data dictionary as "unmapped"; a listed sheet that
#: is absent degrades the relevant analyses to the data-unavailable message.
SHEET_SPECS: Final[tuple[SheetSpec, ...]] = (
    SheetSpec(
        key="seasonality_teja",
        keywords=("seasonality", "index", "guntur", "teja"),
        layout="year_month_matrix",
        description=(
            "Guntur Teja monthly average price by calendar year, with the "
            "workbook's own multi-year average and seasonality index rows."
        ),
    ),
    SheetSpec(
        key="guntur_variety_prices",
        keywords=("guntur", "varietywise", "daily", "price"),
        layout="daily_multivariety",
        description=(
            "Daily Guntur spot prices per variety. Two header rows: variety "
            "name spans four sub-columns (Low, High, Average, Difference)."
        ),
    ),
    SheetSpec(
        key="variety_correlation_workbook",
        keywords=("guntur", "variety", "correlation"),
        layout="square_matrix",
        description=(
            "Correlation matrix between Guntur varieties as supplied in the "
            "workbook. Displayed for reference alongside the values this "
            "application recomputes from the daily price sheet."
        ),
    ),
    SheetSpec(
        key="usd_inr",
        keywords=("usd", "inr", "exchange"),
        layout="daily_series",
        description="Daily USD/INR reference rate.",
    ),
    SheetSpec(
        key="guntur_daily_arrivals",
        keywords=("guntur", "daily", "arrivals"),
        layout="daily_series",
        description="Guntur mandi daily arrivals and offtake, in bags.",
    ),
    SheetSpec(
        key="warangal",
        keywords=("warangal",),
        layout="daily_ohlc_arrivals",
        description="Warangal Teja daily low/high/average price and arrivals.",
    ),
    SheetSpec(
        key="khammam_non_cold",
        keywords=("khammam", "non", "cold"),
        layout="daily_ohlc_arrivals",
        description=(
            "Khammam Teja daily price and arrivals for non-cold-storage "
            "(fresh) lots."
        ),
    ),
    SheetSpec(
        key="khammam_cold",
        keywords=("khammam", "cold"),
        layout="daily_ohlc_arrivals",
        description=(
            "Khammam Teja daily price and arrivals for cold-storage lots."
        ),
    ),
    SheetSpec(
        key="exports",
        keywords=("exports",),
        layout="month_year_matrix",
        description="Red chilli exports by calendar month and year.",
    ),
    SheetSpec(
        key="balance_sheet",
        keywords=("balance", "sheet"),
        layout="particulars_by_year",
        description=(
            "National red chilli supply/demand balance sheet by calendar "
            "year, in lakh tonnes."
        ),
    ),
    SheetSpec(
        key="guntur_monthly_arrivals",
        keywords=("guntur", "monthly", "arrivals"),
        layout="stacked_month_year_matrix",
        description=(
            "Guntur monthly arrivals, supplied twice: once in bags and once "
            "in the workbook's converted unit."
        ),
    ),
    SheetSpec(
        key="apy",
        keywords=("apy",),
        layout="state_year_metric_matrix",
        description=(
            "Area, production and yield by state and year (APY statistics)."
        ),
    ),
    SheetSpec(
        key="cold_storage_stock",
        keywords=("cold", "storage"),
        layout="sparse_records",
        description=(
            "Reported cold storage stock positions by state/market and month."
        ),
    ),
)


# --------------------------------------------------------------------------
# Column-role recognition keywords
# --------------------------------------------------------------------------

#: Tokens (lower-cased, substring match) used to classify a column's role.
#: Order matters: the first matching role wins, so more specific patterns
#: must precede more general ones.
COLUMN_ROLE_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("date", ("date", "month", "year")),
    ("low", ("low",)),
    ("high", ("high",)),
    ("average", ("avg", "average", "mean")),
    ("difference", ("difference", "diff", "spread")),
    ("offtake", ("offtake", "off take")),
    ("arrivals", ("arrival", "arrivals")),
    ("rate", ("usd", "inr", "rate")),
)

#: Cell values that mean "the market did not trade", not "zero".
#: Treated as missing observations; never imputed (strict data rule 4/5).
NON_NUMERIC_MISSING_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "closed",
        "holiday",
        "na",
        "n/a",
        "nil",
        "-",
        "--",
        "",
        "no trade",
        "notrade",
        "no data",
    }
)

#: Regex that pulls a bag-to-weight conversion out of a column header.
#: Copes with every phrasing the workbook uses: "Arrival(Bags) 1 bag=40 kg",
#: "Arrival(cold storage) 1 bag = 40 kg", " Arrivals bag = 45 Kg" and
#: "Offtake one bag = 45 Kg". Conversions are therefore read from the
#: workbook rather than assumed.
BAG_WEIGHT_PATTERN: Final[str] = (
    r"(?:1|one)?\s*bags?\s*=\s*(\d+(?:\.\d+)?)\s*(?:kgs?|kilograms?)\b"
)

#: Month abbreviations used to interpret matrix-layout sheets.
MONTH_ABBREVIATIONS: Final[tuple[str, ...]] = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)

#: Row labels in matrix sheets that are aggregates, not observations.
AGGREGATE_ROW_LABELS: Final[frozenset[str]] = frozenset(
    {"total", "average", "avg", "mean", "sum", "overall average", "grand total"}
)


# --------------------------------------------------------------------------
# The two headline varieties the brief singles out
# --------------------------------------------------------------------------

#: Recognition keywords for the two focus varieties. The *display* name is
#: always taken from the workbook header; these are only used to locate the
#: matching column.
FOCUS_VARIETY_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "LCA 334": ("lca", "334"),
    "Teja": ("teja",),
}

#: Markets analysed for integration/lead-lag, mapped to their dataset keys.
MARKET_DATASETS: Final[dict[str, str]] = {
    "Guntur": "guntur_variety_prices",
    "Warangal": "warangal",
    "Khammam (cold storage)": "khammam_cold",
    "Khammam (non cold storage)": "khammam_non_cold",
}


# --------------------------------------------------------------------------
# Statistical parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticsConfig:
    """Tunable parameters for the statistical layer.

    These are *method* choices, not data. They are surfaced in the UI's
    assumptions panel so that every number remains reproducible.
    """

    #: Significance level used for every hypothesis test and interval.
    alpha: float = 0.05
    #: Rolling windows offered in the UI, in observations.
    rolling_windows: tuple[int, ...] = (7, 14, 30, 60, 90, 180)
    #: Default rolling window.
    default_rolling_window: int = 30
    #: Maximum lead/lag in periods explored by cross-correlation.
    max_lag: int = 60
    #: Maximum lag tested by Granger causality.
    granger_max_lag: int = 10
    #: Minimum paired observations before a correlation is reported.
    min_obs_correlation: int = 30
    #: Minimum observations before a stationarity test is reported.
    min_obs_stationarity: int = 30
    #: Minimum observations before cointegration is attempted.
    min_obs_cointegration: int = 60
    #: Z-score beyond which an observation is flagged as an outlier.
    outlier_z_threshold: float = 3.0
    #: Number of quantile buckets used for arrival threshold analysis.
    threshold_buckets: int = 5
    #: Seasonal period by resample frequency, in periods per year.
    seasonal_periods: dict[str, int] = field(
        default_factory=lambda: {"D": 5, "W": 52, "SME": 24, "ME": 12}
    )


#: Offset alias used for the fortnightly view. pandas labels ``SME`` buckets by
#: their start boundary whereas ``W`` and ``ME`` label by the end, so
#: :func:`utils.resample_series` implements this frequency itself (half-month
#: periods ending on the 15th and the last day) to keep all three views on the
#: same labelling convention. ``pd.date_range(freq="SME")`` is still used to
#: extend a forecast forward, which is correct once anchored on a boundary.
FORTNIGHT_FREQ: Final[str] = "SME"


@dataclass(frozen=True)
class ForecastConfig:
    """Parameters governing model fitting, backtesting and selection."""

    #: Forecast horizons offered per frequency, expressed in periods, and
    #: chosen so that each reaches six months ahead (strict brief requirement).
    horizons: dict[str, int] = field(
        default_factory=lambda: {"W": 26, "SME": 12, "ME": 6}
    )
    #: Human labels for the frequencies.
    frequency_labels: dict[str, str] = field(
        default_factory=lambda: {
            "W": "Weekly",
            "SME": "Fortnightly",
            "ME": "Monthly",
        }
    )
    #: Confidence level for the mean-forecast (confidence) interval.
    confidence_level: float = 0.80
    #: Confidence level for the wider prediction interval.
    prediction_level: float = 0.95
    #: Number of rolling-origin backtest folds.
    backtest_folds: int = 5
    #: Minimum observations a model needs before it is even attempted.
    min_obs_arima: int = 30
    min_obs_sarima: int = 36
    min_obs_seasonal_cycles: int = 2
    min_obs_var: int = 40
    min_obs_vecm: int = 60
    #: Grids searched for (p, d, q) and seasonal (P, D, Q).
    arima_p_range: tuple[int, ...] = (0, 1, 2)
    arima_d_range: tuple[int, ...] = (0, 1)
    arima_q_range: tuple[int, ...] = (0, 1, 2)
    seasonal_p_range: tuple[int, ...] = (0, 1)
    seasonal_d_range: tuple[int, ...] = (0, 1)
    seasonal_q_range: tuple[int, ...] = (0, 1)
    #: Metric used to pick the winning model during backtesting.
    selection_metric: str = "RMSE"


ANALYTICS = AnalyticsConfig()
FORECAST = ForecastConfig()


# --------------------------------------------------------------------------
# Theming
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """A complete colour palette for both the Qt widgets and the charts."""

    name: str
    window: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_soft: str
    positive: str
    negative: str
    neutral: str
    warning: str
    grid: str
    #: Ordered categorical palette used for multi-series charts.
    series: tuple[str, ...]
    #: Diverging colormap name for correlation heatmaps.
    diverging_cmap: str
    #: Sequential colormap name for intensity heatmaps.
    sequential_cmap: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


DARK_THEME: Final[Theme] = Theme(
    name="dark",
    window="#0d1117",
    surface="#151b23",
    surface_alt="#1c242e",
    border="#2a3440",
    text="#e6edf3",
    text_muted="#8b98a5",
    accent="#f0883e",
    accent_soft="#3d2b1c",
    positive="#3fb950",
    negative="#f85149",
    neutral="#58a6ff",
    warning="#d29922",
    grid="#232c37",
    series=(
        "#f0883e",  # amber   - primary / Teja
        "#58a6ff",  # blue    - LCA 334
        "#3fb950",  # green
        "#bc8cff",  # violet
        "#f85149",  # red
        "#39c5cf",  # cyan
        "#d29922",  # gold
        "#ff7b72",  # salmon
        "#7ee787",  # mint
        "#a5d6ff",  # sky
    ),
    diverging_cmap="RdYlBu_r",
    sequential_cmap="inferno",
)

LIGHT_THEME: Final[Theme] = Theme(
    name="light",
    window="#f4f6f8",
    surface="#ffffff",
    surface_alt="#eef1f5",
    border="#d5dbe2",
    text="#1a2027",
    text_muted="#5c6b7a",
    accent="#c2410c",
    accent_soft="#fdebe0",
    positive="#15803d",
    negative="#b91c1c",
    neutral="#1d4ed8",
    warning="#a16207",
    grid="#e3e8ee",
    series=(
        "#c2410c",
        "#1d4ed8",
        "#15803d",
        "#7c3aed",
        "#b91c1c",
        "#0e7490",
        "#a16207",
        "#be123c",
        "#4d7c0f",
        "#0369a1",
    ),
    diverging_cmap="RdYlBu_r",
    sequential_cmap="viridis",
)

THEMES: Final[dict[str, Theme]] = {"dark": DARK_THEME, "light": LIGHT_THEME}
DEFAULT_THEME: Final[str] = "dark"


# --------------------------------------------------------------------------
# UI layout constants
# --------------------------------------------------------------------------

WINDOW_MIN_WIDTH: Final[int] = 1280
WINDOW_MIN_HEIGHT: Final[int] = 800
WINDOW_DEFAULT_WIDTH: Final[int] = 1680
WINDOW_DEFAULT_HEIGHT: Final[int] = 980
SIDEBAR_WIDTH: Final[int] = 250
CHART_DPI: Final[int] = 110
CHART_EXPORT_DPI: Final[int] = 300

#: Navigation entries: (page key, display label, unicode glyph).
#: The page order mirrors the analytical narrative in the brief.
NAV_ITEMS: Final[tuple[tuple[str, str, str], ...]] = (
    ("executive", "Executive Summary", "◉"),
    ("overview", "Market Overview", "▦"),
    ("price", "Price Analysis", "↗"),
    ("arrivals", "Arrival Analysis", "▤"),
    ("integration", "Market Integration", "⬡"),
    ("correlation", "Correlation Studio", "▩"),
    ("seasonality", "Seasonality", "❂"),
    ("exports", "Export Analysis", "✈"),
    ("balance", "Balance Sheet", "⚖"),
    ("forecast", "Forecast Center", "▶"),
    ("insights", "Automated Insights", "✦"),
    ("dictionary", "Data Dictionary", "☷"),
)
