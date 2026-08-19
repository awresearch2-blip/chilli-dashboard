# Chilli Intelligence Desktop

An interactive **PySide6 desktop application** for the Indian red chilli market,
built for traders, exporters, processors, cold-storage operators, analysts,
banks and management.

Every number, chart, statistic and forecast in the application is derived at
runtime from a single Excel workbook — `Chilli mastersheet for dashboard.xlsx`.
Nothing is fetched from the internet, nothing is hardcoded, and nothing is
estimated to fill a gap. Where the workbook cannot support an analysis, the
application says so in those words:

> Data not available in uploaded workbook.

---

## Contents

- [Quick start](#quick-start)
- [What the application answers](#what-the-application-answers)
- [Pages](#pages)
- [Architecture](#architecture)
- [Data rules and how they are enforced](#data-rules-and-how-they-are-enforced)
- [Known limitations of this workbook](#known-limitations-of-this-workbook)
- [Command-line options](#command-line-options)
- [Troubleshooting](#troubleshooting)
- [Further documentation](#further-documentation)

---

## Quick start

### 1. Requirements

- Python **3.11 or newer** (developed and verified on 3.13.3, Windows 11)
- Roughly 400 MB of disk space for the dependencies
- A desktop session — this is a native windowed application, not a web app

### 2. Create a virtual environment and install

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\python.exe -m pip install -r chilli_desktop\requirements.txt
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r chilli_desktop/requirements.txt
```

### 3. Place the workbook

The application searches these locations in order and uses the first that
exists:

1. the path in the `CHILLI_WORKBOOK` environment variable
2. `~/OneDrive/Desktop/Desktop/Shahaniya/AgriWatch/chilli dashboard/Chilli mastersheet for dashboard.xlsx`
3. `Chilli mastersheet for dashboard.xlsx` in the project root
4. the same filename inside the `chilli_desktop/` package directory

### 4. Run

```bash
.venv\Scripts\python.exe -m chilli_desktop.main
```

Or point it at a specific workbook and start in the light theme:

```bash
.venv\Scripts\python.exe -m chilli_desktop.main --workbook "D:\data\chilli master.xlsx" --theme light
```

A splash screen appears while the workbook is read (about 3–5 seconds for the
current file), then the main window opens on the Executive Summary.

---

## What the application answers

| Question | Where |
|---|---|
| Which variety leads price discovery? | Correlation Studio, Automated Insights |
| Does Guntur influence Warangal and Khammam? | Market Integration |
| What are the lead-lag relationships, in days? | Market Integration, Correlation Studio |
| How do arrivals affect prices — immediately, with a lag, above a threshold? | Arrival Analysis |
| What drives LCA 334 and Teja? | Forecast Center (driver attribution) |
| What are prices likely to do over the next week, fortnight, month and six months? | Forecast Center |
| How reliable is that forecast? | Forecast Center (backtest scoreboard) |
| What are the key risks and opportunities? | Executive Summary, Automated Insights |
| What does the workbook *not* support? | Automated Insights (DATA GAP items), Data Dictionary |

---

## Pages

| # | Page | Contents |
|---|---|---|
| 1 | **Executive Summary** | Latest prices with WoW/MoM/QoQ/YoY change, composite sentiment gauge and its five components, price history, supply and trade cards, forecast summary for both focus varieties, strongest insights |
| 2 | **Market Overview** | Prices and arrivals across all markets, arrivals converted to tonnes using each sheet's own bag weight, full dataset-coverage table |
| 3 | **Price Analysis** | Moving averages and ±2σ bands, rolling volatility, variety comparison, rebased relative performance, descriptive statistics, change by horizon, STL decomposition, ADF/KPSS, ACF/PACF, outlier register |
| 4 | **Arrival Analysis** | Price against arrivals, arrivals against offtake, scatter with fit, elasticity by lag, quintile threshold analysis, lagged-impact profile, monthly arrivals calendar, derived season classification |
| 5 | **Market Integration** | Influence diagram, pairwise direction of influence, leadership ranking, lead-lag matrix in periods and days, Johansen and Engle-Granger cointegration, spread-to-Guntur chart |
| 6 | **Correlation Studio** | Pearson and Spearman heatmaps, the workbook's own matrix side by side, a reconciliation table between the two, rolling correlations, cross-correlation lag explorer, driver VIF |
| 7 | **Seasonality** | Seasonal index by month, distribution boxplots, calendar heatmap, comparison against the workbook's own seasonality sheet, harvest/lean classification, day-of-week statistics |
| 8 | **Export Analysis** | Export trend, calendar heatmap, annual totals with partial years flagged, exports against price, cross-correlation and lead-lag, rolling correlation, Granger test, USD/INR channel |
| 9 | **Balance Sheet** | The balance sheet as supplied, stacked supply and demand build-up, ending stock and stock-to-use, stock-to-use against annual price, APY by state and year, cold-storage stock |
| 10 | **Forecast Center** | Model sweep across ARIMA / SARIMA / SARIMAX / Holt-Winters / VAR / VECM, backtest scoreboard, selection rationale, fan chart with both intervals, forecast table, decomposition, driver attribution, VIF, stationarity, assumptions |
| 11 | **Automated Insights** | Every finding the workbook supports, filterable by strength and category, each with its evidence and source sheet |
| 12 | **Data Dictionary** | Auto-generated field-level dictionary, coverage by dataset, units and conversions read from the workbook, unmapped sheets, data-quality notes, Markdown export |

---

## Architecture

```
chilli_desktop/
├── main.py            entry point, CLI, splash, exception hook
├── ui.py              window shell, navigation, global filters, all 12 pages
├── data_loader.py     reads the workbook once; one parser per sheet layout
├── preprocessing.py   DataService: canonical series, filtering, resampling, caching
├── analytics.py       every statistical test, decomposition and relationship measure
├── forecasting.py     model fitting, rolling-origin backtesting, selection, explanation
├── charts.py          interactive matplotlib canvas embedded in Qt
├── insights.py        automated narrative generation
├── settings.py        configuration, sheet registry, themes (no market data)
├── utils.py           Result type, logging, coercion, formatting
├── requirements.txt
├── README.md
├── USER_GUIDE.md
├── FORECAST_METHODOLOGY.md
└── docs/
    └── DATA_DICTIONARY.md   (generated)
```

### Key design decisions

**`Result[T]` everywhere.** Every analytical function returns a `Result`
carrying *either* a value *or* a machine-readable reason it is unavailable,
plus the source sheet and any assumptions. This is what lets the UI honour the
"Data not available in uploaded workbook." rule without any function ever
inventing a fallback number.

**The workbook is read once.** `data_loader.load_workbook` caches by resolved
path. `DataService` memoises every derived series, resample and panel on top of
that. Cached objects are handed out as **copies**, so one page cannot corrupt
another's data by reindexing in place.

**Sheets are matched by keyword, not by exact name.** The workbook names sheets
inconsistently (`Khammam Teja non cold storage` beside
`KhammamTejacoldstorage`). Specs are resolved most-specific-first so the
non-cold sheet is claimed before the generic `khammam`+`cold` spec looks for
one.

**Background threads for slow work.** The forecast sweep, the insight sweep and
the daily leadership ranking run on a `QThreadPool`. Workers return plain
objects and never touch widgets; pages render results on the main thread.

**Pages are lazily built.** A page's widgets are created the first time it is
shown and rebuilt only when filters or the theme change.

---

## Data rules and how they are enforced

| Rule | Enforcement |
|---|---|
| Use only the uploaded workbook | The only I/O in the application is `pd.read_excel`. There is no network code and no other data file is read. |
| Never fabricate or estimate | `utils.to_number` maps `"Closed"`, `"NA"` and blanks to `NaN`, never to zero. `clean_series` drops missing values and never interpolates. |
| Every chart names its source sheet | `ChartPanel.finish()` requires a source string and renders it as a caption; `DataTable` does the same. |
| Assumptions shown separately | Each panel has a **Notes** button listing the assumptions behind it; the Forecast Center has a dedicated assumptions block. |
| Historical vs projected distinguished | Forecasts are dashed with a labelled forecast-origin divider. Workbook-supplied `(exp)` years are hatched in bar charts and labelled in tables. |
| Reproducible | Every statistic states its sample size, window and method. The Correlation Studio includes a reconciliation table against the workbook's own matrix. |
| No hardcoded values | `settings.py` contains configuration and recognition keywords only. Unit conversions (45 kg/bag at Guntur, 40 kg/bag elsewhere) are parsed from the sheet headers with a regex; the second Guntur arrivals block's conversion factor is *measured* from overlapping cells rather than assumed. |

The one place interpolation happens is STL/classical decomposition, which
requires an evenly spaced grid. It is confined to that calculation, the number
of interpolated periods is printed in the panel's notes, and it never touches
any other statistic.

---

## Known limitations of this workbook

These are properties of the current file, surfaced by the application rather
than hidden:

1. **USD/INR has a 1,119-day hole** (15 May 2019 → 7 June 2022; 2020 and 2021
   are absent entirely, 2019 has 90 days). Any statistic spanning it joins two
   disconnected periods. Flagged on the Export Analysis page, in the data-quality
   notes and as a DATA GAP insight.
2. **Cold-storage stock has 6 reporting months across 6 locations**, at most 3
   observations for any one location. Inventory-versus-price, delayed storage
   effects and seasonal storage behaviour therefore **cannot** be computed. The
   Khammam cold-versus-fresh price premium *can* be, and is.
3. **No festival calendar.** Diwali and Sankranti move through the Gregorian
   year, so festival demand cannot be separated from harvest timing. Stated
   explicitly on the Seasonality page.
4. **The export sheet states no unit.** Values are used as supplied and
   compared only against themselves.
5. **Bag weights differ by market** — 45 kg at Guntur, 40 kg at Warangal and
   Khammam — so raw bag counts are not comparable across markets. Market
   Overview provides the tonnage conversion.
6. **The balance sheet has 10 annual observations** and APY has 12. That is
   enough for description and a directional correlation, not for regression or
   a significance test, and the application says so wherever those tables appear.
7. **The final period is usually partial.** The workbook ends 20 July 2026, so
   the July monthly average covers 20 of 31 days. Disclosed wherever a
   resampled series is used.
8. **Market prices move same-day.** Every pairwise cross-correlation peaks at
   lag 0, so there is no exploitable timing lead between these mandis.
   Leadership is therefore established from the *direction* of predictive power
   rather than from timing.

---

## Command-line options

```
python -m chilli_desktop.main [options]

  --workbook PATH                 Full path to the master workbook.
  --theme {dark,light}            Colour theme at start-up (default: dark).
  --log-level {DEBUG,INFO,WARNING,ERROR}
                                  Console verbosity (default: INFO).
  --export-data-dictionary PATH   Write the data dictionary to PATH as
                                  Markdown and exit without opening a window.
```

Generate the data dictionary without launching the UI:

```bash
.venv\Scripts\python.exe -m chilli_desktop.main --export-data-dictionary chilli_desktop\docs\DATA_DICTIONARY.md
```

---

## Troubleshooting

**"The master workbook could not be read."**
The dialog lists every path that was tried. Either move the file to one of
them or set the environment variable:

```bash
set CHILLI_WORKBOOK=D:\data\Chilli mastersheet for dashboard.xlsx
```

**All text renders as empty boxes (□).**
The Qt offscreen platform plugin has no font database. Unset
`QT_QPA_PLATFORM` so the native plugin is used.

**The window opens but a page shows a red error block.**
The page caught an exception and reported it rather than crashing the
application. The traceback is in `logs/chilli_desktop_<date>.log`. Other pages
remain usable.

**A forecast run takes a long time.**
Monthly runs take a few seconds; weekly runs are slower because the series is
four times longer. Deselect models in the Forecast Center's Models list to
narrow the sweep. Progress is reported in the status bar throughout.

**Charts are unreadable when pasted into a report.**
Exports are always rendered on a white background regardless of the on-screen
theme. Use the panel's **PDF** button for vector output, **PNG** for 300 dpi
raster.

---

## Further documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — how to drive each page, what every
  control does, and how to read the statistics.
- **[FORECAST_METHODOLOGY.md](FORECAST_METHODOLOGY.md)** — model
  specifications, order selection, backtesting protocol, interval
  construction, selection rule and known caveats.
- **`docs/DATA_DICTIONARY.md`** — generated from the workbook; regenerate with
  `--export-data-dictionary` or from the Data Dictionary page.
