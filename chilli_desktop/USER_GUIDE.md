# User Guide — Chilli Intelligence Desktop

This guide explains how to drive the application and, more importantly, how to
read what it shows you. It assumes no statistical background; the technical
detail behind the forecasts is in
[FORECAST_METHODOLOGY.md](FORECAST_METHODOLOGY.md).

---

## Contents

1. [The window](#1-the-window)
2. [Global filters](#2-global-filters)
3. [Working with charts](#3-working-with-charts)
4. [Working with tables](#4-working-with-tables)
5. [Reading the pages](#5-reading-the-pages)
6. [Reading the statistics](#6-reading-the-statistics)
7. [Exporting for a presentation](#7-exporting-for-a-presentation)
8. [Common workflows](#8-common-workflows)

---

## 1. The window

```
┌──────────────┬──────────────────────────────────────────────────────┐
│              │  Page title and subtitle          Latest observation │
│  Navigation  ├──────────────────────────────────────────────────────┤
│  (12 pages)  │                                                      │
│              │  Scrollable page content:                            │
│  ──────────  │    summary cards, charts, tables, notes              │
│              │                                                      │
│  Global      │                                                      │
│  Filters     │                                                      │
│              │                                                      │
│  ──────────  │                                                      │
│  Theme       │                                                      │
│  Reload      │                                                      │
├──────────────┴──────────────────────────────────────────────────────┤
│ Status · active filters · data-quality count │ progress │ workbook  │
└─────────────────────────────────────────────────────────────────────┘
```

**Status bar.** Left: the current filter selection and how many data-quality
notes are outstanding. Centre-right: a progress bar during background work.
Right: the workbook filename, how many sheets were mapped, and the read time.

**Header.** The right-hand corner always shows the latest observation date
anywhere in the workbook and when the workbook was read. If a colleague updates
the file while the application is open, press **Reload workbook**.

**Theme.** The button at the bottom of the sidebar switches between dark and
light. Dark suits a trading desk; light suits a projector. Chart exports are
always white-background regardless.

**Resizing.** The window is fully resizable and every chart re-lays out. For
presentation, maximise before you begin.

---

## 2. Global filters

Filters live in the sidebar and apply to **every** page. Nothing is applied
until you press **Apply**.

| Control | Effect |
|---|---|
| **Date range** | Restricts every series to this window. The two fields are bounded by the workbook's own span. |
| **1Y / 3Y / 5Y / All** | Presets. Apply immediately. |
| **Analysis frequency** | Daily, Weekly, Fortnightly or Monthly. Determines the resampling of every chart and statistic. See the note below. |
| **Varieties** | Multi-select. Empty means all. The first selection is treated as the "target" variety on the single-variety pages. |
| **Market** | Restricts market-level panels. |
| **Season (months)** | Multi-select calendar months. Use it to isolate, say, harvest-month behaviour. |
| **Peak / Lean / Clear** | Selects the months the workbook's **own arrivals data** classifies as peak or lean. Not a hardcoded crop calendar. |
| **Price range** | Excludes observations outside the band. Applies only to price series. |
| **Arrival range** | Excludes observations outside the band. Applies only to arrival series. |
| **Apply / Reset** | Apply rebuilds the current page; Reset restores the full history. |

### Choosing a frequency

| Frequency | Use it for | Watch out for |
|---|---|---|
| **Daily** | Lead-lag timing, day-to-day volatility | Noisy; seasonal models will not fit |
| **Weekly** | The general-purpose default | Seasonal period is 52, too long for SARIMA |
| **Fortnightly** | Procurement planning cadence | Halves the sample relative to weekly |
| **Monthly** | Seasonality, exports, balance-sheet work, the most reliable forecasts | Hides intra-month moves |

Fortnightly periods run 1st–15th and 16th–month-end, labelled by the period
**end** — the same convention as weekly and monthly, so the three views are
directly comparable.

> **The Market Integration page always uses daily data**, whatever the sidebar
> says. Averaging to weeks would erase the one- or two-day leads that page
> exists to measure. The page states this at the top.

### Filters never create data

A filter can only narrow. If a filter leaves a panel with too few observations,
the panel says so and tells you the count — it does not quietly widen the
window or pad the series.

---

## 3. Working with charts

Every chart sits in a panel with the same furniture:

```
┌─────────────────────────────────────────────────────────┐
│ Chart title                      [PNG] [PDF] [Notes(3)] │
│                                                         │
│                    the chart                            │
│                                                         │
│ 🏠 ← → ✥ 🔍 ⚙                                          │
│ Source: Guntur Varietywise daily price                  │
│ •  assumption text (when Notes is toggled on)           │
└─────────────────────────────────────────────────────────┘
```

| Control | Action |
|---|---|
| **Hover** | Shows the nearest data point: series name, date and exact value |
| **🏠 Home** | Resets zoom and pan to the full view |
| **← →** | Steps back and forward through your zoom history |
| **✥ Pan** | Click to arm, then drag to pan; right-drag zooms an axis |
| **🔍 Zoom** | Click to arm, then drag a rectangle |
| **⚙ Subplots** | Adjusts margins — useful when long labels are clipped |
| **PNG** | 300 dpi raster export |
| **PDF** | Vector export |
| **Notes (n)** | Toggles the assumptions and caveats behind this chart |

**Always read the Source caption.** It names the workbook sheet the chart was
built from. When a chart combines sheets, all of them are listed.

**Always open Notes on a chart you intend to present.** That is where the
sample size, the interpolation count, the significance caveats and the unit
statements live.

### The unavailable state

A panel that cannot be drawn shows a dashed box:

> **Data not available in uploaded workbook.**
>
> Only 6 paired observation(s) overlap; a scatter plot needs at least 3.

The second paragraph is always the specific reason. This is a deliberate,
informative state — not a bug.

---

## 4. Working with tables

- **CSV** exports the table exactly as displayed, index included.
- Numbers are right-aligned; text is left-aligned.
- Signed columns (change, correlation, elasticity, beta) are **green when
  positive, red when negative**.
- p-values below 0.05 are highlighted in the accent colour.
- Boolean columns read Yes/No, green for Yes.
- The winning model in the Forecast Center's scoreboard is marked **SELECTED**
  in the accent colour.
- A dash (—) means no value exists. It never means zero.
- Each table carries the same Source caption and note bullets as charts.

---

## 5. Reading the pages

### 1 · Executive Summary

Start here. The four cards give the latest quote per variety with week-,
month- and year-on-year change. The sentiment gauge is a composite of five
components — trend, momentum, range position, arrivals pressure and seasonal
setup — each scored between −1 and +1 and listed individually in the table
beneath. **Read the components, not just the needle:** a "strongly bullish"
reading driven only by range position is a very different situation from one
where all five agree.

The forecast summary and the key insights populate a few seconds after the page
opens; both run in the background.

### 2 · Market Overview

Prices and arrivals for every market, plus the coverage table — which is worth
a look before you trust anything else, because it shows exactly how far each
sheet reaches. Note the tonnage table: bag weights differ between markets, so
raw bag counts are not comparable and this table is the conversion.

### 3 · Price Analysis

The single-variety deep dive. Moving averages with ±2σ bands, annualised
rolling volatility, a rebased relative-performance chart (all varieties indexed
to 100 at the window start — this is the quickest way to see which variety has
actually outperformed), STL decomposition, and the diagnostics that determine
how the series must be modelled.

The **outlier register** lists flagged observations. Nothing is ever removed
from any calculation; the register is for you to check against the source sheet.

### 4 · Arrival Analysis

The supply-pressure page. Read it in this order:

1. **Price against arrivals** — the visual relationship.
2. **Elasticity by lag** — the percentage price response to a 1% arrivals
   change, at lags 0 to 4.
3. **Threshold effects** — arrivals split into quintiles, with the average
   next-period price change in each. This answers "above what level do prices
   come under pressure?"
4. **Lagged impact** — how long the effect takes to appear.

On the current workbook the elasticity is small and the quintile difference is
not statistically significant, which is itself the finding: **arrivals alone
are not a reliable price signal here.** The page says so.

### 5 · Market Integration

The influence diagram is the summary: node size and colour follow the
leadership score, solid arrows are one-way influence, dashed double-headed
arrows are two-way feedback.

Beneath it:

- **Pairwise direction of influence** — Granger causality run *both* ways for
  every pair, classified One-way / Feedback / Independent. Only one-way
  relationships are evidence of leadership.
- **Leadership ranking** — the composite score and its inputs.
- **Lead-lag matrix** — peak cross-correlation in periods and days.
- **Cointegration** — Johansen for the system, Engle-Granger for each pair.

**How to read cointegration:** if the markets are cointegrated they share a
long-run equilibrium and cannot drift apart permanently, so an unusually wide
spread is a temporary dislocation rather than a new level. That is the
statistical definition of an integrated market, and it is what makes the
spread-to-Guntur chart tradeable.

### 6 · Correlation Studio

Pearson (linear co-movement) and Spearman (rank agreement, robust to outliers)
side by side, then **the workbook's own matrix**, then a **reconciliation
table** between the two. Small differences are expected — the application
correlates the filtered window at the selected frequency, the workbook used its
own fixed sample. Large differences are worth investigating.

The rolling-correlation panels show whether a relationship holds through time.
A correlation that flips sign cannot be relied on for positioning.

The cross-correlation stems are the lag explorer: positive lag means the first
series leads. Both series are differenced first, because correlating two
trending price levels produces a flat near-1.0 curve whose peak means nothing.

### 7 · Seasonality

Seasonal index by month (month mean ÷ overall mean), the distribution of
monthly averages across years, the calendar heatmap, and a direct comparison
against the workbook's own seasonality sheet.

**Read the boxplot alongside the index.** A month with a high index but a very
wide box is not dependable.

Harvest and lean months are **derived** by ranking calendar months on the
workbook's arrivals data, not taken from an external crop calendar.

Festival effects are explicitly not computable — the workbook has no festival
dates, and those dates move through the Gregorian year.

### 8 · Export Analysis

Export trend, calendar grid, annual totals (partial years hatched — do not
compare them with complete years), then the relationship with price: scatter,
correlation, cross-correlation, rolling correlation and a Granger test. The
currency channel is at the bottom, with the USD/INR coverage gap flagged.

### 9 · Balance Sheet

The balance sheet as supplied, stacked supply and demand, ending stock and
stock-to-use, and stock-to-use against annual average price.

**Two cautions.** Years the workbook marks `(exp)` are the *workbook's own*
expectations, not this application's forecasts; they are hatched in charts and
labelled in tables. And with only ten annual observations, the
stock-to-use-versus-price relationship is a directional observation, not a
finding — the page states this.

Cold-storage stock is shown for reference with an explicit statement that
inventory-versus-price analysis cannot be performed on 6 sparse months.

### 10 · Forecast Center

The flagship page.

**Controls:** variety, frequency, horizon (defaults reach six months), training
window, and which models to attempt. Press **Run model sweep**.

**Then read in this order:**

1. **Model selection** — the plain-language rationale: which model won, on what
   metric, by how much, and whether the margin was meaningful.
2. **Model comparison** — the scoreboard. Lower RMSE/MAE/MAPE is better; higher
   R² and directional accuracy are better. Models that could not be applied are
   listed with the reason.
3. **The fan chart** — solid white is actual history, dashed amber is the
   projection, the dotted vertical line is the forecast origin. The inner band
   is the 80% confidence interval, the outer the 95% prediction interval.
4. **Forecast table** — every row is a projection. None is historical.
5. **Forecast explanation** — direction, the structure of the history, the
   significant drivers, the model's track record and the planning range.
6. **Assumptions** — read before acting.
7. **Decomposition, driver attribution, VIF, stationarity** — the supporting
   evidence.

**Plan against the prediction interval, not the line.** The 95% band is
calibrated against the error the model actually made in backtesting, which is
what makes it the honest planning range.

**Watch for a negative R².** It means that over the backtest windows a flat
line at the mean of the held-out data would have scored better. The level
forecast may still be the best available, but confidence in it should be low —
and the page tells you when this happens.

### 11 · Automated Insights

Every finding, filterable by strength (Strong / Moderate / Weak /
Informational / Data gap) and by category. Each card carries the statistic
behind it, the source sheet, and an expandable **Evidence** block.

**Do not skip the DATA GAP cards.** They record what the workbook cannot answer
— which is exactly the information you need before promising an analysis to
someone.

### 12 · Data Dictionary

Generated from the workbook at runtime: every sheet, every field, its inferred
role, type, population rate, range and unit. Also lists the units and
conversions parsed from sheet headers, any unmapped sheets, and all
data-quality notes. **Export data dictionary as Markdown** writes it to a file.

---

## 6. Reading the statistics

### Correlation

| \|r\| | Description |
|---|---|
| ≥ 0.90 | very strong |
| 0.70–0.90 | strong |
| 0.50–0.70 | moderate |
| 0.30–0.50 | weak |
| < 0.30 | negligible |

Correlation is not causation, and correlation between two trending series is
mostly shared trend. That is why the direction tests are run on differences.

### p-values

A p-value below 0.05 means the pattern is unlikely to be chance alone. Above
it, treat the effect as unproven. p-values are shown to three decimals and
values under 0.001 print as `<0.001`.

### Stationarity (ADF and KPSS)

The two tests have opposite null hypotheses, so reading them together is more
informative than either alone:

| ADF | KPSS | Meaning |
|---|---|---|
| rejects | fails to reject | Stationary — can be modelled in levels |
| fails to reject | rejects | Unit root — must be differenced |
| rejects | rejects | Trend-stationary |
| fails to reject | fails to reject | Inconclusive; too little information |

### Granger causality

"X Granger-causes Y" means past values of X improve the prediction of Y. It is
**predictive precedence, not proof of a mechanism**. Read it alongside the
direction test: X→Y significant while Y→X is not is much stronger evidence than
both being significant.

### Elasticity

Estimated in logs, so the coefficient reads directly: an elasticity of −0.15
means a 1% rise in arrivals is associated with a 0.15% lower price.

### Directional accuracy

The share of backtest steps where the model called up-or-down correctly from
the last observed value. Above ~55% is useful; near 50% is a coin toss, and the
page says so when that is the case.

---

## 7. Exporting for a presentation

| Need | Do this |
|---|---|
| A chart for a slide | Panel **PNG** (300 dpi) |
| A chart for a print report | Panel **PDF** (vector) |
| Numbers for a spreadsheet | Table **CSV** |
| The full field reference | Data Dictionary → Export as Markdown |

Everything is written to `exports/` in the project root by default.

Exports are rendered on a white background whichever theme is on screen, so a
dark-theme session still produces report-ready output.

**Before presenting a chart, open its Notes.** The caveats belong on the slide
or in the speaker notes — particularly sample size, the USD/INR gap, and
whether the final period is partial.

---

## 8. Common workflows

### "What is happening in Teja right now?"

Executive Summary → read the Teja card, then the sentiment components table.
Then Price Analysis with frequency **Weekly** for the trend and volatility
context.

### "Should I buy now or wait?"

1. Seasonality → is the month ahead seasonally firm or soft?
2. Arrival Analysis → are arrivals running above or below the seasonal norm?
3. Forecast Center → monthly forecast, and read the **prediction interval**.
4. Balance Sheet → is the stock-to-use buffer thin or comfortable?

Treat a bullish signal as strong only when several of those agree.

### "Is the Warangal–Guntur spread worth trading?"

Market Integration → the spread-to-Guntur chart for the current level, then the
cointegration result. Cointegrated means the spread mean-reverts and an extreme
reading is a dislocation. Not cointegrated means the spread has no anchor and
can keep widening.

### "Which variety should I watch as the market's bellwether?"

Correlation Studio → the variety with the highest average correlation with the
rest of the complex is the best single proxy. Automated Insights states it
directly, and Market Integration's leadership ranking gives the directional
evidence.

### "How much do exports actually matter?"

Export Analysis → the correlation, then the **rolling** correlation. If the
rolling line changes sign across the sample, the headline correlation is not a
stable relationship and should not carry weight in a decision.

### "What can this workbook not tell me?"

Automated Insights → filter to **Data gaps only**. Then Data Dictionary for the
field-level coverage. Do this before committing to an analysis for someone else.
