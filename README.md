> **Looking for the current dashboard?** The desktop app (`python -m
> chilli_desktop.main`) and its browser counterpart (`chilli_web`, see
> [chilli_web/README.md](chilli_web/README.md)) are the active, supported
> applications for this workbook. Everything below documents an earlier
> Streamlit-based pipeline/dashboard phase (`ingestion/`, `pipeline/`,
> `analytics/`, `dashboard/`) kept here for reference.

# Chilli Intelligence Platform — Phase 1 + Phase 2a/2b/2c + Phase 3a/3b + Phase 4a

Foundation layer for a long-lived commodity intelligence platform built on
`Chilli mastersheet for dashboard.xlsx`. Phase 1 covers ingestion, file
monitoring, validation, and cleaning. Phase 2a adds the core analytics
engine (Price, Arrival, Price-vs-Arrivals, Seasonality) for LCA334 ("334")
and Teja in the Guntur market. Phase 2b adds the relational modules: Variety
Analysis, Market Comparison, Export Analysis, Balance Sheet, USD/INR, and
Cold Storage. Phase 2c adds transparent composite scoring (Market Strength,
Supply/Arrival Pressure, Demand, Price Stability, Bullish/Bearish, Risk,
Confidence, Composite Commodity Index) on top of everything above. Phase 3a
adds the forecasting engine: real model comparison (Seasonal Naive,
Holt-Winters, SARIMA, XGBoost) with walk-forward backtesting and automatic
per-horizon model selection, for the brief's full 7/14/30/60/90/120/180-day
horizon set. Phase 3b adds explainability on top of that: a model-specific
explanation for every forecast, key drivers pulled from Phase 2c's own
signals, real historical analogs, an empirical bullish/bearish probability,
a deterministic risk checklist, and a historical accuracy tracker that
starts honestly empty. Phase 4a adds the interactive Streamlit dashboard —
six views covering every module above, with variety/market/date-range/
forecast-horizon filters, interactive Plotly charts, and a manual Refresh
button — see "What's not here yet" below for what's still to come.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running

```bash
# one refresh cycle, then exit
.venv\Scripts\python run_refresh.py --once

# watch the workbook and refresh automatically on every save
.venv\Scripts\python run_refresh.py --watch

# watch, plus a fallback refresh every N seconds regardless of file events
.venv\Scripts\python run_refresh.py --watch --interval 3600

# launch the interactive dashboard (separate process/terminal from the above)
.venv\Scripts\streamlit run run_dashboard.py
```

The dashboard is a pure read layer over `data/clean/`, `data/analytical/`,
and `data/forecasts/` — it never runs the watchdog observer itself (that
doesn't compose cleanly with Streamlit's own script-rerun execution model).
Run `run_refresh.py --watch` in a separate terminal for continuous
background updates, or just use the dashboard's own "\U0001f504 Refresh
Now" button (sidebar) to trigger a refresh on demand — both write to the
same files, and the dashboard's cache is keyed on each file's modification
time, so it always reflects whichever process refreshed most recently.

## What a refresh produces

- `data/raw/latest/*.pkl` — every configured sheet, reshaped into tidy long
  format but with values exactly as read (no type coercion, no cleanup).
- `data/clean/latest/*.parquet` — the same sheets after validation-driven,
  logged, deterministic cleaning. Missing values are `NaN`, never imputed.
- `logs/data_quality/<refresh_id>.json` (+ `latest.json`) — the Data Quality
  Report: missing counts, unresolved text contamination, negative values on
  fields that should never be negative, duplicate rows, unconfigured/skipped
  sheets, and stale-formula-cache warnings.
- `logs/cleaning_log.jsonl` — one line per cleaning action ever taken,
  appended across refreshes, with the original value.
- `logs/pipeline.log` — rotating operational log.
- `data/analytical/*.json` — eleven analytics modules (Phase 2a:
  `price_analysis`, `arrival_analysis`, `price_vs_arrivals`, `seasonality`;
  Phase 2b: `variety_analysis`, `market_comparison`, `export_analysis`,
  `balance_sheet`, `fx_analysis`, `cold_storage`; Phase 2c:
  `composite_indices`), recomputed every refresh from `data/clean/latest/`.
  Every metric that needs a minimum sample size (a correlation, a rolling
  window, a CAGR) reports `"status": "insufficient_evidence"` instead of a
  number when there isn't enough real data — never a guess.
  `composite_indices.json` is the one exception to "computed straight from
  the workbook" — it's transparent scoring *on top of* the other ten
  modules' outputs; every score there includes a `components`/`signals`
  breakdown, and the module's docstring spells out every formula and bound
  used (see `analytics/composite_index_module.py`).
- `data/forecasts/{teja,334}_forecast.json` — the forecasting engine's
  output: a point forecast + confidence interval for each of the 7 required
  horizons (7/14/30/60/90/120/180 days), the model actually selected for
  that specific horizon (backtested, not hardcoded), that model's real
  backtested accuracy (RMSE/MAE/MAPE/SMAPE), and (Phase 3b) per horizon: a
  `model_explanation` in that model's own real terms, a `probability`
  (bullish/bearish %, empirical from that model+horizon's own backtest
  residuals), `similar_historical_periods` (real nearest-neighbor analog
  dates and what actually happened after them), and `risks_and_limitations`
  (a deterministic checklist). Each variety's file also gets a top-level
  `key_drivers` pulled from Phase 2c's own composite signals. Model training
  only re-runs when the clean price data's latest date has actually
  advanced since the last run (tracked in
  `data/forecasts/_last_trained_through.json`) — a `--watch` save that
  added no new price rows just skips it.
- `data/forecasts/forecast_log.jsonl` — every forecast ever actually
  produced (not skipped by the gate), append-only, with its target date.
- `data/forecasts/realized_accuracy.json` — scores logged forecasts whose
  target date has passed against the real realized price, recomputed on
  *every* refresh regardless of the retrain gate. On a freshly-started
  system every target date is still in the future, so this correctly and
  honestly reports `insufficient_evidence` — it starts filling in only as
  the system keeps running and forecasts' target dates actually arrive.

## Architecture

```
config/sheets.yaml       declarative parsing spec, one entry per known sheet
ingestion/                workbook_reader (safe open + retries), sheet_parser
                          (reshape per spec), watcher (watchdog, debounced)
validation/                observes only: rules.py (checks), quality_report.py
cleaning/                  cleaners.py -- deterministic-only, every action logged
pipeline/refresh.py        orchestrates discover -> parse -> validate -> clean
                          -> persist -> report; the single entrypoint reused
                          by the CLI, the watcher, and the dashboard's manual
                          Refresh button
data/{raw,clean,analytical,forecasts}  four dataset tiers
analytics/                 timeseries_stats.py (generic, unit-tested statistical
                          primitives: CAGR, SMA/EMA, drawdown, momentum, trend
                          strength, percentiles, correlations, rolling z-score,
                          lag correlogram, log-log elasticity, seasonal index)
                          + one module per analytics area (price, arrival,
                          price_vs_arrivals, seasonality, variety,
                          market_comparison, export, balance_sheet, fx,
                          cold_storage) + composite_index_module.py (scoring
                          on top of the above, in-memory composition, no new
                          data reads) + run_analytics.py orchestrator
forecasting/               feature_engineering.py (no-leakage lag/rolling/
                          calendar features) + models/ (seasonal_naive,
                          holt_winters, sarima, xgboost_direct) + backtest.py
                          (walk-forward validation) + model_selection.py
                          (best model per horizon by backtested RMSE) +
                          explainability.py (model-specific explanation,
                          key drivers, risks) + similar_periods.py (real
                          nearest-neighbor historical analogs) +
                          probability.py (empirical bullish/bearish %) +
                          accuracy_tracker.py (append-only forecast log +
                          realized-accuracy scoring) + forecast_engine.py
                          (orchestrates all of the above) + run_forecast.py
                          (new-data-gated entrypoint)
dashboard/                 data_loader.py (mtime-keyed st.cache_data readers,
                          Streamlit-free) + charts.py (reusable Plotly figure
                          builders, Streamlit-free, directly unit-tested) +
                          components.py (caveat banner, metric cards, the
                          Refresh Now button) + views/ (one module per
                          section: price_arrivals, seasonality,
                          variety_market, trade_supply, composite_scores,
                          forecasts) + app.py (sidebar nav + global filters)
run_dashboard.py           `streamlit run` entrypoint wrapper
```

## Adding a new sheet to the workbook

Any sheet not listed in `config/sheets.yaml` is detected automatically but
**not parsed** — it shows up under `unconfigured_sheets` in the next Data
Quality Report. To bring it in, add an entry to `sheets.yaml` describing its
layout (see the existing entries for examples of each layout type: `long`,
`wide_pivot_year_month`, `wide_pivot_year_cols`, `wide_pivot_merged_header`,
`sparse_tracker`, or `derived_skip` for precomputed/derived tables). This is
a config edit, not a code change.

## Known data quirks worth remembering (see `sheets.yaml` notes for detail)

- **Guntur Monthly Arrivals** contains a second table further down the same
  sheet with the same data in a different unit (bags → Qtl via ×0.45).
  Intentionally not ingested twice — see the note in `sheets.yaml`.
- **Khammam Teja Price & Arrivals** has the worst data quality of all sheets
  (highest missingness, `#DIV/0!` error strings, and an "NR" token that
  isn't in the invalid-token list — surfaced as text contamination rather
  than guessed at).
- **Guntur Variety correlation** is a precomputed matrix, not ingested as a
  raw source — Phase 2 should recompute correlations from the price sheets.
- **APY** and **Red Chilli Balance sheet** each have a trailing `(exp)` year
  that is the workbook author's own forecast — tagged `is_estimate=True`
  rather than mixed in with actuals.
- Bag sizes differ by market (Guntur = 45kg, Warangal = 40kg, Khammam =
  38kg) — never compare raw arrival volumes across markets without
  converting first.
- **cold storage** has only 6 rows as of this writing — too sparse for real
  analysis; downstream consumers should report "Insufficient evidence"
  rather than analyze it until it accumulates more history.
- **"Seasonality index_Guntur Teja"** is mislabeled: its values are plain
  monthly-average Teja prices, not a normalized seasonal index (verified —
  its numbers match the actual daily average almost exactly, e.g. Dec-2020
  sheet value 15752 vs actual 15752.38). The seasonality module reports it
  as-is for reference but computes its own real normalized index separately.
- **Price-vs-arrivals pairing is an approximation**: Guntur Daily arrivals
  is a total-market figure (all varieties combined), not variety-specific,
  because no per-variety arrivals series exists. Module 3 pairs it against
  each variety's own price anyway (the best available option) and states
  this caveat directly in its own output.
- **The variety/year "group" fields used to be a silent type-mixing risk**:
  Excel stores some header labels as numbers (the "334" variety, most
  years) and others as text (`"2026(exp)"`) in the very same header row.
  `sheet_parser.py` now casts every such group/year value to `str` at parse
  time — found because "334" only ever matched *by accident*, via a
  parquet-write fallback that happened to stringify the column. Fixed at
  the source so it no longer depends on that side effect.
- **"Import" in the Balance Sheet is exactly 0.0 every year** (India doesn't
  import chilli) — a real finding, but it makes any correlation against it
  mathematically undefined (zero variance). `pearson_corr`/`spearman_corr`
  now detect constant input and report `insufficient_evidence` instead of a
  silent `NaN` r value.
- **USD/INR vs price correlation (r≈0.77) is mostly a shared-trend
  artifact**: both series have trended upward for most of the 12-year
  history, so the full-sample correlation overstates real short-term
  sensitivity — `fx_analysis` carries this caveat directly in its output.
- **Khammam's Teja "Average Price" columns are structurally ambiguous**
  between cold-storage and non-cold-storage after unmerging (both collapse
  to the same bare label) — Module 6 uses explicit `column_overrides` from
  Phase 1, not generic reconstruction, and non-cold-storage as the primary
  "open market" comparison price.
- **"Arbitrage" in Module 6 is a spread-anomaly flag, not a trade signal**:
  it flags a regional price spread wider than that pair's own recent history
  (|z| > 2) — there's no transport-cost or quality-differential data to
  determine whether it's actually exploitable.
- **Balance Sheet correlations use `min_n=6`, not the default 30** — there
  are at most ~9 real annual observations per category. Always reported
  with `n` alongside `r`, with a standing caveat that these are directional
  signals only, never statistically robust at this sample size.
- **Bullish Score and Composite Commodity Index can (correctly) disagree.**
  As of the last real refresh: Bullish Score was 75/100 for both varieties
  (driven by price trend/momentum/seasonal-deviation signals), while
  Composite Commodity Index sat only around 57 — because Demand Index was
  low (offtake down sharply YoY) and Supply Pressure wasn't as low as the
  YoY arrival collapse alone would suggest (current rolling arrivals are
  actually ~16% *above* this month's own seasonal norm, per
  `rolling_arrivals_vs_seasonal_norm` — last year's same month was likely
  anomalously high). Each score's `components`/`signals` breakdown is what
  explains the difference; they're intentionally not forced to agree.
- **Forecast horizons are calendar days, looked up by exact date, not row
  position.** The price series has ~9 gap days out of ~2612; a "30-day
  forecast" means the target is 30 calendar days from the origin, and
  training targets for dates that land exactly on a gap are dropped rather
  than approximated from a nearby row.
- **Model selection differs by horizon, by design.** On the last real run,
  SARIMA won at the 7-day horizon for Teja while Holt-Winters won at every
  other horizon for both varieties — nothing is hardcoded to one model;
  each horizon's winner comes from its own backtested RMSE.
- **Holt-Winters and SARIMA fit on different resamplings of the same
  series** — Holt-Winters on weekly means (52-week seasonality, for fitting
  stability; `horizon_days` is rounded to the nearest whole week, a
  documented approximation) and SARIMA directly on the daily series
  reindexed to explicit daily frequency (`asfreq("D")`, so the handful of
  gap days are handled as missing observations by the Kalman filter, not
  invented values) with a 7-day (weekly) seasonal component — yearly
  seasonality for SARIMA is deliberately left to Holt-Winters and the ML
  model's calendar features instead of asking one model to do everything.
- **Confidence intervals are native where the library provides them**
  (Holt-Winters via simulation, SARIMA via `get_forecast`) **and empirical
  otherwise** (Seasonal Naive, XGBoost get a band from that specific
  model+horizon's actual out-of-sample backtest residuals) — every forecast
  says which kind it got via `ci_is_empirical_from_backtest_residuals`.
- **Retraining is gated on new data, not on every refresh** — a full
  backtest+refit across both varieties takes ~90 seconds; a refresh where
  the clean price data's latest date hasn't advanced skips it entirely
  (confirmed: dropped to ~18 seconds on a same-data re-run).
- **USD/INR is forward-filled onto the price series' calendar, price and
  arrivals are not.** USD/INR has its own trading calendar (2277 dates vs
  the price sheet's ~2621) — a plain reindex left ~75% of `fx_rollmean_30`
  windows (and even some individual lag values, including at the most
  recent date) as `NaN`, which silently broke `similar_historical_periods`
  entirely and likely handicapped XGBoost's candidacy in backtesting. A
  currency rate genuinely persists across a day with no fresh quote, so
  forward-filling it is real information, not fabrication — unlike price
  or arrivals, where a gap usually means "the mandi was closed" and there
  is no real value to carry forward, so those are left as real gaps.
- **A single scattered price/arrivals gap no longer disables an entire
  rolling window.** `rolling(window)`'s pandas default requires every value
  in the window to be non-null; one real "Closed" day used to silently
  zero out the next up-to-90 rows' rolling stats, including possibly the
  most recent (most important) row. Rolling calculations now tolerate a
  partial window (`min_periods = max(3, window // 2)`) — still computed
  purely from real observed values, just not thrown away over one gap.
- **`similar_historical_periods` falls back to the nearest earlier complete
  day when the origin's own single-day lag is gapped**, and says so
  explicitly via a `note` field — it never silently substitutes a different
  date without disclosing it. This is expected to trigger periodically
  (found on the very first real Phase 3b run: the origin's own row was
  incomplete due to a gap 2 days prior).
- **Historical analogs can (correctly) disagree with the model's own point
  forecast.** On the last real run, both varieties' point forecasts trend
  gently upward, but the closest real historical analogs (Oct 2022 for
  Teja, Aug 2023 for 334) mostly saw slight *declines* over the same
  horizon — a genuinely useful tension, not a contradiction to resolve;
  it's exactly the kind of counter-signal this layer exists to surface.

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

Tests run against small synthetic fixture workbooks built in
`tests/fixtures/` (not real market data) that replicate the structural
patterns above: merged headers, error tokens, trailing blank rows, a second
stacked table, and an unconfigured sheet. `test_timeseries_stats.py` checks
every statistical primitive against hand-computed or synthetically-injected
expected values (e.g. a known doubling series for CAGR, a known seasonal
pattern for the seasonal index, a known lead/lag relationship for the
correlogram). `test_forecasting.py` checks the properties that matter most
for a forecasting engine's *correctness*, not just its output: feature
construction never leaks future information into an earlier row (verified
by planting a detectable spike at a known future date), target lookup uses
exact calendar dates rather than approximating across a gap, backtest
origin selection respects the training-history and horizon-buffer
constraints, and model selection picks the lowest-error candidate from a
synthetic backtest result set. `test_phase3b.py` checks the explainability
layer against small fixtures: each model type's explanation returns real
data from a fake-but-realistic fitted state (never a generic sentence),
the similar-periods search respects its recent-past exclusion window and
correctly falls back (with disclosure) when the origin's own row is
gapped, the bullish/bearish probability reflects a known residual bias,
and the accuracy tracker correctly distinguishes "nothing logged yet" from
"logged but not yet due" from "scoreable now." `test_dashboard.py` covers
`charts.py` (every builder returns a valid figure on both real and
empty/None input) and `data_loader.py` (honest `insufficient_evidence` on
a missing file, and cache invalidation when a file's content and mtime
change) — no Streamlit UI tests, which isn't meaningfully testable without
a much heavier UI-testing framework; the dashboard was instead verified
live (see below).

Phase 4a's dashboard was launched for real (`streamlit run
run_dashboard.py`) and driven through a browser: all six views were
confirmed to render real numbers matching what earlier phases already
verified (e.g. Teja's ₹20,750 latest price and 5.79% CAGR, the -61.8%
arrival YoY collapse, Bullish Score 75/100, the Balance Sheet's annual
correlation caveat), every documented `insufficient_evidence` case (Cold
Storage, Realized Accuracy) rendered its honest caveat rather than a blank
section, and the "Refresh Now" button was clicked live and confirmed to
update the displayed refresh ID end-to-end (~13 seconds, since forecasting
correctly skipped with no new price data).

## What's not here yet

Phase 4b (Export PDF, Export Excel, and custom-branded dark/light theming
— all additive polish on a working dashboard, not prerequisites for it)
and alerts/consultancy-report generation are future work per the original
brief's own phasing.
