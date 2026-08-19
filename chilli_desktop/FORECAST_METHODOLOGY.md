# Forecast Methodology

Complete specification of how *Chilli Intelligence Desktop* produces, validates
and selects its price forecasts. Everything described here is implemented in
`forecasting.py`, with the supporting diagnostics in `analytics.py`.

---

## Contents

1. [Scope and inputs](#1-scope-and-inputs)
2. [Series construction](#2-series-construction)
3. [Model specifications](#3-model-specifications)
4. [Order selection](#4-order-selection)
5. [Backtesting protocol](#5-backtesting-protocol)
6. [Accuracy metrics](#6-accuracy-metrics)
7. [Model selection rule](#7-model-selection-rule)
8. [Uncertainty intervals](#8-uncertainty-intervals)
9. [Explainability](#9-explainability)
10. [Skip conditions](#10-skip-conditions)
11. [Assumptions](#11-assumptions)
12. [Known caveats](#12-known-caveats)
13. [Reproducing a forecast by hand](#13-reproducing-a-forecast-by-hand)

---

## 1. Scope and inputs

**Targets.** Any variety quoted on the Guntur variety-wise daily price sheet.
The brief singles out **Guntur LCA 334** and **Guntur Teja**, which the
Executive Summary forecasts automatically.

**Frequencies and horizons.** Each default horizon reaches six months:

| Frequency | Label | Periods/year | Default horizon | Reach |
|---|---|---|---|---|
| `W` | Weekly | 52 | 26 | ~6 months |
| `SME` | Fortnightly | 24 | 12 | ~6 months |
| `ME` | Monthly | 12 | 6 | 6 months |

**Exogenous drivers** (used by SARIMAX and by driver attribution), all from the
workbook:

| Driver | Source sheet | Available at |
|---|---|---|
| Guntur arrivals | Guntur Daily arrivals | all frequencies |
| Guntur offtake | Guntur Daily arrivals | all frequencies |
| USD/INR | USD to INR exchange rate | all frequencies |
| Exports | Red chilli exports | **monthly only** |

Exports are recorded monthly. Using them at weekly frequency would require
inventing intra-month values, so they are offered only on the monthly view and
the reason is displayed.

**Companion series** (used by VAR and VECM): up to three other varieties from
the same panel with at least 60% overlap with the target.

---

## 2. Series construction

1. Read the daily average price for the variety from the two-header-row price
   sheet. `"Closed"` and blank cells become missing observations — never zero.
2. Apply the global filters (date range, calendar months, price band).
3. Resample to the analysis frequency by **mean**, dropping empty periods.
   Nothing is interpolated: a fortnight with no trading produces no observation.
4. All three frequencies are labelled by the **end** of the period. Fortnightly
   periods are the 1st–15th and 16th–month-end, implemented directly rather than
   via pandas' `SME` rule, which labels buckets by their start and would shift
   the fortnightly view by one period relative to weekly and monthly.
5. If a training window is set, keep only the most recent *N* periods.

**Partial final period.** The workbook ends mid-month, so the final resampled
bucket usually covers only part of a period. This is detected, quantified
("20 of 31 days") and attached as a note to every forecast built from that
series.

---

## 3. Model specifications

All six are fitted through `statsmodels`.

### ARIMA(p, d, q)

Non-seasonal. Captures level, momentum and shock persistence. The baseline
against which everything else must justify its extra complexity.

### SARIMA(p, d, q)(P, D, Q, m)

ARIMA plus a seasonal component at period *m* (12 monthly, 24 fortnightly).

### SARIMAX(p, d, q)(P, D, Q, m) + exogenous

SARIMA plus the exogenous drivers above, aligned to the target's index. The
target and every driver must be present in the same period for it to enter the
estimation sample.

### Holt-Winters exponential smoothing

Additive trend, additive seasonality, seasonal period *m*, initialisation
estimated. Unlike SARIMA it handles a long seasonal period (52 weeks) without a
state vector explosion, which makes it the seasonal option on the weekly view.

### VAR(p)

Vector autoregression across the target and its companions. VAR requires
stationary inputs, so the panel is **first-differenced**, the model is fitted
and forecast in differences, and the target's path is **cumulated back** onto
its last observed level. Lag order is chosen by AIC, bounded by the sample size
relative to the number of series.

### VECM(rank, k_ar_diff)

Vector error correction: a VAR in differences plus an error-correction term that
pulls the series back toward their long-run equilibrium. Used **only** when the
Johansen trace test finds at least one cointegrating relationship in the panel.
It models levels directly, so no re-integration is needed.

VECM is frequently the winner on this data, which is what one would expect:
the mandi prices are cointegrated, so a specification that knows about the
long-run relationship should beat one that does not.

---

## 4. Order selection

Orders are selected **once on the full training history**, then held **fixed**
across every backtest fold. Re-searching orders inside each fold would let the
model tune itself against its own test window and would make the backtest
optimistic.

1. **Differencing order `d`** comes from the ADF sequence in
   `analytics.stationarity_tests` — the number of differences needed before ADF
   rejects a unit root, capped at 2.
2. **Seasonal differencing `D`** is set to 1 when seasonally differencing
   materially reduces variance versus a first difference, otherwise 0.
3. **`p`, `q` ∈ {0, 1, 2}** and **`P`, `Q` ∈ {0, 1}** are grid-searched by AIC,
   capped at 60 fits. `enforce_stationarity` and `enforce_invertibility` are
   disabled so near-boundary parameterisations still fit.
4. VAR lag order is chosen by AIC over a sample-size-bounded range;
   `k_ar_diff` for VECM is scaled to the sample and number of series.

---

## 5. Backtesting protocol

**Rolling-origin evaluation** with 5 folds by default.

```
full history ────────────────────────────────────────────────►
fold 1: [────── train ──────][ test ]
fold 2: [──────── train ────────][ test ]
fold 3: [────────── train ──────────][ test ]
fold 4: [──────────── train ────────────][ test ]
fold 5: [────────────── train ──────────────][ test ]
```

- Origins are spaced one horizon apart through the tail of the sample.
- Minimum training length is the greater of 20 periods and 50% of the series.
- At each origin the model is **refitted from scratch** on data up to that point
  only, then asked for `horizon` steps. No future information enters the fit.
- For SARIMAX, exogenous drivers are **held at their last observed value**
  across the test window, exactly as they must be in a live forecast. Feeding
  the true future driver values would be look-ahead leakage and would flatter
  SARIMAX against every other model.
- A fold that fails to converge is skipped and the remaining folds still count;
  the fold count is reported.

Errors are pooled across folds for the headline metrics, and also retained
**per horizon step** to calibrate the prediction interval (§8).

---

## 6. Accuracy metrics

With *aₜ* actual, *fₜ* forecast, *n* pooled held-out observations, and *anchor*
the last observed value before each fold's origin:

| Metric | Definition | Reading |
|---|---|---|
| **RMSE** | √(Σ(aₜ−fₜ)²/n) | Same units as price. Penalises large misses. **Selection metric.** |
| **MAE** | Σ\|aₜ−fₜ\|/n | Same units. Typical miss. |
| **MAPE** | 100·Σ\|(aₜ−fₜ)/aₜ\|/n | Percent. Comparable across varieties. Zero actuals excluded. |
| **R²** | 1 − SSE/SST | Share of held-out variance explained. **Can be negative** — see below. |
| **Directional accuracy** | % of steps where sign(fₜ−anchor) = sign(aₜ−anchor) | Did it call up-or-down correctly *from today*? |

**Negative R² is meaningful, not a bug.** It says that over the backtest
windows, a flat line at the mean of the held-out data would have scored better
than the model. The level forecast may still be the best available from this
workbook, but confidence in it should be low. The application states this
explicitly in the selection rationale whenever it occurs.

**Directional accuracy is measured against the last observed value**, not
step-to-step within the forecast path, because "from where we are now, is it up
or down?" is the question a desk actually asks.

---

## 7. Model selection rule

1. Keep every model with a usable backtest (at least one completed fold and a
   finite RMSE).
2. Select the **lowest out-of-sample RMSE**.
3. Generate a plain-language rationale stating the winner, its RMSE, the fold
   and observation count, the runner-up and the percentage gap.
4. If the gap to the runner-up is **under 5%**, the rationale says the two are
   effectively tied and advises comparing their forecast paths.
5. Add the directional-accuracy reading, and a warning if R² is negative.
6. List every skipped model with its reason.

**Fallback.** If no model backtests successfully — the history is too short to
hold out folds — the lowest-AIC model is selected instead and the rationale
states that it has **not** been validated out of sample and should be treated as
indicative.

RMSE is the default because large misses matter disproportionately in a
procurement or hedging decision. The metric is configurable in
`settings.ForecastConfig.selection_metric`.

---

## 8. Uncertainty intervals

Two bands are reported and **they mean different things**.

### Confidence interval — default 80%

The model's own analytic forecast-error band, from the state-space filter for
the ARIMA family, VAR and VECM. It reflects what the fitted model believes about
its own uncertainty, conditional on the model form being correct — which is why
it is generally too narrow.

Holt-Winters provides no analytic band. For it, the in-sample residual standard
deviation is widened by √h (standard random-walk error growth) and the
substitution is recorded in the panel's notes.

### Prediction interval — default 95%

The analytic variance combined **in quadrature** with the error the model
actually made during backtesting, at each horizon step:

```
σ_combined(h) = √( σ_model(h)² + RMSE_backtest(h)² )

PI(h) = forecast(h) ± z_0.975 · σ_combined(h)
```

where `RMSE_backtest(h)` is the pooled RMSE at horizon step *h* across folds.
Beyond the backtested horizon it is extrapolated as
`RMSE(h_max) · √(h / h_max)`.

**This is the band to plan against.** A model that has historically been wide
gets a wide band regardless of how confident its likelihood is. When no usable
backtest exists, the prediction band falls back to the model's own variance and
is explicitly labelled a lower bound on the true uncertainty.

Both levels are configurable (`confidence_level`, `prediction_level`).

---

## 9. Explainability

Every forecast is accompanied by:

**Headline** — the projected level at the end of the horizon, the percentage
change from the latest actual, and both dates.

**Structural decomposition** — STL on the history, reporting the variance share
of trend, seasonal and residual, with a plain-language reading. A
trend-dominated series means the level matters more than the calendar; a large
residual share means shocks dominate and point forecasts should be held loosely.

**Driver attribution** — OLS of the target on the workbook's exogenous drivers,
estimated on **first differences** to avoid spurious regression, reporting
coefficients, standard errors, p-values and **standardised betas** so drivers on
different scales can be ranked. Accompanied by VIF, because two collinear
drivers cannot be separately attributed.

**Track record** — the model's backtest MAPE, RMSE and directional accuracy.

**Planning range** — the prediction interval at the end of the horizon, with the
instruction to treat the band rather than the line as the forecast.

**Assumptions** — interval construction, exogenous-driver handling, companion
series, the stationarity verdict, any partial final period, and the standing
caveat that the forecast contains no information about weather, policy or any
other event not represented in the workbook.

---

## 10. Skip conditions

A model that cannot be applied is never silently dropped. It appears in the
comparison table with `Not applied — <reason>`.

| Model | Skipped when |
|---|---|
| *all* | fewer than 30 observations at the chosen frequency |
| SARIMA | seasonal period > 24 (e.g. 52 at weekly): the state vector would need 52 terms the sample cannot identify. Holt-Winters covers seasonality at that frequency, and the monthly view supports SARIMA directly |
| SARIMA | fewer than 2 full seasonal cycles + 10 observations |
| SARIMA | the AIC search found no seasonal term that beat plain ARIMA, so SARIMA would duplicate it |
| SARIMAX | no exogenous driver overlaps the series at this frequency |
| SARIMAX | fewer than max(30, 5·(k+1)) periods with the target and all drivers present |
| SARIMAX | every candidate driver is constant over the overlap |
| VAR / VECM | no multivariate panel, or fewer than 2 series |
| VAR / VECM | fewer than 40 periods with every panel series present simultaneously |
| VECM | the Johansen test finds no cointegrating relationship — an error-correction term would have nothing to correct toward, and VAR on differences is the correct specification |
| VECM | fewer than 60 overlapping periods |
| *any* | the fit did not converge (the exception is reported) |

---

## 11. Assumptions

1. The statistical relationships in the workbook's history continue over the
   forecast horizon.
2. Exogenous drivers stay at their **last observed value** across the horizon.
   The workbook contains no forward projection for arrivals, offtake, exports or
   the exchange rate, so this is stated as an assumption rather than modelled.
3. Missing observations are genuinely missing, not zero. Closed-market days
   carry no price.
4. The mean is the right within-period summary for price (a period's average,
   not its close).
5. Prices are in the workbook's own unit throughout — INR per quintal as
   recorded on the source sheet. No conversion is applied.
6. Backtest performance is informative about future performance, which assumes
   no structural break between the backtest window and the forecast horizon.

---

## 12. Known caveats

**Same-day co-movement.** Every pairwise cross-correlation between these
markets peaks at lag zero. There is no timing lead to exploit; leadership is
established from the direction of predictive power instead.

**Cointegration favours VECM.** Because the varieties and markets are
cointegrated, VECM often wins. That is a genuine finding, but it also means the
forecast inherits VECM's assumption that the historical long-run relationship
persists. A structural break in the relationship would break the forecast.

**Weekly seasonality is approximate.** Fifty-two-period seasonality on ~590
weekly observations is roughly eleven cycles. Holt-Winters can fit it, but the
seasonal estimate for any single week of the year rests on about eleven
observations. **The monthly view is the more reliable basis for seasonal work.**

**Negative out-of-sample R² is common at weekly frequency.** Chilli prices are
close to a random walk week to week. The level forecast remains the best
available, but the interval — not the line — is the usable output.

**The USD/INR gap.** The exchange-rate series is missing 15 May 2019 to 7 June
2022. SARIMAX runs on the periods where the target and all drivers overlap, so
that gap silently shortens the SARIMAX estimation sample. The overlap count is
reported; compare it against the ARIMA sample before reading SARIMAX
coefficients as authoritative.

**The final period is usually partial.** The workbook ends mid-month, so the
most recent monthly average is computed from part of a month. It is the forecast
origin, so this matters — the note is attached to every affected forecast.

**Small annual samples.** The balance sheet has 10 annual observations and APY
12. They inform context and driver intuition but are far too short for
regression or significance testing, and are not used as forecast inputs.

---

## 13. Reproducing a forecast by hand

Every forecast is reproducible from the workbook alone:

1. **Series.** Open the variety price sheet. Take column *Avg Price* for the
   variety. Drop `Closed` and blank rows. Restrict to the filter window.
2. **Resample.** Average within each period, using period-end labels. Drop empty
   periods. Compare your count with the "Observations" figure the application
   reports.
3. **Differencing.** Run ADF; the number of differences until it rejects is `d`.
   The Price Analysis page shows this.
4. **Orders.** Grid-search `(p,d,q)(P,D,Q,m)` by AIC over
   `p,q ∈ {0,1,2}`, `P,Q ∈ {0,1}`. The winning order is printed in the model
   label, e.g. `SARIMA(0, 1, 2)(0, 0, 1, 12)`.
5. **Fit and forecast.** Fit on the full series, request `horizon` steps, take
   `conf_int(alpha=0.20)` for the 80% band.
6. **Backtest.** Repeat at 5 origins spaced one horizon apart, refitting each
   time, and pool the errors. Your RMSE should match the scoreboard.
7. **Prediction interval.** Combine the analytic σ with the per-step backtest
   RMSE in quadrature and take ±1.96σ.

Every intermediate quantity — sample size, differencing order, chosen orders,
fold count, per-metric scores — is displayed in the application, so any step can
be checked independently.
