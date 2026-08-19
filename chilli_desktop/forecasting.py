"""Forecast engine: fit, backtest, compare and select time-series models.

Model families
--------------
============  ===============================================================
ARIMA         Non-seasonal autoregressive integrated moving average.
SARIMA        ARIMA with a seasonal component.
SARIMAX       SARIMA plus exogenous drivers taken from the workbook
              (arrivals, offtake, exports, USD/INR).
Holt-Winters  Exponential smoothing with additive trend and seasonality.
VAR           Vector autoregression across markets or varieties.
VECM          Vector error correction, used when the panel is cointegrated.
============  ===============================================================

A model that cannot be applied is never silently dropped: it is returned in
:attr:`ModelComparison.skipped` with the reason, and the UI lists it.

Interval convention
-------------------
Two bands are reported and they mean different things.

*Confidence interval* (default 80%)
    The model's own analytic forecast-error band. It reflects what the fitted
    model believes about its uncertainty and is generally too narrow, because
    it assumes the model form is correct.

*Prediction interval* (default 95%)
    The analytic variance combined in quadrature with the *realised*
    out-of-sample error the model made during rolling-origin backtesting, at
    each horizon step. This is calibrated against how the model has actually
    performed on this data and is the band a trader should plan against.

Where a model provides no analytic band (Holt-Winters), both are derived from
residual and backtest error instead, and the fact is recorded in the notes.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from . import settings
from .analytics import decompose, driver_regression, stationarity_tests
from .utils import LOG, Result, periods_per_year, safe_analysis

CFG = settings.FORECAST
ACFG = settings.ANALYTICS

# Mandi series are irregular by nature -- markets close for holidays and
# harvest breaks -- so a resampled index legitimately carries no fixed pandas
# frequency. statsmodels warns about that on every fit; the warning is expected
# and the forward index is built explicitly by `_future_index`, so it is
# silenced here rather than left to flood the log.
try:  # pragma: no cover - depends on statsmodels internals
    from statsmodels.tools.sm_exceptions import ValueWarning as _SMValueWarning

    warnings.filterwarnings("ignore", category=_SMValueWarning)
except ImportError:  # pragma: no cover
    pass

#: Seasonal periods above this are not attempted with SARIMA: the state
#: vector becomes larger than the sample can identify.
MAX_SEASONAL_PERIOD_FOR_SARIMA = 24


# ==========================================================================
# Containers
# ==========================================================================


@dataclass
class BacktestMetrics:
    """Out-of-sample accuracy from rolling-origin evaluation."""

    rmse: float = float("nan")
    mae: float = float("nan")
    mape: float = float("nan")
    r2: float = float("nan")
    directional_accuracy: float = float("nan")
    folds: int = 0
    points: int = 0
    #: RMSE by horizon step, used to calibrate the prediction interval.
    rmse_by_step: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def as_row(self, model: str) -> dict[str, Any]:
        return {
            "Model": model,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE %": self.mape,
            "R²": self.r2,
            "Directional accuracy %": self.directional_accuracy,
            "Backtest folds": self.folds,
            "Points evaluated": self.points,
        }

    @property
    def usable(self) -> bool:
        return self.folds > 0 and np.isfinite(self.rmse)


@dataclass
class ForecastResult:
    """A fitted model, its forecast, its intervals and its track record."""

    model: str
    label: str
    history: pd.Series
    forecast: pd.Series
    conf_lower: pd.Series
    conf_upper: pd.Series
    pred_lower: pd.Series
    pred_upper: pd.Series
    metrics: BacktestMetrics
    fitted_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    residuals: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    aic: float = float("nan")
    bic: float = float("nan")
    exog_used: list[str] = field(default_factory=list)
    companion_series: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    frequency: str = "ME"

    def table(self) -> pd.DataFrame:
        """The forecast as a presentation-ready table."""
        frame = pd.DataFrame(
            {
                "Forecast": self.forecast,
                f"Lower {CFG.confidence_level:.0%} (confidence)": self.conf_lower,
                f"Upper {CFG.confidence_level:.0%} (confidence)": self.conf_upper,
                f"Lower {CFG.prediction_level:.0%} (prediction)": self.pred_lower,
                f"Upper {CFG.prediction_level:.0%} (prediction)": self.pred_upper,
            }
        )
        frame.index.name = "Period ending"
        if not self.history.empty:
            last = float(self.history.iloc[-1])
            frame.insert(1, "Change vs latest", frame["Forecast"] - last)
            frame.insert(
                2,
                "Change vs latest %",
                (frame["Forecast"] - last) / last * 100 if last else np.nan,
            )
        return frame


@dataclass
class ModelComparison:
    """Every model tried, the winner, and why it won."""

    results: list[ForecastResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    best: ForecastResult | None = None
    selection_reason: str = ""
    target_name: str = ""
    frequency: str = "ME"
    source: str = ""

    def comparison_table(self) -> pd.DataFrame:
        """Ranked scoreboard of every model that was successfully fitted."""
        rows = [r.metrics.as_row(r.label) for r in self.results]
        for name, reason in self.skipped:
            rows.append(
                {
                    "Model": name,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MAPE %": np.nan,
                    "R²": np.nan,
                    "Directional accuracy %": np.nan,
                    "Backtest folds": 0,
                    "Points evaluated": 0,
                    "Status": f"Not applied — {reason}",
                }
            )
        frame = pd.DataFrame(rows)
        if "Status" not in frame.columns:
            frame["Status"] = ""
        frame["Status"] = frame["Status"].fillna("")
        for result in self.results:
            mask = frame["Model"] == result.label
            frame.loc[mask, "Status"] = (
                "SELECTED" if self.best and result.label == self.best.label else "Fitted"
            )
        return frame.sort_values(
            ["RMSE"], ascending=True, na_position="last"
        ).reset_index(drop=True)


# ==========================================================================
# Metrics
# ==========================================================================


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE that ignores zero actuals rather than returning infinity."""
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _directional_accuracy(
    actual: np.ndarray, predicted: np.ndarray, anchor: float
) -> float:
    """Share of steps whose direction versus the last observed value is right.

    This is the question a desk actually asks -- "from where we are now, does
    the model call up or down correctly?" -- rather than step-to-step
    direction within the forecast path.
    """
    if actual.size == 0 or not np.isfinite(anchor):
        return float("nan")
    actual_dir = np.sign(actual - anchor)
    pred_dir = np.sign(predicted - anchor)
    mask = actual_dir != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(actual_dir[mask] == pred_dir[mask]) * 100)


def _score(actual: np.ndarray, predicted: np.ndarray, anchor: float) -> dict[str, float]:
    errors = actual - predicted
    sse = float(np.sum(errors**2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": _safe_mape(actual, predicted),
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "directional_accuracy": _directional_accuracy(actual, predicted, anchor),
    }


# ==========================================================================
# Order selection
# ==========================================================================


def _select_arima_order(
    series: pd.Series, seasonal_period: int = 0, max_fits: int = 60
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], float]:
    """Grid-search (p,d,q)(P,D,Q,m) by AIC on the full history.

    Orders are chosen once and then held fixed across backtest folds, which
    keeps evaluation honest (no re-tuning against each test window) and keeps
    the search affordable.
    """
    stationarity = stationarity_tests(series)
    d = int(stationarity.value.attrs.get("integration_order", 1)) if stationarity else 1
    d = max(0, min(d, 2))

    seasonal_d = 0
    if seasonal_period > 1 and len(series) >= 2 * seasonal_period + 10:
        seasonal_diff = series.diff(seasonal_period).dropna()
        # Seasonal differencing helps when it materially reduces variance.
        if len(seasonal_diff) > 10 and seasonal_diff.var() < series.diff().dropna().var():
            seasonal_d = 1

    best: tuple[float, tuple[int, int, int], tuple[int, int, int, int]] = (
        np.inf,
        (1, d, 0),
        (0, 0, 0, 0),
    )
    fits = 0
    seasonal_grid: list[tuple[int, int, int, int]]
    if seasonal_period > 1:
        seasonal_grid = [
            (p, seasonal_d, q, seasonal_period)
            for p in CFG.seasonal_p_range
            for q in CFG.seasonal_q_range
        ]
    else:
        seasonal_grid = [(0, 0, 0, 0)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (p, q), seasonal in itertools.product(
            itertools.product(CFG.arima_p_range, CFG.arima_q_range), seasonal_grid
        ):
            if p == 0 and q == 0 and seasonal[0] == 0 and seasonal[2] == 0:
                continue
            if fits >= max_fits:
                break
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
                fits += 1
                if np.isfinite(fitted.aic) and fitted.aic < best[0]:
                    best = (float(fitted.aic), (p, d, q), seasonal)
            except Exception:  # noqa: BLE001 - a failed combination is simply skipped
                fits += 1
                continue

    return best[1], best[2], best[0]


# ==========================================================================
# Rolling-origin backtest
# ==========================================================================


def _rolling_backtest(
    series: pd.Series,
    horizon: int,
    fit_predict: Callable[[pd.Series, int], np.ndarray | None],
    folds: int | None = None,
    min_train: int | None = None,
) -> BacktestMetrics:
    """Evaluate a model by repeatedly forecasting held-out windows.

    Origins are spaced evenly through the tail of the sample. At each origin
    the model is refitted on data up to that point only, so no future
    information leaks into the evaluation.
    """
    folds = folds or CFG.backtest_folds
    n = len(series)
    min_train = min_train or max(20, int(n * 0.5))

    usable_folds = min(folds, max(0, (n - min_train) // max(1, horizon)))
    if usable_folds < 1:
        return BacktestMetrics()

    all_actual: list[float] = []
    all_pred: list[float] = []
    anchors: list[float] = []
    per_step_errors: dict[int, list[float]] = {}
    per_fold: list[dict[str, float]] = []
    completed = 0

    for fold in range(usable_folds):
        end = n - (usable_folds - fold - 1) * horizon - horizon
        if end < min_train:
            continue
        train = series.iloc[:end]
        test = series.iloc[end : end + horizon]
        if len(test) == 0:
            continue
        try:
            predicted = fit_predict(train, len(test))
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Backtest fold %d failed: %s", fold, exc)
            continue
        if predicted is None or len(predicted) != len(test):
            continue
        predicted = np.asarray(predicted, dtype=float)
        if not np.all(np.isfinite(predicted)):
            continue

        actual = test.to_numpy(dtype=float)
        anchor = float(train.iloc[-1])
        all_actual.extend(actual.tolist())
        all_pred.extend(predicted.tolist())
        anchors.extend([anchor] * len(actual))
        for step, (a, p) in enumerate(zip(actual, predicted), start=1):
            per_step_errors.setdefault(step, []).append(float(a - p))
        per_fold.append(_score(actual, predicted, anchor))
        completed += 1

    if completed == 0 or not all_actual:
        return BacktestMetrics()

    actual_arr = np.asarray(all_actual)
    pred_arr = np.asarray(all_pred)
    anchor_arr = np.asarray(anchors)

    errors = actual_arr - pred_arr
    sse = float(np.sum(errors**2))
    sst = float(np.sum((actual_arr - np.mean(actual_arr)) ** 2))
    actual_dir = np.sign(actual_arr - anchor_arr)
    pred_dir = np.sign(pred_arr - anchor_arr)
    dir_mask = actual_dir != 0

    rmse_by_step = pd.Series(
        {step: float(np.sqrt(np.mean(np.square(errs)))) for step, errs in per_step_errors.items()}
    ).sort_index()

    return BacktestMetrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        mape=_safe_mape(actual_arr, pred_arr),
        r2=1.0 - sse / sst if sst > 0 else float("nan"),
        directional_accuracy=(
            float(np.mean(actual_dir[dir_mask] == pred_dir[dir_mask]) * 100)
            if dir_mask.any()
            else float("nan")
        ),
        folds=completed,
        points=len(actual_arr),
        rmse_by_step=rmse_by_step,
    )


# ==========================================================================
# Interval construction
# ==========================================================================


def _future_index(history: pd.Series, steps: int, freq: str) -> pd.DatetimeIndex:
    """Build the forward date index for a forecast."""
    last = history.index[-1]
    try:
        return pd.date_range(start=last, periods=steps + 1, freq=freq)[1:]
    except (ValueError, TypeError):
        spacing = pd.Series(history.index).diff().dt.days.median()
        spacing = 30.0 if not np.isfinite(spacing) or spacing <= 0 else float(spacing)
        return pd.DatetimeIndex(
            [last + pd.Timedelta(days=spacing * (i + 1)) for i in range(steps)]
        )


def _build_intervals(
    point: pd.Series,
    analytic_lower: pd.Series | None,
    analytic_upper: pd.Series | None,
    metrics: BacktestMetrics,
    residual_sigma: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, list[str]]:
    """Assemble the confidence and prediction bands.

    The confidence band is analytic where available. The prediction band adds
    the model's realised backtest error, by horizon step, in quadrature -- so
    a model that has historically been wide gets a wide band regardless of
    how confident its own likelihood is.
    """
    notes: list[str] = []
    steps = len(point)
    z_conf = float(sp_stats.norm.ppf(0.5 + CFG.confidence_level / 2))
    z_pred = float(sp_stats.norm.ppf(0.5 + CFG.prediction_level / 2))

    if analytic_lower is not None and analytic_upper is not None:
        conf_lower = pd.Series(np.asarray(analytic_lower, dtype=float), index=point.index)
        conf_upper = pd.Series(np.asarray(analytic_upper, dtype=float), index=point.index)
        model_sigma = (conf_upper - conf_lower) / (2 * z_conf)
        notes.append(
            f"Confidence band is the model's analytic {CFG.confidence_level:.0%} "
            "forecast-error interval."
        )
    else:
        # No analytic band: widen the in-sample residual sd with the square
        # root of the horizon, the standard random-walk error growth.
        growth = np.sqrt(np.arange(1, steps + 1, dtype=float))
        model_sigma = pd.Series(residual_sigma * growth, index=point.index)
        conf_lower = point - z_conf * model_sigma
        conf_upper = point + z_conf * model_sigma
        notes.append(
            "This model provides no analytic interval, so the confidence band "
            "is built from the in-sample residual standard deviation, widened "
            "by the square root of the horizon."
        )

    empirical = pd.Series(np.zeros(steps), index=point.index)
    if metrics.usable and not metrics.rmse_by_step.empty:
        by_step = metrics.rmse_by_step
        values = []
        for step in range(1, steps + 1):
            if step in by_step.index:
                values.append(float(by_step.loc[step]))
            else:
                # Beyond the backtested horizon, grow the last observed error
                # with the square root of the extra distance.
                last_step = int(by_step.index.max())
                values.append(
                    float(by_step.loc[last_step]) * np.sqrt(step / max(last_step, 1))
                )
        empirical = pd.Series(values, index=point.index)
        notes.append(
            f"Prediction band widens the model variance with the realised "
            f"out-of-sample error from {metrics.folds} backtest fold(s) "
            f"(RMSE {metrics.rmse:,.0f}), combined in quadrature and taken to "
            f"{CFG.prediction_level:.0%}."
        )
    else:
        notes.append(
            "No usable backtest, so the prediction band is the model's own "
            f"variance taken to {CFG.prediction_level:.0%}. Treat it as a "
            "lower bound on the true uncertainty."
        )

    combined_sigma = np.sqrt(model_sigma.to_numpy() ** 2 + empirical.to_numpy() ** 2)
    combined_sigma = pd.Series(combined_sigma, index=point.index)
    pred_lower = point - z_pred * combined_sigma
    pred_upper = point + z_pred * combined_sigma

    return conf_lower, conf_upper, pred_lower, pred_upper, notes


# ==========================================================================
# Individual model builders
# ==========================================================================


def _fit_arima_family(
    series: pd.Series,
    horizon: int,
    freq: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    label: str,
    model_name: str,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    source: str = "",
) -> ForecastResult:
    """Fit one member of the ARIMA/SARIMA/SARIMAX family and forecast."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=300)

        def fit_predict(train: pd.Series, steps: int) -> np.ndarray | None:
            train_exog = exog.loc[train.index] if exog is not None else None
            sub = SARIMAX(
                train,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            if exog is not None:
                # Hold the drivers at their last observed value across the
                # test window: a genuine out-of-sample run cannot know them.
                held = np.repeat(
                    exog.loc[train.index].iloc[[-1]].to_numpy(), steps, axis=0
                )
                return np.asarray(sub.get_forecast(steps, exog=held).predicted_mean)
            return np.asarray(sub.get_forecast(steps).predicted_mean)

        metrics = _rolling_backtest(series, horizon, fit_predict)

        if exog is not None:
            if future_exog is None or len(future_exog) < horizon:
                held = np.repeat(exog.iloc[[-1]].to_numpy(), horizon, axis=0)
            else:
                held = future_exog.iloc[:horizon].to_numpy()
            forecast_obj = fitted.get_forecast(horizon, exog=held)
        else:
            forecast_obj = fitted.get_forecast(horizon)

        point_values = np.asarray(forecast_obj.predicted_mean, dtype=float)
        conf = forecast_obj.conf_int(alpha=1 - CFG.confidence_level)

    index = _future_index(series, horizon, freq)
    point = pd.Series(point_values, index=index, name="Forecast")
    conf_array = np.asarray(conf, dtype=float)
    residuals = pd.Series(np.asarray(fitted.resid, dtype=float), index=series.index)
    sigma = float(np.nanstd(residuals.to_numpy()))

    conf_lower, conf_upper, pred_lower, pred_upper, notes = _build_intervals(
        point,
        pd.Series(conf_array[:, 0], index=index),
        pd.Series(conf_array[:, 1], index=index),
        metrics,
        sigma,
    )

    exog_names = list(exog.columns) if exog is not None else []
    if exog_names:
        notes.append(
            "Exogenous drivers are held at their latest observed value across "
            "the forecast horizon, because the workbook contains no forward "
            "projection for them. This is stated as an assumption rather than "
            "modelled."
        )

    return ForecastResult(
        model=model_name,
        label=label,
        history=series,
        forecast=point,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        metrics=metrics,
        fitted_values=pd.Series(np.asarray(fitted.fittedvalues), index=series.index),
        residuals=residuals,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        exog_used=exog_names,
        notes=notes,
        source=source,
        frequency=freq,
    )


def _fit_holt_winters(
    series: pd.Series, horizon: int, freq: str, period: int, source: str = ""
) -> ForecastResult:
    """Additive Holt-Winters exponential smoothing."""
    seasonal = "add" if period > 1 and len(series) >= 2 * period + 1 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=period if seasonal else None,
            initialization_method="estimated",
        ).fit()

        def fit_predict(train: pd.Series, steps: int) -> np.ndarray | None:
            use_seasonal = seasonal if len(train) >= 2 * period + 1 else None
            sub = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=use_seasonal,
                seasonal_periods=period if use_seasonal else None,
                initialization_method="estimated",
            ).fit()
            return np.asarray(sub.forecast(steps))

        metrics = _rolling_backtest(series, horizon, fit_predict)
        point_values = np.asarray(fitted.forecast(horizon), dtype=float)

    index = _future_index(series, horizon, freq)
    point = pd.Series(point_values, index=index, name="Forecast")
    residuals = pd.Series(np.asarray(fitted.resid, dtype=float), index=series.index)
    sigma = float(np.nanstd(residuals.to_numpy()))

    conf_lower, conf_upper, pred_lower, pred_upper, notes = _build_intervals(
        point, None, None, metrics, sigma
    )
    label = (
        f"Holt-Winters (additive trend + additive seasonal, m={period})"
        if seasonal
        else "Holt-Winters (additive trend, no seasonal term)"
    )
    if not seasonal:
        notes.append(
            f"Seasonality was omitted: {len(series)} observations cannot "
            f"support a seasonal period of {period}."
        )

    return ForecastResult(
        model="Holt-Winters",
        label=label,
        history=series,
        forecast=point,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        metrics=metrics,
        fitted_values=pd.Series(np.asarray(fitted.fittedvalues), index=series.index),
        residuals=residuals,
        aic=float(getattr(fitted, "aic", np.nan)),
        bic=float(getattr(fitted, "bic", np.nan)),
        notes=notes,
        source=source,
        frequency=freq,
    )


def _fit_var(
    panel: pd.DataFrame, target: str, horizon: int, freq: str, source: str = ""
) -> ForecastResult:
    """Vector autoregression on first differences, re-integrated to levels.

    VAR requires stationary inputs. The panel is differenced, forecast, and
    the target column cumulated back onto its last observed level.
    """
    differenced = panel.diff().dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_lags = int(max(1, min(8, len(differenced) // (5 * panel.shape[1]))))
        model = VAR(differenced)
        order = model.select_order(max_lags)
        lags = int(order.selected_orders.get("aic", 1) or 1)
        lags = max(1, min(lags, max_lags))
        fitted = model.fit(lags)

        def fit_predict(train: pd.Series, steps: int) -> np.ndarray | None:
            sub_panel = panel.loc[: train.index[-1]]
            sub_diff = sub_panel.diff().dropna()
            if len(sub_diff) < lags * (panel.shape[1] + 2):
                return None
            sub_fit = VAR(sub_diff).fit(lags)
            forecast_diff = sub_fit.forecast(sub_diff.values[-lags:], steps)
            column = list(panel.columns).index(target)
            return float(sub_panel[target].iloc[-1]) + np.cumsum(forecast_diff[:, column])

        metrics = _rolling_backtest(panel[target].dropna(), horizon, fit_predict)

        mid, lower, upper = fitted.forecast_interval(
            differenced.values[-lags:], horizon, alpha=1 - CFG.confidence_level
        )

    column = list(panel.columns).index(target)
    anchor = float(panel[target].iloc[-1])
    index = _future_index(panel[target].dropna(), horizon, freq)
    point = pd.Series(anchor + np.cumsum(mid[:, column]), index=index, name="Forecast")
    # Interval half-widths accumulate with the differenced path.
    half = (upper[:, column] - lower[:, column]) / 2.0
    cumulative_half = np.sqrt(np.cumsum(half**2))
    residuals = pd.Series(
        np.asarray(fitted.resid[target]) if target in fitted.resid else np.asarray(fitted.resid)[:, column],
        index=differenced.index[-len(fitted.resid) :],
    )

    conf_lower, conf_upper, pred_lower, pred_upper, notes = _build_intervals(
        point,
        point - cumulative_half,
        point + cumulative_half,
        metrics,
        float(np.nanstd(residuals.to_numpy())),
    )
    notes.append(
        f"Fitted on first differences with {lags} lag(s) chosen by AIC, then "
        "cumulated back to price levels."
    )

    return ForecastResult(
        model="VAR",
        label=f"VAR({lags}) on {panel.shape[1]} series",
        history=panel[target].dropna(),
        forecast=point,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        metrics=metrics,
        residuals=residuals,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        companion_series=[c for c in panel.columns if c != target],
        notes=notes,
        source=source,
        frequency=freq,
    )


def _fit_vecm(
    panel: pd.DataFrame,
    target: str,
    horizon: int,
    freq: str,
    rank: int,
    source: str = "",
) -> ForecastResult:
    """Vector error correction model, used when the panel is cointegrated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_ar_diff = int(max(1, min(4, len(panel) // (10 * panel.shape[1]))))
        fitted = VECM(
            panel, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci"
        ).fit()

        def fit_predict(train: pd.Series, steps: int) -> np.ndarray | None:
            sub_panel = panel.loc[: train.index[-1]]
            if len(sub_panel) < max(CFG.min_obs_vecm, 10 * panel.shape[1]):
                return None
            sub_fit = VECM(
                sub_panel, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci"
            ).fit()
            column = list(panel.columns).index(target)
            return np.asarray(sub_fit.predict(steps=steps))[:, column]

        metrics = _rolling_backtest(panel[target].dropna(), horizon, fit_predict)

        mid, lower, upper = fitted.predict(
            steps=horizon, alpha=1 - CFG.confidence_level
        )

    column = list(panel.columns).index(target)
    index = _future_index(panel[target].dropna(), horizon, freq)
    point = pd.Series(np.asarray(mid)[:, column], index=index, name="Forecast")
    residuals = pd.Series(
        np.asarray(fitted.resid)[:, column],
        index=panel.index[-np.asarray(fitted.resid).shape[0] :],
    )

    conf_lower, conf_upper, pred_lower, pred_upper, notes = _build_intervals(
        point,
        pd.Series(np.asarray(lower)[:, column], index=index),
        pd.Series(np.asarray(upper)[:, column], index=index),
        metrics,
        float(np.nanstd(residuals.to_numpy())),
    )
    notes.append(
        f"Cointegration rank {rank} detected across "
        f"{', '.join(map(str, panel.columns))}, so the model includes an "
        "error-correction term that pulls the series back toward their "
        "long-run equilibrium."
    )

    return ForecastResult(
        model="VECM",
        label=f"VECM(rank={rank}, k_ar_diff={k_ar_diff}) on {panel.shape[1]} series",
        history=panel[target].dropna(),
        forecast=point,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        metrics=metrics,
        residuals=residuals,
        companion_series=[c for c in panel.columns if c != target],
        notes=notes,
        source=source,
        frequency=freq,
    )


# ==========================================================================
# Orchestration
# ==========================================================================


def run_all_models(
    series: pd.Series,
    freq: str,
    horizon: int,
    *,
    target_name: str = "",
    exog: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    source: str = "",
    progress: Callable[[str, int], None] | None = None,
    models: Sequence[str] | None = None,
    history_notes: Sequence[str] | None = None,
) -> ModelComparison:
    """Fit every applicable model, backtest each, and select the best.

    Parameters
    ----------
    series:
        The target price series, already resampled to ``freq``.
    freq:
        Pandas offset alias: ``W``, ``SME`` or ``ME``.
    horizon:
        Number of periods to forecast.
    exog:
        Candidate exogenous drivers, aligned to ``series``. Enables SARIMAX.
    panel:
        Multivariate panel including the target column. Enables VAR/VECM.
    progress:
        Optional callback ``(message, percent)`` for the UI progress bar.
    models:
        Restrict the run to these model names. ``None`` runs all applicable.
    history_notes:
        Caveats about the input series (an incomplete final period, coverage
        gaps) that must travel with every forecast built from it.
    """
    comparison = ModelComparison(
        target_name=target_name or str(series.name or "series"),
        frequency=freq,
        source=source,
    )
    wanted = set(models) if models else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    def report(message: str, percent: int) -> None:
        if progress:
            try:
                progress(message, percent)
            except Exception:  # noqa: BLE001 - never let the UI break a fit
                pass

    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    period = periods_per_year(freq)

    if n < CFG.min_obs_arima:
        comparison.selection_reason = (
            f"Only {n} observation(s) at {settings.FORECAST.frequency_labels.get(freq, freq)} "
            f"frequency; at least {CFG.min_obs_arima} are needed to fit any model."
        )
        for name in ("ARIMA", "SARIMA", "SARIMAX", "Holt-Winters", "VAR", "VECM"):
            comparison.skipped.append((name, comparison.selection_reason))
        return comparison

    # ---- order selection (once, shared by ARIMA / SARIMA / SARIMAX) ----
    report("Selecting model orders by AIC…", 5)
    seasonal_ok = (
        period > 1
        and period <= MAX_SEASONAL_PERIOD_FOR_SARIMA
        and n >= CFG.min_obs_seasonal_cycles * period + 10
    )
    order, seasonal_order, _aic = _select_arima_order(
        clean, period if seasonal_ok else 0
    )

    # ---- ARIMA ----
    if want("ARIMA"):
        report("Fitting ARIMA…", 15)
        try:
            comparison.results.append(
                _fit_arima_family(
                    clean, horizon, freq, order, (0, 0, 0, 0),
                    label=f"ARIMA{order}", model_name="ARIMA", source=source,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("ARIMA failed")
            comparison.skipped.append(("ARIMA", f"the fit did not converge ({exc})"))

    # ---- SARIMA ----
    if want("SARIMA"):
        report("Fitting SARIMA…", 30)
        if not seasonal_ok:
            if period > MAX_SEASONAL_PERIOD_FOR_SARIMA:
                reason = (
                    f"a seasonal period of {period} at "
                    f"{settings.FORECAST.frequency_labels.get(freq, freq).lower()} "
                    f"frequency needs a state vector of {period} terms, which "
                    f"{n} observations cannot identify. Holt-Winters covers "
                    "seasonality at this frequency, and the monthly view "
                    "supports SARIMA directly."
                )
            else:
                reason = (
                    f"{n} observation(s) is below the "
                    f"{CFG.min_obs_seasonal_cycles * period + 10} needed for "
                    f"{CFG.min_obs_seasonal_cycles} full seasonal cycles of "
                    f"{period} periods."
                )
            comparison.skipped.append(("SARIMA", reason))
        elif seasonal_order == (0, 0, 0, 0):
            comparison.skipped.append(
                (
                    "SARIMA",
                    "the AIC search found no seasonal term that improved on "
                    "the non-seasonal ARIMA, so SARIMA would duplicate it.",
                )
            )
        else:
            try:
                comparison.results.append(
                    _fit_arima_family(
                        clean, horizon, freq, order, seasonal_order,
                        label=f"SARIMA{order}{seasonal_order}",
                        model_name="SARIMA", source=source,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("SARIMA failed")
                comparison.skipped.append(("SARIMA", f"the fit did not converge ({exc})"))

    # ---- SARIMAX ----
    if want("SARIMAX"):
        report("Fitting SARIMAX with exogenous drivers…", 45)
        if exog is None or exog.empty:
            comparison.skipped.append(
                (
                    "SARIMAX",
                    "no exogenous driver in the workbook overlaps this series "
                    "at this frequency.",
                )
            )
        else:
            aligned = pd.concat([clean.rename("__target__"), exog], axis=1).dropna()
            drivers = aligned.drop(columns="__target__")
            drivers = drivers.loc[:, drivers.std(ddof=0) > 0]
            if len(aligned) < max(CFG.min_obs_arima, 5 * (drivers.shape[1] + 1)):
                comparison.skipped.append(
                    (
                        "SARIMAX",
                        f"only {len(aligned)} period(s) have both the price and "
                        f"all {exog.shape[1]} driver(s) present; the overlap is "
                        "too short to estimate the driver coefficients.",
                    )
                )
            elif drivers.empty:
                comparison.skipped.append(
                    ("SARIMAX", "every candidate driver is constant over the overlap.")
                )
            else:
                try:
                    # Restore the caller's series name: it is what every
                    # headline and chart label is built from.
                    target_aligned = aligned["__target__"].rename(clean.name)
                    comparison.results.append(
                        _fit_arima_family(
                            target_aligned, horizon, freq, order,
                            seasonal_order if seasonal_ok else (0, 0, 0, 0),
                            label=(
                                f"SARIMAX{order}"
                                f"{seasonal_order if seasonal_ok else ''} + "
                                f"{len(drivers.columns)} driver(s)"
                            ),
                            model_name="SARIMAX",
                            exog=drivers,
                            source=source,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("SARIMAX failed")
                    comparison.skipped.append(
                        ("SARIMAX", f"the fit did not converge ({exc})")
                    )

    # ---- Holt-Winters ----
    if want("Holt-Winters"):
        report("Fitting Holt-Winters…", 60)
        try:
            comparison.results.append(
                _fit_holt_winters(clean, horizon, freq, period, source)
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Holt-Winters failed")
            comparison.skipped.append(
                ("Holt-Winters", f"the fit did not converge ({exc})")
            )

    # ---- VAR / VECM ----
    report("Testing the panel for cointegration…", 75)
    var_panel: pd.DataFrame | None = None
    panel_reason = ""
    target_column = ""
    if panel is not None and not panel.empty:
        candidate = panel.apply(pd.to_numeric, errors="coerce").dropna()
        matches = [c for c in candidate.columns if str(c) == str(clean.name)]
        target_column = matches[0] if matches else (candidate.columns[0] if len(candidate.columns) else "")
        if candidate.shape[1] < 2:
            panel_reason = "the panel needs at least two series."
        elif not target_column:
            panel_reason = "the target series is not present in the panel."
        elif len(candidate) < CFG.min_obs_var:
            panel_reason = (
                f"only {len(candidate)} period(s) have every panel series "
                f"present simultaneously; at least {CFG.min_obs_var} are needed."
            )
        else:
            var_panel = candidate
    else:
        panel_reason = "no multivariate panel was supplied for this target."

    rank = 0
    if var_panel is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jo = coint_johansen(var_panel.values, det_order=0, k_ar_diff=1)
            rank = int(sum(1 for i in range(len(jo.lr1)) if jo.lr1[i] > jo.cvt[i, 1]))
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Johansen test failed: %s", exc)
            rank = 0

    if want("VAR"):
        report("Fitting VAR…", 82)
        if var_panel is None:
            comparison.skipped.append(("VAR", panel_reason))
        else:
            try:
                comparison.results.append(
                    _fit_var(var_panel, target_column, horizon, freq, source)
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("VAR failed")
                comparison.skipped.append(("VAR", f"the fit did not converge ({exc})"))

    if want("VECM"):
        report("Fitting VECM…", 90)
        if var_panel is None:
            comparison.skipped.append(("VECM", panel_reason))
        elif rank < 1:
            comparison.skipped.append(
                (
                    "VECM",
                    "the Johansen test finds no cointegrating relationship in "
                    "this panel, so an error-correction term would have "
                    "nothing to correct toward. VAR on differences is the "
                    "correct specification here.",
                )
            )
        elif len(var_panel) < CFG.min_obs_vecm:
            comparison.skipped.append(
                (
                    "VECM",
                    f"only {len(var_panel)} overlapping period(s); at least "
                    f"{CFG.min_obs_vecm} are needed.",
                )
            )
        else:
            try:
                comparison.results.append(
                    _fit_vecm(var_panel, target_column, horizon, freq, rank, source)
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("VECM failed")
                comparison.skipped.append(("VECM", f"the fit did not converge ({exc})"))

    # Caveats about the input series belong on every forecast derived from it.
    for note in history_notes or ():
        if note:
            for result in comparison.results:
                result.notes.append(note)

    report("Selecting the best model…", 96)
    _select_best(comparison)
    report("Done", 100)
    return comparison


def _select_best(comparison: ModelComparison) -> None:
    """Pick the winning model and record a plain-language justification."""
    scored = [r for r in comparison.results if r.metrics.usable]

    if not scored:
        if comparison.results:
            # Nothing backtested; fall back to in-sample information criteria.
            with_aic = [r for r in comparison.results if np.isfinite(r.aic)]
            if with_aic:
                best = min(with_aic, key=lambda r: r.aic)
                comparison.best = best
                comparison.selection_reason = (
                    f"No model produced a usable rolling-origin backtest — the "
                    f"history is too short to hold out {CFG.backtest_folds} "
                    f"windows. {best.label} was selected on the lowest AIC "
                    f"({best.aic:,.1f}) instead. Because it has not been "
                    "validated out of sample, treat the forecast as indicative."
                )
            else:
                comparison.best = comparison.results[0]
                comparison.selection_reason = (
                    "No model could be scored on either backtest accuracy or "
                    "AIC; the first successful fit is shown."
                )
        else:
            comparison.selection_reason = (
                "No model could be fitted to this series. See the skipped-model "
                "list for the reason in each case."
            )
        return

    metric = CFG.selection_metric.lower()
    key = {"rmse": "rmse", "mae": "mae", "mape": "mape"}.get(metric, "rmse")
    best = min(scored, key=lambda r: getattr(r.metrics, key))
    comparison.best = best

    ranked = sorted(scored, key=lambda r: getattr(r.metrics, key))
    parts = [
        f"{best.label} was selected because it produced the lowest "
        f"out-of-sample {CFG.selection_metric} "
        f"({getattr(best.metrics, key):,.0f}) across "
        f"{best.metrics.folds} rolling-origin backtest fold(s) covering "
        f"{best.metrics.points} held-out observation(s)."
    ]
    if len(ranked) > 1:
        runner = ranked[1]
        gap = getattr(runner.metrics, key) - getattr(best.metrics, key)
        relative = gap / getattr(best.metrics, key) if getattr(best.metrics, key) else 0
        parts.append(
            f"The next best model, {runner.label}, was "
            f"{relative:.1%} worse ({getattr(runner.metrics, key):,.0f})."
        )
        if relative < 0.05:
            parts.append(
                "That margin is narrow enough that the two models are "
                "effectively tied; compare their forecast paths before relying "
                "on either alone."
            )
    if np.isfinite(best.metrics.directional_accuracy):
        parts.append(
            f"It called the direction of the move correctly "
            f"{best.metrics.directional_accuracy:.0f}% of the time in backtesting"
            + (
                " — better than a coin toss."
                if best.metrics.directional_accuracy > 55
                else " — no better than a coin toss, so use the level forecast "
                "rather than the direction."
                if best.metrics.directional_accuracy < 55
                else "."
            )
        )
    if np.isfinite(best.metrics.r2) and best.metrics.r2 < 0:
        parts.append(
            f"Its out-of-sample R² is negative ({best.metrics.r2:.2f}), meaning "
            "that over the backtest windows a flat line at the mean of the "
            "held-out data would have scored better. The level forecast is "
            "still the best available from this workbook, but the confidence "
            "in it should be low."
        )
    if comparison.skipped:
        parts.append(
            f"{len(comparison.skipped)} model(s) were not applicable: "
            + "; ".join(f"{name} ({reason})" for name, reason in comparison.skipped)
            + "."
        )
    comparison.selection_reason = " ".join(parts)


# ==========================================================================
# Explainability
# ==========================================================================


@dataclass
class ForecastExplanation:
    """Everything needed to justify a forecast to a non-statistician."""

    headline: str
    direction: str
    components: Result[pd.DataFrame]
    drivers: Result[dict[str, Any]]
    stationarity: Result[pd.DataFrame]
    assumptions: list[str] = field(default_factory=list)
    plain_language: list[str] = field(default_factory=list)


def explain(
    result: ForecastResult,
    exog: pd.DataFrame | None = None,
    source: str = "",
) -> ForecastExplanation:
    """Build a trader-readable explanation of a forecast.

    Combines the structural decomposition of the history, a driver
    regression against the workbook's exogenous variables, the stationarity
    diagnosis that determined the differencing, and the model's own backtest
    record.
    """
    history = result.history
    freq = result.frequency
    components = decompose(history, freq, source=source)
    stationarity = stationarity_tests(history, source=source)
    drivers = (
        driver_regression(history, exog, source=source)
        if exog is not None and not exog.empty
        else Result.unavailable(
            "No exogenous driver from the workbook overlaps this series at "
            "this frequency, so driver attribution cannot be performed.",
            source,
        )
    )

    last = float(history.iloc[-1]) if not history.empty else float("nan")
    final = float(result.forecast.iloc[-1]) if not result.forecast.empty else float("nan")
    change = final - last
    change_pct = (change / last * 100) if last else float("nan")
    direction = "higher" if change > 0 else "lower" if change < 0 else "broadly unchanged"

    horizon_label = settings.FORECAST.frequency_labels.get(freq, freq).lower()
    headline = (
        f"{result.label} projects {result.history.name or 'the series'} at "
        f"{final:,.0f} by {result.forecast.index[-1]:%b %Y} — "
        f"{abs(change_pct):.1f}% {direction} than the latest observed "
        f"{last:,.0f} ({history.index[-1]:%d %b %Y})."
        if np.isfinite(final) and np.isfinite(last)
        else "The forecast could not be summarised."
    )

    plain: list[str] = []
    if np.isfinite(change_pct):
        plain.append(
            f"Direction: prices are projected {direction}"
            + (f" by about {abs(change_pct):.1f}% over the horizon." if change != 0 else ".")
        )
    if components:
        shares = components.value.attrs.get("variance_shares", {})
        if shares:
            plain.append(
                "Structure of the history: "
                f"{shares.get('Trend', 0):.0%} of the variation is trend, "
                f"{shares.get('Seasonal', 0):.0%} is repeating seasonal "
                f"pattern and {shares.get('Residual', 0):.0%} is irregular. "
                + (
                    "A trend-dominated series means the level matters more "
                    "than the calendar."
                    if shares.get("Trend", 0) > 0.6
                    else "A strong seasonal share means the calendar carries "
                    "real information for timing."
                    if shares.get("Seasonal", 0) > 0.25
                    else "A large irregular share means shocks dominate and "
                    "point forecasts should be held loosely."
                )
            )
    if drivers:
        payload = drivers.unwrap()
        coefficients = payload["coefficients"]
        significant = coefficients[
            coefficients["Significant"] & (coefficients.index != "const")
        ]
        if not significant.empty:
            ordered = significant.reindex(
                significant["Standardised beta"].abs().sort_values(ascending=False).index
            )
            described = []
            for name, row in ordered.iterrows():
                sign = "rises" if row["Coefficient"] > 0 else "falls"
                described.append(f"price {sign} when {name} rises")
            plain.append(
                "Drivers with a statistically significant link over the common "
                f"history: {'; '.join(described)}. Together they explain "
                f"{payload['r_squared']:.0%} of period-on-period movement."
            )
        else:
            plain.append(
                "None of the workbook's exogenous drivers shows a "
                "statistically significant link to this series' movements, so "
                "the forecast rests on the price series' own history."
            )
    else:
        plain.append(drivers.reason)

    if result.metrics.usable:
        plain.append(
            f"Track record: across {result.metrics.folds} backtest fold(s) this "
            f"model was off by {result.metrics.mape:.1f}% on average "
            f"(RMSE {result.metrics.rmse:,.0f}) and called the direction "
            f"correctly {result.metrics.directional_accuracy:.0f}% of the time."
        )
    else:
        plain.append(
            "This model has no usable backtest record on this data, so its "
            "accuracy is unproven."
        )

    if not result.pred_lower.empty:
        plain.append(
            f"Planning range: the {CFG.prediction_level:.0%} prediction "
            f"interval at the end of the horizon runs from "
            f"{result.pred_lower.iloc[-1]:,.0f} to "
            f"{result.pred_upper.iloc[-1]:,.0f}. Treat that band, not the "
            "single line, as the forecast."
        )

    assumptions = list(result.notes)
    assumptions.append(
        "The forecast assumes the statistical relationships in the workbook's "
        "history continue. It contains no information about weather, policy, "
        "or any other event not represented in the workbook."
    )
    if result.exog_used:
        assumptions.append(
            "Exogenous drivers used: " + ", ".join(result.exog_used) + "."
        )
    if result.companion_series:
        assumptions.append(
            "Modelled jointly with: " + ", ".join(map(str, result.companion_series)) + "."
        )
    if stationarity:
        assumptions.extend(n for n in stationarity.notes if n)

    return ForecastExplanation(
        headline=headline,
        direction=direction,
        components=components,
        drivers=drivers,
        stationarity=stationarity,
        assumptions=assumptions,
        plain_language=plain,
    )
