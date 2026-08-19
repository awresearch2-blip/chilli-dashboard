"""Forecast Center -- web equivalent of ``ui.ForecastCenterPage``.

The flagship page: choose a variety, frequency, horizon, training window and
model subset, then run the full model sweep. Every applicable model is
fitted, backtested and ranked by :func:`forecasting.run_all_models` -- the
exact function the desktop app calls -- and the result renders as a fan
chart, a scoreboard, a forecast table and a plain-language explanation.

One behavioural note versus the desktop app: changing a sidebar filter does
not automatically clear a forecast already on screen here (the desktop page
rebuilds from scratch on every filter change and loses its result). Pressing
"Run model sweep" always uses the *current* filters, so the result is never
stale -- it just isn't auto-cleared before you press it again.
"""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, dcc, html

from chilli_desktop import forecasting, settings
from chilli_desktop.settings import FORTNIGHT_FREQ
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/forecast", name="Forecast Center",
    description="Weekly, fortnightly and monthly projections to six months, with model comparison",
    order=10,
)

_MODEL_NAMES = ("ARIMA", "SARIMA", "SARIMAX", "Holt-Winters", "VAR", "VECM")
_FREQ_OPTIONS = [
    {"label": settings.FORECAST.frequency_labels.get(alias, alias), "value": alias}
    for alias in ("W", FORTNIGHT_FREQ, "ME")
]


def layout(**_kwargs):
    from chilli_web import server_state

    service = server_state.get_service()
    varieties = service.varieties()
    focus = service.focus_varieties()
    default_variety = next(iter(focus.values()), varieties[0] if varieties else "")

    controls = html.Div(
        [
            html.Div(
                [
                    html.Div("Variety", className="filter-label"),
                    dcc.Dropdown(id="fc-variety", options=[{"label": v, "value": v} for v in varieties],
                                value=default_variety, clearable=False),
                ]
            ),
            html.Div(
                [
                    html.Div("Frequency", className="filter-label"),
                    dcc.Dropdown(id="fc-frequency", options=_FREQ_OPTIONS, value="ME", clearable=False),
                ]
            ),
            html.Div(
                [
                    html.Div("Horizon (periods)", className="filter-label"),
                    dcc.Input(id="fc-horizon", type="number", min=1, max=60, value=settings.FORECAST.horizons["ME"], style={"width": "100%"}),
                ]
            ),
            html.Div(
                [
                    html.Div("Training window (0 = all)", className="filter-label"),
                    dcc.Input(id="fc-window", type="number", min=0, value=0, style={"width": "100%"}),
                ]
            ),
            html.Div(
                [
                    html.Div("Models", className="filter-label"),
                    dcc.Checklist(
                        id="fc-models",
                        options=[{"label": f" {m}", "value": m} for m in _MODEL_NAMES],
                        value=list(_MODEL_NAMES), labelStyle={"display": "block", "fontSize": "11px"},
                    ),
                ]
            ),
        ],
        className="grid-cols-3",
    )

    return html.Div(
        [
            C.section_header(
                "Forecast controls",
                "Choose the target, the frequency and the horizon, then run the model sweep. "
                "Every applicable model is fitted, backtested and ranked; the winner is "
                "selected on out-of-sample error.",
            ),
            html.Div(
                [controls, html.Button("Run model sweep", id="fc-run", className="primary-button", n_clicks=0,
                                       style={"width": "220px", "marginTop": "12px"})],
                className="chart-card",
            ),
            html.Div(id="fc-status", children=C.info_box(
                "Press “Run model sweep” to fit and compare every applicable model. "
                "A monthly run takes a few seconds; weekly takes longer.", tone="muted",
            )),
            dcc.Loading(
                [
                    html.Div(id="fc-forecast-chart", children=C.chart_card(
                        "Forecast", None,
                        source="No forecast has been run yet. Configure the controls above and press “Run model sweep”.",
                        height=400,
                    )),
                    html.Div(id="fc-selection-box"),
                    html.Div(id="fc-comparison-table"),
                    html.Div(id="fc-forecast-table"),
                    html.Div(id="fc-skipped-box"),
                    html.Div(id="fc-explain-section"),
                ],
                type="circle",
            ),
        ]
    )


@callback(Output("fc-horizon", "value"), Input("fc-frequency", "value"))
def _sync_horizon(freq):
    return settings.FORECAST.horizons.get(freq, 6)


def _compute(service, filters, variety, freq, horizon, window, models, progress=None):
    """Identical logic to ``ui.ForecastCenterPage._compute``."""
    result = service.variety_series(variety)
    if not result:
        return {"error": result.reason}
    raw = service.apply_filters(result.unwrap(), filters, kind="price")
    series = service.series_at(raw, freq)
    series.name = variety
    history_notes = [service.partial_last_period(raw, freq)]
    if window:
        if len(series) > window:
            history_notes.append(
                f"Training restricted to the most recent {window} period(s) of {len(series)} available, as selected."
            )
        series = series.tail(window)
    if series.empty:
        return {"error": "The selected series is empty after filtering."}

    exog = service.exogenous_matrix(freq)
    panel = service.variety_panel(freq, filters)
    panel_frame = panel.unwrap() if panel else None
    if panel_frame is not None and variety in panel_frame.columns:
        companions = [c for c in panel_frame.columns if c != variety and panel_frame[c].notna().sum() >= len(series) * 0.6][:3]
        panel_frame = panel_frame[[variety] + companions]

    comparison = forecasting.run_all_models(
        series, freq, horizon, target_name=variety, exog=exog.unwrap() if exog else None,
        panel=panel_frame, source=result.source, progress=progress, models=models, history_notes=history_notes,
    )
    explanation = (
        forecasting.explain(comparison.best, exog.unwrap() if exog else None, result.source)
        if comparison.best else None
    )
    return {
        "comparison": comparison, "explanation": explanation,
        "exog": exog.unwrap() if exog else None, "exog_source": exog.source if exog else "",
        "variety": variety, "freq": freq,
    }


@callback(
    Output("fc-status", "children"),
    Output("fc-forecast-chart", "children"),
    Output("fc-selection-box", "children"),
    Output("fc-comparison-table", "children"),
    Output("fc-forecast-table", "children"),
    Output("fc-skipped-box", "children"),
    Output("fc-explain-section", "children"),
    Input("fc-run", "n_clicks"),
    State("filters-store", "data"),
    State("theme-store", "data"),
    State("fc-variety", "value"),
    State("fc-frequency", "value"),
    State("fc-horizon", "value"),
    State("fc-window", "value"),
    State("fc-models", "value"),
    background=True,
    running=[(Output("fc-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def _run(_n_clicks, filters_data, theme_name, variety, freq, horizon, window, models):
    service, filters, theme = PG.current(filters_data, theme_name)
    payload = _compute(service, filters, variety, freq, int(horizon or 1), int(window or 0), models or None)

    if payload.get("error"):
        status = C.info_box(f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{payload['error']}", tone="danger")
        return status, C.chart_card("Forecast", None, source=payload["error"]), *([html.Div()] * 5)

    comparison: forecasting.ModelComparison = payload["comparison"]
    freq_label = PG.frequency_label(freq)
    best = comparison.best

    comparison_table = C.dataframe_table(
        comparison.comparison_table(), theme, title="Model comparison and backtest scores", source=comparison.source,
        notes=[
            f"Ranked by out-of-sample {settings.FORECAST.selection_metric} from "
            f"{settings.FORECAST.backtest_folds}-fold rolling-origin backtesting. Lower "
            "RMSE, MAE and MAPE are better; higher R² and directional accuracy are better.",
            "A negative R² means a flat line at the mean of the held-out window would have "
            "scored better than the model.",
        ], id="fc-comparison",
    )
    skipped_box = (
        C.info_box_items("Models not applied to this series, and why:", [f"{name}: {reason}" for name, reason in comparison.skipped])
        if comparison.skipped else html.Div()
    )

    if best is None:
        status = C.info_box(f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{comparison.selection_reason}", tone="danger")
        return status, C.chart_card("Forecast", None, source=comparison.selection_reason), html.Div(), comparison_table, html.Div(), skipped_box, html.Div()

    status = C.info_box(
        f"{variety} — {freq_label.lower()} forecast, {len(best.forecast)} period(s) to "
        f"{best.forecast.index[-1]:%d %b %Y}. {len(comparison.results)} model(s) fitted, "
        f"{len(comparison.skipped)} skipped.",
        tone="muted",
    )

    fig = PC.forecast_figure(
        best.history, best.forecast, theme, conf=(best.conf_lower, best.conf_upper),
        pred=(best.pred_lower, best.pred_upper), label=best.label, ylabel="INR per quintal",
        history_window=min(len(best.history), max(60, len(best.forecast) * 8)),
    )
    forecast_chart = C.chart_card(f"{variety} — {freq_label} forecast · {best.label}", fig, source=best.source, notes=best.notes, height=420)

    selection_box = C.info_box(f"Model selection\n\n{comparison.selection_reason}", tone="info")

    forecast_table = C.dataframe_table(
        best.table(), theme, title="Forecast table", source=best.source,
        notes=[
            f"Confidence interval is the model's analytic {settings.FORECAST.confidence_level:.0%} "
            f"band. Prediction interval is the {settings.FORECAST.prediction_level:.0%} band "
            "after widening for the error the model actually made in backtesting — plan "
            "against the prediction interval, not the single line.",
            "All rows are projections. No row is historical data.",
        ], id="fc-forecast-table",
    )

    explanation = payload.get("explanation")
    explain_children = []
    if explanation is not None:
        explain_children.append(C.section_header("Forecast explanation", "Why the model projects what it projects, in plain language."))
        explain_children.append(C.info_box(explanation.headline, tone="info"))
        explain_children.append(C.info_box_items("", explanation.plain_language, tone="info"))
        explain_children.append(C.info_box_items("Assumptions and caveats behind this forecast:", explanation.assumptions, tone="warning"))

        if explanation.components:
            fig = PC.decomposition_figure(explanation.components.unwrap(), theme)
            explain_children.append(C.chart_card(
                "Historical trend, seasonal and residual components", fig,
                source=explanation.components.source, notes=explanation.components.notes, height=460,
            ))
        else:
            explain_children.append(C.chart_card("Historical trend, seasonal and residual components", None, source=explanation.components.source))

        if explanation.drivers:
            driver_payload = explanation.drivers.unwrap()
            explain_children.append(C.dataframe_table(
                driver_payload["coefficients"], theme, title="Driver attribution", source=payload.get("exog_source", ""),
                notes=list(explanation.drivers.notes) + [
                    f"Model R² {driver_payload['r_squared']:.3f} (adjusted "
                    f"{driver_payload['adj_r_squared']:.3f}) on {driver_payload['n_obs']} "
                    f"observations; overall F-test p={driver_payload['f_pvalue']:.4f}.",
                ], id="fc-drivers",
            ))
            vif = driver_payload.get("vif")
            if vif is not None and vif:
                explain_children.append(C.dataframe_table(vif.unwrap(), theme, title="Driver collinearity (VIF)", source=payload.get("exog_source", ""), notes=vif.notes, id="fc-vif"))
        else:
            explain_children.append(C.chart_card("Driver attribution", None, source=explanation.drivers.source))

        if explanation.stationarity:
            explain_children.append(C.dataframe_table(
                explanation.stationarity.unwrap(), theme, title="Stationarity diagnosis",
                source=explanation.stationarity.source, notes=explanation.stationarity.notes, id="fc-stationarity",
            ))

    return status, forecast_chart, selection_box, comparison_table, forecast_table, skipped_box, html.Div(explain_children)
