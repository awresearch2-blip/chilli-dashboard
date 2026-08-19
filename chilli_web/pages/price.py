"""Price Analysis -- web equivalent of ``ui.PriceAnalysisPage``."""

from __future__ import annotations

import dash
import numpy as np
from dash import Input, Output, callback, html

from chilli_desktop import analytics, settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/price", name="Price Analysis",
    description="Trend, rolling statistics, decomposition and time-series diagnostics",
    order=3,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="pr-filter-note"),
            html.Div(id="pr-partial-note"),
            C.section_header("Price and moving averages"),
            html.Div(id="pr-rolling-chart"),
            html.Div(id="pr-volatility-chart"),
            C.section_header("Comparison across varieties", "Select varieties in the sidebar filter."),
            html.Div(id="pr-compare-chart"),
            html.Div(id="pr-relative-chart"),
            C.section_header("Descriptive statistics and change"),
            html.Div(id="pr-stats-grid"),
            C.section_header(
                "Trend, seasonal and irregular components",
                "STL decomposition separates the persistent level from the repeating calendar "
                "pattern and the unexplained remainder.",
            ),
            html.Div(id="pr-decomposition"),
            C.section_header(
                "Time-series diagnostics",
                "These determine how the series must be modelled — see the Forecast Center "
                "for the models themselves.",
            ),
            html.Div(id="pr-stationarity"),
            html.Div(id="pr-acf"),
            C.section_header("Outliers", "Flagged for review; never removed from any calculation."),
            html.Div(id="pr-outliers"),
        ]
    )


@callback(
    Output("pr-filter-note", "children"),
    Output("pr-partial-note", "children"),
    Output("pr-rolling-chart", "children"),
    Output("pr-volatility-chart", "children"),
    Output("pr-compare-chart", "children"),
    Output("pr-relative-chart", "children"),
    Output("pr-stats-grid", "children"),
    Output("pr-decomposition", "children"),
    Output("pr-stationarity", "children"),
    Output("pr-acf", "children"),
    Output("pr-outliers", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    variety = PG.target_variety(service, filters)
    if not variety:
        empty = C.info_box(
            f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo variety is available under the current filters.",
            tone="danger",
        )
        return note, html.Div(), empty, *([html.Div()] * 9)

    result = service.variety_series(variety)
    raw = service.apply_filters(result.unwrap(), filters, kind="price")
    series = service.series_at(raw, filters.frequency)
    series.name = variety
    source = result.source
    window = settings.ANALYTICS.default_rolling_window

    partial = service.partial_last_period(raw, filters.frequency)
    partial_note = C.info_box(partial, tone="warning") if partial else html.Div()

    rolling = analytics.rolling_statistics(series, window, source)
    if rolling:
        frame = rolling.unwrap()
        fig = PC.line_figure(
            {
                variety: frame["Value"],
                f"Moving average ({window})": frame[f"Moving average ({window})"],
                f"EMA ({window})": frame[f"EMA ({window})"],
            },
            theme, ylabel="INR per quintal", highlight=variety,
            fill_between=(frame["Lower band (-2 sd)"], frame["Upper band (+2 sd)"]),
        )
        rolling_chart = C.chart_card(
            f"{variety} price with rolling statistics", fig, source=source, notes=rolling.notes, height=380
        )
        vol_column = next((c for c in frame.columns if c.startswith("Rolling volatility")), None)
        if vol_column:
            vol_fig = PC.line_figure({"Annualised volatility": frame[vol_column] * 100}, theme, ylabel="% annualised")
            vol_chart = C.chart_card("Rolling volatility (annualised)", vol_fig, source=source, notes=rolling.notes, height=280)
        else:
            vol_chart = html.Div()
    else:
        rolling_chart = C.chart_card(f"{variety} price with rolling statistics", None, source=rolling.source)
        vol_chart = html.Div()

    variety_panel = service.variety_panel(filters.frequency, filters)
    if variety_panel:
        frame = variety_panel.unwrap()
        fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, theme, ylabel="INR per quintal", highlight=variety)
        compare_chart = C.chart_card("Variety comparison", fig, source=variety_panel.source, height=350)

        normalised = {}
        for column in frame.columns:
            values = frame[column].dropna()
            if values.empty or values.iloc[0] == 0:
                continue
            normalised[str(column)] = values / values.iloc[0] * 100
        if normalised:
            rel_fig = PC.line_figure(normalised, theme, ylabel="Index (first period = 100)", highlight=variety)
            relative_chart = C.chart_card(
                "Relative performance (first period = 100)", rel_fig, source=variety_panel.source,
                notes=[
                    "Each series is rebased to 100 at its own first observation in the "
                    "filtered window, so the lines show relative performance rather than "
                    "absolute price.",
                ], height=320,
            )
        else:
            relative_chart = C.chart_card("Relative performance (first period = 100)", None, source=variety_panel.source)
    else:
        compare_chart = C.chart_card("Variety comparison", None, source=variety_panel.source)
        relative_chart = C.chart_card("Relative performance (first period = 100)", None, source=variety_panel.source)

    stats = analytics.descriptive_stats(series, source)
    changes = analytics.change_summary(raw, source)
    stats_tables = []
    if stats:
        values = stats.unwrap().drop(labels=["First observation", "Last observation"], errors="ignore")
        stats_tables.append(
            C.dataframe_table(values.to_frame("Value"), theme, title="Descriptive statistics", source=source, notes=stats.notes, id="pr-stats")
        )
    else:
        stats_tables.append(C.chart_card("Descriptive statistics", None, source=stats.source))
    if changes:
        cframe = changes.unwrap().copy()
        cframe["Change %"] = cframe["Change %"] * 100
        stats_tables.append(
            C.dataframe_table(cframe.set_index("Horizon"), theme, title="Change by horizon", source=source, notes=changes.notes, id="pr-changes")
        )
    else:
        stats_tables.append(C.chart_card("Change by horizon", None, source=changes.source))
    stats_grid = C.grid_of(stats_tables, 2)

    decomposition = analytics.decompose(series, filters.frequency, source=source)
    if decomposition:
        fig = PC.decomposition_figure(decomposition.unwrap(), theme)
        decomposition_chart = C.chart_card("Decomposition", fig, source=source, notes=decomposition.notes, height=520)
    else:
        decomposition_chart = C.chart_card("Decomposition", None, source=decomposition.source)

    stationarity = analytics.stationarity_tests(series, source)
    stationarity_table = (
        C.dataframe_table(stationarity.unwrap(), theme, title="Stationarity tests", source=source, notes=stationarity.notes, id="pr-stationarity")
        if stationarity else C.chart_card("Stationarity tests", None, source=stationarity.source)
    )

    autocorrelation = analytics.autocorrelation(series, 40, source)
    if autocorrelation:
        fig = PC.acf_pacf_figure(autocorrelation.unwrap(), theme)
        acf_chart = C.chart_card("Autocorrelation and partial autocorrelation", fig, source=source, notes=autocorrelation.notes, height=320)
    else:
        acf_chart = C.chart_card("Autocorrelation and partial autocorrelation", None, source=autocorrelation.source)

    outliers = analytics.zscore_and_outliers(series, window, source)
    if outliers:
        frame = outliers.unwrap()
        flagged = frame[frame["Z outlier"] | frame["IQR outlier"]]
        if flagged.empty:
            outliers_table = html.Div(
                [
                    html.Div("Flagged observations", className="card-title"),
                    C.unavailable_panel(
                        "No observation in this window is flagged as an outlier by either the "
                        "z-score or the IQR rule. That is a result, not a gap.",
                        source,
                    ),
                ],
                className="chart-card",
            )
        else:
            outliers_table = C.dataframe_table(flagged, theme, title="Flagged observations", source=source, notes=outliers.notes, id="pr-outliers")
    else:
        outliers_table = C.chart_card("Flagged observations", None, source=outliers.source)

    return (
        note, partial_note, rolling_chart, vol_chart, compare_chart, relative_chart,
        stats_grid, decomposition_chart, stationarity_table, acf_chart, outliers_table,
    )
