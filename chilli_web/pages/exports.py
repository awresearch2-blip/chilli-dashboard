"""Export Analysis -- web equivalent of ``ui.ExportAnalysisPage``."""

from __future__ import annotations

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, callback, html

from chilli_desktop import analytics, settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/exports", name="Export Analysis",
    description="Export volume, its relationship with price, and the currency channel", order=8,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="ex-filter-note"),
            html.Div(id="ex-unit-note"),
            C.section_header("Export volume over time"),
            html.Div(id="ex-trend-chart"),
            html.Div(id="ex-calendar-chart"),
            html.Div(id="ex-annual-chart"),
            html.Div(id="ex-seasonal-chart"),
            C.section_header("Exports against price", "Both on a monthly basis, the finest frequency the export sheet supports."),
            html.Div(id="ex-dual-chart"),
            html.Div(id="ex-scatter-chart"),
            C.section_header("Statistical relationship"),
            html.Div(id="ex-corr-table"),
            html.Div(id="ex-cross-chart"),
            html.Div(id="ex-leadlag-box"),
            html.Div(id="ex-rolling-chart"),
            html.Div(id="ex-granger-table"),
            C.section_header(
                "Currency channel",
                "A weaker rupee raises the rupee proceeds of an export sale, which is the "
                "mechanism connecting the exchange rate to domestic prices.",
            ),
            html.Div(id="ex-fx-chart"),
            html.Div(id="ex-fx-gaps"),
            html.Div(id="ex-fx-corr"),
            html.Div(id="ex-fx-granger"),
        ]
    )


@callback(
    Output("ex-filter-note", "children"),
    Output("ex-unit-note", "children"),
    Output("ex-trend-chart", "children"),
    Output("ex-calendar-chart", "children"),
    Output("ex-annual-chart", "children"),
    Output("ex-seasonal-chart", "children"),
    Output("ex-dual-chart", "children"),
    Output("ex-scatter-chart", "children"),
    Output("ex-corr-table", "children"),
    Output("ex-cross-chart", "children"),
    Output("ex-leadlag-box", "children"),
    Output("ex-rolling-chart", "children"),
    Output("ex-granger-table", "children"),
    Output("ex-fx-chart", "children"),
    Output("ex-fx-gaps", "children"),
    Output("ex-fx-corr", "children"),
    Output("ex-fx-granger", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    exports_result = service.exports_monthly()
    if not exports_result:
        empty = C.info_box(exports_result.message(), tone="danger")
        return note, html.Div(), empty, *([html.Div()] * 14)

    exports = exports_result.unwrap()
    matrix = service.exports_matrix()

    unit_note = C.info_box(
        "The export sheet does not state its unit of measure. Values are used exactly as "
        "supplied and compared only against themselves; no conversion is applied and no "
        "cross-series arithmetic assumes a unit.",
        tone="warning",
    )

    trend_fig = PC.line_figure({"Exports": exports}, theme, ylabel="As supplied (unit not stated)")
    trend_chart = C.chart_card("Monthly exports", trend_fig, source=exports_result.source, notes=exports_result.notes, height=320)

    calendar_chart = html.Div()
    annual_chart = html.Div()
    seasonal_chart = html.Div()
    if matrix:
        frame = matrix.unwrap()
        fig = PC.heatmap_figure(frame, theme, diverging=False, value_format=",.0f", cbar_label="As supplied")
        calendar_chart = C.chart_card(
            "Exports by month and year", fig, source=matrix.source,
            notes=["Blank cells are months the workbook has not yet recorded."], height=380,
        )

        annual = frame.sum(axis=1, min_count=1)
        complete = frame.notna().all(axis=1)
        fig = PC.bar_figure(
            annual, theme, ylabel="As supplied (unit not stated)", colour_negative=False,
            value_labels=True, projected=[str(y) for y in frame.index[~complete]],
        )
        annual_chart = C.chart_card(
            "Annual export total", fig, source=matrix.source,
            notes=["Years with any missing month are hatched: their totals are partial and "
                   "must not be compared with complete years."], height=320,
        )

        fig = PC.bar_figure(
            frame.mean(axis=0), theme, ylabel="Mean monthly exports", colour_negative=False,
            reference=float(frame.mean(axis=0).mean()), reference_label="All-month average",
        )
        seasonal_chart = C.chart_card("Average exports by calendar month", fig, source=matrix.source, notes=["Averaged across every year present in the sheet."], height=300)

    variety = PG.target_variety(service, filters)
    price_result = service.variety_series(variety) if variety else None
    if not price_result:
        warn = C.info_box("Export-price analysis needs a price series; none is available.", tone="warning")
        return note, unit_note, trend_chart, calendar_chart, annual_chart, seasonal_chart, warn, *([html.Div()] * 10)

    price = service.series_at(service.apply_filters(price_result.unwrap(), filters, kind="price"), "ME")
    source = f"{exports_result.source}; {price_result.source}"

    dual_fig = PC.dual_axis_figure(price, exports, theme, primary_label=f"{variety} price (INR/quintal)", secondary_label="Exports (as supplied)", secondary_as_bars=True)
    dual_chart = C.chart_card("Exports and price", dual_fig, source=source, height=340)

    scatter_fig = PC.scatter_fit_figure(exports, price, theme, xlabel="Monthly exports (as supplied)", ylabel=f"{variety} price (INR/quintal)")
    scatter_chart = C.chart_card("Exports against price", scatter_fig, source=source, height=340)

    pair = analytics.correlation_pair(price, exports, source)
    corr_table = (
        C.dataframe_table(pair.unwrap().to_frame("Value"), theme, title="Correlation", source=source, notes=pair.notes, id="ex-corr")
        if pair else C.chart_card("Correlation", None, source=pair.source)
    )

    cross = analytics.cross_correlation(exports, price, 12, source, differenced=True)
    if cross:
        table = cross.unwrap()
        fig = PC.stem_figure(
            table["correlation"], theme, xlabel="Lag in months (positive = exports lead price)",
            ylabel="Correlation of month-on-month changes", band=float(table["upper_95"].iloc[0]),
            highlight_index=int(table["correlation"].abs().idxmax()),
        )
        cross_chart = C.chart_card("Cross-correlation: exports leading price", fig, source=source, notes=cross.notes, height=320)
    else:
        cross_chart = C.chart_card("Cross-correlation: exports leading price", None, source=cross.source)

    reading = analytics.lead_lag(exports, price, "Exports", f"{variety} price", "ME", 12, source)
    leadlag_box = (
        C.info_box_items("Lead-lag reading:", [reading.unwrap().sentence()] + list(reading.notes))
        if reading else html.Div()
    )

    rolling = analytics.rolling_correlation(price, exports, 24, source)
    if rolling:
        fig = PC.line_figure({"Rolling correlation": rolling.unwrap()}, theme, ylabel="Correlation")
        rolling_chart = C.chart_card(
            "24-month rolling correlation", fig, source=source,
            notes=list(rolling.notes) + ["A relationship that changes sign over time cannot be relied on for positioning."],
            height=300,
        )
    else:
        rolling_chart = C.chart_card("24-month rolling correlation", None, source=rolling.source)

    granger = analytics.granger_causality(exports.rename("Exports"), price.rename(f"{variety} price"), 6, source)
    granger_table = (
        C.dataframe_table(granger.unwrap(), theme, title="Granger causality: exports → price", source=source, notes=granger.notes, id="ex-granger")
        if granger else C.chart_card("Granger causality: exports → price", None, source=granger.source)
    )

    fx_result = service.usd_inr()
    fx_chart = html.Div()
    fx_gaps = html.Div()
    fx_corr = html.Div()
    fx_granger = html.Div()
    if fx_result:
        fx = service.series_at(fx_result.unwrap(), "ME")
        gaps = service.coverage_gaps(fx_result.unwrap())
        fx_notes = []
        if not gaps.empty:
            worst = gaps.iloc[0]
            fx_notes.append(
                f"The exchange-rate sheet has {len(gaps)} gap(s) over 45 days; the largest "
                f"spans {int(worst['days'])} days ({worst['from']} to {worst['to']}). The "
                "line is drawn across the gap but no value exists inside it."
            )
        fig = PC.dual_axis_figure(price, fx, theme, primary_label=f"{variety} price (INR/quintal)", secondary_label="USD/INR")
        fx_chart = C.chart_card("USD/INR and price", fig, source=f"{price_result.source}; {fx_result.source}", notes=fx_notes, height=340)
        if not gaps.empty:
            fx_gaps = C.dataframe_table(
                gaps, theme, title="Exchange-rate coverage gaps", source=fx_result.source,
                notes=["Any statistic spanning these breaks joins disconnected periods."], id="ex-fxgaps",
            )
        fx_pair = analytics.correlation_pair(price, fx, f"{price_result.source}; {fx_result.source}")
        if fx_pair:
            fx_corr = C.dataframe_table(
                fx_pair.unwrap().to_frame("Value"), theme, title="USD/INR against price",
                source=f"{price_result.source}; {fx_result.source}",
                notes=list(fx_pair.notes) + [
                    "Both series trend upward across this sample, so part of this "
                    "correlation is shared trend rather than a mechanism. The Granger test "
                    "below is the directional check.",
                ], id="ex-fxcorr",
            )
        fx_granger_result = analytics.granger_causality(fx.rename("USD/INR"), price.rename(f"{variety} price"), 6, f"{price_result.source}; {fx_result.source}")
        if fx_granger_result:
            fx_granger = C.dataframe_table(
                fx_granger_result.unwrap(), theme, title="Granger causality: USD/INR → price",
                source=f"{price_result.source}; {fx_result.source}", notes=fx_granger_result.notes, id="ex-fxgranger",
            )
    else:
        fx_chart = C.chart_card("USD/INR and price", None, source=fx_result.source)

    return (
        note, unit_note, trend_chart, calendar_chart, annual_chart, seasonal_chart,
        dual_chart, scatter_chart, corr_table, cross_chart, leadlag_box, rolling_chart,
        granger_table, fx_chart, fx_gaps, fx_corr, fx_granger,
    )
