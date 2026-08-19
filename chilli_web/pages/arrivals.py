"""Arrival Analysis -- web equivalent of ``ui.ArrivalAnalysisPage``."""

from __future__ import annotations

import dash
import numpy as np
from dash import Input, Output, callback, html

from chilli_desktop import analytics, settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/arrivals", name="Arrival Analysis",
    description="How arrivals move prices: elasticity, thresholds, timing and seasonality",
    order=4,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="ar-filter-note"),
            html.Div(id="ar-basis-note"),
            C.section_header("Price against arrivals"),
            html.Div(id="ar-dual-chart"),
            html.Div(id="ar-flow-chart"),
            C.section_header(
                "Scatter and fitted relationship",
                "Colour shows when each observation occurred, which reveals whether the "
                "relationship has shifted over time.",
            ),
            html.Div(id="ar-scatter"),
            C.section_header(
                "Elasticity",
                "Estimated on log differences, so each coefficient reads as the percentage "
                "price response to a 1% change in arrivals.",
            ),
            html.Div(id="ar-elasticity-table"),
            html.Div(id="ar-elasticity-chart"),
            C.section_header(
                "Threshold effects",
                "Arrivals split into quintiles: at what level do prices start to come under pressure?",
            ),
            html.Div(id="ar-threshold-table"),
            html.Div(id="ar-threshold-chart"),
            C.section_header(
                "Timing of the arrivals effect",
                "Correlation between the change in arrivals and the change in price at successive lags.",
            ),
            html.Div(id="ar-lag-chart"),
            html.Div(id="ar-lag-table"),
            C.section_header(
                "Arrivals seasonality",
                "The monthly arrivals sheet, as a calendar grid and as a season classification derived from it.",
            ),
            html.Div(id="ar-calendar"),
            html.Div(id="ar-season-table"),
        ]
    )


@callback(
    Output("ar-filter-note", "children"),
    Output("ar-basis-note", "children"),
    Output("ar-dual-chart", "children"),
    Output("ar-flow-chart", "children"),
    Output("ar-scatter", "children"),
    Output("ar-elasticity-table", "children"),
    Output("ar-elasticity-chart", "children"),
    Output("ar-threshold-table", "children"),
    Output("ar-threshold-chart", "children"),
    Output("ar-lag-chart", "children"),
    Output("ar-lag-table", "children"),
    Output("ar-calendar", "children"),
    Output("ar-season-table", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    variety = PG.target_variety(service, filters)
    price_result = service.variety_series(variety) if variety else None
    arrivals_result = service.guntur_arrivals()

    if not price_result or not arrivals_result:
        message = (
            (price_result.message() if price_result and not price_result else "")
            or (arrivals_result.message() if not arrivals_result else "")
            or settings.DATA_UNAVAILABLE_MESSAGE
        )
        empty = C.info_box(message, tone="danger")
        return note, empty, *([html.Div()] * 11)

    source = f"{price_result.source}; {arrivals_result.source}"
    price = service.series_at(service.apply_filters(price_result.unwrap(), filters, kind="price"), filters.frequency)
    arrivals = service.series_at(service.apply_filters(arrivals_result.unwrap(), filters, kind="arrivals"), filters.frequency, "sum")
    offtake_result = service.guntur_offtake()

    basis_note = C.info_box(
        f"Prices are {variety} averages; arrivals are Guntur mandi totals summed within each "
        f"{PG.frequency_label(filters.frequency).lower()} period, in bags of "
        f"{service.market_bag_weight('Guntur') or '—'} kg as stated on the arrivals sheet.",
        tone="muted",
    )

    dual_fig = PC.dual_axis_figure(
        price, arrivals, theme, primary_label=f"{variety} price (INR/quintal)",
        secondary_label="Arrivals (bags)", secondary_as_bars=True,
    )
    dual_chart = C.chart_card("Price and arrivals", dual_fig, source=source, height=380)

    if offtake_result:
        offtake = service.series_at(service.apply_filters(offtake_result.unwrap(), filters), filters.frequency, "sum")
        flow_fig = PC.line_figure({"Arrivals": arrivals, "Offtake": offtake}, theme, ylabel="Bags per period")
        flow_chart = C.chart_card(
            "Arrivals against offtake", flow_fig, source=arrivals_result.source,
            notes=["Offtake is the quantity actually lifted. Arrivals persistently above "
                   "offtake means stock building in the mandi."], height=300,
        )
    else:
        flow_chart = html.Div()

    scatter_fig = PC.scatter_fit_figure(arrivals, price, theme, xlabel="Arrivals (bags per period)", ylabel=f"{variety} price (INR/quintal)")
    scatter_chart = C.chart_card("Arrivals against price", scatter_fig, source=source, height=380)

    elasticity = analytics.elasticity(price, arrivals, source)
    if elasticity:
        frame = elasticity.unwrap()
        elasticity_table = C.dataframe_table(frame, theme, title="Elasticity by arrivals lag", source=source, notes=elasticity.notes, id="ar-elasticity")
        significant = frame[frame["Significant"]]
        highlight = int(significant["p-value"].idxmin()) if not significant.empty else None
        stem_fig = PC.stem_figure(frame["Elasticity"], theme, xlabel="Arrivals lag (periods)", ylabel="Elasticity (% price per % arrivals)", highlight_index=highlight)
        elasticity_chart = C.chart_card("Elasticity by lag", stem_fig, source=source, notes=elasticity.notes, height=300)
    else:
        elasticity_table = C.chart_card("Elasticity by arrivals lag", None, source=elasticity.source)
        elasticity_chart = html.Div()

    thresholds = analytics.threshold_effects(price, arrivals, source=source)
    if thresholds:
        frame = thresholds.unwrap()
        threshold_table = C.dataframe_table(frame, theme, title="Price behaviour by arrivals bucket", source=source, notes=thresholds.notes, id="ar-threshold")
        labels = [f"{row['Arrivals from']:,.0f}–{row['Arrivals to']:,.0f}" for _, row in frame.iterrows()]
        import pandas as pd
        values = pd.Series(frame["Mean next-period change %"].to_numpy(), index=pd.Index(labels, name="Arrivals range (bags)"))
        bar_fig = PC.bar_figure(
            values, theme, ylabel="Mean next-period price change (%)", reference=0.0,
            reference_label="No change", value_labels=True, value_format="+.2f",
        )
        threshold_chart = C.chart_card("Next-period price change by arrivals bucket", bar_fig, source=source, notes=thresholds.notes, height=320)
    else:
        threshold_table = C.chart_card("Price behaviour by arrivals bucket", None, source=thresholds.source)
        threshold_chart = html.Div()

    lagged = analytics.lagged_impact(price, arrivals, 8, source)
    if lagged:
        frame = lagged.unwrap()
        band = 1.96 / np.sqrt(float(frame["Observations"].max()))
        stem_fig = PC.stem_figure(frame["Correlation with price change"], theme, xlabel="Arrivals lag (periods)", ylabel="Correlation with price change", band=band)
        lag_chart = C.chart_card("Lagged impact of arrivals on price change", stem_fig, source=source, notes=lagged.notes, height=320)
        lag_table = C.dataframe_table(frame, theme, title="Lagged impact detail", source=source, notes=lagged.notes, id="ar-lagged")
    else:
        lag_chart = C.chart_card("Lagged impact of arrivals on price change", None, source=lagged.source)
        lag_table = html.Div()

    monthly = service.data.get("guntur_monthly_arrivals")
    if monthly is not None:
        fig = PC.heatmap_figure(
            monthly.frame, theme, diverging=False, value_format=",.0f", cbar_label="Bags",
        )
        calendar_chart = C.chart_card(
            "Monthly arrivals by year", fig, source=monthly.sheet_name,
            notes=[monthly.meta.get("primary_unit", ""), monthly.meta.get("secondary_unit", "")],
            height=400,
        )
    else:
        calendar_chart = C.chart_card(
            "Monthly arrivals by year", None,
            source="The Guntur monthly arrivals sheet is not present in the workbook.",
        )

    season = service.season_profile()
    if season:
        frame = season.unwrap().set_index("month_name")
        frame.index.name = "Month"
        season_table = C.dataframe_table(frame, theme, title="Season classification by month", source=season.source, notes=season.notes, id="ar-season")
    else:
        season_table = C.chart_card("Season classification by month", None, source=season.source)

    return (
        note, basis_note, dual_chart, flow_chart, scatter_chart, elasticity_table,
        elasticity_chart, threshold_table, threshold_chart, lag_chart, lag_table,
        calendar_chart, season_table,
    )
