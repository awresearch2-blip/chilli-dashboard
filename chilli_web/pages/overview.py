"""Market Overview -- web equivalent of ``ui.MarketOverviewPage``."""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, html

from chilli_desktop import settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/overview", name="Market Overview",
    description="Prices, arrivals and coverage across every market in the workbook",
    order=2,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="ov-filter-note"),
            C.section_header("Prices by market", "Teja basis, so all markets are comparable."),
            html.Div(id="ov-price-chart"),
            C.section_header("Arrivals by market", "Volumes are in each sheet's own bag unit."),
            html.Div(id="ov-arrivals-chart"),
            html.Div(id="ov-tonnes-table"),
            C.section_header(
                "Workbook coverage",
                "What each sheet contains and how far it reaches. Every analysis in this "
                "application is bounded by these spans.",
            ),
            html.Div(id="ov-coverage-table"),
            html.Div(id="ov-quality-box"),
        ]
    )


@callback(
    Output("ov-filter-note", "children"),
    Output("ov-price-chart", "children"),
    Output("ov-arrivals-chart", "children"),
    Output("ov-tonnes-table", "children"),
    Output("ov-coverage-table", "children"),
    Output("ov-quality-box", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    price_panel = service.price_panel(filters.frequency, filters)
    if price_panel:
        frame = price_panel.unwrap()
        notes = [service.market_note(m) for m in frame.columns if service.market_note(m)]
        fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, theme, ylabel="INR per quintal")
        price_chart = C.chart_card("Teja price by market", fig, source=price_panel.source, notes=notes, height=380)
    else:
        price_chart = C.chart_card("Teja price by market", None, source=price_panel.source)

    arrivals_panel = service.arrivals_panel(filters.frequency, filters)
    if arrivals_panel:
        frame = arrivals_panel.unwrap()
        notes = [
            f"{market}: 1 bag = {service.market_bag_weight(market):g} kg per sheet header."
            for market in frame.columns if service.market_bag_weight(market)
        ]
        notes.append(
            "Bag weights differ between markets (Guntur 45 kg, Warangal and Khammam 40 kg), "
            "so bag counts are not directly comparable across markets. The tonnage table "
            "below converts them."
        )
        fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, theme, ylabel="Bags per period")
        arrivals_chart = C.chart_card("Arrivals by market", fig, source=arrivals_panel.source, notes=notes, height=340)
    else:
        arrivals_chart = C.chart_card("Arrivals by market", None, source=arrivals_panel.source)

    rows: dict[str, pd.Series] = {}
    sources: list[str] = []
    unavailable: list[str] = []
    for market in service.markets():
        converted = service.market_arrivals_tonnes(market)
        if not converted:
            unavailable.append(f"{market}: {converted.reason}")
            continue
        series = service.apply_filters(converted.unwrap(), filters, kind="arrivals")
        annual = series.resample("YE").sum()
        annual.index = annual.index.year
        rows[market] = annual
        if converted.source not in sources:
            sources.append(converted.source)
    if rows:
        tonnes_table = C.dataframe_table(
            pd.DataFrame(rows), theme, title="Arrivals in tonnes, by market and year",
            source="; ".join(sources),
            notes=[
                "Converted using each sheet's stated kilograms-per-bag. Years are calendar "
                "years and partial years appear short.",
            ] + unavailable,
            id="ov-tonnes",
        )
    else:
        tonnes_table = html.Div(
            [
                html.Div("Arrivals in tonnes, by market and year", className="card-title"),
                C.unavailable_panel(
                    "No market states a kilograms-per-bag conversion, so tonnage cannot be "
                    "derived. " + " ".join(unavailable)
                ),
            ],
            className="chart-card",
        )

    coverage_table = C.dataframe_table(
        service.coverage_table(), theme, title="Dataset coverage", source=service.data.path.name,
        notes=[
            f"Workbook read in {service.data.load_seconds:.2f}s at "
            f"{service.data.loaded_at:%d %b %Y %H:%M}.",
        ],
        id="ov-coverage",
    )

    quality = service.data_quality_notes()
    quality_box = (
        C.info_box_items("Coverage limitations that affect the analyses in this application:", quality, tone="warning")
        if quality else html.Div()
    )

    return note, price_chart, arrivals_chart, tonnes_table, coverage_table, quality_box
