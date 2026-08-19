"""Balance Sheet -- web equivalent of ``ui.BalanceSheetPage``."""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, html

from chilli_desktop import settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/balance", name="Balance Sheet",
    description="Production, supply, consumption, carry-forward and inventory", order=9,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="bs-unit-note"),
            C.section_header("The balance sheet as supplied"),
            html.Div(id="bs-table"),
            C.section_header("Supply and demand build-up", "Opening stock plus production against consumption plus exports."),
            html.Div(id="bs-supply-chart"),
            html.Div(id="bs-demand-chart"),
            C.section_header(
                "Stocks and the stock-to-use ratio",
                "The tightness measure: a thin buffer historically coincides with firmer, more volatile prices.",
            ),
            html.Div(id="bs-stock-chart"),
            html.Div(id="bs-ratio-chart"),
            html.Div(id="bs-scatter"),
            C.section_header("Area, production and yield (APY)", "State-level crop statistics as supplied in the workbook."),
            html.Div(id="bs-apy-table"),
            html.Div(id="bs-apy-charts"),
            C.section_header("Cold storage stock", "Reported positions by state and market."),
            html.Div(id="bs-cold-table"),
            html.Div(id="bs-cold-warning"),
            html.Div(id="bs-cold-chart"),
        ]
    )


@callback(
    Output("bs-unit-note", "children"),
    Output("bs-table", "children"),
    Output("bs-supply-chart", "children"),
    Output("bs-demand-chart", "children"),
    Output("bs-stock-chart", "children"),
    Output("bs-ratio-chart", "children"),
    Output("bs-scatter", "children"),
    Output("bs-apy-table", "children"),
    Output("bs-apy-charts", "children"),
    Output("bs-cold-table", "children"),
    Output("bs-cold-warning", "children"),
    Output("bs-cold-chart", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)

    balance = service.balance_sheet()
    if not balance:
        empty = C.info_box(balance.message(), tone="danger")
        unit_note, table, supply_chart, demand_chart, stock_chart, ratio_chart, scatter = (
            empty, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), html.Div()
        )
    else:
        frame = balance.unwrap()
        dataset = service.data.datasets["balance_sheet"]
        projected = dataset.meta.get("projected_years", [])
        unit_note = C.info_box(
            f"Units: {dataset.meta.get('unit_note', 'not stated')}. "
            + (
                f"Year(s) {', '.join(str(y) for y in projected)} are marked as expected in "
                "the workbook — these are the workbook's own projections, not this "
                "application's forecasts, and are hatched wherever they appear in a chart."
                if projected else "All years are realised data."
            ),
            tone="warning" if projected else "muted",
        )
        table = C.dataframe_table(frame, theme, title="Red chilli balance sheet", source=balance.source, notes=balance.notes, id="bs-full")

        supply_rows = [r for r in frame.index if any(k in str(r).lower() for k in ("openingstock", "opening stock", "production", "import"))]
        demand_rows = [r for r in frame.index if any(k in str(r).lower() for k in ("consumption", "export"))]
        supply_chart = html.Div()
        if supply_rows:
            fig = PC.grouped_bar_figure(frame.loc[supply_rows].T, theme, ylabel=dataset.meta.get("unit_note", ""), stacked=True)
            supply_chart = C.chart_card(
                "Supply components by year", fig, source=balance.source,
                notes=["Stacked to show total supply." + (f" {', '.join(str(y) for y in projected)} is the workbook's expectation." if projected else "")],
                height=320,
            )
        demand_chart = html.Div()
        if demand_rows:
            fig = PC.grouped_bar_figure(frame.loc[demand_rows].T, theme, ylabel=dataset.meta.get("unit_note", ""), stacked=True)
            demand_chart = C.chart_card("Demand components by year", fig, source=balance.source, notes=["Stacked to show total offtake."], height=320)

        ending_row = service.balance_sheet_row("Ending Stock")
        stock_chart = html.Div()
        if ending_row:
            values = ending_row.unwrap()
            fig = PC.bar_figure(
                values, theme, ylabel=dataset.meta.get("unit_note", ""), reference=float(values.mean()),
                reference_label="Sample average", colour_negative=False, value_labels=True,
                value_format=",.2f", projected=[str(y) for y in projected],
            )
            stock_chart = C.chart_card("Ending stock by year", fig, source=ending_row.source, notes=ending_row.notes, height=300)

        stock_use = service.balance_sheet_row("Stock to Use")
        ratio_chart = html.Div()
        if stock_use:
            values = stock_use.unwrap()
            fig = PC.bar_figure(
                values, theme, ylabel="%", reference=float(values.mean()), reference_label="Sample average",
                colour_negative=False, value_labels=True, value_format=",.2f", projected=[str(y) for y in projected],
            )
            ratio_chart = C.chart_card("Stock-to-use ratio by year", fig, source=stock_use.source, notes=stock_use.notes, height=300)

        scatter = html.Div()
        price_result = service.variety_series("Teja")
        if price_result and stock_use:
            annual_price = service.series_at(price_result.unwrap(), "YE")
            annual_price = pd.Series(annual_price.to_numpy(), index=pd.Index(annual_price.index.year, name="year"))
            joined = pd.concat([annual_price.rename("Annual average price"), stock_use.unwrap().rename("Stock-to-use %")], axis=1).dropna()
            if len(joined) >= 4:
                fig = PC.scatter_fit_figure(
                    joined["Stock-to-use %"], joined["Annual average price"], theme,
                    xlabel="Stock-to-use ratio (%)", ylabel="Annual average Teja price (INR/quintal)",
                    colour_by_date=False,
                )
                scatter = html.Div([
                    C.chart_card(
                        "Stock-to-use ratio against annual average price", fig,
                        source=f"{balance.source}; {price_result.source}",
                        notes=[f"Only {len(joined)} paired years are available. That is far "
                               "too few for a significance test or a regression — read the "
                               "fit as a directional observation, not a finding."], height=340,
                    ),
                    C.dataframe_table(
                        joined, theme, title="Paired annual data", source=f"{balance.source}; {price_result.source}",
                        notes=["Calendar-year average of the daily Teja quotes."], id="bs-paired",
                    ),
                ])
            else:
                scatter = C.chart_card(
                    "Stock-to-use ratio against annual average price", None,
                    source=f"Only {len(joined)} year(s) have both a balance-sheet ratio and a price average.",
                )

    apy = service.apy()
    apy_table = html.Div()
    apy_charts = html.Div()
    if apy:
        frame = apy.unwrap()
        dataset = service.data.datasets["apy"]
        projected = dataset.meta.get("projected_years", [])
        apy_table = C.dataframe_table(
            frame, theme, title="APY by state and year", source=apy.source,
            notes=[
                f"Year(s) marked expected in the workbook: {', '.join(str(y) for y in projected) or 'none'}.",
                "The most recent expected year carries an area figure only; production and "
                "yield are absent and are shown as blank rather than estimated.",
            ], id="bs-apy",
        )
        national = dataset.meta.get("national_row")
        charts = []
        for metric, unit in (("Production (MT)", "tonnes"), ("Area (Ha)", "hectares"), ("Yield (t/Ha)", "tonnes per hectare")):
            if metric not in frame.columns:
                continue
            pivot = frame.pivot_table(index="year", columns="state", values=metric)
            if national and national in pivot.columns:
                fig = PC.bar_figure(
                    pivot[national].dropna(), theme, ylabel=f"{metric} ({unit})", colour_negative=False,
                    projected=[str(y) for y in projected],
                )
                charts.append(C.chart_card(f"{metric} by year", fig, source=apy.source, notes=[f"All-India total ('{national}' row). State detail is in the table above."], height=300))
            else:
                states = [c for c in pivot.columns if c != national]
                fig = PC.grouped_bar_figure(pivot[states], theme, ylabel=f"{metric} ({unit})", stacked=True)
                charts.append(C.chart_card(f"{metric} by year", fig, source=apy.source, height=300))
        apy_charts = C.grid_of(charts, 2) if charts else html.Div()
    else:
        apy_table = C.info_box(apy.message(), tone="warning")

    cold = service.cold_storage()
    cold_table = html.Div()
    cold_warning = html.Div()
    cold_chart = html.Div()
    if cold:
        frame = cold.unwrap()
        dataset = service.data.datasets["cold_storage_stock"]
        cold_table = C.dataframe_table(
            frame, theme, title="Cold storage stock as reported", source=cold.source,
            notes=list(cold.notes) + [dataset.meta.get("unit", "")], id="bs-cold",
        )
        coverage = dataset.meta.get("observations_per_location", {})
        cold_warning = C.info_box(
            f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n"
            f"Inventory-versus-price analysis, delayed storage effects and seasonal storage "
            f"behaviour cannot be computed. The sheet holds {len(frame)} reporting month(s) "
            f"across {frame.shape[1]} location(s), with at most "
            f"{max(coverage.values(), default=0)} observation(s) for any single location — "
            f"against the minimum this application requires before reporting any "
            "correlation. The levels above are shown for reference only.\n\n"
            "The one storage relationship the workbook does support is the Khammam "
            "cold-storage price premium over fresh lots, which is computed from the two "
            "Khammam price sheets and reported on the Automated Insights page.",
            tone="warning",
        )
        longest = max(coverage.items(), key=lambda kv: kv[1], default=(None, 0))
        if longest[0] and longest[1] >= 2:
            fig = PC.bar_figure(frame[longest[0]].dropna(), theme, ylabel="Bags", colour_negative=False, value_labels=True)
            cold_chart = C.chart_card(
                "Reported stock by location", fig, source=cold.source,
                notes=[f"'{longest[0]}' is the best-covered column with {longest[1]} "
                       "observation(s). Shown as discrete readings, not a time series, "
                       "because the reporting is too sparse to join into a line."], height=300,
            )
        else:
            cold_chart = C.chart_card("Reported stock by location", None, source="No location has even two observations.")
    else:
        cold_table = C.chart_card("Cold storage stock as reported", None, source=cold.source)

    return (
        unit_note, table, supply_chart, demand_chart, stock_chart, ratio_chart, scatter,
        apy_table, apy_charts, cold_table, cold_warning, cold_chart,
    )
