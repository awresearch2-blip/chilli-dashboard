"""Seasonality -- web equivalent of ``ui.SeasonalityPage``."""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, html

from chilli_desktop import analytics, settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/seasonality", name="Seasonality",
    description="Monthly and weekly patterns, seasonal indices and the harvest calendar", order=7,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="se-filter-note"),
            C.section_header(
                "Monthly seasonal index",
                "Month mean divided by the overall mean. Above 1.00 marks a seasonally firm month.",
            ),
            html.Div(id="se-index-chart"),
            html.Div(id="se-stats-table"),
            C.section_header(
                "Distribution by month",
                "Boxes show the spread of monthly averages across years, so a reliable "
                "seasonal month can be told from a volatile one.",
            ),
            html.Div(id="se-box-chart"),
            C.section_header(
                "Calendar grid",
                "Every month of every year. Vertical banding marks a seasonal pattern; "
                "horizontal banding marks a strong year effect.",
            ),
            html.Div(id="se-calendar"),
            C.section_header("The workbook's own seasonality sheet", "Shown for comparison with the indices computed above."),
            html.Div(id="se-workbook-chart"),
            html.Div(id="se-workbook-table"),
            html.Div(id="se-compare-chart"),
            html.Div(id="se-legend-box"),
            C.section_header(
                "Harvest and lean seasons",
                "Derived by ranking calendar months on arrivals — the workbook's own harvest "
                "signature rather than an external crop calendar.",
            ),
            html.Div(id="se-season-chart"),
            html.Div(id="se-season-table"),
            C.section_header("Weekly pattern"),
            html.Div(id="se-weekday-table"),
            html.Div(id="se-festival-box"),
        ]
    )


@callback(
    Output("se-filter-note", "children"),
    Output("se-index-chart", "children"),
    Output("se-stats-table", "children"),
    Output("se-box-chart", "children"),
    Output("se-calendar", "children"),
    Output("se-workbook-chart", "children"),
    Output("se-workbook-table", "children"),
    Output("se-compare-chart", "children"),
    Output("se-legend-box", "children"),
    Output("se-season-chart", "children"),
    Output("se-season-table", "children"),
    Output("se-weekday-table", "children"),
    Output("se-festival-box", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))
    festival_box = C.info_box(
        f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nFestival effects cannot be isolated. Diwali, "
        "Sankranti and comparable festivals move through the Gregorian calendar from year to "
        "year, so separating festival demand from harvest timing requires a festival-date "
        "calendar. The workbook contains none, so the monthly patterns above necessarily "
        "blend the two.",
        tone="warning",
    )

    variety = PG.target_variety(service, filters)
    result = service.variety_series(variety) if variety else None
    if not result:
        empty = C.info_box(result.message() if result else settings.DATA_UNAVAILABLE_MESSAGE, tone="danger")
        return note, empty, *([html.Div()] * 10), festival_box

    series = service.apply_filters(result.unwrap(), filters, kind="price")
    source = result.source

    indices = analytics.seasonal_indices(series, source)
    if indices:
        table = indices.unwrap()
        fig = PC.bar_figure(
            table["Seasonal index"], theme, ylabel="Seasonal index (1.00 = average)", reference=1.0,
            reference_label="All-month average", colour_negative=False, value_labels=True, value_format=".3f",
        )
        index_chart = C.chart_card(f"Seasonal index by month — {variety}", fig, source=source, notes=indices.notes, height=320)
        stats_table = C.dataframe_table(table, theme, title="Seasonal statistics by month", source=source, notes=indices.notes, id="se-stats")
    else:
        index_chart = C.chart_card("Seasonal index by month", None, source=indices.source)
        stats_table = html.Div()

    monthly = service.series_at(series, "ME")
    if len(monthly) >= 24:
        groups = {
            settings.MONTH_ABBREVIATIONS[m - 1].title(): monthly[monthly.index.month == m] for m in range(1, 13)
        }
        fig = PC.box_by_group_figure(groups, theme, ylabel="Monthly average price (INR/quintal)")
        box_chart = C.chart_card(
            "Monthly price distribution", fig, source=source,
            notes=["Each box covers one calendar month across all years in the filtered "
                   "window. A narrow box means a dependable seasonal level; a wide one means "
                   "the month's outcome varies."], height=340,
        )
    else:
        box_chart = C.chart_card(
            "Monthly price distribution", None,
            source=f"Only {len(monthly)} monthly observation(s) after filtering; at least 24 are needed.",
        )

    if len(monthly) >= 12:
        grid = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "value": monthly.to_numpy()}).pivot_table(
            index="year", columns="month", values="value"
        )
        grid.columns = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in grid.columns]
        grid.index.name = "Year"
        fig = PC.heatmap_figure(grid, theme, diverging=False, value_format=",.0f", cbar_label="INR per quintal")
        calendar_chart = C.chart_card(
            f"{variety} monthly average price by year", fig, source=source,
            notes=["Monthly averages of the daily quotes. Blank cells are months with no recorded trade."], height=400,
        )
    else:
        calendar_chart = C.chart_card("Monthly average price by year", None, source="At least twelve monthly observations are needed.")

    workbook_seasonality = service.workbook_seasonality()
    workbook_chart = html.Div()
    workbook_table = html.Div()
    compare_chart = html.Div()
    legend_box = html.Div()
    if workbook_seasonality:
        dataset = workbook_seasonality.unwrap()
        fig = PC.heatmap_figure(dataset.frame, theme, diverging=False, value_format=",.0f", cbar_label="INR per quintal")
        workbook_chart = C.chart_card(
            "Workbook seasonality grid", fig, source=dataset.sheet_name,
            notes=[
                "Read directly from the workbook, not recomputed.",
                "Cells the workbook records as 'Closed' appear blank; they are treated as "
                "missing observations, never as zero.",
            ], height=380,
        )
        supplied = dataset.meta.get("workbook_supplied_rows")
        if supplied is not None and not supplied.empty:
            workbook_table = C.dataframe_table(
                supplied, theme, title="The workbook's own average and seasonality index rows",
                source=dataset.sheet_name,
                notes=["These are the workbook's figures. The chart above compares them with "
                       "this application's independent calculation from the daily price sheet."],
                id="se-workbook",
            )
            index_row = next((i for i in supplied.index if "season" in str(i).lower()), None)
            if index_row is not None and indices:
                ours = indices.unwrap()["Seasonal index"]
                theirs = pd.to_numeric(supplied.loc[index_row], errors="coerce")
                comparison = pd.DataFrame({"This application": ours, "Workbook": theirs}).dropna()
                if not comparison.empty:
                    comparison["Difference"] = comparison["This application"] - comparison["Workbook"]
                    fig = PC.grouped_bar_figure(comparison[["This application", "Workbook"]], theme, ylabel="Seasonal index")
                    compare_chart = C.chart_card(
                        "Seasonal index: this application against the workbook", fig,
                        source=f"{source}; {dataset.sheet_name}",
                        notes=[
                            "Both series should broadly agree. Differences arise because this "
                            "application uses the filtered daily price sheet while the "
                            "workbook used its own fixed sample.",
                            f"Median absolute difference: {comparison['Difference'].abs().median():.3f}.",
                        ], height=320,
                    )
        legend = dataset.meta.get("legend_text") or []
        if legend:
            legend_box = C.info_box_items("The sheet's own colour legend describes these bands:", legend, tone="muted")
    else:
        workbook_chart = C.chart_card("Workbook seasonality grid", None, source=workbook_seasonality.source)

    season = service.season_profile()
    if season:
        table = season.unwrap()
        values = pd.Series(table["mean_arrivals"].to_numpy(), index=pd.Index(table["month_name"], name="Month"))
        fig = PC.bar_figure(
            values, theme, ylabel="Mean arrivals (bags)", reference=float(table["mean_arrivals"].mean()),
            reference_label="All-month average", colour_negative=False,
        )
        season_chart = C.chart_card("Mean arrivals by month", fig, source=season.source, notes=season.notes, height=320)
        season_table = C.dataframe_table(
            table.set_index("month_name").rename_axis("Month"), theme, title="Season classification",
            source=season.source, notes=season.notes, id="se-season",
        )
    else:
        season_chart = C.chart_card("Mean arrivals by month", None, source=season.source)
        season_table = html.Div()

    weekday = analytics.weekday_seasonality(series, source)
    weekday_table = (
        C.dataframe_table(weekday.unwrap(), theme, title="Day-of-week statistics", source=source, notes=weekday.notes, id="se-weekday")
        if weekday else C.chart_card("Day-of-week statistics", None, source=weekday.source)
    )

    return (
        note, index_chart, stats_table, box_chart, calendar_chart, workbook_chart,
        workbook_table, compare_chart, legend_box, season_chart, season_table,
        weekday_table, festival_box,
    )
