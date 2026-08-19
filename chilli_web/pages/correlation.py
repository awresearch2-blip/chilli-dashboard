"""Correlation Studio -- web equivalent of ``ui.CorrelationStudioPage``."""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, html

from chilli_desktop import analytics, settings
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__, path="/correlation", name="Correlation Studio",
    description="Pearson, Spearman, rolling, lag and cross-correlation", order=6,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="cs-filter-note"),
            C.section_header(
                "Correlation matrices",
                "Recomputed from the daily price sheet at the selected frequency. Pearson "
                "measures linear co-movement; Spearman measures rank agreement and is robust "
                "to outliers.",
            ),
            html.Div(id="cs-matrices"),
            html.Div(id="cs-workbook-matrix"),
            html.Div(id="cs-reconciliation"),
            C.section_header("Rolling correlation", "Whether a relationship holds through time, or breaks down."),
            html.Div(id="cs-rolling"),
            C.section_header(
                "Cross-correlation and lag explorer",
                "Positive lag means the first series leads. Both series are differenced "
                "first, so a shared trend cannot manufacture a peak.",
            ),
            html.Div(id="cs-cross"),
            C.section_header(
                "Multicollinearity among drivers",
                "Two drivers carrying the same information cannot be separately attributed "
                "in any regression.",
            ),
            html.Div(id="cs-vif"),
        ]
    )


@callback(
    Output("cs-filter-note", "children"),
    Output("cs-matrices", "children"),
    Output("cs-workbook-matrix", "children"),
    Output("cs-reconciliation", "children"),
    Output("cs-rolling", "children"),
    Output("cs-cross", "children"),
    Output("cs-vif", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    panel_result = service.variety_panel(filters.frequency, filters)
    if not panel_result:
        empty = C.info_box(panel_result.message(), tone="danger")
        return note, empty, *([html.Div()] * 5)
    frame = panel_result.unwrap()
    source = panel_result.source

    matrices = []
    for method in ("pearson", "spearman"):
        result = analytics.correlation_matrix(frame, method, source)
        if result:
            fig = PC.heatmap_figure(result.unwrap(), theme, diverging=True, vmin=-1, vmax=1, cbar_label="Correlation")
            matrices.append(C.chart_card(f"{method.title()} correlation", fig, source=source, notes=result.notes, height=400))
        else:
            matrices.append(C.chart_card(f"{method.title()} correlation", None, source=result.source))
    matrices_block = C.grid_of(matrices, 2)

    workbook_matrix = service.workbook_variety_correlation()
    if workbook_matrix:
        fig = PC.heatmap_figure(workbook_matrix.unwrap(), theme, diverging=True, vmin=-1, vmax=1, cbar_label="Correlation")
        workbook_chart = C.chart_card(
            "The workbook's own correlation matrix, for comparison", fig, source=workbook_matrix.source,
            notes=list(workbook_matrix.notes) + [
                "This matrix is read directly from the workbook and is not recomputed. It "
                "covers more varieties than the daily price sheet quotes, so some rows and "
                "columns have no counterpart above.",
            ], height=400,
        )
    else:
        workbook_chart = C.chart_card("The workbook's own correlation matrix, for comparison", None, source=workbook_matrix.source)

    reconciliation_block = html.Div()
    if workbook_matrix:
        recomputed = analytics.correlation_matrix(frame, "pearson", source)
        if recomputed:
            ours = recomputed.unwrap()
            theirs = workbook_matrix.unwrap()
            rows = []
            for row_label in ours.index:
                match_row = next((r for r in theirs.index if str(r).lower() == str(row_label).lower()), None)
                if match_row is None:
                    continue
                for column_label in ours.columns:
                    if str(column_label) == str(row_label):
                        continue
                    match_col = next((c for c in theirs.columns if str(c).lower() == str(column_label).lower()), None)
                    if match_col is None:
                        continue
                    mine = ours.loc[row_label, column_label]
                    yours = theirs.loc[match_row, match_col]
                    if pd.isna(mine) or pd.isna(yours):
                        continue
                    rows.append({"Pair": f"{row_label} / {column_label}", "This application": float(mine), "Workbook": float(yours), "Difference": float(mine) - float(yours)})
            if rows:
                reconciliation = (
                    pd.DataFrame(rows).drop_duplicates(subset="Pair").set_index("Pair")
                    .sort_values("Difference", key=abs, ascending=False)
                )
                reconciliation_block = C.dataframe_table(
                    reconciliation, theme, title="Reconciliation against the workbook",
                    source=f"{source}; {workbook_matrix.source}",
                    notes=[
                        "Differences are expected: this application correlates "
                        f"{PG.frequency_label(filters.frequency).lower()} averages over the "
                        "filtered window, while the workbook's matrix was computed over its "
                        "own sample. Large gaps are worth investigating; small ones confirm "
                        "both are measuring the same thing.",
                        f"Median absolute difference: {reconciliation['Difference'].abs().median():.3f}.",
                    ], id="cs-reconciliation",
                )

    columns = list(frame.columns)
    pairs = [(columns[0], c) for c in columns[1:4]] if len(columns) > 1 else []
    markets = service.price_panel(filters.frequency, filters)
    market_frame = markets.unwrap() if markets and markets.unwrap().shape[1] > 1 else None
    if market_frame is not None:
        market_columns = list(market_frame.columns)
        for other in market_columns[1:3]:
            pairs.append((market_columns[0], other))

    window = max(12, min(52, len(frame) // 6))
    rolling_charts = []
    cross_charts = []
    for left, right in pairs[:4]:
        left_series = frame[left] if left in frame.columns else market_frame[left]
        right_series = frame[right] if right in frame.columns else market_frame[right]

        rolling_result = analytics.rolling_correlation(left_series, right_series, window, source)
        if rolling_result:
            fig = PC.line_figure({f"{window}-period correlation": rolling_result.unwrap()}, theme, ylabel="Correlation")
            rolling_charts.append(C.chart_card(f"{left} vs {right} — rolling correlation", fig, source=source, notes=rolling_result.notes, height=280))
        else:
            rolling_charts.append(C.chart_card(f"{left} vs {right} — rolling correlation", None, source=rolling_result.source))

        cross_result = analytics.cross_correlation(left_series, right_series, 20, source, differenced=True)
        if cross_result:
            table = cross_result.unwrap()
            peak = int(table["correlation"].abs().idxmax())
            fig = PC.stem_figure(
                table["correlation"], theme, xlabel=f"Lag ({left} leading →)", ylabel="Correlation of changes",
                band=float(table["upper_95"].iloc[0]), highlight_index=peak,
            )
            cross_charts.append(C.chart_card(f"{left} → {right} — cross-correlation", fig, source=source, notes=cross_result.notes, height=300))
        else:
            cross_charts.append(C.chart_card(f"{left} → {right} — cross-correlation", None, source=cross_result.source))

    rolling_block = C.grid_of(rolling_charts, 2) if rolling_charts else C.info_box(
        f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nAt least two series are needed for a rolling correlation.",
        tone="warning",
    )
    cross_block = C.grid_of(cross_charts, 2) if cross_charts else html.Div()

    exog = service.exogenous_matrix("ME")
    if exog:
        vif = analytics.variance_inflation(exog.unwrap(), exog.source)
        vif_block = (
            C.dataframe_table(vif.unwrap(), theme, title="Variance inflation factors", source=exog.source, notes=vif.notes, id="cs-vif")
            if vif else C.chart_card("Variance inflation factors", None, source=vif.source)
        )
    else:
        vif_block = C.chart_card("Variance inflation factors", None, source=exog.source)

    return note, matrices_block, workbook_chart, reconciliation_block, rolling_block, cross_block, vif_block
