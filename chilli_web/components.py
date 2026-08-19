"""Reusable Dash building blocks -- the web equivalents of the Qt widgets in
``chilli_desktop.ui`` (``ChartPanel``, ``DataTable``, ``SummaryCard``,
``InsightCard``, ``InfoBox``, ``SectionHeader``).

Every card, table and chart wrapper carries the same three things the desktop
widgets were built to always show: a Source caption naming the workbook sheet
(strict data rule 8), an expandable Notes/assumptions block (strict data rule
9), and a dashed "Data not available in uploaded workbook." panel in place of
an empty chart whenever the backend returns an unavailable ``Result``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Sequence

import numpy as np
import pandas as pd
from dash import ALL, MATCH, Input, Output, State, callback, dash_table, dcc, html
import plotly.graph_objects as go

from chilli_desktop import insights as insights_mod
from chilli_desktop import settings
from chilli_desktop.settings import Theme
from chilli_desktop.utils import Result

from . import plotly_charts

# ==========================================================================
# Source / notes / unavailable
# ==========================================================================


def source_caption(source: str) -> html.Div:
    return html.Div(f"Source: {source or '—'}", className="source-caption")


def notes_block(notes: Sequence[str] | None) -> html.Details | None:
    usable = [n for n in (notes or []) if n]
    if not usable:
        return None
    return html.Details(
        [
            html.Summary(f"Notes ({len(usable)})", className="notes-summary"),
            html.Ul([html.Li(n) for n in usable], className="notes-list"),
        ],
        className="notes-block",
    )


def unavailable_panel(message_or_result: Any, source: str = "", *, height: int = 200) -> html.Div:
    """The dashed "no data" panel. Mirrors ``ChartPanel.show_unavailable``."""
    if isinstance(message_or_result, Result):
        message = message_or_result.message()
        source = source or message_or_result.source
    else:
        message = str(message_or_result)
        if settings.DATA_UNAVAILABLE_MESSAGE not in message:
            message = f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{message}"
    return html.Div(
        [
            html.Div(
                [html.Span(line) for chunk in message.split("\n") for line in [chunk, html.Br()]][:-1],
                className="unavailable-message",
            ),
            source_caption(source),
        ],
        className="unavailable-panel",
        style={"minHeight": f"{height}px"},
    )


# ==========================================================================
# Chart card
# ==========================================================================

_card_counter = 0


def _next_id(prefix: str) -> str:
    global _card_counter
    _card_counter += 1
    return f"{prefix}-{_card_counter}"


def chart_card(
    title: str,
    figure: go.Figure | None,
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
    height: int = 360,
    id: str | None = None,
) -> html.Div:
    """A titled chart panel. Mirrors ``charts.ChartPanel``.

    Pass ``figure=None`` (with ``notes``/``source`` still set from the
    unavailable ``Result``) to render the dashed unavailable state instead.
    """
    if figure is None:
        return html.Div(
            [html.Div(title, className="card-title"), unavailable_panel(source or "", height=height)],
            className="chart-card",
        )
    graph_id = id or _next_id("chart")
    return html.Div(
        [
            html.Div(title, className="card-title"),
            dcc.Graph(
                id=graph_id,
                figure=figure,
                config=plotly_charts.figure_config(),
                style={"height": f"{height}px"},
            ),
            source_caption(source),
            notes_block(notes),
        ],
        className="chart-card",
    )


def chart_or_unavailable(
    title: str,
    result: Result[Any],
    build: Any,
    *,
    height: int = 360,
    id: str | None = None,
) -> html.Div:
    """Build a chart card from a ``Result``, dispatching to the unavailable
    state automatically. ``build`` receives the unwrapped value and must
    return a ``go.Figure``."""
    if not result:
        return chart_card(title, None, source=result.source, height=height)
    return chart_card(title, build(result.unwrap()), source=result.source, notes=result.notes,
                      height=height, id=id)


# ==========================================================================
# Tables
# ==========================================================================


def _format_cell(value: Any) -> str:
    """Mirrors ``ui.PandasModel._format`` exactly, so a number reads the same
    way whether you're looking at the desktop app or the browser."""
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, pd.Timestamp):
        return "—" if pd.isna(value) else value.strftime("%d %b %Y")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return "—"
        magnitude = abs(number)
        if magnitude != 0 and magnitude < 0.001:
            return f"{number:.2e}"
        if magnitude < 1:
            return f"{number:,.4f}"
        if magnitude < 1000:
            return f"{number:,.2f}"
        return f"{number:,.0f}"
    text = str(value)
    return "—" if text in ("nan", "NaT", "None", "") else text


#: Column-name fragments that pin two-decimal formatting (percentages,
#: errors) so "0.0000" never sits beside "59.62" -- same rule as the desktop
#: table model.
_TWO_DECIMAL_HINTS = ("%", "percent", "change", "ratio", "mape", "rmse", "mae")

_SIGNED_HINTS = ("change", "elasticity", "coefficient", "beta", "correlation", "r²", "r2")


def _cell_colour(column: str, raw: Any, theme: Theme) -> str | None:
    lowered = column.lower()
    if isinstance(raw, (bool, np.bool_)):
        return theme.positive if bool(raw) else theme.text_muted
    if isinstance(raw, str) and raw.strip().upper() == "SELECTED":
        return theme.accent
    if isinstance(raw, (float, np.floating, int, np.integer)) and not isinstance(raw, (bool, np.bool_)):
        number = float(raw)
        if not np.isfinite(number):
            return None
        if any(h in lowered for h in _SIGNED_HINTS):
            if number > 0:
                return theme.positive
            if number < 0:
                return theme.negative
        if "p-value" in lowered and number < settings.ANALYTICS.alpha:
            return theme.accent
    return None


def dataframe_table(
    frame: pd.DataFrame,
    theme: Theme,
    *,
    title: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    id: str | None = None,
    page_size: int = 15,
) -> html.Div:
    """A titled, coloured, exportable table. Mirrors ``ui.DataTable``."""
    if frame is None or frame.empty:
        return html.Div(
            [html.Div(title, className="card-title"), unavailable_panel("The table contains no rows.", source)],
            className="chart-card",
        )

    table_id = id or _next_id("table")
    has_named_index = not isinstance(frame.index, pd.RangeIndex)
    index_name = frame.index.name or "Index"
    display = frame.reset_index() if has_named_index else frame.copy()
    if has_named_index:
        display.columns = [index_name] + list(frame.columns)

    columns = [{"name": str(c), "id": str(c)} for c in display.columns]
    records: list[dict[str, str]] = []
    style_data_conditional: list[dict[str, Any]] = []
    for row_index in range(len(display)):
        record: dict[str, str] = {}
        for column in display.columns:
            raw = display.iloc[row_index][column]
            record[str(column)] = _format_cell(raw)
            colour = _cell_colour(str(column), raw, theme)
            if colour:
                style_data_conditional.append(
                    {"if": {"row_index": row_index, "column_id": str(column)}, "color": colour}
                )
        records.append(record)

    csv_text = frame.to_csv(index=has_named_index)

    table = dash_table.DataTable(
        id=f"{table_id}-grid",
        columns=columns,
        data=records,
        page_size=page_size,
        sort_action="native",
        filter_action="native" if len(display) > 12 else "none",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": theme.surface_alt,
            "color": theme.text_muted,
            "fontWeight": "600",
            "fontSize": "11px",
            "border": f"1px solid {theme.border}",
            "textTransform": "uppercase",
        },
        style_cell={
            "backgroundColor": theme.surface,
            "color": theme.text,
            "fontSize": "11px",
            "padding": "6px 10px",
            "border": f"1px solid {theme.border}",
            "textAlign": "right",
            "fontFamily": "Segoe UI, Calibri, sans-serif",
        },
        style_cell_conditional=[
            {"if": {"column_id": str(display.columns[0])}, "textAlign": "left", "fontWeight": "600"}
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": theme.surface_alt},
            *style_data_conditional,
        ],
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(title, className="card-title"),
                    html.Button(
                        "CSV", id={"type": "csv-btn", "index": table_id}, className="chip-button", n_clicks=0
                    ),
                ],
                className="card-header-row",
            ),
            table,
            dcc.Store(id={"type": "csv-data", "index": table_id}, data=csv_text),
            dcc.Store(id={"type": "csv-name", "index": table_id}, data=f"{title or table_id}.csv"),
            dcc.Download(id={"type": "csv-download", "index": table_id}),
            source_caption(source),
            notes_block(notes),
        ],
        className="chart-card",
    )


@callback(
    Output({"type": "csv-download", "index": MATCH}, "data"),
    Input({"type": "csv-btn", "index": MATCH}, "n_clicks"),
    State({"type": "csv-data", "index": MATCH}, "data"),
    State({"type": "csv-name", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def _download_csv(n_clicks: int, csv_text: str, filename: str):
    """One pattern-matching callback serves every table's CSV button."""
    return dict(content=csv_text, filename=filename)


# ==========================================================================
# Cards
# ==========================================================================


def summary_card(
    label: str, value: str, delta: str = "", caption: str = "", *, tone: str = "neutral"
) -> html.Div:
    """Mirrors ``ui.SummaryCard``."""
    return html.Div(
        [
            html.Div(label, className="summary-label"),
            html.Div(value, className="summary-value"),
            html.Div(delta, className="summary-delta") if delta else None,
            html.Div(caption, className="summary-caption") if caption else None,
        ],
        className=f"summary-card tone-{tone}",
    )


def card_row(cards: Sequence[html.Div]) -> html.Div:
    return html.Div(list(cards), className="card-row")


def grid_of(widgets: Sequence[html.Div], columns: int = 2) -> html.Div:
    return html.Div(list(widgets), className=f"grid-cols-{columns}")


_INSIGHT_TONE = {
    "strong": "positive",
    "moderate": "accent",
    "weak": "warning",
    "informational": "neutral",
    "data gap": "negative",
}


def insight_card(insight: insights_mod.Insight, *, id: str | None = None) -> html.Div:
    """Mirrors ``ui.InsightCard``."""
    tone = _INSIGHT_TONE.get(insight.strength, "neutral")
    header = [
        html.Span(insight.strength.upper(), className=f"insight-badge tone-{tone}"),
        html.Span(insight.category, className="insight-category"),
    ]
    if insight.direction not in ("n/a", ""):
        header.append(html.Span(insight.direction.upper(), className="insight-direction"))

    body: list[Any] = [
        html.Div(header, className="insight-header"),
        html.Div(insight.headline, className="insight-headline"),
    ]
    if insight.detail:
        body.append(html.Div(insight.detail, className="insight-detail"))
    body.append(source_caption(insight.source))
    if insight.evidence:
        card_id = id or _next_id("insight")
        body.append(
            html.Details(
                [
                    html.Summary(f"Evidence ({len(insight.evidence)})", className="notes-summary"),
                    html.Ul([html.Li(e) for e in insight.evidence], className="notes-list"),
                ],
                className="notes-block",
            )
        )
    return html.Div(body, className=f"insight-card tone-{tone}")


def info_box(text: str = "", *, title: str = "", tone: str = "info") -> html.Div:
    """Mirrors ``ui.InfoBox``."""
    children: list[Any] = []
    if title:
        children.append(html.Div(title, className="info-title"))
    children.append(html.Div([html.Span(line) for chunk in text.split("\n") for line in [chunk, html.Br()]][:-1]))
    return html.Div(children, className=f"info-box tone-{tone}")


def info_box_items(title: str, items: Sequence[str], *, tone: str = "info") -> html.Div:
    usable = [i for i in items if i]
    if not usable:
        return html.Div(style={"display": "none"})
    return html.Div(
        [
            html.Div(title, className="info-title") if title else None,
            html.Ul([html.Li(i) for i in usable]),
        ],
        className=f"info-box tone-{tone}",
    )


def section_header(title: str, subtitle: str = "") -> html.Div:
    """Mirrors ``ui.SectionHeader``."""
    return html.Div(
        [
            html.Div(title, className="section-title"),
            html.Div(subtitle, className="section-subtitle") if subtitle else None,
        ],
        className="section-header",
    )


def filter_note(filters_describe: str, frequency_label: str) -> html.Div:
    return info_box(f"Sample: {filters_describe}  Analysis frequency: {frequency_label}.", tone="muted")
