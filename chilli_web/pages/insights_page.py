"""Automated Insights -- web equivalent of ``ui.AutomatedInsightsPage``.

The sweep across every generator in :mod:`chilli_desktop.insights` runs once
as a background callback and is cached in a ``dcc.Store``; the strength and
category filters then re-render instantly from that store without
re-running any statistics, which is actually snappier than the desktop
page's approach of rebuilding the same widgets on every dropdown change.
"""

from __future__ import annotations

import types

import dash
from dash import Input, Output, callback, dcc, html

from chilli_desktop import insights as insights_mod
from chilli_desktop import settings
from chilli_web import components as C
from chilli_web import page_common as PG

dash.register_page(
    __name__, path="/insights", name="Automated Insights",
    description="Every finding the workbook supports, ranked by strength of evidence", order=11,
)

_STRENGTH_OPTIONS = [
    {"label": "All findings", "value": "all"},
    {"label": "Strong only", "value": "strong"},
    {"label": "Strong and moderate", "value": "strong_moderate"},
    {"label": "Data gaps only", "value": "data gap"},
]


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="in-filter-note"),
            C.info_box(
                "Findings are generated from the workbook only. Each carries the statistic "
                "behind it and the sheet it came from. Items marked DATA GAP record "
                "questions the workbook cannot answer — they are findings too.",
                tone="info",
            ),
            html.Div(
                [
                    html.Div("Show:", style={"marginRight": "8px", "color": "var(--text-muted)", "fontSize": "11px"}),
                    dcc.Dropdown(id="in-strength-filter", options=_STRENGTH_OPTIONS, value="all", clearable=False, style={"width": "220px"}),
                    dcc.Dropdown(id="in-category-filter", options=[{"label": "All categories", "value": "all"}], value="all", clearable=False, style={"width": "220px", "marginLeft": "8px"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
            ),
            html.Div(id="in-status", children=C.info_box("Generating insights…", tone="muted")),
            dcc.Store(id="in-store", data=[]),
            html.Div(id="in-failures-box"),
            dcc.Loading(html.Div(id="in-cards"), type="dot"),
        ]
    )


@callback(Output("in-filter-note", "children"), Input("filters-store", "data"))
def _render_note(filters_data):
    service, filters, _theme = PG.current(filters_data, None)
    return C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))


@callback(
    Output("in-store", "data"),
    Output("in-status", "children"),
    Output("in-failures-box", "children"),
    Output("in-category-filter", "options"),
    Input("filters-store", "data"),
    background=True,
)
def _sweep(filters_data):
    service, filters, _theme = PG.current(filters_data, None)
    variety = PG.target_variety(service, filters)
    found, failures = insights_mod.generate_all(service, filters, variety)

    counts: dict[str, int] = {}
    for insight in found:
        counts[insight.strength] = counts.get(insight.strength, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in counts.items())
    status = C.info_box(f"{len(found)} finding(s): {summary}.", tone="muted")

    failures_box = (
        C.info_box_items("Some generators could not complete:", failures, tone="warning")
        if failures else html.Div()
    )

    categories = sorted({i.category for i in found})
    category_options = [{"label": "All categories", "value": "all"}] + [{"label": c, "value": c} for c in categories]

    serialised = [
        {
            "category": i.category, "headline": i.headline, "detail": i.detail,
            "source": i.source, "strength": i.strength, "direction": i.direction,
            "evidence": i.evidence,
        }
        for i in found
    ]
    return serialised, status, failures_box, category_options


@callback(
    Output("in-cards", "children"),
    Input("in-store", "data"),
    Input("in-strength-filter", "value"),
    Input("in-category-filter", "value"),
)
def _apply_filter(stored, strength_choice, category_choice):
    if not stored:
        return html.Div()

    allowed = {
        "all": None, "strong": {"strong"}, "strong_moderate": {"strong", "moderate"}, "data gap": {"data gap"},
    }[strength_choice or "all"]

    cards = []
    for record in stored:
        if allowed is not None and record["strength"] not in allowed:
            continue
        if category_choice not in (None, "all") and record["category"] != category_choice:
            continue
        cards.append(C.insight_card(types.SimpleNamespace(**record)))

    if not cards:
        return C.info_box("No finding matches the current selection.", tone="muted")
    return html.Div(cards)
