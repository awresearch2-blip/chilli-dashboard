"""Market Integration -- web equivalent of ``ui.MarketIntegrationPage``.

Runs on **daily** data regardless of the sidebar frequency, for the same
reason the desktop page does: averaging to weeks or months would erase a
one- or two-day lead, which is exactly what this page exists to measure. The
Granger-causality battery behind the influence diagram is the single most
expensive computation in the application, so it runs as a background
callback, same as the desktop page dispatches it to a worker thread.
"""

from __future__ import annotations

import dash
import numpy as np
from dash import Input, Output, callback, dcc, html

from chilli_desktop import analytics
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

ANALYSIS_FREQ = "D"

dash.register_page(
    __name__, path="/integration", name="Market Integration",
    description="Does Guntur drive Warangal and Khammam? Direction, strength and cointegration",
    order=5,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="mi-filter-note"),
            C.info_box(
                "Lead-lag and causality on this page are computed on DAILY data regardless "
                "of the sidebar frequency: averaging to weeks or months would hide a one- or "
                "two-day lead, which is exactly what is being measured here. Cointegration is "
                "also run on daily levels.",
                tone="info",
            ),
            html.Div(id="mi-price-chart"),
            html.Div(id="mi-spread-chart"),
            C.section_header(
                "Influence structure",
                "Computed with Granger causality run in both directions for every pair. "
                "This can take a few seconds.",
            ),
            dcc.Loading(html.Div(id="mi-network"), type="circle"),
            html.Div(id="mi-influence-table"),
            html.Div(id="mi-leadership-table"),
            html.Div(id="mi-leadership-box"),
            C.section_header("Lead-lag timing", "Peak cross-correlation of the differenced series, in periods and in days."),
            html.Div(id="mi-leadlag-table"),
            C.section_header(
                "Cointegration",
                "The formal test of market integration: cointegrated markets share a "
                "long-run equilibrium and their spreads mean-revert.",
            ),
            html.Div(id="mi-johansen-table"),
            html.Div(id="mi-pairwise-table"),
            html.Div(id="mi-cointegration-box"),
        ]
    )


@callback(
    Output("mi-filter-note", "children"),
    Output("mi-price-chart", "children"),
    Output("mi-spread-chart", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render_fast(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    panel_result = service.price_panel(ANALYSIS_FREQ, filters)
    if not panel_result:
        return note, C.info_box(panel_result.message(), tone="danger"), html.Div()

    frame = panel_result.unwrap()
    source = panel_result.source
    fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, theme, ylabel="INR per quintal")
    price_chart = C.chart_card("Teja price by market (daily)", fig, source=source, height=340)

    reference = "Guntur" if "Guntur" in frame.columns else str(frame.columns[0])
    spreads = {f"{column} − {reference}": (frame[column] - frame[reference]).dropna() for column in frame.columns if column != reference}
    if spreads:
        fig = PC.line_figure(spreads, theme, ylabel="INR per quintal")
        spread_chart = C.chart_card(
            f"Spread to {reference}", fig, source=source,
            notes=["A spread that oscillates around a stable level indicates integrated "
                   "markets. A spread that trends away indicates divergence."], height=300,
        )
    else:
        spread_chart = html.Div()

    return note, price_chart, spread_chart


def _compute(filters_data):
    service, filters, _theme = PG.current(filters_data, None)
    panel_result = service.price_panel(ANALYSIS_FREQ, filters)
    if not panel_result:
        return {"error": panel_result.message()}
    frame = panel_result.unwrap()
    source = panel_result.source
    return {
        "leadership": analytics.leadership_ranking(frame, ANALYSIS_FREQ, source, 20),
        "leadlag": analytics.lead_lag_matrix(frame, ANALYSIS_FREQ, 20, source),
        "cointegration": analytics.cointegration(frame.dropna(), source),
        "source": source,
    }


@callback(
    Output("mi-network", "children"),
    Output("mi-influence-table", "children"),
    Output("mi-leadership-table", "children"),
    Output("mi-leadership-box", "children"),
    Output("mi-leadlag-table", "children"),
    Output("mi-johansen-table", "children"),
    Output("mi-pairwise-table", "children"),
    Output("mi-cointegration-box", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
    background=True,
)
def _render_battery(filters_data, theme_name):
    _service, _filters, theme = PG.current(filters_data, theme_name)
    payload = _compute(filters_data)
    if payload.get("error"):
        empty = C.info_box(payload["error"], tone="danger")
        return empty, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), html.Div()

    source = payload["source"]
    leadership = payload["leadership"]
    if leadership:
        table = leadership.unwrap()
        leadership_table = C.dataframe_table(table, theme, title="Market leadership ranking", source=source, notes=leadership.notes, id="mi-leadership")
        pairs = table.attrs.get("pairs")
        if pairs is not None and not pairs.empty:
            influence_table = C.dataframe_table(pairs.drop(columns=["Note"], errors="ignore"), theme, title="Pairwise direction of influence", source=source, id="mi-influence")
            edges = []
            for _, row in pairs.iterrows():
                verdict = str(row["Verdict"])
                direction = str(row["Direction"])
                if verdict == "One-way" and "->" in direction:
                    start, end = [s.strip() for s in direction.split("->")]
                    strength = -np.log10(max(float(row["A -> B best p"]), 1e-12))
                    edges.append((start, end, float(strength), "one-way"))
                elif verdict.startswith("Feedback") and "->" in direction:
                    start = direction.split("dominant")[0].strip()
                    end = direction.split("->")[-1].strip()
                    strength = -np.log10(max(min(float(row["A -> B best p"]), float(row["B -> A best p"])), 1e-12))
                    edges.append((start, end, float(strength), "feedback"))
            scores = {str(row["Series"]): float(row["Leadership score"]) for _, row in table.iterrows()}
            fig = PC.influence_network_figure(list(scores.keys()), edges, theme, node_scores=scores)
            network = C.chart_card("Market influence diagram", fig, source=source, notes=leadership.notes, height=460)
        else:
            network = C.chart_card("Market influence diagram", None, source=source)
            influence_table = html.Div()
        leadership_box = C.info_box_items("Reading of the influence structure:", leadership.notes)
    else:
        network = C.chart_card("Market influence diagram", None, source=leadership.source)
        influence_table = C.chart_card("Pairwise direction of influence", None, source=leadership.source)
        leadership_table = C.chart_card("Market leadership ranking", None, source=leadership.source)
        leadership_box = html.Div()

    leadlag = payload["leadlag"]
    leadlag_table = (
        C.dataframe_table(leadlag.unwrap().drop(columns=["Note"], errors="ignore"), theme, title="Pairwise lead-lag", source=source, notes=leadlag.notes, id="mi-leadlag")
        if leadlag else C.chart_card("Pairwise lead-lag", None, source=leadlag.source)
    )

    cointegration = payload["cointegration"]
    if cointegration:
        data = cointegration.unwrap()
        johansen = data.get("johansen")
        johansen_table = (
            C.dataframe_table(johansen, theme, title="Johansen system test", source=source, notes=cointegration.notes, id="mi-johansen")
            if johansen is not None and not johansen.empty
            else C.chart_card("Johansen system test", None, source="The Johansen test could not be run on this panel.")
        )
        pairwise = data.get("pairwise")
        pairwise_table = (
            C.dataframe_table(pairwise, theme, title="Engle-Granger pairwise test", source=source, id="mi-pairwise")
            if pairwise is not None and not pairwise.empty
            else C.chart_card("Engle-Granger pairwise test", None, source="No pair had enough overlapping observations.")
        )
        cointegration_box = C.info_box_items("Interpretation:", cointegration.notes)
    else:
        johansen_table = C.chart_card("Johansen system test", None, source=cointegration.source)
        pairwise_table = C.chart_card("Engle-Granger pairwise test", None, source=cointegration.source)
        cointegration_box = C.info_box(cointegration.message(), tone="warning")

    return network, influence_table, leadership_table, leadership_box, leadlag_table, johansen_table, pairwise_table, cointegration_box
