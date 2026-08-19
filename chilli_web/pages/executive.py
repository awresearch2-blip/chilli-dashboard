"""Executive Summary -- the web equivalent of ``ui.ExecutiveSummaryPage``.

Every number and chart here comes from the same calls the desktop page makes:
``insights.variety_snapshots``, ``insights.compute_sentiment``,
``service.variety_panel``, ``service.guntur_arrivals`` /
``guntur_offtake`` / ``exports_monthly`` / ``balance_sheet_row``,
``forecasting.run_all_models`` and ``insights.generate_all``. This module only
lays them out reactively instead of building them once into a Qt layout.

The two slow sections (the forecast summary and the automated insights) run
as Dash *background callbacks* -- the same reason the desktop page dispatches
them to a ``QThreadPool`` worker instead of blocking the window.
"""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html

from chilli_desktop import forecasting, insights, settings
from chilli_desktop.preprocessing import DataService, FilterState
from chilli_desktop.utils import fmt_date, fmt_number, fmt_pct

from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import plotly_charts as PC

dash.register_page(
    __name__,
    path="/",
    name="Executive Summary",
    description="Where the market stands, what is driving it, and where it is heading",
    order=1,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="exec-filter-note"),
            C.section_header("Latest prices", "Most recent quote per variety with change over several horizons."),
            html.Div(id="exec-price-cards"),
            html.Div(id="exec-change-table"),
            C.section_header(
                "Market sentiment",
                "Composite bullish/bearish reading for the lead variety, built from five workbook-derived components.",
            ),
            html.Div(id="exec-sentiment"),
            C.section_header("Price history", "All varieties over the filtered window."),
            html.Div(id="exec-price-chart"),
            C.section_header("Supply and trade", "Arrivals, offtake and export volume."),
            html.Div(id="exec-supply-cards"),
            html.Div(id="exec-arrivals-chart"),
            C.section_header(
                "Forecast summary",
                "Best-performing model per focus variety. The Forecast Center has the full model comparison.",
            ),
            html.Div(id="exec-forecast-box"),
            dcc.Loading(html.Div(id="exec-forecast-table"), type="dot"),
            C.section_header("Key insights", "The strongest findings the data supports."),
            html.Div(id="exec-insight-box"),
            dcc.Loading(html.Div(id="exec-insight-cards"), type="dot"),
        ]
    )


# ==========================================================================
# Latest prices
# ==========================================================================


@callback(
    Output("exec-filter-note", "children"),
    Output("exec-price-cards", "children"),
    Output("exec-change-table", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render_prices(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    note = C.filter_note(filters.describe(), PG.frequency_label(filters.frequency))

    focus = service.focus_varieties()
    snapshots = insights.variety_snapshots(service, filters)
    focus_labels = set(focus.values())
    ordered = [s for s in snapshots if s.name in focus_labels] + [
        s for s in snapshots if s.name not in focus_labels
    ]

    if not ordered:
        return note, C.info_box(
            f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo variety price survived the current filters.",
            tone="danger",
        ), html.Div()

    cards = []
    for snap in ordered[:4]:
        wow = snap.change_pct("Week on week")
        mom = snap.change_pct("Month on month")
        yoy = snap.change_pct("Year on year")
        cards.append(
            C.summary_card(
                snap.name,
                fmt_number(snap.latest, 0),
                f"WoW {fmt_pct(wow, 1, signed=True)}",
                f"MoM {fmt_pct(mom, 1, signed=True)} · YoY {fmt_pct(yoy, 1, signed=True)}\n"
                f"as at {fmt_date(snap.latest_date)}",
                tone=PG.tone_for_change(wow),
            )
        )
    card_block = html.Div(
        [
            C.card_row(cards),
            C.info_box(
                "Prices are in the workbook's own unit (INR per quintal as recorded on "
                f"'{ordered[0].source}'). Changes compare the latest quote with the last quote "
                "on or before the target date; no value is interpolated.",
                tone="muted",
            ),
        ]
    )

    rows: list[dict] = []
    for snap in ordered:
        row: dict = {"Variety": snap.name, "Latest": snap.latest, "As at": snap.latest_date}
        for horizon in (
            "Previous observation", "Week on week", "Fortnight",
            "Month on month", "Quarter on quarter", "Year on year",
        ):
            row[horizon] = snap.change_pct(horizon) * 100
        rows.append(row)
    table = C.dataframe_table(
        pd.DataFrame(rows).set_index("Variety"), theme,
        title="Change by horizon — all varieties", source=ordered[0].source,
        notes=["Percentage change over each horizon."], id="exec-change",
    )
    return note, card_block, table


# ==========================================================================
# Sentiment
# ==========================================================================


@callback(
    Output("exec-sentiment", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render_sentiment(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    focus = service.focus_varieties()
    snapshots = insights.variety_snapshots(service, filters)
    primary = focus.get("Teja") or (snapshots[0].name if snapshots else "")
    if not primary:
        return C.info_box(settings.DATA_UNAVAILABLE_MESSAGE, tone="danger")

    sentiment = insights.compute_sentiment(service, primary, filters)
    gauge_fig = (
        PC.sentiment_gauge_figure(sentiment.unwrap().score, sentiment.unwrap().label, theme)
        if sentiment
        else PC.sentiment_gauge_figure(float("nan"), "", theme)
    )
    gauge = dcc.Graph(figure=gauge_fig, config=PC.figure_config(), style={"height": "220px"})

    if not sentiment:
        return html.Div([gauge, C.info_box(sentiment.message(), tone="warning")])

    value = sentiment.unwrap()
    component_frame = pd.DataFrame(
        [
            {"Component": c.name, "Score": c.score, "Reading": c.explanation, "Source": c.source}
            for c in value.components
        ]
    ).set_index("Component")
    table = C.dataframe_table(
        component_frame, theme, title="Sentiment components", source=sentiment.source,
        notes=value.notes, id="exec-sentiment-table",
    )
    return html.Div([gauge, table])


# ==========================================================================
# Price history
# ==========================================================================


@callback(
    Output("exec-price-chart", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render_price_chart(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    focus = service.focus_varieties()
    primary = focus.get("Teja") or ""
    variety_panel = service.variety_panel(filters.frequency, filters)
    if not variety_panel:
        return C.chart_card("Guntur variety prices", None, source=variety_panel.source)
    frame = variety_panel.unwrap()
    fig = PC.line_figure(
        {str(c): frame[c] for c in frame.columns}, theme, ylabel="INR per quintal", highlight=primary,
    )
    return C.chart_card(
        "Guntur variety prices", fig, source=variety_panel.source,
        notes=[f"{PG.frequency_label(filters.frequency)} averages of the daily quotes."],
        height=380,
    )


# ==========================================================================
# Supply and trade
# ==========================================================================


@callback(
    Output("exec-supply-cards", "children"),
    Output("exec-arrivals-chart", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
)
def _render_supply(filters_data, theme_name):
    service, filters, theme = PG.current(filters_data, theme_name)
    cards = []

    arrivals = service.guntur_arrivals()
    if arrivals:
        series = service.apply_filters(arrivals.unwrap(), filters, kind="arrivals")
        monthly = service.series_at(series, "ME", "sum")
        if len(monthly) >= 13:
            latest = float(monthly.iloc[-1])
            same_month = monthly[monthly.index.month == monthly.index[-1].month].iloc[:-1]
            norm = float(same_month.mean()) if not same_month.empty else float("nan")
            delta = (latest - norm) / norm if norm else float("nan")
            cards.append(
                C.summary_card(
                    "Arrivals, latest month", f"{fmt_number(latest, 0)} bags",
                    f"{fmt_pct(delta, 0, signed=True)} vs same-month average",
                    f"{monthly.index[-1]:%b %Y}. Bag weight per sheet header: "
                    f"{service.market_bag_weight('Guntur') or '—'} kg.",
                    tone=PG.tone_for_change(-delta if delta == delta else float("nan")),
                )
            )

    offtake = service.guntur_offtake()
    if offtake and arrivals:
        arr = service.series_at(service.apply_filters(arrivals.unwrap(), filters, kind="arrivals"), "ME", "sum")
        off = service.series_at(service.apply_filters(offtake.unwrap(), filters), "ME", "sum")
        joined = pd.concat([arr.rename("a"), off.rename("o")], axis=1).dropna()
        if not joined.empty and joined["a"].iloc[-1]:
            ratio = float(joined["o"].iloc[-1] / joined["a"].iloc[-1])
            mean_ratio = float((joined["o"] / joined["a"]).mean())
            cards.append(
                C.summary_card(
                    "Offtake / arrivals", fmt_pct(ratio, 0), f"average {fmt_pct(mean_ratio, 0)}",
                    "Share of arriving material actually lifted. A falling ratio means stock "
                    "is accumulating in the mandi.",
                    tone="positive" if ratio > mean_ratio else "warning",
                )
            )

    exports = service.exports_monthly()
    if exports:
        series = exports.unwrap()
        latest = float(series.iloc[-1])
        year_ago = series[series.index <= series.index[-1] - pd.DateOffset(years=1)]
        delta = (
            (latest - float(year_ago.iloc[-1])) / float(year_ago.iloc[-1])
            if not year_ago.empty and float(year_ago.iloc[-1]) else float("nan")
        )
        cards.append(
            C.summary_card(
                "Exports, latest month", fmt_number(latest, 0), f"YoY {fmt_pct(delta, 0, signed=True)}",
                f"{series.index[-1]:%b %Y}. Units are not stated on the sheet, so this is "
                "comparable only against itself.",
                tone=PG.tone_for_change(delta),
            )
        )

    stock_use = service.balance_sheet_row("Stock to Use")
    if stock_use:
        values = stock_use.unwrap()
        latest = float(values.iloc[-1])
        mean = float(values.iloc[:-1].mean())
        projected = service.data.datasets["balance_sheet"].meta.get("projected_years", [])
        cards.append(
            C.summary_card(
                f"Stock-to-use {values.index[-1]}", f"{latest:,.1f}%", f"average {mean:,.1f}%",
                ("Workbook projection, not realised data." if values.index[-1] in projected else "Realised.")
                + " A thin buffer supports prices.",
                tone="positive" if latest < mean else "negative",
            )
        )

    card_block = (
        C.card_row(cards) if cards else
        C.info_box(
            f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNone of the supply or trade series is "
            "available under the current filters.",
            tone="warning",
        )
    )

    focus = service.focus_varieties()
    primary = focus.get("Teja") or ""
    variety_panel = service.variety_panel(filters.frequency, filters)
    if arrivals and variety_panel and primary:
        price_result = service.variety_series(primary)
        price = service.series_at(service.apply_filters(price_result.unwrap(), filters, kind="price"), filters.frequency)
        arr_series = service.series_at(service.apply_filters(arrivals.unwrap(), filters, kind="arrivals"), filters.frequency, "sum")
        fig = PC.dual_axis_figure(
            price, arr_series, theme, primary_label=f"{primary} price (INR/quintal)",
            secondary_label="Arrivals (bags)", secondary_as_bars=True,
        )
        arrivals_chart = C.chart_card(
            "Price against arrivals", fig, source=f"{variety_panel.source}; {arrivals.source}",
            notes=["Arrivals are summed within each period; price is averaged."], height=320,
        )
    else:
        arrivals_chart = C.chart_card("Price against arrivals", None, source="")

    return card_block, arrivals_chart


# ==========================================================================
# Forecast summary (background)
# ==========================================================================


def _compute_forecast_summary(service: DataService, filters: FilterState, targets: list[str]) -> list[dict]:
    """Fit the monthly model sweep for each focus variety. Identical to
    ``ui.ExecutiveSummaryPage._compute_forecast_summary``."""
    freq = "ME"
    horizon = settings.FORECAST.horizons[freq]
    rows: list[dict] = []
    for variety in targets:
        result = service.variety_series(variety)
        if not result:
            rows.append({"Variety": variety, "Status": result.reason})
            continue
        raw = service.apply_filters(result.unwrap(), filters, kind="price")
        series = service.series_at(raw, freq)
        series.name = variety
        if len(series) < settings.FORECAST.min_obs_arima:
            rows.append({"Variety": variety, "Status": f"Only {len(series)} monthly observation(s) after filtering."})
            continue
        exog = service.exogenous_matrix(freq)
        panel = service.variety_panel(freq, filters)
        comparison = forecasting.run_all_models(
            series, freq, horizon, target_name=variety,
            exog=exog.unwrap() if exog else None,
            panel=panel.unwrap() if panel else None,
            source=result.source,
            history_notes=[service.partial_last_period(raw, freq)],
        )
        best = comparison.best
        if best is None:
            rows.append({"Variety": variety, "Status": comparison.selection_reason})
            continue
        final = float(best.forecast.iloc[-1])
        latest = float(series.iloc[-1])
        rows.append(
            {
                "Variety": variety, "Latest": latest, "Model": best.label,
                f"Forecast {best.forecast.index[-1]:%b %Y}": final,
                "Change %": (final - latest) / latest * 100 if latest else float("nan"),
                "Lower 95%": float(best.pred_lower.iloc[-1]),
                "Upper 95%": float(best.pred_upper.iloc[-1]),
                "Backtest MAPE %": best.metrics.mape,
                "Directional accuracy %": best.metrics.directional_accuracy,
                "Status": "OK", "_source": best.source, "_reason": comparison.selection_reason,
            }
        )
    return rows


@callback(
    Output("exec-forecast-box", "children"),
    Output("exec-forecast-table", "children"),
    Input("filters-store", "data"),
    background=True,
)
def _render_forecast_summary(filters_data):
    service, filters, theme = PG.current(filters_data, None)
    focus = service.focus_varieties()
    targets = list(focus.values())
    if not targets:
        return (
            C.info_box(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNeither focus variety is present in the workbook.",
                tone="danger",
            ),
            html.Div(),
        )

    rows = _compute_forecast_summary(service, filters, targets)
    usable = [r for r in rows if r.get("Status") == "OK"]
    if not usable:
        return (
            C.info_box(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n" + "  ".join(str(r.get("Status", "")) for r in rows),
                tone="warning",
            ),
            html.Div(),
        )

    source = usable[0]["_source"]
    reasons = [r["_reason"] for r in usable]
    for row in rows:
        row.pop("_source", None)
        row.pop("_reason", None)
    frame = pd.DataFrame([r for r in rows if r.get("Status") == "OK"]).drop(columns=["Status"], errors="ignore")
    if "Variety" in frame.columns:
        frame = frame.set_index("Variety")

    table = C.dataframe_table(
        frame, theme, title="Forecast summary", source=source,
        notes=[
            "Six months ahead on monthly data, using the model with the lowest backtest RMSE for each variety.",
            "Lower/Upper 95% is the backtest-calibrated prediction interval at the end of the horizon.",
        ],
        id="exec-forecast",
    )
    box = C.info_box_items("Model selection", reasons)
    return box, table


# ==========================================================================
# Key insights (background)
# ==========================================================================


@callback(
    Output("exec-insight-box", "children"),
    Output("exec-insight-cards", "children"),
    Input("filters-store", "data"),
    background=True,
)
def _render_insights(filters_data):
    service, filters, theme = PG.current(filters_data, None)
    focus = service.focus_varieties()
    primary = focus.get("Teja") or (service.varieties()[0] if service.varieties() else "")
    found, failures = insights.generate_all(service, filters, primary)
    highlights = insights.executive_highlights(found, limit=6)
    if not highlights:
        return (
            C.info_box(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo insight reached the moderate-or-better "
                "evidence threshold under these filters.",
                tone="warning",
            ),
            html.Div(),
        )
    box_text = (
        f"{len(found)} finding(s) generated; the {len(highlights)} strongest are shown. "
        "The Automated Insights page lists all of them."
        + (f"  {len(failures)} generator(s) reported a problem." if failures else "")
    )
    cards = [C.insight_card(insight) for insight in highlights]
    return C.info_box(box_text, tone="muted"), html.Div(cards)
