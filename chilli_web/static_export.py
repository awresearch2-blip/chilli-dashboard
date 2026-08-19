"""Render a self-contained, static HTML snapshot of the dashboard.

Unlike the live Dash app, this has no server behind it: every chart and table
is computed once, right now, using the exact same
:class:`chilli_desktop.preprocessing.DataService`, :mod:`analytics`,
:mod:`forecasting` and :mod:`insights` calls the live pages use, and baked
into one HTML file with Plotly's JS runtime embedded inline. The result keeps
its charts' hover/zoom/pan interactivity in any browser, indefinitely, with
no Python process running anywhere -- which is the point: it cannot go down
when a background process does.

Run:

    .venv\\Scripts\\python.exe -m chilli_web.static_export --out snapshot.html

Trade-off, stated plainly: this is a point-in-time snapshot. The sidebar
filters, the "Run model sweep" button and anything else that recomputes on
demand are not present here -- there is nothing to recompute against. Every
number is real and already verified against the live app; none of it updates
itself.
"""

from __future__ import annotations

import argparse
import html as html_lib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.offline as pyo

from chilli_desktop import analytics, forecasting, insights, settings
from chilli_desktop.preprocessing import DataService, default_filters
from chilli_desktop.utils import fmt_date, fmt_number, fmt_pct

from . import plotly_charts as PC
from . import server_state
from . import theme as theme_mod

DARK = settings.DARK_THEME
_CHART_TEMPLATE = theme_mod.template_name(DARK)


# ==========================================================================
# Chart -> HTML fragment
# ==========================================================================

_chart_counter = 0

#: Static-export-only chart config: the live Dash app's shared
#: ``PC.figure_config()`` leaves Plotly's built-in "download as PNG" modebar
#: button enabled, which works fine there (a real server-backed browser
#: session). An Artifact's sandbox blocks any download the page starts
#: itself, so that same button would sit in the toolbar looking clickable and
#: silently do nothing -- worse than not having it. Hover, zoom, pan and reset
#: all still work; only the non-functional download affordance is removed.
_STATIC_CONFIG = {
    **PC.figure_config(),
    "modeBarButtonsToRemove": [*PC.figure_config()["modeBarButtonsToRemove"], "toImage"],
}


def chart_html(fig, height: int = 380) -> str:
    """One Plotly figure as an embeddable fragment, Plotly.js excluded
    (embedded once, globally, in the page shell)."""
    global _chart_counter
    _chart_counter += 1
    fig.update_layout(template=_CHART_TEMPLATE, height=height)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=_STATIC_CONFIG,
        div_id=f"chart-{_chart_counter}",
    )


def plotly_js_bundle() -> str:
    return f"<script>{pyo.get_plotlyjs()}</script>"


# ==========================================================================
# Table -> HTML fragment
# ==========================================================================


def _format_cell(value: Any) -> str:
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


_SIGNED_HINTS = ("change", "elasticity", "coefficient", "beta", "correlation", "r²", "r2")


def _cell_class(column: str, raw: Any) -> str:
    lowered = column.lower()
    if isinstance(raw, (bool, np.bool_)):
        return "num pos" if bool(raw) else "num muted"
    if isinstance(raw, str) and raw.strip().upper() == "SELECTED":
        return "tag-selected"
    if isinstance(raw, (float, np.floating, int, np.integer)) and not isinstance(raw, (bool, np.bool_)):
        number = float(raw)
        if np.isfinite(number):
            if any(h in lowered for h in _SIGNED_HINTS):
                if number > 0:
                    return "num pos"
                if number < 0:
                    return "num neg"
            if "p-value" in lowered and number < settings.ANALYTICS.alpha:
                return "num accent"
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, (bool, np.bool_)):
        return "num"
    return ""


def table_html(frame: pd.DataFrame, *, caption: str = "", max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return '<p class="empty-note">No rows.</p>'
    has_named_index = not isinstance(frame.index, pd.RangeIndex)
    display = frame.reset_index() if has_named_index else frame.copy()
    if has_named_index:
        display.columns = [frame.index.name or "Index"] + list(frame.columns)
    truncated = len(display) > max_rows
    shown = display.head(max_rows)

    head = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(
            f'<td class="{_cell_class(str(col), row[col])}">{html_lib.escape(_format_cell(row[col]))}</td>'
            for col in shown.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    caption_html = f"<caption>{html_lib.escape(caption)}</caption>" if caption else ""
    note = (
        f'<p class="empty-note">Showing the first {max_rows} of {len(display)} rows.</p>'
        if truncated else ""
    )
    return (
        f'<div class="table-wrap">{caption_html}<table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>{note}"
    )


def source_html(source: str, notes: Sequence[str] = ()) -> str:
    notes_html = ""
    usable = [n for n in notes if n]
    if usable:
        items = "".join(f"<li>{html_lib.escape(n)}</li>" for n in usable)
        notes_html = f'<details class="notes"><summary>Notes ({len(usable)})</summary><ul>{items}</ul></details>'
    return f'<div class="source">Source: {html_lib.escape(source or "—")}</div>{notes_html}'


def unavailable_html(message: str, source: str = "") -> str:
    return (
        f'<div class="unavailable">{html_lib.escape(message)}</div>{source_html(source)}'
    )


def card(title: str, body: str, *, source: str = "", notes: Sequence[str] = ()) -> str:
    return (
        f'<div class="card"><div class="card-title">{html_lib.escape(title)}</div>'
        f"{body}{source_html(source, notes)}</div>"
    )


def kpi(label: str, value: str, delta: str = "", caption: str = "", tone: str = "") -> str:
    delta_html = f'<div class="kpi-delta">{html_lib.escape(delta)}</div>' if delta else ""
    caption_html = f'<div class="kpi-caption">{html_lib.escape(caption)}</div>' if caption else ""
    return (
        f'<div class="kpi tone-{tone}"><div class="kpi-label">{html_lib.escape(label)}</div>'
        f'<div class="kpi-value">{html_lib.escape(value)}</div>{delta_html}{caption_html}</div>'
    )


def tone_for(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return "pos" if value > 0 else "neg" if value < 0 else ""


def info_box(text: str, *, title: str = "", tone: str = "info") -> str:
    title_html = f'<div class="info-title">{html_lib.escape(title)}</div>' if title else ""
    body = "".join(f"<p>{html_lib.escape(p)}</p>" for p in text.split("\n") if p)
    return f'<div class="info-box tone-{tone}">{title_html}{body}</div>'


def info_list(title: str, items: Sequence[str], *, tone: str = "info") -> str:
    usable = [i for i in items if i]
    if not usable:
        return ""
    lis = "".join(f"<li>{html_lib.escape(i)}</li>" for i in usable)
    return f'<div class="info-box tone-{tone}"><div class="info-title">{html_lib.escape(title)}</div><ul>{lis}</ul></div>'


# ==========================================================================
# Section builders -- each returns (anchor_id, nav_label, html_body)
# ==========================================================================


def build_executive(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    focus = service.focus_varieties()
    snapshots = insights.variety_snapshots(service, filters)
    focus_labels = set(focus.values())
    ordered = [s for s in snapshots if s.name in focus_labels] + [s for s in snapshots if s.name not in focus_labels]

    kpis = []
    for snap in ordered[:4]:
        wow = snap.change_pct("Week on week")
        mom = snap.change_pct("Month on month")
        yoy = snap.change_pct("Year on year")
        kpis.append(kpi(
            snap.name, fmt_number(snap.latest, 0), f"WoW {fmt_pct(wow, 1, signed=True)}",
            f"MoM {fmt_pct(mom, 1, signed=True)} · YoY {fmt_pct(yoy, 1, signed=True)} · as at {fmt_date(snap.latest_date)}",
            tone=tone_for(wow),
        ))
    parts.append(f'<div class="kpi-row">{"".join(kpis)}</div>')

    primary = focus.get("Teja") or (ordered[0].name if ordered else "")
    if primary:
        sentiment = insights.compute_sentiment(service, primary, filters)
        if sentiment:
            value = sentiment.unwrap()
            rows = "".join(
                f"<tr><td>{html_lib.escape(c.name)}</td><td class='num'>{c.score:+.2f}</td>"
                f"<td>{html_lib.escape(c.explanation)}</td></tr>"
                for c in value.components
            )
            gauge_pct = (value.score + 1) / 2 * 100
            parts.append(
                f'<div class="card"><div class="card-title">Market sentiment — {html_lib.escape(primary)}</div>'
                f'<div class="gauge"><div class="gauge-track"><div class="gauge-fill" style="width:{gauge_pct:.1f}%"></div>'
                f'<div class="gauge-marker" style="left:{gauge_pct:.1f}%"></div></div>'
                f'<div class="gauge-label">{html_lib.escape(value.label)} ({value.score:+.2f})</div></div>'
                f'<table class="simple"><thead><tr><th>Component</th><th>Score</th><th>Reading</th></tr></thead>'
                f"<tbody>{rows}</tbody></table>{source_html(sentiment.source, value.notes)}</div>"
            )

    variety_panel = service.variety_panel(filters.frequency, filters)
    if variety_panel:
        frame = variety_panel.unwrap()
        fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, DARK, ylabel="INR per quintal", highlight=primary)
        parts.append(card("Guntur variety prices", chart_html(fig, 380), source=variety_panel.source))

    found, failures = insights.generate_all(service, filters, primary)
    highlights = insights.executive_highlights(found, limit=6)
    if highlights:
        cards = "".join(_insight_card_html(i) for i in highlights)
        parts.append(
            f'<div class="section-lead">{len(found)} finding(s) generated across the whole workbook; '
            f"the {len(highlights)} strongest are shown here.</div>"
            f'<div class="insight-grid">{cards}</div>'
        )

    return "executive", "Executive Summary", "".join(parts)


def _insight_card_html(insight: insights.Insight) -> str:
    tone_map = {"strong": "pos", "moderate": "accent", "weak": "warn", "informational": "", "data gap": "neg"}
    tone = tone_map.get(insight.strength, "")
    direction_html = (
        f'<span class="insight-direction">{html_lib.escape(insight.direction.upper())}</span>'
        if insight.direction not in ("n/a", "") else ""
    )
    detail_html = f"<p>{html_lib.escape(insight.detail)}</p>" if insight.detail else ""
    evidence_html = ""
    if insight.evidence:
        lis = "".join(f"<li>{html_lib.escape(e)}</li>" for e in insight.evidence)
        evidence_html = f'<details class="notes"><summary>Evidence ({len(insight.evidence)})</summary><ul>{lis}</ul></details>'
    return (
        f'<div class="insight-card tone-{tone}"><div class="insight-header">'
        f'<span class="insight-badge">{html_lib.escape(insight.strength.upper())}</span>'
        f'<span class="insight-category">{html_lib.escape(insight.category)}</span>{direction_html}</div>'
        f'<div class="insight-headline">{html_lib.escape(insight.headline)}</div>'
        f"{detail_html}{source_html(insight.source)}{evidence_html}</div>"
    )


def build_price(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    for variety in ("Teja", "LCA 334"):
        resolved = service.resolve_variety(variety)
        if not resolved:
            continue
        result = service.variety_series(resolved)
        if not result:
            parts.append(card(f"{resolved} — price", unavailable_html(result.reason)))
            continue
        raw = service.apply_filters(result.unwrap(), filters, kind="price")
        series = service.series_at(raw, "W")
        series.name = resolved
        window = settings.ANALYTICS.default_rolling_window
        rolling = analytics.rolling_statistics(series, window, result.source)
        if rolling:
            frame = rolling.unwrap()
            fig = PC.line_figure(
                {resolved: frame["Value"], f"MA({window})": frame[f"Moving average ({window})"]},
                DARK, ylabel="INR per quintal", highlight=resolved,
                fill_between=(frame["Lower band (-2 sd)"], frame["Upper band (+2 sd)"]),
            )
            parts.append(card(f"{resolved} — price with rolling statistics", chart_html(fig, 340), source=result.source, notes=rolling.notes))
            vol_col = next((c for c in frame.columns if c.startswith("Rolling volatility")), None)
            if vol_col:
                fig = PC.line_figure({"Annualised volatility": frame[vol_col] * 100}, DARK, ylabel="% annualised")
                parts.append(card(f"{resolved} — rolling volatility", chart_html(fig, 240)))

    variety_panel = service.variety_panel("W", filters)
    if variety_panel:
        frame = variety_panel.unwrap()
        normalised = {}
        for column in frame.columns:
            values = frame[column].dropna()
            if not values.empty and values.iloc[0] != 0:
                normalised[str(column)] = values / values.iloc[0] * 100
        if normalised:
            fig = PC.line_figure(normalised, DARK, ylabel="Index (first period = 100)")
            parts.append(card(
                "Relative performance across all varieties (first period = 100)", chart_html(fig, 360),
                source=variety_panel.source,
                notes=["Each series is rebased to 100 at its own first observation in the filtered window."],
            ))
    return "price", "Price Analysis", "".join(parts)


def build_integration(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    panel_result = service.price_panel("D", filters)
    if not panel_result:
        return "integration", "Market Integration", unavailable_html(panel_result.message())
    frame = panel_result.unwrap()
    source = panel_result.source
    fig = PC.line_figure({str(c): frame[c] for c in frame.columns}, DARK, ylabel="INR per quintal")
    parts.append(card("Teja price by market (daily)", chart_html(fig, 340), source=source))

    leadership = analytics.leadership_ranking(frame, "D", source, 20)
    if leadership:
        table = leadership.unwrap()
        pairs = table.attrs.get("pairs")
        edges = []
        if pairs is not None and not pairs.empty:
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
            scores = {str(r["Series"]): float(r["Leadership score"]) for _, r in table.iterrows()}
            fig = PC.influence_network_figure(list(scores.keys()), edges, DARK, node_scores=scores)
            parts.append(card("Market influence diagram", chart_html(fig, 460), source=source, notes=leadership.notes))
        parts.append(card("Market leadership ranking", table_html(table), source=source))

    cointegration = analytics.cointegration(frame.dropna(), source)
    if cointegration:
        data = cointegration.unwrap()
        johansen = data.get("johansen")
        if johansen is not None and not johansen.empty:
            parts.append(card("Johansen cointegration test", table_html(johansen), source=source, notes=cointegration.notes))
    return "integration", "Market Integration", "".join(parts)


def build_correlation(service: DataService, filters) -> tuple[str, str, str]:
    panel_result = service.variety_panel("W", filters)
    if not panel_result:
        return "correlation", "Correlation Studio", unavailable_html(panel_result.message())
    frame = panel_result.unwrap()
    source = panel_result.source
    result = analytics.correlation_matrix(frame, "pearson", source)
    parts = []
    if result:
        fig = PC.heatmap_figure(result.unwrap(), DARK, diverging=True, vmin=-1, vmax=1, cbar_label="Correlation")
        parts.append(card("Pearson correlation across varieties", chart_html(fig, 420), source=source, notes=result.notes))
    return "correlation", "Correlation Studio", "".join(parts)


def build_seasonality(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    resolved = service.resolve_variety("Teja")
    result = service.variety_series(resolved) if resolved else None
    if result:
        series = service.apply_filters(result.unwrap(), filters, kind="price")
        indices = analytics.seasonal_indices(series, result.source)
        if indices:
            table = indices.unwrap()
            fig = PC.bar_figure(
                table["Seasonal index"], DARK, ylabel="Seasonal index (1.00 = average)", reference=1.0,
                reference_label="All-month average", colour_negative=False, value_labels=True, value_format=".3f",
            )
            parts.append(card(f"Seasonal index by month — {resolved}", chart_html(fig, 320), source=result.source, notes=indices.notes))

    season = service.season_profile()
    if season:
        table = season.unwrap()
        values = pd.Series(table["mean_arrivals"].to_numpy(), index=pd.Index(table["month_name"], name="Month"))
        fig = PC.bar_figure(values, DARK, ylabel="Mean arrivals (bags)", reference=float(table["mean_arrivals"].mean()), reference_label="All-month average", colour_negative=False)
        parts.append(card("Mean arrivals by month (harvest / lean classification)", chart_html(fig, 320), source=season.source, notes=season.notes))
        parts.append(card("Season classification", table_html(table.set_index("month_name").rename_axis("Month"))))
    return "seasonality", "Seasonality", "".join(parts)


def build_exports(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    exports_result = service.exports_monthly()
    if not exports_result:
        return "exports", "Export Analysis", unavailable_html(exports_result.message())
    exports = exports_result.unwrap()
    fig = PC.line_figure({"Exports": exports}, DARK, ylabel="As supplied (unit not stated)")
    parts.append(card("Monthly exports", chart_html(fig, 300), source=exports_result.source, notes=list(exports_result.notes) + ["The export sheet does not state its unit of measure."]))

    resolved = service.resolve_variety("Teja")
    price_result = service.variety_series(resolved) if resolved else None
    if price_result:
        price = service.series_at(service.apply_filters(price_result.unwrap(), filters, kind="price"), "ME")
        source = f"{exports_result.source}; {price_result.source}"
        fig = PC.dual_axis_figure(price, exports, DARK, primary_label=f"{resolved} price (INR/quintal)", secondary_label="Exports (as supplied)", secondary_as_bars=True)
        parts.append(card("Exports and price", chart_html(fig, 340), source=source))

        fx_result = service.usd_inr()
        if fx_result:
            fx = service.series_at(fx_result.unwrap(), "ME")
            fig = PC.dual_axis_figure(price, fx, DARK, primary_label=f"{resolved} price (INR/quintal)", secondary_label="USD/INR")
            gaps = service.coverage_gaps(fx_result.unwrap())
            notes = []
            if not gaps.empty:
                worst = gaps.iloc[0]
                notes.append(f"The exchange-rate sheet has a {int(worst['days'])}-day gap ({worst['from']} to {worst['to']}).")
            parts.append(card("USD/INR and price (currency channel)", chart_html(fig, 320), source=f"{price_result.source}; {fx_result.source}", notes=notes))
    return "exports", "Export Analysis", "".join(parts)


def build_balance(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    balance = service.balance_sheet()
    if not balance:
        return "balance", "Balance Sheet", unavailable_html(balance.message())
    frame = balance.unwrap()
    dataset = service.data.datasets["balance_sheet"]
    projected = dataset.meta.get("projected_years", [])
    parts.append(info_box(
        f"Units: {dataset.meta.get('unit_note', 'not stated')}. "
        + (f"Year(s) {', '.join(str(y) for y in projected)} are the workbook's own projection, not this report's forecast."
           if projected else "All years are realised data."),
        tone="warn" if projected else "info",
    ))
    parts.append(card("Red chilli balance sheet, as supplied", table_html(frame), source=balance.source))

    stock_use = service.balance_sheet_row("Stock to Use")
    if stock_use:
        values = stock_use.unwrap()
        fig = PC.bar_figure(values, DARK, ylabel="%", reference=float(values.mean()), reference_label="Sample average", colour_negative=False, value_labels=True, value_format=",.2f", projected=[str(y) for y in projected])
        parts.append(card("Stock-to-use ratio by year", chart_html(fig, 300), source=stock_use.source, notes=stock_use.notes))
    return "balance", "Balance Sheet", "".join(parts)


def build_forecast(service: DataService, filters) -> tuple[str, str, str]:
    parts = []
    freq = "ME"
    horizon = settings.FORECAST.horizons[freq]
    for variety in ("Teja", "LCA 334"):
        resolved = service.resolve_variety(variety)
        if not resolved:
            continue
        result = service.variety_series(resolved)
        if not result:
            continue
        raw = service.apply_filters(result.unwrap(), filters, kind="price")
        series = service.series_at(raw, freq)
        series.name = resolved
        exog = service.exogenous_matrix(freq)
        panel = service.variety_panel(freq, filters)
        panel_frame = panel.unwrap() if panel else None
        if panel_frame is not None and resolved in panel_frame.columns:
            companions = [c for c in panel_frame.columns if c != resolved and panel_frame[c].notna().sum() >= len(series) * 0.6][:3]
            panel_frame = panel_frame[[resolved] + companions]

        comparison = forecasting.run_all_models(
            series, freq, horizon, target_name=resolved, exog=exog.unwrap() if exog else None,
            panel=panel_frame, source=result.source, history_notes=[service.partial_last_period(raw, freq)],
        )
        best = comparison.best
        if best is None:
            parts.append(card(f"{resolved} — forecast", unavailable_html(comparison.selection_reason)))
            continue

        fig = PC.forecast_figure(
            best.history, best.forecast, DARK, conf=(best.conf_lower, best.conf_upper),
            pred=(best.pred_lower, best.pred_upper), label=best.label, ylabel="INR per quintal",
            history_window=min(len(best.history), max(60, len(best.forecast) * 8)),
        )
        parts.append(card(f"{resolved} — {best.label} forecast", chart_html(fig, 400), source=best.source, notes=best.notes))
        parts.append(info_box(comparison.selection_reason, title="Model selection"))
        parts.append(card("Model comparison and backtest scores", table_html(comparison.comparison_table())))
        parts.append(card("Forecast table", table_html(best.table())))

        explanation = forecasting.explain(best, exog.unwrap() if exog else None, result.source)
        parts.append(info_box(explanation.headline, title="What this means"))
        parts.append(info_list("", explanation.plain_language, tone="info"))
        parts.append(info_list("Assumptions and caveats", explanation.assumptions, tone="warn"))

    return "forecast", "Forecast Center", "".join(parts)


def build_insights_all(service: DataService, filters) -> tuple[str, str, str]:
    resolved = service.resolve_variety("Teja") or (service.varieties()[0] if service.varieties() else "")
    found, failures = insights.generate_all(service, filters, resolved)
    strength_order = {"strong": 0, "moderate": 1, "weak": 2, "informational": 3, "data gap": 4}
    found = sorted(found, key=lambda i: strength_order.get(i.strength, 9))
    counts: dict[str, int] = {}
    for i in found:
        counts[i.strength] = counts.get(i.strength, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items())
    parts = [f'<div class="section-lead">{len(found)} finding(s): {html_lib.escape(summary)}.</div>']
    cards = "".join(_insight_card_html(i) for i in found)
    parts.append(f'<div class="insight-grid">{cards}</div>')
    return "insights", "Automated Insights", "".join(parts)


SECTION_BUILDERS = (
    build_executive, build_price, build_integration, build_correlation,
    build_seasonality, build_exports, build_balance, build_forecast, build_insights_all,
)


# ==========================================================================
# Page shell
# ==========================================================================

_CSS = """
:root {
  --ink: #f4f6f8; --surface: #ffffff; --surface-2: #eef1f5; --line: #d5dbe2;
  --text: #1a2027; --text-muted: #5c6b7a;
  --accent: #c2410c; --accent-soft: #fdebe0;
  --good: #15803d; --bad: #b91c1c; --info: #1d4ed8; --warn: #a16207;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #0d1117; --surface: #151b23; --surface-2: #1c242e; --line: #2a3440;
    --text: #e6edf3; --text-muted: #8b98a5;
    --accent: #f0883e; --accent-soft: #3d2b1c;
    --good: #3fb950; --bad: #f85149; --info: #58a6ff; --warn: #d29922;
  }
}
:root[data-theme="dark"] {
  --ink: #0d1117; --surface: #151b23; --surface-2: #1c242e; --line: #2a3440;
  --text: #e6edf3; --text-muted: #8b98a5;
  --accent: #f0883e; --accent-soft: #3d2b1c;
  --good: #3fb950; --bad: #f85149; --info: #58a6ff; --warn: #d29922;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; background: var(--ink); color: var(--text);
  font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
  font-size: 14px; line-height: 1.5;
}
.mono, td.num, .kpi-value, .gauge-label, th { font-variant-numeric: tabular-nums; }
td.num, .kpi-value {
  font-family: "Cascadia Code", SFMono-Regular, Consolas, "Roboto Mono", monospace;
}
a { color: var(--info); }
.shell { display: flex; min-height: 100vh; }
.rail {
  width: 240px; flex: 0 0 240px; background: var(--surface);
  border-right: 1px solid var(--line); position: sticky; top: 0; height: 100vh;
  overflow-y: auto; padding: 20px 0;
}
.brand { padding: 0 18px 16px 18px; border-bottom: 1px solid var(--line); margin-bottom: 10px; }
.brand-name { font-size: 15px; font-weight: 800; color: var(--accent); letter-spacing: 0.3px; }
.brand-sub { font-size: 10.5px; color: var(--text-muted); margin-top: 2px; }
.rail nav { display: flex; flex-direction: column; gap: 2px; padding: 0 10px; }
.rail a {
  display: block; padding: 8px 10px; border-radius: 6px; color: var(--text-muted);
  text-decoration: none; font-size: 12.5px; font-weight: 600;
}
.rail a:hover { background: var(--surface-2); color: var(--text); }
.rail-note {
  margin: 16px 10px 0 10px; padding: 10px 12px; font-size: 10.5px; color: var(--text-muted);
  background: var(--surface-2); border-radius: 6px; border: 1px solid var(--line);
}
.main { flex: 1; min-width: 0; }
.masthead {
  padding: 34px 40px 26px 40px; border-bottom: 1px solid var(--line); background: var(--surface);
}
.masthead h1 { margin: 0 0 6px 0; font-size: 26px; font-weight: 800; text-wrap: balance; }
.masthead p { margin: 0; color: var(--text-muted); font-size: 13px; max-width: 62ch; }
.meta-row { display: flex; gap: 18px; margin-top: 14px; flex-wrap: wrap; }
.meta-chip {
  font-size: 11px; color: var(--text-muted); background: var(--surface-2);
  border: 1px solid var(--line); border-radius: 20px; padding: 4px 12px;
}
section { padding: 30px 40px; border-bottom: 1px solid var(--line); }
section:last-child { border-bottom: none; }
section h2 {
  font-size: 19px; font-weight: 800; margin: 0 0 4px 0;
  text-transform: uppercase; letter-spacing: 0.4px;
}
.section-lead { color: var(--text-muted); font-size: 12.5px; margin: 0 0 16px 0; }
.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 16px;
}
.card-title { font-size: 13px; font-weight: 700; margin-bottom: 10px; }
.source { color: var(--text-muted); font-size: 10.5px; font-style: italic; margin-top: 8px; }
.notes { margin-top: 6px; }
.notes summary { color: var(--text-muted); font-size: 10.5px; cursor: pointer; }
.notes ul {
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 6px;
  padding: 8px 10px 8px 24px; margin: 6px 0 0 0; font-size: 10.5px; color: var(--text-muted);
}
.empty-note { color: var(--text-muted); font-size: 10.5px; }
.unavailable {
  background: var(--surface-2); border: 1px dashed var(--line); border-radius: 8px;
  padding: 20px; color: var(--text-muted); font-size: 12.5px; text-align: center;
}
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
table.simple { width: auto; min-width: 60%; }
caption { text-align: left; font-size: 11px; color: var(--text-muted); margin-bottom: 6px; caption-side: top; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th {
  color: var(--text-muted); font-weight: 700; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.3px; border-bottom: 2px solid var(--line);
}
tr:nth-child(even) td { background: var(--surface-2); }
td.num { text-align: right; }
td.num.pos { color: var(--good); }
td.num.neg { color: var(--bad); }
td.num.accent { color: var(--accent); }
td.num.muted { color: var(--text-muted); }
td.tag-selected { color: var(--accent); font-weight: 700; }
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }
.kpi {
  flex: 1 1 190px; min-width: 180px; background: var(--surface); border: 1px solid var(--line);
  border-left: 3px solid var(--text-muted); border-radius: 8px; padding: 13px 15px;
}
.kpi.tone-pos { border-left-color: var(--good); }
.kpi.tone-neg { border-left-color: var(--bad); }
.kpi-label { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }
.kpi-value { font-size: 22px; font-weight: 700; margin-top: 3px; }
.kpi-delta { font-size: 11px; font-weight: 600; margin-top: 3px; color: var(--text-muted); }
.kpi-caption { font-size: 10px; color: var(--text-muted); margin-top: 4px; }
.gauge { margin-bottom: 14px; }
.gauge-track {
  position: relative; height: 10px; border-radius: 6px; margin-bottom: 8px;
  background: linear-gradient(90deg, var(--bad), var(--text-muted), var(--good));
}
.gauge-marker {
  position: absolute; top: -3px; width: 16px; height: 16px; border-radius: 50%;
  background: var(--surface); border: 2px solid var(--text); transform: translateX(-8px);
}
.gauge-label { text-align: center; font-weight: 700; font-size: 13px; }
.info-box {
  background: var(--surface-2); border: 1px solid var(--line); border-left: 3px solid var(--info);
  border-radius: 8px; padding: 12px 15px; margin-bottom: 14px; font-size: 12px; color: var(--text-muted);
}
.info-box.tone-warn { border-left-color: var(--warn); }
.info-box p { margin: 4px 0; }
.info-box ul { margin: 6px 0 0 0; padding-left: 18px; }
.info-title { font-weight: 700; color: var(--accent); font-size: 11.5px; margin-bottom: 4px; }
.insight-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 1000px) { .insight-grid { grid-template-columns: 1fr; } }
.insight-card {
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--text-muted);
  border-radius: 8px; padding: 13px 15px;
}
.insight-card.tone-pos { border-left-color: var(--good); }
.insight-card.tone-neg { border-left-color: var(--bad); }
.insight-card.tone-warn { border-left-color: var(--warn); }
.insight-card.tone-accent { border-left-color: var(--accent); }
.insight-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.insight-badge { font-size: 9px; font-weight: 800; border: 1px solid var(--text-muted); color: var(--text-muted); border-radius: 3px; padding: 1px 6px; }
.insight-category { font-size: 10px; color: var(--text-muted); font-weight: 600; }
.insight-direction { font-size: 9px; font-weight: 800; border: 1px solid var(--text-muted); color: var(--text-muted); border-radius: 3px; padding: 1px 6px; margin-left: auto; }
.insight-headline { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
.insight-card p { font-size: 11px; color: var(--text-muted); margin: 0 0 6px 0; }
footer { padding: 22px 40px; color: var(--text-muted); font-size: 10.5px; text-align: center; }
"""


def build_page(service: DataService, filters, sections: Sequence[tuple[str, str, str]]) -> str:
    data = service.data
    nav = "".join(f'<a href="#{sid}">{html_lib.escape(label)}</a>' for sid, label, _ in sections)
    body = "".join(
        f'<section id="{sid}"><h2>{html_lib.escape(label)}</h2>{content}</section>'
        for sid, label, content in sections
    )
    generated = pd.Timestamp.now()
    latest = service.latest_observation_date()

    # Deliberately no <!doctype>/<html>/<head>/<body> wrapper: this fragment
    # is written directly to a file and published through the Artifact tool,
    # which supplies that skeleton itself. The <title> and <style> tags still
    # work sitting directly in the body -- browsers parse them regardless of
    # a formal <head>, and the Artifact publisher scans the first 8KB of the
    # file for a <title> tag by text search, not by requiring a <head>.
    return f"""<meta charset="utf-8">
<title>Chilli Intelligence Snapshot</title>
<style>{_CSS}</style>
{plotly_js_bundle()}
<div class="shell">
  <aside class="rail">
    <div class="brand">
      <div class="brand-name">\U0001f336 CHILLI INTELLIGENCE</div>
      <div class="brand-sub">Snapshot report · {settings.ORG_NAME}</div>
    </div>
    <nav>{nav}</nav>
    <div class="rail-note">
      Point-in-time snapshot generated {generated:%d %b %Y %H:%M}. Charts are interactive
      (hover, zoom, pan) but nothing here recomputes — there is no server behind this page.
    </div>
  </aside>
  <main class="main">
    <div class="masthead">
      <h1>Red Chilli Market Intelligence</h1>
      <p>Guntur, Warangal and Khammam price, arrivals, forecasts and automated findings,
         generated from {html_lib.escape(data.path.name)}.</p>
      <div class="meta-row">
        <span class="meta-chip">Latest observation: {fmt_date(latest)}</span>
        <span class="meta-chip">Workbook read {data.loaded_at:%d %b %Y %H:%M}</span>
        <span class="meta-chip">{len(data.datasets)}/{len(data.raw_shapes)} sheets mapped</span>
        <span class="meta-chip">{filters.describe()}</span>
      </div>
    </div>
    {body}
    <footer>
      Chilli Intelligence Desktop &amp; Web · Static snapshot · Source: {html_lib.escape(data.path.name)},
      data through {fmt_date(latest)}
    </footer>
  </main>
</div>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    service = server_state.get_service()
    filters = default_filters(service)

    sections = [builder(service, filters) for builder in SECTION_BUILDERS]
    html_text = build_page(service, filters, sections)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"written: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
