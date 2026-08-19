"""The page shell: sidebar navigation, global filters, header, status bar.

Mirrors ``ui.MainWindow``'s ``_build_sidebar`` / ``_build_filter_controls`` /
``_build_header``, translated from Qt widgets + signal/slot wiring to Dash
components + callbacks. The filter *semantics* -- what each control means, the
preset buttons, the Apply/Reset behaviour -- are copied exactly; only the
widget toolkit differs.
"""

from __future__ import annotations

from typing import Any

import dash
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from chilli_desktop import settings
from chilli_desktop.preprocessing import FilterState
from chilli_desktop.settings import FORTNIGHT_FREQ
from chilli_desktop.utils import fmt_date

from . import filters_io, server_state

_FREQUENCY_OPTIONS = [
    {"label": "Daily", "value": "D"},
    {"label": "Weekly", "value": "W"},
    {"label": "Fortnightly", "value": FORTNIGHT_FREQ},
    {"label": "Monthly", "value": "ME"},
]

_NAV_ICON_BY_LABEL = {label: icon for _key, label, icon in settings.NAV_ITEMS}


# ==========================================================================
# Sidebar
# ==========================================================================


def _nav_links() -> html.Div:
    """Plain ``href``-bearing links, no per-link component ids.

    The active link is highlighted by a client-side callback matching
    ``href`` against the current pathname (see ``02_clientside.js``), rather
    than a Python pattern-matching callback keyed on a dict id. Dash renders
    a dict id as a literal JSON-blob ``id`` attribute in the DOM (e.g.
    ``id='{"index":"/","type":"nav-link"}'``); that string breaks any
    ``querySelector``-based tooling that treats ``id`` as a normal CSS
    identifier -- including this session's own browser-automation tool -- for
    no benefit here, since nothing needs to target one specific nav link.
    """
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 0))
    links = [
        dcc.Link(
            f"{_NAV_ICON_BY_LABEL.get(page['name'], '›')}   {page['name']}",
            href=page["relative_path"],
            className="nav-link",
        )
        for page in pages
    ]
    return html.Div(links, id="nav-list", className="nav-list")


def _filter_controls() -> html.Div:
    service = server_state.get_service()
    start, end = service.full_date_span()
    span_start = start or pd.Timestamp("2014-01-01")
    span_end = end or pd.Timestamp.now()

    def label(text: str) -> html.Div:
        return html.Div(text, className="filter-label")

    return html.Div(
        [
            html.Div("GLOBAL FILTERS", className="sidebar-section"),
            html.Div(
                [
                    label("Date range"),
                    dcc.DatePickerRange(
                        id="filter-date-range",
                        min_date_allowed=span_start.date(),
                        max_date_allowed=span_end.date(),
                        start_date=span_start.date(),
                        end_date=span_end.date(),
                        display_format="DD MMM YYYY",
                        style={"fontSize": "10px"},
                    ),
                    html.Div(
                        [
                            html.Button("1Y", id={"type": "date-preset", "index": "1Y"}, className="chip-button", n_clicks=0),
                            html.Button("3Y", id={"type": "date-preset", "index": "3Y"}, className="chip-button", n_clicks=0),
                            html.Button("5Y", id={"type": "date-preset", "index": "5Y"}, className="chip-button", n_clicks=0),
                            html.Button("All", id={"type": "date-preset", "index": "All"}, className="chip-button", n_clicks=0),
                        ],
                        className="preset-row",
                    ),
                ]
            ),
            html.Div(
                [
                    label("Analysis frequency"),
                    dcc.Dropdown(
                        id="filter-frequency", options=_FREQUENCY_OPTIONS, value="W",
                        clearable=False, className="dash-dropdown",
                    ),
                ]
            ),
            html.Div(
                [
                    label("Varieties (none = all)"),
                    dcc.Dropdown(
                        id="filter-varieties",
                        options=[{"label": v, "value": v} for v in service.varieties()],
                        value=[], multi=True, placeholder="All varieties",
                    ),
                ]
            ),
            html.Div(
                [
                    label("Market"),
                    dcc.Dropdown(
                        id="filter-market",
                        options=[{"label": m, "value": m} for m in service.markets()],
                        value=None, placeholder="All markets", clearable=True,
                    ),
                ]
            ),
            html.Div(
                [
                    label("Season (months)"),
                    dcc.Dropdown(
                        id="filter-months",
                        options=[
                            {"label": settings.MONTH_ABBREVIATIONS[i].title(), "value": i + 1}
                            for i in range(12)
                        ],
                        value=[], multi=True, placeholder="All months",
                    ),
                    html.Div(
                        [
                            html.Button("Peak", id={"type": "season-preset", "index": "Peak arrivals"}, className="chip-button", n_clicks=0),
                            html.Button("Lean", id={"type": "season-preset", "index": "Lean arrivals"}, className="chip-button", n_clicks=0),
                            html.Button("Clear", id={"type": "season-preset", "index": "Clear"}, className="chip-button", n_clicks=0),
                        ],
                        className="preset-row",
                    ),
                ]
            ),
            html.Div(
                [
                    label("Price range (INR/quintal)"),
                    html.Div(
                        [
                            dcc.Input(id="filter-price-min", type="number", placeholder="min", style={"width": "48%"}),
                            dcc.Input(id="filter-price-max", type="number", placeholder="max", style={"width": "48%"}),
                        ],
                        className="preset-row",
                    ),
                ]
            ),
            html.Div(
                [
                    label("Arrival range (bags)"),
                    html.Div(
                        [
                            dcc.Input(id="filter-arrival-min", type="number", placeholder="min", style={"width": "48%"}),
                            dcc.Input(id="filter-arrival-max", type="number", placeholder="max", style={"width": "48%"}),
                        ],
                        className="preset-row",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Button("Apply", id="filter-apply", className="primary-button", n_clicks=0,
                               style={"marginBottom": "4px"}),
                    html.Button("Reset", id="filter-reset", className="secondary-button", n_clicks=0),
                ]
            ),
        ],
        className="filter-block",
    )


def sidebar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("🌶  CHILLI INTELLIGENCE", className="brand-title"),
                    html.Div(
                        f"Web {settings.APP_VERSION} · {settings.ORG_NAME}",
                        className="brand-subtitle",
                    ),
                ],
                className="brand",
            ),
            _nav_links(),
            _filter_controls(),
            html.Div(
                [
                    html.Button("☀  Switch to light theme", id="theme-toggle", className="secondary-button", n_clicks=0),
                    html.Button("↻  Reload workbook", id="reload-workbook", className="secondary-button", n_clicks=0),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(id="page-title", className="page-title"),
                    html.Div(id="page-subtitle", className="page-subtitle"),
                ]
            ),
            html.Div(id="header-meta", className="header-meta"),
        ],
        className="header",
    )


def status_bar() -> html.Div:
    return html.Div(
        [html.Div(id="status-left"), html.Div(id="status-right")],
        className="status-bar",
    )


def app_shell(page_container: Any) -> html.Div:
    return html.Div(
        [
            dcc.Store(id="filters-store", data=filters_io.to_dict(server_state.initial_filters())),
            dcc.Store(id="theme-store", data=settings.DEFAULT_THEME),
            dcc.Store(id="reload-tick", data=0),
            html.Div(
                [
                    sidebar(),
                    html.Div(
                        [header(), html.Div(page_container, className="page-body"), status_bar()],
                        className="main-column",
                    ),
                ],
                className="app-shell",
            ),
        ],
        id="theme-root",
        className="theme-root",
        **{"data-theme": settings.DEFAULT_THEME},
    )


# ==========================================================================
# Shell callbacks
# ==========================================================================


@callback(Output("page-title", "children"), Output("page-subtitle", "children"), Input("url", "pathname"))
def _sync_header_text(pathname: str | None):
    page = dash.page_registry.get(_match_page(pathname))
    if not page:
        return "", ""
    return page["name"], page.get("description", "")


def _match_page(pathname: str | None) -> str | None:
    pathname = pathname or "/"
    for module, page in dash.page_registry.items():
        if page["relative_path"] == pathname:
            return module
    return None


dash.clientside_callback(
    "window.dash_clientside.chilli.highlightNav",
    Output("nav-list", "data-active-path"),
    Input("url", "pathname"),
)


@callback(Output("header-meta", "children"), Input("filters-store", "data"), Input("reload-tick", "data"))
def _sync_header_meta(_filters_data, _tick):
    service = server_state.get_service()
    latest = service.latest_observation_date()
    data = service.data
    return (
        f"Latest observation: {fmt_date(latest)}\n"
        f"Workbook read {data.loaded_at:%d %b %Y %H:%M}"
    )


@callback(
    Output("status-left", "children"), Output("status-right", "children"),
    Input("filters-store", "data"), Input("reload-tick", "data"),
)
def _sync_status_bar(filters_data, _tick):
    service = server_state.get_service()
    filters = filters_io.from_dict(filters_data)
    warnings_count = len(service.data_quality_notes())
    data = service.data
    left = f"Ready · {filters.describe()}" + (
        f" · {warnings_count} data-quality note(s)" if warnings_count else ""
    )
    right = (
        f"{data.path.name} · {len(data.datasets)}/{len(data.raw_shapes)} sheets · "
        f"read in {data.load_seconds:.2f}s"
    )
    return left, right


# -- date presets -----------------------------------------------------------


@callback(
    Output("filter-date-range", "start_date"),
    Output("filter-date-range", "end_date"),
    Input({"type": "date-preset", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _apply_date_preset(_clicks):
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update, dash.no_update
    service = server_state.get_service()
    start, end = service.full_date_span()
    if end is None:
        return dash.no_update, dash.no_update
    preset = triggered["index"]
    if preset == "All":
        target = pd.Timestamp(start) if start is not None else end
    else:
        years = {"1Y": 1, "3Y": 3, "5Y": 5}[preset]
        target = pd.Timestamp(end) - pd.DateOffset(years=years)
        if start is not None:
            target = max(pd.Timestamp(start), target)
    return target.date(), pd.Timestamp(end).date()


# -- season presets -----------------------------------------------------------


@callback(
    Output("filter-months", "value"),
    Input({"type": "season-preset", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _apply_season_preset(_clicks):
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update
    season = triggered["index"]
    if season == "Clear":
        return []
    service = server_state.get_service()
    profile = service.season_profile()
    if not profile:
        return dash.no_update
    table = profile.unwrap()
    return sorted(table[table["season"] == season].index.tolist())


# -- apply / reset ------------------------------------------------------------


@callback(
    Output("filters-store", "data", allow_duplicate=True),
    Input("filter-apply", "n_clicks"),
    State("filter-date-range", "start_date"),
    State("filter-date-range", "end_date"),
    State("filter-frequency", "value"),
    State("filter-varieties", "value"),
    State("filter-market", "value"),
    State("filter-months", "value"),
    State("filter-price-min", "value"),
    State("filter-price-max", "value"),
    State("filter-arrival-min", "value"),
    State("filter-arrival-max", "value"),
    prevent_initial_call=True,
)
def _apply_filters(
    _n, start_date, end_date, frequency, varieties, market, months,
    price_min, price_max, arrival_min, arrival_max,
):
    filters = FilterState(
        start=None if not start_date else pd.Timestamp(start_date),
        end=None if not end_date else pd.Timestamp(end_date),
        varieties=tuple(varieties or ()),
        market=market or "",
        price_min=price_min,
        price_max=price_max,
        arrival_min=arrival_min,
        arrival_max=arrival_max,
        months=tuple(months or ()),
        frequency=frequency or "W",
    )
    return filters_io.to_dict(filters)


@callback(
    Output("filters-store", "data", allow_duplicate=True),
    Output("filter-date-range", "start_date", allow_duplicate=True),
    Output("filter-date-range", "end_date", allow_duplicate=True),
    Output("filter-frequency", "value", allow_duplicate=True),
    Output("filter-varieties", "value", allow_duplicate=True),
    Output("filter-market", "value", allow_duplicate=True),
    Output("filter-months", "value", allow_duplicate=True),
    Output("filter-price-min", "value", allow_duplicate=True),
    Output("filter-price-max", "value", allow_duplicate=True),
    Output("filter-arrival-min", "value", allow_duplicate=True),
    Output("filter-arrival-max", "value", allow_duplicate=True),
    Input("filter-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _reset_filters(_n):
    filters = server_state.initial_filters()
    start = filters.start.date() if filters.start is not None else None
    end = filters.end.date() if filters.end is not None else None
    return (filters_io.to_dict(filters), start, end, filters.frequency, [], None, [], None, None, None, None)


# -- theme toggle -------------------------------------------------------------


@callback(
    Output("theme-store", "data"),
    Output("theme-toggle", "children"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def _toggle_theme(_n, current):
    new_name = "light" if current == "dark" else "dark"
    label = "🌙  Switch to dark theme" if new_name == "light" else "☀  Switch to light theme"
    return new_name, label


@callback(Output("theme-root", "data-theme"), Input("theme-store", "data"))
def _sync_theme_attr(theme_name):
    """The single owner of the wrapper's data-theme attribute.

    Runs on initial load too (no ``prevent_initial_call``), so the CSS
    variable block for the store's default theme is active before any click.
    """
    return theme_name


# -- reload workbook ----------------------------------------------------------


@callback(
    Output("reload-tick", "data"),
    Output("filters-store", "data", allow_duplicate=True),
    Input("reload-workbook", "n_clicks"),
    State("reload-tick", "data"),
    prevent_initial_call=True,
)
def _reload_workbook(_n, tick):
    server_state.get_service(force_reload=True)
    return (tick or 0) + 1, filters_io.to_dict(server_state.initial_filters())
