"""Plotly figure builders -- the web equivalents of every chart in
:mod:`chilli_desktop.charts`.

Each function takes the same *already-computed* data the desktop builder took
(a series, a frame, notes, a theme) and returns a ``go.Figure`` instead of
populating a Qt ``ChartPanel``. The statistics behind every chart are computed
exactly once, by :mod:`chilli_desktop.analytics` / ``forecasting`` / the
:class:`~chilli_desktop.preprocessing.DataService` -- nothing here recomputes
anything.

Plotly's built-in mode bar gives every figure zoom, pan, box-select, reset and
PNG export for free, which is more capable than the hand-built
``HoverTracker`` + ``NavigationToolbar2QT`` wrapper the desktop app needed for
the same features inside a matplotlib-in-Qt canvas.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chilli_desktop import settings
from chilli_desktop.settings import Theme

from . import theme as theme_mod

_MODEBAR_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


def figure_config() -> dict[str, Any]:
    """The ``dcc.Graph`` config every chart should share."""
    return dict(_MODEBAR_CONFIG)


def _base_layout(theme: Theme, ylabel: str = "", xlabel: str = "") -> dict[str, Any]:
    return dict(
        template=theme_mod.template_name(theme),
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


def _series_colour(theme: Theme, index: int) -> str:
    return theme.series[index % len(theme.series)]


def empty_figure(theme: Theme, message: str) -> go.Figure:
    """A blank canvas carrying only an annotation -- used sparingly.

    Most "no data" cases are handled one level up by showing a text panel
    instead of a graph at all (mirroring ``ChartPanel.show_unavailable``);
    this exists for the few spots (e.g. inside a subplot grid) where a figure
    object is structurally required.
    """
    fig = go.Figure()
    fig.update_layout(
        template=theme_mod.template_name(theme),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text=message, showarrow=False, font=dict(color=theme.text_muted, size=12))
        ],
    )
    return fig


# ==========================================================================
# Line / forecast / dual-axis
# ==========================================================================


def line_figure(
    series: dict[str, pd.Series],
    theme: Theme,
    *,
    ylabel: str = "",
    highlight: str | None = None,
    fill_between: tuple[pd.Series, pd.Series] | None = None,
    value_suffix: str = "",
) -> go.Figure:
    """Multi-series line chart. Mirrors ``charts.draw_line``."""
    fig = go.Figure()

    if fill_between is not None:
        lower, upper = fill_between
        common = lower.index.intersection(upper.index)
        if len(common):
            fig.add_trace(
                go.Scatter(
                    x=common, y=upper.loc[common], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=common, y=lower.loc[common], mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=_hex_to_rgba(theme.accent, 0.13),
                    showlegend=False, hoverinfo="skip",
                )
            )

    for index, (name, raw) in enumerate(series.items()):
        values = pd.to_numeric(raw, errors="coerce").dropna()
        if values.empty:
            continue
        is_highlight = highlight is not None and name == highlight
        colour = theme.accent if is_highlight else _series_colour(theme, index)
        fig.add_trace(
            go.Scatter(
                x=values.index, y=values.to_numpy(), mode="lines", name=str(name),
                line=dict(color=colour, width=2.4 if is_highlight else 1.6),
                hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>%{{y:,.2f}}{value_suffix}<extra></extra>",
            )
        )

    fig.update_layout(**_base_layout(theme, ylabel))
    return fig


def forecast_figure(
    history: pd.Series,
    forecast: pd.Series,
    theme: Theme,
    *,
    conf: tuple[pd.Series, pd.Series] | None = None,
    pred: tuple[pd.Series, pd.Series] | None = None,
    label: str = "Forecast",
    ylabel: str = "",
    history_window: int | None = None,
) -> go.Figure:
    """Fan chart: history, projection and both uncertainty bands.

    Mirrors ``charts.draw_forecast``. History is solid, the projection is
    dashed and bridged to the last actual point, and a vertical marker names
    the forecast origin -- the same "never confuse actual with projected"
    convention as the desktop app (strict data rule 10).
    """
    history = pd.to_numeric(history, errors="coerce").dropna()
    forecast = pd.to_numeric(forecast, errors="coerce").dropna()
    if history_window:
        history = history.tail(history_window)

    fig = go.Figure()

    if pred is not None:
        lower, upper = pred
        idx = forecast.index
        fig.add_trace(go.Scatter(x=idx, y=upper.reindex(idx), mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=idx, y=lower.reindex(idx), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(theme.neutral, 0.14),
            name=f"{settings.FORECAST.prediction_level:.0%} prediction interval",
        ))
    if conf is not None:
        lower, upper = conf
        idx = forecast.index
        fig.add_trace(go.Scatter(x=idx, y=upper.reindex(idx), mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=idx, y=lower.reindex(idx), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(theme.accent, 0.22),
            name=f"{settings.FORECAST.confidence_level:.0%} confidence interval",
        ))

    fig.add_trace(go.Scatter(
        x=history.index, y=history.to_numpy(), mode="lines", name="Historical (actual)",
        line=dict(color=theme.text, width=1.8),
        hovertemplate="Historical<br>%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
    ))
    bridge = pd.concat([history.tail(1), forecast])
    fig.add_trace(go.Scatter(
        x=bridge.index, y=bridge.to_numpy(), mode="lines+markers",
        name=f"{label} (projected)",
        line=dict(color=theme.accent, width=2.6, dash="dash"),
        marker=dict(size=4, color=theme.accent),
        hovertemplate=f"{label} (projected)<br>%{{x|%d %b %Y}}<br>%{{y:,.0f}}<extra></extra>",
    ))

    if not history.empty:
        origin = history.index[-1]
        fig.add_vline(x=origin, line=dict(color=theme.text_muted, width=1, dash="dot"))
        fig.add_annotation(
            x=origin, yref="paper", y=1.0, text="forecast origin", showarrow=False,
            textangle=-90, xanchor="left", yanchor="top",
            font=dict(color=theme.text_muted, size=9),
        )

    fig.update_layout(**_base_layout(theme, ylabel))
    return fig


def dual_axis_figure(
    primary: pd.Series,
    secondary: pd.Series,
    theme: Theme,
    *,
    primary_label: str = "",
    secondary_label: str = "",
    secondary_as_bars: bool = False,
) -> go.Figure:
    """Two series on independent y-axes. Mirrors ``charts.draw_dual_axis``."""
    left = pd.to_numeric(primary, errors="coerce").dropna()
    right = pd.to_numeric(secondary, errors="coerce").dropna()

    fig = go.Figure()
    if not left.empty:
        fig.add_trace(go.Scatter(
            x=left.index, y=left.to_numpy(), mode="lines", name=primary_label or "Primary",
            line=dict(color=theme.accent, width=2.0), yaxis="y1",
            hovertemplate=f"{primary_label}<br>%{{x|%d %b %Y}}<br>%{{y:,.0f}}<extra></extra>",
        ))
    if not right.empty:
        if secondary_as_bars:
            fig.add_trace(go.Bar(
                x=right.index, y=right.to_numpy(), name=secondary_label or "Secondary",
                marker_color=theme.neutral, opacity=0.42, yaxis="y2",
                hovertemplate=f"{secondary_label}<br>%{{x|%d %b %Y}}<br>%{{y:,.0f}}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=right.index, y=right.to_numpy(), mode="lines", name=secondary_label or "Secondary",
                line=dict(color=theme.neutral, width=1.6), yaxis="y2",
                hovertemplate=f"{secondary_label}<br>%{{x|%d %b %Y}}<br>%{{y:,.0f}}<extra></extra>",
            ))

    fig.update_layout(
        template=theme_mod.template_name(theme),
        yaxis=dict(title=primary_label, color=theme.accent),
        yaxis2=dict(title=secondary_label, overlaying="y", side="right", color=theme.neutral,
                     showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ==========================================================================
# Scatter
# ==========================================================================


def scatter_fit_figure(
    x: pd.Series,
    y: pd.Series,
    theme: Theme,
    *,
    xlabel: str = "",
    ylabel: str = "",
    colour_by_date: bool = True,
) -> go.Figure:
    """Scatter with an OLS fit line. Mirrors ``charts.draw_scatter_with_fit``."""
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    fig = go.Figure()
    if len(joined) < 3:
        return empty_figure(theme, "Not enough paired observations for a scatter plot.")

    if colour_by_date and isinstance(joined.index, pd.DatetimeIndex):
        ordinals = joined.index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        colourscale, reversed_ = theme_mod.sequential_colorscale(theme)
        fig.add_trace(go.Scatter(
            x=joined["x"], y=joined["y"], mode="markers",
            marker=dict(
                color=ordinals, colorscale=colourscale, reversescale=reversed_, size=7,
                colorbar=dict(
                    title="Date",
                    tickvals=np.linspace(ordinals.min(), ordinals.max(), 5),
                    ticktext=[
                        pd.Timestamp.fromordinal(int(v)).strftime("%b %Y")
                        for v in np.linspace(ordinals.min(), ordinals.max(), 5)
                    ],
                ),
                line=dict(width=0),
            ),
            customdata=joined.index.strftime("%d %b %Y"),
            hovertemplate="%{customdata}<br>x=%{x:,.0f}<br>y=%{y:,.0f}<extra></extra>",
            name="Observations", showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=joined["x"], y=joined["y"], mode="markers",
            marker=dict(color=theme.neutral, size=7, line=dict(width=0)),
            hovertemplate="x=%{x:,.0f}<br>y=%{y:,.0f}<extra></extra>",
            name="Observations", showlegend=False,
        ))

    slope, intercept = np.polyfit(joined["x"], joined["y"], 1)
    grid = np.linspace(joined["x"].min(), joined["x"].max(), 100)
    r = float(joined["x"].corr(joined["y"]))
    fig.add_trace(go.Scatter(
        x=grid, y=slope * grid + intercept, mode="lines",
        line=dict(color=theme.accent, width=2.2),
        name=f"Fit: y = {slope:,.3f}x + {intercept:,.0f}  (R² = {r ** 2:.3f})",
    ))
    fig.update_layout(**_base_layout(theme, ylabel, xlabel))
    return fig


# ==========================================================================
# Heatmaps
# ==========================================================================


def heatmap_figure(
    matrix: pd.DataFrame,
    theme: Theme,
    *,
    diverging: bool = True,
    value_format: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    cbar_label: str = "",
) -> go.Figure:
    """Annotated matrix heatmap. Mirrors ``charts.draw_heatmap`` (also used
    for the calendar-grid views)."""
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.notna().sum().sum() == 0:
        return empty_figure(theme, "The matrix contains no numeric values.")

    values = numeric.to_numpy(dtype=float)
    if diverging:
        limit = float(np.nanmax(np.abs(values))) if vmax is None else vmax
        limit = 1.0 if not np.isfinite(limit) or limit == 0 else limit
        zmin, zmax = (-limit if vmin is None else vmin), limit
        colourscale, reversed_ = theme_mod.diverging_colorscale(theme)
    else:
        zmin = float(np.nanmin(values)) if vmin is None else vmin
        zmax = float(np.nanmax(values)) if vmax is None else vmax
        colourscale, reversed_ = theme_mod.sequential_colorscale(theme)

    text = np.where(
        np.isfinite(values),
        np.vectorize(lambda v: format(v, value_format) if np.isfinite(v) else "")(values),
        "",
    ) if annotate else None

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=[str(c) for c in numeric.columns],
            y=[str(i) for i in numeric.index],
            colorscale=colourscale,
            reversescale=reversed_,
            zmin=zmin, zmax=zmax,
            text=text,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=10),
            hovertemplate="%{y} / %{x}<br>%{z:,.3f}<extra></extra>",
            colorbar=dict(title=cbar_label),
            xgap=2, ygap=2,
        )
    )
    fig.update_layout(
        template=theme_mod.template_name(theme),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="bottom"),
    )
    return fig


# ==========================================================================
# Bars
# ==========================================================================


def bar_figure(
    values: pd.Series,
    theme: Theme,
    *,
    ylabel: str = "",
    reference: float | None = None,
    reference_label: str = "",
    colour_negative: bool = True,
    horizontal: bool = False,
    value_labels: bool = False,
    value_format: str = ",.0f",
    projected: Sequence[Any] = (),
) -> go.Figure:
    """Bar chart, optionally flagging projected/expected categories with a
    hatch pattern. Mirrors ``charts.draw_bar``."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return empty_figure(theme, "There are no values to plot.")

    projected_set = {str(p) for p in projected}
    colours: list[str] = []
    patterns: list[str] = []
    for label, value in numeric.items():
        if colour_negative and value < 0:
            colours.append(theme.negative)
        elif reference is not None and value < reference:
            colours.append(theme.warning)
        else:
            colours.append(theme.accent)
        patterns.append("/" if str(label) in projected_set else "")

    text = [format(v, value_format) for v in numeric.to_numpy()] if value_labels else None
    bar_kwargs = dict(
        marker=dict(
            color=colours,
            pattern=dict(shape=patterns, fgcolor=theme.text_muted, size=6, solidity=0.3),
        ),
        text=text,
        textposition="outside" if value_labels else None,
        hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>" if not horizontal
        else "%{y}<br>%{x:,.3f}<extra></extra>",
    )
    if horizontal:
        fig = go.Figure(go.Bar(y=[str(i) for i in numeric.index], x=numeric.to_numpy(),
                                orientation="h", **bar_kwargs))
    else:
        fig = go.Figure(go.Bar(x=[str(i) for i in numeric.index], y=numeric.to_numpy(),
                                **bar_kwargs))

    if reference is not None:
        line_fn = fig.add_vline if horizontal else fig.add_hline
        line_fn(
            **({"x": reference} if horizontal else {"y": reference}),
            line=dict(color=theme.text_muted, dash="dash", width=1.2),
            annotation_text=reference_label, annotation_font=dict(color=theme.text_muted, size=9),
        )

    fig.update_layout(**_base_layout(theme, "" if horizontal else ylabel, ylabel if horizontal else ""))
    fig.update_layout(showlegend=False)
    return fig


def grouped_bar_figure(
    frame: pd.DataFrame,
    theme: Theme,
    *,
    ylabel: str = "",
    stacked: bool = False,
) -> go.Figure:
    """Grouped or stacked bars. Mirrors ``charts.draw_grouped_bar``."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.notna().sum().sum() == 0:
        return empty_figure(theme, "There are no values to plot.")

    fig = go.Figure()
    for index, column in enumerate(numeric.columns):
        fig.add_trace(go.Bar(
            x=[str(i) for i in numeric.index], y=numeric[column].to_numpy(),
            name=str(column), marker_color=_series_colour(theme, index),
            hovertemplate=f"%{{x}}<br>{column}: %{{y:,.3f}}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(theme, ylabel))
    fig.update_layout(barmode="relative" if stacked else "group")
    fig.add_hline(y=0, line=dict(color=theme.border, width=1))
    return fig


# ==========================================================================
# Stems (elasticity / lag / ACF-PACF)
# ==========================================================================


def _stem_traces(
    x: Sequence[Any], y: np.ndarray, theme: Theme, *, highlight_index: Any = None
) -> list[go.Scatter]:
    """Vertical stems from zero to each value, plus tip markers.

    Plotly has no built-in stem plot, so each stem is one line segment; all
    segments share a single trace via ``None`` separators, which keeps the
    figure light regardless of how many lags are drawn.
    """
    xs: list[Any] = []
    ys: list[float] = []
    for xi, yi in zip(x, y):
        xs.extend([xi, xi, None])
        ys.extend([0, yi, None])
    colours = [
        theme.accent if highlight_index is not None and xi == highlight_index else theme.neutral
        for xi in x
    ]
    return [
        go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.neutral, width=1.6),
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=list(x), y=list(y), mode="markers", marker=dict(color=colours, size=7),
                   showlegend=False, hovertemplate="lag %{x}<br>%{y:,.4f}<extra></extra>"),
    ]


def stem_figure(
    values: pd.Series,
    theme: Theme,
    *,
    xlabel: str = "",
    ylabel: str = "",
    band: float | None = None,
    highlight_index: Any = None,
) -> go.Figure:
    """Stem plot for lag profiles. Mirrors ``charts.draw_stem``."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return empty_figure(theme, "There are no values to plot.")

    fig = go.Figure()
    for trace in _stem_traces(numeric.index, numeric.to_numpy(), theme, highlight_index=highlight_index):
        fig.add_trace(trace)
    fig.add_hline(y=0, line=dict(color=theme.border, width=1))
    fig.add_vline(x=0, line=dict(color=theme.text_muted, width=1, dash="dot"))
    if band is not None and np.isfinite(band):
        fig.add_hrect(y0=-abs(band), y1=abs(band), fillcolor=theme.text_muted, opacity=0.15,
                       line_width=0, annotation_text="95% significance band",
                       annotation_font=dict(size=9, color=theme.text_muted))
    fig.update_layout(**_base_layout(theme, ylabel, xlabel))
    fig.update_layout(showlegend=False)
    return fig


def acf_pacf_figure(frame: pd.DataFrame, theme: Theme) -> go.Figure:
    """Side-by-side ACF/PACF stem plots. Mirrors ``charts.draw_acf_pacf``."""
    if frame is None or frame.empty:
        return empty_figure(theme, "Autocorrelation could not be computed.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelation (ACF)", "Partial autocorrelation (PACF)"))
    band = float(frame["Upper 95%"].iloc[0]) if "Upper 95%" in frame.columns else None
    for col_index, column in enumerate(("ACF", "PACF"), start=1):
        if column not in frame.columns:
            continue
        values = frame[column]
        for trace in _stem_traces(values.index, values.to_numpy(dtype=float), theme):
            fig.add_trace(trace, row=1, col=col_index)
        if band is not None:
            fig.add_hrect(y0=-band, y1=band, fillcolor=theme.text_muted, opacity=0.14,
                          line_width=0, row=1, col=col_index)
        fig.add_hline(y=0, line=dict(color=theme.border, width=1), row=1, col=col_index)

    fig.update_layout(template=theme_mod.template_name(theme), showlegend=False)
    fig.update_xaxes(title_text="Lag")
    return fig


# ==========================================================================
# Decomposition
# ==========================================================================


def decomposition_figure(frame: pd.DataFrame, theme: Theme, *, ylabel: str = "") -> go.Figure:
    """Stacked observed/trend/seasonal/residual panels. Mirrors
    ``charts.draw_decomposition``."""
    if frame is None or frame.empty:
        return empty_figure(theme, "The decomposition could not be computed.")
    components = [c for c in ("Observed", "Trend", "Seasonal", "Residual") if c in frame.columns]
    if not components:
        return empty_figure(theme, "The decomposition returned no components.")

    fig = make_subplots(rows=len(components), cols=1, shared_xaxes=True,
                        subplot_titles=components, vertical_spacing=0.05)
    for index, component in enumerate(components, start=1):
        values = pd.to_numeric(frame[component], errors="coerce")
        colour = theme.text if component == "Observed" else _series_colour(theme, index - 1)
        if component == "Residual":
            for trace in _stem_traces(values.index, values.to_numpy(dtype=float), theme):
                fig.add_trace(trace, row=index, col=1)
            fig.add_hline(y=0, line=dict(color=theme.border, width=1), row=index, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=values.index, y=values.to_numpy(), mode="lines",
                           line=dict(color=colour, width=1.8), showlegend=False,
                           hovertemplate=f"{component}<br>%{{x|%d %b %Y}}<br>%{{y:,.2f}}<extra></extra>"),
                row=index, col=1,
            )
    fig.update_layout(template=theme_mod.template_name(theme), height=140 * len(components) + 80)
    return fig


# ==========================================================================
# Box plots
# ==========================================================================


def box_by_group_figure(groups: dict[str, pd.Series], theme: Theme, *, ylabel: str = "") -> go.Figure:
    """Distribution by category. Mirrors ``charts.draw_box_by_group``."""
    usable = {k: pd.to_numeric(v, errors="coerce").dropna() for k, v in groups.items()}
    usable = {k: v for k, v in usable.items() if len(v) >= 2}
    if not usable:
        return empty_figure(theme, "No group has at least two observations.")

    fig = go.Figure()
    for index, (name, values) in enumerate(usable.items()):
        fig.add_trace(go.Box(
            y=values.to_numpy(), name=str(name), marker_color=_series_colour(theme, index),
            boxmean=False,
        ))
    fig.update_layout(**_base_layout(theme, ylabel))
    fig.update_layout(showlegend=False)
    return fig


# ==========================================================================
# Influence network
# ==========================================================================


def influence_network_figure(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float, str]],
    theme: Theme,
    *,
    node_scores: dict[str, float] | None = None,
) -> go.Figure:
    """Influence diagram. Mirrors ``charts.draw_influence_network``.

    One-way edges get an arrowhead annotation; two-way feedback edges are
    drawn as a plain dashed line -- Plotly annotations only arrowhead one end,
    so a double-headed relationship is distinguished by dash style instead of
    a second arrowhead, with a legend explaining the convention.
    """
    if not nodes:
        return empty_figure(theme, "No series had enough overlapping history to place on the diagram.")

    count = len(nodes)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, count, endpoint=False)
    positions = {str(n): (float(np.cos(a)), float(np.sin(a))) for n, a in zip(nodes, angles)}

    scores = node_scores or {}
    finite = [v for v in scores.values() if np.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    spread = (hi - lo) or 1.0

    fig = go.Figure()
    annotations: list[dict[str, Any]] = []
    strengths = [abs(s) for *_, s, _ in edges] if edges else []
    max_strength = max(strengths) if strengths else 1.0

    for start, end, strength, kind in edges:
        if start not in positions or end not in positions:
            continue
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        width = 1.2 + 3.2 * (abs(strength) / max_strength if max_strength else 0)
        if kind == "feedback":
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=theme.neutral, width=width, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=width,
                arrowcolor=theme.accent, standoff=24, startstandoff=24,
            ))

    for name, (x, y) in positions.items():
        score = scores.get(name, float("nan"))
        normalised = (score - lo) / spread if np.isfinite(score) else 0.0
        size = 34 + 26 * normalised
        colour = (
            theme.accent if normalised > 0.66 else theme.neutral if normalised > 0.33 else theme.surface_alt
        )
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", marker=dict(size=size, color=colour,
                       line=dict(color=theme.border, width=1.5)),
            text=[name if np.isnan(score) else f"{name}<br>{score:.2f}"],
            textposition="middle center", textfont=dict(color=theme.text, size=10),
            showlegend=False, hovertemplate=f"{name}<br>score %{{customdata:.2f}}<extra></extra>",
            customdata=[score],
        ))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=theme.accent, width=2),
                             name="One-way influence"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=theme.neutral, width=2, dash="dash"),
                             name="Two-way feedback"))

    fig.update_layout(
        template=theme_mod.template_name(theme),
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x", scaleratio=1),
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        height=460,
    )
    return fig


# ==========================================================================
# Sentiment gauge
# ==========================================================================


def sentiment_gauge_figure(score: float, label: str, theme: Theme) -> go.Figure:
    """Bearish-to-bullish gauge. Web equivalent of the hand-painted
    ``ui.SentimentGauge`` QPainter widget.

    Uses ``mode="gauge"`` only, with the formatted score folded into the
    title text, rather than Plotly's built-in ``+number`` mode: the bundled
    plotly.js in this Dash release renders ``number.valueformat`` as the raw
    unformatted float (confirmed by inspecting the figure JSON, which carries
    the correct format string -- the bug is in the front-end renderer, not
    here). Formatting the text ourselves sidesteps it entirely and also
    reproduces the desktop widget's single combined line more faithfully.
    """
    if not np.isfinite(score):
        return empty_figure(theme, settings.DATA_UNAVAILABLE_MESSAGE)

    title_text = f"{label}   ({score:+.2f})" if label else f"{score:+.2f}"
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        gauge=dict(
            axis=dict(range=[-1, 1], tickvals=[-1, 0, 1], ticktext=["Bearish", "Neutral", "Bullish"],
                      tickfont=dict(color=theme.text_muted, size=10)),
            bar=dict(color=theme.text, thickness=0.25),
            bgcolor=theme.surface,
            steps=[
                dict(range=[-1, -0.15], color=theme.negative),
                dict(range=[-0.15, 0.15], color=theme.text_muted),
                dict(range=[0.15, 1], color=theme.positive),
            ],
        ),
        title=dict(text=title_text, font=dict(color=theme.text, size=16)),
    ))
    fig.update_layout(template=theme_mod.template_name(theme), height=220,
                      margin=dict(l=30, r=30, t=50, b=10))
    return fig


# ==========================================================================
# Small helpers
# ==========================================================================


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
