"""Dark/light theming for the web app, built directly from the desktop
palette in :mod:`chilli_desktop.settings`.

The colour tokens themselves are not reinvented: :data:`settings.DARK_THEME`
and :data:`settings.LIGHT_THEME` are plain hex-string dataclasses with no Qt
dependency, so they are imported and reused verbatim. This module only adds
the translation from "a Theme dataclass" to "a Plotly template" and "a set of
CSS custom properties", which the desktop app never needed because Qt
stylesheets consumed the dataclass directly.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from chilli_desktop import settings
from chilli_desktop.settings import Theme

#: Plotly's built-in colorscale names line up with the matplotlib colormap
#: names already chosen in settings.py, so no manual colour sampling is
#: needed -- only the "reversed" (`_r`) suffix needs translating.
_DIVERGING = {"RdYlBu_r": ("RdYlBu", True), "RdYlBu": ("RdYlBu", False)}
_SEQUENTIAL = {"inferno": ("Inferno", False), "viridis": ("Viridis", False)}


def diverging_colorscale(theme: Theme) -> tuple[str, bool]:
    return _DIVERGING.get(theme.diverging_cmap, ("RdYlBu", True))


def sequential_colorscale(theme: Theme) -> tuple[str, bool]:
    return _SEQUENTIAL.get(theme.sequential_cmap, ("Inferno", False))


def plotly_template(theme: Theme) -> go.layout.Template:
    """Build a Plotly template so every figure inherits the theme in one place."""
    axis = dict(
        gridcolor=theme.grid,
        zerolinecolor=theme.border,
        linecolor=theme.border,
        tickfont=dict(color=theme.text_muted, size=10),
        title=dict(font=dict(color=theme.text_muted, size=11)),
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=theme.surface,
            plot_bgcolor=theme.surface,
            font=dict(color=theme.text, family="Segoe UI, Calibri, sans-serif", size=11),
            title=dict(font=dict(color=theme.text, size=13)),
            colorway=list(theme.series),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                bgcolor=theme.surface_alt,
                bordercolor=theme.border,
                borderwidth=1,
                font=dict(color=theme.text, size=10),
            ),
            margin=dict(l=56, r=24, t=36, b=40),
            hoverlabel=dict(
                bgcolor=theme.surface_alt,
                bordercolor=theme.accent,
                font=dict(color=theme.text, size=11),
            ),
        )
    )


# Register both templates once, at import time, so every chart builder can
# simply set `fig.update_layout(template=f"chilli_{theme.name}")`.
for _name, _theme in settings.THEMES.items():
    pio.templates[f"chilli_{_name}"] = plotly_template(_theme)


def template_name(theme: Theme) -> str:
    return f"chilli_{theme.name}"


def css_variables(theme: Theme, selector: str = ":root") -> str:
    """Render the theme as CSS custom properties for the page stylesheet.

    ``selector`` scopes the block to a wrapper element rather than the real
    document root, because the toggle is implemented as a plain Dash callback
    setting a ``data-theme`` attribute on a wrapper ``html.Div`` -- Dash
    components have no handle on the actual ``<html>`` tag, and reaching for
    client-side JS to touch ``document.documentElement`` would be the only
    alternative for no real benefit.
    """
    tokens = {
        "--window": theme.window,
        "--surface": theme.surface,
        "--surface-alt": theme.surface_alt,
        "--border": theme.border,
        "--text": theme.text,
        "--text-muted": theme.text_muted,
        "--accent": theme.accent,
        "--accent-soft": theme.accent_soft,
        "--positive": theme.positive,
        "--negative": theme.negative,
        "--neutral": theme.neutral,
        "--warning": theme.warning,
        "--grid": theme.grid,
    }
    body = "\n".join(f"  {k}: {v};" for k, v in tokens.items())
    return f'{selector}[data-theme="{theme.name}"] {{\n{body}\n}}'


def all_theme_css(selector: str = ".theme-root") -> str:
    return "\n\n".join(css_variables(t, selector) for t in settings.THEMES.values())


ALL_THEME_CSS = all_theme_css()
