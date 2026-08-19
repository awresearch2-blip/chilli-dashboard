"""A Qt-free stand-in for :class:`chilli_desktop.charts.ChartPanel`.

``chilli_desktop.charts`` holds every chart-drawing function (``draw_line``,
``draw_forecast``, ``draw_heatmap``, ...), and each one only touches a handful
of things on the ``panel`` argument it receives: a matplotlib ``Figure`` via
``.begin()``, the active ``Theme`` via ``.theme``, and ``.finish()`` /
``.show_unavailable()`` to record the outcome. None of that is Qt-specific --
the Qt-specific part is ``ChartPanel`` itself, which wraps those calls in
widgets, buttons and a navigation toolbar for the desktop window.

:class:`WebChartPanel` implements the same small surface with a plain
matplotlib ``Figure`` and no widget behind it, so every ``draw_*`` function
runs completely unmodified and produces the same figure, in the same theme
colours, that the desktop app would show. Streamlit then renders that figure
as an image with ``st.pyplot``.

The trade-off, stated plainly: the desktop app's hover tooltips, click-to-zoom
and pan come from Qt's event loop and matplotlib's interactive backend,
neither of which exist in a browser tab showing a static image. Those three
interactions are the one piece of "current functionality" this web version
does not preserve. Everything else -- every chart, every statistic, every
insight, every forecast -- is produced by the identical code path.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass, field
from typing import Any, Sequence

import streamlit as st
from matplotlib.figure import Figure

from chilli_desktop import settings
from chilli_desktop.settings import Theme
from chilli_desktop.utils import Result


class _NullCanvas:
    """Satisfies the handful of canvas calls ``charts.py`` makes in passing.

    ``HoverTracker`` (built for the desktop's live cursor) connects to
    ``canvas.mpl_connect`` and later calls ``canvas.draw_idle()``. Those calls
    are harmless no-ops here: the figure is rendered once, as a static image,
    after the drawing function returns.
    """

    def mpl_connect(self, *_args: Any, **_kwargs: Any) -> int:
        return 0

    def mpl_disconnect(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def draw_idle(self) -> None:
        return None


@dataclass
class WebChartPanel:
    """Drop-in replacement for ``ChartPanel`` with no Qt widget behind it."""

    theme: Theme
    figure: Figure = field(default_factory=lambda: Figure(figsize=(7.6, 3.6), dpi=220))
    canvas: _NullCanvas = field(default_factory=_NullCanvas)
    title: str = ""
    source: str = ""
    notes: list[str] = field(default_factory=list)
    unavailable_message: str | None = None

    def begin(self) -> Figure:
        self.figure.clear()
        self.unavailable_message = None
        return self.figure

    def finish(self, source: str = "", notes: Sequence[str] | None = None, hover: Any = None) -> None:
        self.source = source
        self.notes = [n for n in (notes or []) if n]

    def show_unavailable(self, result_or_message: Any, source: str = "") -> None:
        if isinstance(result_or_message, Result):
            self.unavailable_message = result_or_message.message()
            self.source = source or result_or_message.source
        else:
            message = str(result_or_message) or settings.DATA_UNAVAILABLE_MESSAGE
            if settings.DATA_UNAVAILABLE_MESSAGE not in message:
                message = f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{message}"
            self.unavailable_message = message
            self.source = source
        self.notes = []
        self.figure.clear()

    def set_title(self, title: str) -> None:
        self.title = title

    def set_source(self, source: str) -> None:
        self.source = source

    def set_notes(self, notes: Sequence[str]) -> None:
        self.notes = [n for n in notes if n]


def new_panel(theme: Theme, figsize: tuple[float, float] = (7.6, 3.6)) -> WebChartPanel:
    """Create a fresh panel sized for the current theme.

    Rendered at 220 DPI. Streamlit's ``use_container_width=True`` stretches
    the chart's raster image to fill a wide page column with pure CSS scaling
    -- it does not re-render the figure at a larger pixel size. At the
    previous 118 DPI (~900 px wide) that stretch visibly upscaled and blurred
    every chart, which is what reads as "looks like a screenshot". 220 DPI
    renders each chart at roughly 1650-2200 px wide, comfortably past the
    width of the content column on any normal monitor, so the browser is
    always scaling *down* (or not at all) rather than up.
    """
    return WebChartPanel(theme=theme, figure=Figure(figsize=figsize, dpi=220))


def render_panel(panel: WebChartPanel, *, key: str, height_hint: str = "") -> None:
    """Draw a panel's outcome into the Streamlit page: chart, or the
    standard unavailable message, plus the source and notes caption.

    ``key`` must be unique per call site on a page -- it namespaces the PNG
    download button so two charts on one page don't collide.
    """
    if panel.title:
        st.markdown(f"**{panel.title}**")
    if panel.unavailable_message is not None:
        st.info(panel.unavailable_message)
    elif panel.figure.get_axes():
        st.pyplot(panel.figure, use_container_width=True)
        buffer = io.BytesIO()
        panel.figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight",
                             facecolor=settings.THEMES["light"].surface)
        st.download_button(
            "Download PNG", buffer.getvalue(), file_name=f"{key}.png",
            mime="image/png", key=f"dl_{key}",
        )
    else:
        st.info(settings.DATA_UNAVAILABLE_MESSAGE)

    caption = f"Source: {panel.source}" if panel.source else "Source: —"
    if panel.notes:
        caption += "\n\n" + "\n".join(f"- {n}" for n in panel.notes)
    st.caption(caption)


def render_result_table(
    result: Result[Any],
    *,
    empty_ok: bool = False,
    height: int | None = None,
) -> None:
    """Render a Result-wrapped DataFrame/Series with its source and notes.

    Mirrors the desktop's ``DataTable``: the unavailable state is the mandated
    message plus the specific reason, never a blank table.
    """
    if not result:
        st.info(f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{result.reason}")
        return
    value = result.unwrap()
    frame = value.to_frame() if hasattr(value, "to_frame") and not hasattr(value, "columns") else value
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        if not empty_ok:
            st.info(f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nThe table contains no rows.")
            return
    dataframe_kwargs: dict[str, Any] = {"use_container_width": True}
    if height is not None:
        dataframe_kwargs["height"] = height
    st.dataframe(frame, **dataframe_kwargs)
    caption = f"Source: {result.source}" if result.source else "Source: —"
    if result.notes:
        caption += "\n\n" + "\n".join(f"- {n}" for n in result.notes if n)
    st.caption(caption)


def inject_theme_css(theme: Theme) -> None:
    """Approximate the desktop app's palette inside Streamlit's own chrome.

    The block must reach ``st.markdown`` with **no leading whitespace** on any
    line. Markdown treats a 4-space indent as a fenced code block, which would
    render the whole ``<style>`` tag as inert, visible text instead of CSS --
    ``textwrap.dedent`` strips the indentation this source file's formatting
    otherwise leaves in place.
    """
    st.markdown(
        textwrap.dedent(
            f"""\
            <style>
            .stApp {{ background-color: {theme.window}; }}
            section[data-testid="stSidebar"] {{ background-color: {theme.surface}; }}
            div[data-testid="stMetric"] {{
                background-color: {theme.surface};
                border: 1px solid {theme.border};
                border-left: 3px solid {theme.accent};
                border-radius: 8px;
                padding: 10px 14px;
            }}
            div[data-testid="stMetricValue"] {{ color: {theme.text}; }}
            div[data-testid="stMetricLabel"] {{ color: {theme.text_muted}; }}
            .streamlit-expanderHeader {{ color: {theme.text}; }}
            h1, h2, h3 {{ color: {theme.text}; }}
            p, label, .stMarkdown {{ color: {theme.text}; }}
            .stCaption, [data-testid="stCaptionContainer"] {{ color: {theme.text_muted} !important; }}
            div[data-testid="stAlert"] {{ background-color: {theme.surface_alt}; }}
            .insight-card {{
                background-color: {theme.surface};
                border: 1px solid {theme.border};
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 10px;
            }}
            .insight-badge {{
                display: inline-block; font-size: 10px; font-weight: 700;
                border-radius: 3px; padding: 1px 6px; margin-right: 6px;
            }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


_STRENGTH_COLOUR = {
    "strong": "#3fb950",
    "moderate": "#f0883e",
    "weak": "#d29922",
    "informational": "#58a6ff",
    "data gap": "#f85149",
}


def render_insight_card(insight: Any) -> None:
    """Render one Insight as a small bordered card, matching the desktop look.

    Built as one unindented line, deliberately. A multi-line, indented
    f-string here would trip markdown's "4 spaces = code block" rule (as
    ``inject_theme_css`` above did until it was fixed) and print raw ``<div>``
    tags instead of rendering them.
    """
    colour = _STRENGTH_COLOUR.get(insight.strength, "#8b98a5")
    direction_html = ""
    if insight.direction not in ("n/a", ""):
        dir_colour = (
            "#3fb950" if insight.direction == "bullish"
            else "#f85149" if insight.direction == "bearish"
            else "#8b98a5"
        )
        direction_html = (
            f'<span class="insight-badge" style="color:{dir_colour};border:1px solid {dir_colour}">'
            f"{insight.direction.upper()}</span>"
        )
    html = (
        f'<div class="insight-card" style="border-left:3px solid {colour}">'
        f'<span class="insight-badge" style="color:{colour};border:1px solid {colour}">{insight.strength.upper()}</span>'
        f'<span style="font-size:11px;color:#8b98a5;font-weight:600">{insight.category}</span>'
        f"{direction_html}"
        f'<div style="font-size:14px;font-weight:600;margin-top:6px;">{insight.headline}</div>'
        f'<div style="font-size:12px;color:#8b98a5;margin-top:4px;">{insight.detail}</div>'
        f'<div style="font-size:10px;color:#8b98a5;font-style:italic;margin-top:6px;">Source: {insight.source or "—"}</div>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    if insight.evidence:
        with st.expander(f"Evidence ({len(insight.evidence)})"):
            for item in insight.evidence:
                st.markdown(f"- {item}")
