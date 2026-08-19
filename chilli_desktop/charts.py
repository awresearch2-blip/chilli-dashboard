"""Interactive, theme-aware charts embedded in Qt.

Built directly on ``Figure`` + ``FigureCanvasQTAgg`` rather than pyplot: pyplot
keeps global figure state, which leaks memory and fights Qt when panels are
rebuilt on every filter change.

Every chart is wrapped in a :class:`ChartPanel`, which supplies:

*   a title and, mandatorily, the **source sheet** caption (strict data rule 8);
*   a matplotlib navigation toolbar -- zoom, pan, reset home, configure;
*   PNG and PDF export at print resolution;
*   hover tooltips showing the exact value under the cursor;
*   a collapsible assumptions/notes area (strict data rule 9);
*   the standard "Data not available in uploaded workbook." state, which is
    what a panel shows instead of an empty axis when the workbook cannot
    support it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from PySide6 import QtCore, QtGui, QtWidgets

from . import settings
from .settings import Theme
from .utils import LOG, Result, ensure_dir, fmt_number

# --------------------------------------------------------------------------
# Figure styling
# --------------------------------------------------------------------------


def style_figure(figure: Figure, theme: Theme) -> None:
    """Apply the active theme to a figure and all of its axes."""
    figure.patch.set_facecolor(theme.surface)
    for axes in figure.get_axes():
        style_axes(axes, theme)


def style_axes(axes, theme: Theme, *, grid: bool = True) -> None:
    """Apply the active theme to one axes object."""
    axes.set_facecolor(theme.surface)
    for spine_name, spine in axes.spines.items():
        spine.set_visible(spine_name in ("left", "bottom"))
        spine.set_color(theme.border)
        spine.set_linewidth(0.8)
    axes.tick_params(colors=theme.text_muted, labelsize=8, length=3, width=0.7)
    axes.xaxis.label.set_color(theme.text_muted)
    axes.yaxis.label.set_color(theme.text_muted)
    axes.xaxis.label.set_fontsize(9)
    axes.yaxis.label.set_fontsize(9)
    axes.title.set_color(theme.text)
    axes.title.set_fontsize(10)
    if grid:
        axes.grid(True, color=theme.grid, linewidth=0.6, alpha=0.9)
        axes.set_axisbelow(True)
    else:
        axes.grid(False)


def _style_legend(axes, theme: Theme, **kwargs: Any) -> None:
    """Add a legend that reads correctly in both themes."""
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        return
    defaults: dict[str, Any] = {
        "frameon": True,
        "fontsize": 8,
        "loc": "best",
        "framealpha": 0.92,
    }
    defaults.update(kwargs)
    legend = axes.legend(**defaults)
    legend.get_frame().set_facecolor(theme.surface_alt)
    legend.get_frame().set_edgecolor(theme.border)
    for text in legend.get_texts():
        text.set_color(theme.text)


def _format_date_axis(axes, theme: Theme) -> None:
    """Compact, non-overlapping date ticks."""
    locator = AutoDateLocator(minticks=4, maxticks=9)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    for label in axes.get_xticklabels():
        label.set_color(theme.text_muted)


def _series_colour(theme: Theme, index: int) -> str:
    return theme.series[index % len(theme.series)]


# --------------------------------------------------------------------------
# Hover tooltips
# --------------------------------------------------------------------------


@dataclass
class _HoverTarget:
    """A plotted series registered for hover read-out."""

    label: str
    x: np.ndarray
    y: np.ndarray
    colour: str
    is_date: bool
    formatter: Callable[[float], str] | None = None


class HoverTracker:
    """Shows the nearest data point's value as the cursor moves.

    matplotlib has no built-in tooltip. This keeps a numeric copy of each
    registered series, finds the closest point in display coordinates, and
    draws a single annotation. Working in display space (rather than data
    space) means the "nearest" point is the one that looks nearest, even when
    the two axes have wildly different units.
    """

    #: Maximum pixel distance at which a point is considered hovered.
    PICK_RADIUS_PX = 40

    def __init__(self, canvas: FigureCanvasQTAgg, axes, theme: Theme) -> None:
        self.canvas = canvas
        self.axes = axes
        self.theme = theme
        self.targets: list[_HoverTarget] = []
        self._annotation = axes.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 14),
            textcoords="offset points",
            fontsize=8,
            color=theme.text,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": theme.surface_alt,
                "edgecolor": theme.accent,
                "linewidth": 0.9,
                "alpha": 0.97,
            },
            zorder=1000,
            annotation_clip=False,
        )
        self._annotation.set_visible(False)
        self._marker = Line2D(
            [], [], marker="o", markersize=7, markerfacecolor="none",
            markeredgecolor=theme.accent, markeredgewidth=1.6, linestyle="none",
            zorder=999,
        )
        axes.add_line(self._marker)
        self._marker.set_visible(False)
        self._cid = canvas.mpl_connect("motion_notify_event", self._on_move)
        self._cid_leave = canvas.mpl_connect("axes_leave_event", self._on_leave)

    def register(
        self,
        label: str,
        x: Iterable[Any],
        y: Iterable[Any],
        colour: str,
        *,
        is_date: bool = True,
        formatter: Callable[[float], str] | None = None,
    ) -> None:
        """Register a plotted series for hover read-out."""
        try:
            x_arr = np.asarray(
                [
                    v.timestamp() if hasattr(v, "timestamp") else float(v)
                    for v in pd.to_datetime(list(x)).to_pydatetime()
                ]
                if is_date
                else [float(v) for v in x],
                dtype=float,
            )
        except (TypeError, ValueError):
            return
        try:
            y_arr = np.asarray([float(v) for v in y], dtype=float)
        except (TypeError, ValueError):
            return
        if x_arr.size == 0 or x_arr.size != y_arr.size:
            return
        self.targets.append(
            _HoverTarget(label, x_arr, y_arr, colour, is_date, formatter)
        )

    def register_series(
        self, label: str, series: pd.Series, colour: str,
        formatter: Callable[[float], str] | None = None,
    ) -> None:
        """Convenience wrapper for a date-indexed pandas Series."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return
        is_date = isinstance(clean.index, pd.DatetimeIndex)
        self.register(
            label,
            clean.index if is_date else clean.index.to_numpy(),
            clean.to_numpy(),
            colour,
            is_date=is_date,
            formatter=formatter,
        )

    # -- internals --------------------------------------------------------

    def _to_display(self, target: _HoverTarget) -> np.ndarray:
        """Project a target's data points into pixel coordinates."""
        if target.is_date:
            from matplotlib.dates import date2num

            x_plot = date2num(pd.to_datetime(target.x, unit="s").to_pydatetime())
        else:
            x_plot = target.x
        points = np.column_stack([np.asarray(x_plot, dtype=float), target.y])
        finite = np.isfinite(points).all(axis=1)
        if not finite.any():
            return np.empty((0, 2))
        transformed = self.axes.transData.transform(points[finite])
        result = np.full((points.shape[0], 2), np.nan)
        result[finite] = transformed
        return result

    def _on_move(self, event) -> None:
        if event.inaxes is not self.axes or not self.targets:
            self._hide()
            return

        best: tuple[float, _HoverTarget, int] | None = None
        for target in self.targets:
            display = self._to_display(target)
            if display.size == 0:
                continue
            distances = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
            if not np.isfinite(distances).any():
                continue
            index = int(np.nanargmin(distances))
            distance = float(distances[index])
            if best is None or distance < best[0]:
                best = (distance, target, index)

        if best is None or best[0] > self.PICK_RADIUS_PX:
            self._hide()
            return

        _, target, index = best
        y_value = float(target.y[index])
        if target.is_date:
            stamp = pd.Timestamp(target.x[index], unit="s")
            x_label = f"{stamp:%d %b %Y}"
            from matplotlib.dates import date2num

            x_plot = float(date2num(stamp.to_pydatetime()))
        else:
            x_label = fmt_number(target.x[index], 0)
            x_plot = float(target.x[index])

        formatted = (
            target.formatter(y_value) if target.formatter else fmt_number(y_value, 2)
        )
        self._annotation.xy = (x_plot, y_value)
        self._annotation.set_text(f"{target.label}\n{x_label}\n{formatted}")
        self._annotation.get_bbox_patch().set_edgecolor(target.colour)
        self._annotation.set_visible(True)
        self._marker.set_data([x_plot], [y_value])
        self._marker.set_markeredgecolor(target.colour)
        self._marker.set_visible(True)
        self.canvas.draw_idle()

    def _on_leave(self, _event) -> None:
        self._hide()

    def _hide(self) -> None:
        if self._annotation.get_visible() or self._marker.get_visible():
            self._annotation.set_visible(False)
            self._marker.set_visible(False)
            self.canvas.draw_idle()

    def disconnect(self) -> None:
        for cid in (self._cid, self._cid_leave):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------
# The Qt wrapper
# --------------------------------------------------------------------------


class _Toolbar(NavigationToolbar2QT):
    """Navigation toolbar limited to the actions that make sense here."""

    toolitems = [
        t for t in NavigationToolbar2QT.toolitems
        if t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Subplots")
    ]


class ChartPanel(QtWidgets.QFrame):
    """A titled, exportable, hover-enabled chart with provenance and notes."""

    def __init__(
        self,
        title: str = "",
        *,
        theme: Theme | None = None,
        height: int = 340,
        parent: QtWidgets.QWidget | None = None,
        show_toolbar: bool = True,
    ) -> None:
        super().__init__(parent)
        self.theme = theme or settings.THEMES[settings.DEFAULT_THEME]
        self._title = title
        self._source = ""
        self._notes: list[str] = []
        self._hover: HoverTracker | None = None
        self.setObjectName("chartPanel")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("chartTitle")
        self.title_label.setWordWrap(True)
        header.addWidget(self.title_label, 1)

        self.png_button = QtWidgets.QToolButton()
        self.png_button.setText("PNG")
        self.png_button.setToolTip("Export this chart as a high-resolution PNG")
        self.png_button.clicked.connect(self.export_png)
        header.addWidget(self.png_button)

        self.pdf_button = QtWidgets.QToolButton()
        self.pdf_button.setText("PDF")
        self.pdf_button.setToolTip("Export this chart as a vector PDF")
        self.pdf_button.clicked.connect(self.export_pdf)
        header.addWidget(self.pdf_button)

        self.notes_button = QtWidgets.QToolButton()
        self.notes_button.setText("Notes")
        self.notes_button.setCheckable(True)
        self.notes_button.setToolTip("Show the assumptions behind this chart")
        self.notes_button.toggled.connect(self._toggle_notes)
        header.addWidget(self.notes_button)
        layout.addLayout(header)

        self.figure = Figure(figsize=(7.6, 3.4), dpi=settings.CHART_DPI, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(height)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.canvas, 1)

        self.message_label = QtWidgets.QLabel()
        self.message_label.setObjectName("unavailableMessage")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.message_label.setVisible(False)
        self.message_label.setMinimumHeight(height)
        layout.addWidget(self.message_label, 1)

        self.toolbar = _Toolbar(self.canvas, self) if show_toolbar else None
        if self.toolbar is not None:
            self.toolbar.setIconSize(QtCore.QSize(16, 16))
            layout.addWidget(self.toolbar)

        self.source_label = QtWidgets.QLabel()
        self.source_label.setObjectName("sourceCaption")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        self.notes_label = QtWidgets.QLabel()
        self.notes_label.setObjectName("notesCaption")
        self.notes_label.setWordWrap(True)
        self.notes_label.setVisible(False)
        self.notes_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.notes_label)

        self.apply_theme(self.theme)

    # -- lifecycle --------------------------------------------------------

    def set_title(self, title: str) -> None:
        self._title = title
        self.title_label.setText(title)

    def begin(self) -> Figure:
        """Clear the figure and return it, ready for new artists."""
        if self._hover is not None:
            self._hover.disconnect()
            self._hover = None
        self.figure.clear()
        self.message_label.setVisible(False)
        self.canvas.setVisible(True)
        if self.toolbar is not None:
            self.toolbar.setVisible(True)
            # Discard the old view stack so "Home" resets to the new data.
            self.toolbar.update()
        return self.figure

    def finish(
        self,
        source: str = "",
        notes: Sequence[str] | None = None,
        hover: HoverTracker | None = None,
    ) -> None:
        """Apply theming, record provenance and repaint."""
        self._hover = hover
        style_figure(self.figure, self.theme)
        self.set_source(source)
        self.set_notes(notes or [])
        self.canvas.draw_idle()

    def show_unavailable(self, result_or_message: Any, source: str = "") -> None:
        """Render the mandated unavailable state instead of an empty chart."""
        if isinstance(result_or_message, Result):
            message = result_or_message.message()
            source = source or result_or_message.source
        else:
            message = str(result_or_message) or settings.DATA_UNAVAILABLE_MESSAGE
            if settings.DATA_UNAVAILABLE_MESSAGE not in message:
                message = f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{message}"
        if self._hover is not None:
            self._hover.disconnect()
            self._hover = None
        self.figure.clear()
        self.canvas.setVisible(False)
        if self.toolbar is not None:
            self.toolbar.setVisible(False)
        self.message_label.setText(message)
        self.message_label.setVisible(True)
        self.set_source(source)
        self.set_notes([])

    def set_source(self, source: str) -> None:
        self._source = source
        self.source_label.setText(
            f"Source: {source}" if source else "Source: —"
        )
        self.source_label.setVisible(True)

    def set_notes(self, notes: Sequence[str]) -> None:
        self._notes = [n for n in notes if n]
        has_notes = bool(self._notes)
        self.notes_button.setEnabled(has_notes)
        self.notes_button.setText(f"Notes ({len(self._notes)})" if has_notes else "Notes")
        self.notes_label.setText(
            "\n".join(f"•  {n}" for n in self._notes) if has_notes else ""
        )
        if not has_notes:
            self.notes_button.setChecked(False)
            self.notes_label.setVisible(False)

    def _toggle_notes(self, checked: bool) -> None:
        self.notes_label.setVisible(checked and bool(self._notes))

    # -- theming ----------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.setStyleSheet(
            f"""
            QFrame#chartPanel {{
                background-color: {theme.surface};
                border: 1px solid {theme.border};
                border-radius: 8px;
            }}
            QLabel#chartTitle {{
                color: {theme.text};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#sourceCaption {{
                color: {theme.text_muted};
                font-size: 10px;
                font-style: italic;
            }}
            QLabel#notesCaption {{
                color: {theme.text_muted};
                font-size: 10px;
                background-color: {theme.surface_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
                padding: 7px;
            }}
            QLabel#unavailableMessage {{
                color: {theme.text_muted};
                font-size: 12px;
                background-color: {theme.surface_alt};
                border: 1px dashed {theme.border};
                border-radius: 6px;
                padding: 22px;
            }}
            QToolButton {{
                color: {theme.text_muted};
                background-color: {theme.surface_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
            }}
            QToolButton:hover {{ color: {theme.text}; border-color: {theme.accent}; }}
            QToolButton:checked {{ color: {theme.accent}; border-color: {theme.accent}; }}
            QToolButton:disabled {{ color: {theme.border}; }}
            """
        )
        if self.toolbar is not None:
            self.toolbar.setStyleSheet(
                f"QToolBar {{ background: {theme.surface}; border: none; }}"
                f"QToolButton {{ background: {theme.surface}; border: none; padding: 2px; }}"
                f"QToolButton:hover {{ background: {theme.surface_alt}; }}"
            )
        style_figure(self.figure, theme)
        self.canvas.draw_idle()

    # -- export -----------------------------------------------------------

    def _default_export_name(self, extension: str) -> str:
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in self._title)
        safe = safe.strip().replace(" ", "_") or "chart"
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe}_{stamp}.{extension}"

    def export_png(self) -> None:
        self._export("png", "PNG image (*.png)")

    def export_pdf(self) -> None:
        self._export("pdf", "PDF document (*.pdf)")

    def _export(self, extension: str, filter_text: str) -> None:
        if not self.figure.get_axes():
            QtWidgets.QMessageBox.information(
                self, "Nothing to export",
                "This panel has no chart to export.",
            )
            return
        directory = ensure_dir(settings.EXPORT_DIR)
        suggested = str(directory / self._default_export_name(extension))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export chart as {extension.upper()}", suggested, filter_text
        )
        if not path:
            return
        try:
            self.save_to(Path(path), extension)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Chart export failed")
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"The chart could not be exported:\n{exc}"
            )
            return
        QtWidgets.QMessageBox.information(
            self, "Export complete", f"Chart saved to:\n{path}"
        )

    def save_to(self, path: Path, extension: str = "png") -> None:
        """Write the figure to disk at print resolution.

        Exports are always rendered on a white background, whichever theme is
        active on screen: a dark-theme PNG dropped into a report or printed
        wastes ink and reads poorly.
        """
        original = [
            self.figure.patch.get_facecolor(),
            [axes.get_facecolor() for axes in self.figure.get_axes()],
        ]
        light = settings.THEMES["light"]
        try:
            style_figure(self.figure, light)
            for axes in self.figure.get_axes():
                for text in axes.texts:
                    text.set_color(light.text)
            if extension == "pdf":
                with PdfPages(str(path)) as pdf:
                    pdf.savefig(self.figure, facecolor=light.surface, bbox_inches="tight")
            else:
                self.figure.savefig(
                    str(path),
                    dpi=settings.CHART_EXPORT_DPI,
                    facecolor=light.surface,
                    bbox_inches="tight",
                )
        finally:
            self.figure.patch.set_facecolor(original[0])
            for axes, colour in zip(self.figure.get_axes(), original[1]):
                axes.set_facecolor(colour)
            style_figure(self.figure, self.theme)
            self.canvas.draw_idle()


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------
#
# Each builder takes a ChartPanel and populates it. They are plain functions
# rather than methods so a page can compose them freely.


def draw_line(
    panel: ChartPanel,
    series: dict[str, pd.Series],
    *,
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    value_format: Callable[[float], str] | None = None,
    highlight: str | None = None,
    fill_between: tuple[pd.Series, pd.Series] | None = None,
) -> None:
    """Multi-series line chart with hover read-out."""
    usable = {
        name: pd.to_numeric(s, errors="coerce").dropna()
        for name, s in series.items()
        if s is not None and not s.empty
    }
    usable = {k: v for k, v in usable.items() if not v.empty}
    if not usable:
        panel.show_unavailable(
            "None of the requested series contains any observation "
            "after the current filters were applied.",
            source,
        )
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme
    hover = HoverTracker(panel.canvas, axes, theme)

    if fill_between is not None:
        lower, upper = fill_between
        common = lower.index.intersection(upper.index)
        if len(common):
            axes.fill_between(
                common, lower.loc[common], upper.loc[common],
                color=theme.accent, alpha=0.13, linewidth=0, zorder=1,
            )

    for index, (name, values) in enumerate(usable.items()):
        colour = theme.accent if (highlight and name == highlight) else _series_colour(theme, index)
        width = 2.0 if (highlight and name == highlight) else 1.4
        axes.plot(
            values.index, values.to_numpy(),
            color=colour, linewidth=width, label=str(name), zorder=3,
            solid_capstyle="round",
        )
        hover.register_series(str(name), values, colour, value_format)

    axes.set_ylabel(ylabel)
    _format_date_axis(axes, theme)
    if len(usable) > 1:
        _style_legend(axes, theme)
    panel.finish(source, notes, hover)


def draw_forecast(
    panel: ChartPanel,
    history: pd.Series,
    forecast: pd.Series,
    conf: tuple[pd.Series, pd.Series] | None = None,
    pred: tuple[pd.Series, pd.Series] | None = None,
    *,
    label: str = "Forecast",
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    history_window: int | None = None,
) -> None:
    """Fan chart: history, forecast path and both uncertainty bands.

    Historical and projected values are drawn differently (solid versus
    dashed, with a vertical divider at the forecast origin) so the two can
    never be confused (strict data rule 10).
    """
    history = pd.to_numeric(history, errors="coerce").dropna()
    forecast = pd.to_numeric(forecast, errors="coerce").dropna()
    if history.empty or forecast.empty:
        panel.show_unavailable(
            "The forecast could not be plotted because either the history or "
            "the projection is empty.",
            source,
        )
        return
    if history_window:
        history = history.tail(history_window)

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme
    hover = HoverTracker(panel.canvas, axes, theme)

    if pred is not None:
        lower, upper = pred
        axes.fill_between(
            forecast.index, lower.reindex(forecast.index), upper.reindex(forecast.index),
            color=theme.neutral, alpha=0.14, linewidth=0, zorder=1,
            label=f"{settings.FORECAST.prediction_level:.0%} prediction interval",
        )
    if conf is not None:
        lower, upper = conf
        axes.fill_between(
            forecast.index, lower.reindex(forecast.index), upper.reindex(forecast.index),
            color=theme.accent, alpha=0.22, linewidth=0, zorder=2,
            label=f"{settings.FORECAST.confidence_level:.0%} confidence interval",
        )

    axes.plot(
        history.index, history.to_numpy(), color=theme.text,
        linewidth=1.5, label="Historical (actual)", zorder=4,
    )
    hover.register_series("Historical", history, theme.text)

    # Join the last actual to the first projection so the path is continuous.
    bridge = pd.concat([history.tail(1), forecast])
    axes.plot(
        bridge.index, bridge.to_numpy(), color=theme.accent, linewidth=2.0,
        linestyle="--", label=f"{label} (projected)", zorder=5,
    )
    hover.register_series(f"{label} (projected)", forecast, theme.accent)

    origin = history.index[-1]
    axes.axvline(origin, color=theme.text_muted, linewidth=1.0, linestyle=":", zorder=3)
    axes.annotate(
        "forecast origin",
        xy=(origin, axes.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=7.5,
        color=theme.text_muted,
        rotation=90,
        va="top",
    )

    axes.set_ylabel(ylabel)
    _format_date_axis(axes, theme)
    _style_legend(axes, theme, loc="upper left")
    panel.finish(source, notes, hover)


def draw_scatter_with_fit(
    panel: ChartPanel,
    x: pd.Series,
    y: pd.Series,
    *,
    xlabel: str = "",
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    colour_by_date: bool = True,
) -> None:
    """Scatter plot with an ordinary-least-squares fit line and R²."""
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(joined) < 3:
        panel.show_unavailable(
            f"Only {len(joined)} paired observation(s) overlap; a scatter plot "
            "needs at least 3.",
            source,
        )
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme

    if colour_by_date and isinstance(joined.index, pd.DatetimeIndex):
        shades = joined.index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        scatter = axes.scatter(
            joined["x"], joined["y"], c=shades, cmap=theme.sequential_cmap,
            s=16, alpha=0.75, linewidths=0, zorder=3,
        )
        bar = figure.colorbar(scatter, ax=axes, pad=0.02)
        bar.ax.tick_params(colors=theme.text_muted, labelsize=7)
        bar.outline.set_edgecolor(theme.border)
        ticks = np.linspace(shades.min(), shades.max(), 5)
        bar.set_ticks(ticks)
        bar.set_ticklabels(
            [pd.Timestamp.fromordinal(int(t)).strftime("%b %Y") for t in ticks]
        )
        bar.set_label("Observation date", color=theme.text_muted, fontsize=8)
    else:
        axes.scatter(
            joined["x"], joined["y"], color=theme.neutral, s=16,
            alpha=0.7, linewidths=0, zorder=3,
        )

    slope, intercept = np.polyfit(joined["x"], joined["y"], 1)
    grid = np.linspace(joined["x"].min(), joined["x"].max(), 100)
    correlation = float(joined["x"].corr(joined["y"]))
    axes.plot(
        grid, slope * grid + intercept, color=theme.accent, linewidth=1.8,
        label=f"Fit: y = {slope:,.3f}x + {intercept:,.0f}  (R² = {correlation ** 2:.3f})",
        zorder=4,
    )

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    _style_legend(axes, theme)
    combined = list(notes or []) + [
        f"{len(joined)} paired observation(s). The fit is descriptive; it is "
        "not a forecasting model."
    ]
    panel.finish(source, combined, None)


def draw_heatmap(
    panel: ChartPanel,
    matrix: pd.DataFrame,
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
    diverging: bool = True,
    value_format: str = "{:.2f}",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    cbar_label: str = "",
) -> None:
    """Annotated matrix heatmap (correlations, seasonality grids)."""
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.notna().sum().sum() == 0:
        panel.show_unavailable("The matrix contains no numeric values.", source)
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme

    if diverging:
        limit = float(np.nanmax(np.abs(numeric.to_numpy()))) if vmax is None else vmax
        limit = 1.0 if not np.isfinite(limit) or limit == 0 else limit
        low, high = (-limit, limit) if vmin is None else (vmin, limit)
        cmap = theme.diverging_cmap
    else:
        low = float(np.nanmin(numeric.to_numpy())) if vmin is None else vmin
        high = float(np.nanmax(numeric.to_numpy())) if vmax is None else vmax
        cmap = theme.sequential_cmap

    image = axes.imshow(
        numeric.to_numpy(dtype=float), cmap=cmap, vmin=low, vmax=high,
        aspect="auto", interpolation="nearest",
    )
    axes.set_xticks(range(numeric.shape[1]))
    axes.set_xticklabels([str(c) for c in numeric.columns], rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(numeric.shape[0]))
    axes.set_yticklabels([str(i) for i in numeric.index], fontsize=8)
    axes.grid(False)

    if annotate and numeric.shape[0] * numeric.shape[1] <= 260:
        # Label colour follows cell luminance so text stays readable.
        norm = image.norm
        colormap = image.cmap
        for row in range(numeric.shape[0]):
            for column in range(numeric.shape[1]):
                value = numeric.iat[row, column]
                if not np.isfinite(value):
                    axes.text(
                        column, row, "—", ha="center", va="center",
                        fontsize=7, color=theme.text_muted,
                    )
                    continue
                red, green, blue, _ = colormap(norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                axes.text(
                    column, row, value_format.format(value),
                    ha="center", va="center", fontsize=7,
                    color="#111111" if luminance > 0.6 else "#ffffff",
                )

    bar = figure.colorbar(image, ax=axes, pad=0.02)
    bar.ax.tick_params(colors=theme.text_muted, labelsize=7)
    bar.outline.set_edgecolor(theme.border)
    if cbar_label:
        bar.set_label(cbar_label, color=theme.text_muted, fontsize=8)

    combined = list(notes or [])
    if numeric.isna().any().any():
        combined.append(
            "Blank cells (—) mark pairs with too few overlapping observations "
            "to compute a value."
        )
    panel.finish(source, combined, None)


def draw_bar(
    panel: ChartPanel,
    values: pd.Series,
    *,
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    reference: float | None = None,
    reference_label: str = "",
    colour_negative: bool = True,
    horizontal: bool = False,
    value_labels: bool = False,
    value_format: str = "{:,.0f}",
    projected: Sequence[Any] = (),
) -> None:
    """Bar chart, optionally split by sign and marking projected categories."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        panel.show_unavailable("There are no values to plot.", source)
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme

    projected_set = {str(p) for p in projected}
    colours: list[str] = []
    hatches: list[str] = []
    for label, value in numeric.items():
        if colour_negative and value < 0:
            colours.append(theme.negative)
        elif reference is not None and value < reference:
            colours.append(theme.warning)
        else:
            colours.append(theme.accent)
        hatches.append("////" if str(label) in projected_set else "")

    positions = np.arange(len(numeric))
    labels = [str(i) for i in numeric.index]

    def _coord(x: float, y: float) -> str:
        """Readout for a categorical axis: name the bar, not its tick index."""
        index = int(round(x if not horizontal else y))
        name = labels[index] if 0 <= index < len(labels) else "—"
        value = y if not horizontal else x
        return f"{name}: {value:,.4g}"

    axes.format_coord = _coord

    if horizontal:
        bars = axes.barh(positions, numeric.to_numpy(), color=colours, height=0.72, zorder=3)
        axes.set_yticks(positions)
        axes.set_yticklabels([str(i) for i in numeric.index], fontsize=8)
        axes.set_xlabel(ylabel)
        axes.invert_yaxis()
    else:
        bars = axes.bar(positions, numeric.to_numpy(), color=colours, width=0.72, zorder=3)
        axes.set_xticks(positions)
        axes.set_xticklabels(
            [str(i) for i in numeric.index],
            rotation=45 if len(numeric) > 8 else 0,
            ha="right" if len(numeric) > 8 else "center",
            fontsize=8,
        )
        axes.set_ylabel(ylabel)

    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor(theme.text_muted)
            bar.set_linewidth(0.7)

    if reference is not None:
        line = axes.axhline if not horizontal else axes.axvline
        line(
            reference, color=theme.text_muted, linestyle="--", linewidth=1.1,
            label=reference_label or f"Reference {reference:,.2f}", zorder=4,
        )
        # Bars fill the plotting area, so an in-axes legend always sits on top
        # of a bar or its value label. Anchor it just outside the top edge.
        _style_legend(
            axes, theme, loc="lower right", bbox_to_anchor=(1.0, 1.01),
            ncol=1, borderaxespad=0.0,
        )

    if value_labels:
        for bar, value in zip(bars, numeric.to_numpy()):
            if horizontal:
                axes.text(
                    bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    "  " + value_format.format(value), va="center", ha="left",
                    fontsize=7, color=theme.text_muted,
                )
            else:
                axes.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    value_format.format(value), va="bottom", ha="center",
                    fontsize=7, color=theme.text_muted,
                )

    combined = list(notes or [])
    if projected_set:
        combined.append(
            "Hatched bars are values the workbook marks as expected/projected, "
            "not realised history."
        )
    panel.finish(source, combined, None)


def draw_grouped_bar(
    panel: ChartPanel,
    frame: pd.DataFrame,
    *,
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    stacked: bool = False,
) -> None:
    """Grouped or stacked bars: rows are categories, columns are groups."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.notna().sum().sum() == 0:
        panel.show_unavailable("There are no values to plot.", source)
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme
    positions = np.arange(len(numeric.index))
    n_groups = max(1, numeric.shape[1])

    if stacked:
        cumulative_pos = np.zeros(len(numeric.index))
        cumulative_neg = np.zeros(len(numeric.index))
        for index, column in enumerate(numeric.columns):
            heights = numeric[column].fillna(0).to_numpy(dtype=float)
            base = np.where(heights >= 0, cumulative_pos, cumulative_neg)
            axes.bar(
                positions, heights, bottom=base, width=0.7,
                color=_series_colour(theme, index), label=str(column), zorder=3,
            )
            cumulative_pos = cumulative_pos + np.where(heights >= 0, heights, 0)
            cumulative_neg = cumulative_neg + np.where(heights < 0, heights, 0)
    else:
        width = 0.8 / n_groups
        for index, column in enumerate(numeric.columns):
            axes.bar(
                positions + index * width - 0.4 + width / 2,
                numeric[column].to_numpy(dtype=float),
                width=width, color=_series_colour(theme, index),
                label=str(column), zorder=3,
            )

    axes.set_xticks(positions)
    axes.set_xticklabels(
        [str(i) for i in numeric.index],
        rotation=45 if len(numeric.index) > 8 else 0,
        ha="right" if len(numeric.index) > 8 else "center",
        fontsize=8,
    )
    axes.set_ylabel(ylabel)
    axes.axhline(0, color=theme.border, linewidth=0.8)
    _style_legend(axes, theme, ncol=min(4, n_groups))
    panel.finish(source, notes, None)


def draw_calendar_heatmap(
    panel: ChartPanel,
    year_month: pd.DataFrame,
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
    value_format: str = "{:,.0f}",
    cbar_label: str = "",
) -> None:
    """Year x month grid: the calendar view of a monthly series."""
    draw_heatmap(
        panel, year_month, source=source, notes=notes, diverging=False,
        value_format=value_format, cbar_label=cbar_label,
    )


def draw_acf_pacf(
    panel: ChartPanel,
    frame: pd.DataFrame,
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
) -> None:
    """Side-by-side ACF and PACF stem plots with significance bands."""
    if frame is None or frame.empty:
        panel.show_unavailable("Autocorrelation could not be computed.", source)
        return

    figure = panel.begin()
    theme = panel.theme
    left = figure.add_subplot(121)
    right = figure.add_subplot(122)

    for axes, column, title in (
        (left, "ACF", "Autocorrelation (ACF)"),
        (right, "PACF", "Partial autocorrelation (PACF)"),
    ):
        if column not in frame.columns:
            continue
        values = frame[column]
        axes.vlines(
            values.index, 0, values.to_numpy(dtype=float),
            color=theme.neutral, linewidth=1.5,
        )
        axes.plot(
            values.index, values.to_numpy(dtype=float), "o",
            markersize=3, color=theme.accent,
        )
        axes.axhline(0, color=theme.border, linewidth=0.8)
        if "Upper 95%" in frame.columns:
            band = float(frame["Upper 95%"].iloc[0])
            axes.axhspan(-band, band, color=theme.text_muted, alpha=0.14, zorder=0)
        axes.set_title(title)
        axes.set_xlabel("Lag")
        style_axes(axes, theme)

    panel.finish(source, notes, None)


def draw_decomposition(
    panel: ChartPanel,
    frame: pd.DataFrame,
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
    ylabel: str = "",
) -> None:
    """Stacked observed / trend / seasonal / residual panels."""
    if frame is None or frame.empty:
        panel.show_unavailable("The decomposition could not be computed.", source)
        return

    components = [c for c in ("Observed", "Trend", "Seasonal", "Residual") if c in frame.columns]
    if not components:
        panel.show_unavailable("The decomposition returned no components.", source)
        return

    figure = panel.begin()
    theme = panel.theme
    axes_list = figure.subplots(len(components), 1, sharex=True)
    if len(components) == 1:
        axes_list = [axes_list]

    for index, (axes, component) in enumerate(zip(axes_list, components)):
        values = pd.to_numeric(frame[component], errors="coerce")
        colour = theme.text if component == "Observed" else _series_colour(theme, index)
        if component == "Residual":
            axes.vlines(values.index, 0, values.to_numpy(dtype=float),
                        color=colour, linewidth=0.7, alpha=0.8)
            axes.axhline(0, color=theme.border, linewidth=0.8)
        else:
            axes.plot(values.index, values.to_numpy(dtype=float), color=colour, linewidth=1.3)
        axes.set_ylabel(component, fontsize=8)
        style_axes(axes, theme)

    _format_date_axis(axes_list[-1], theme)
    panel.finish(source, notes, None)


def draw_influence_network(
    panel: ChartPanel,
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float, str]],
    *,
    source: str = "",
    notes: Sequence[str] | None = None,
    node_scores: dict[str, float] | None = None,
) -> None:
    """Influence diagram of which market or variety drives which.

    Nodes are laid out on a circle. Each edge is ``(from, to, strength,
    kind)`` where *kind* is ``"one-way"`` or ``"feedback"``; one-way edges get
    a single arrowhead, feedback edges a double one. Node size and colour
    follow the leadership score, so the dominant market is visually obvious.
    """
    if not nodes:
        panel.show_unavailable(
            "No market or variety had enough overlapping history to place on "
            "the influence diagram.",
            source,
        )
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme
    axes.set_aspect("equal")
    axes.axis("off")

    count = len(nodes)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, count, endpoint=False)
    positions = {
        str(name): (float(np.cos(a)), float(np.sin(a)))
        for name, a in zip(nodes, angles)
    }

    scores = node_scores or {}
    finite = [v for v in scores.values() if np.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    spread = (hi - lo) or 1.0

    strengths = [abs(s) for *_, s, _ in [(a, b, s, k) for a, b, s, k in edges]] if edges else []
    max_strength = max(strengths) if strengths else 1.0

    for start, end, strength, kind in edges:
        if start not in positions or end not in positions:
            continue
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        weight = 0.9 + 2.6 * (abs(strength) / max_strength if max_strength else 0)
        feedback = kind == "feedback"
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=0.16",
            arrowstyle="<|-|>" if feedback else "-|>",
            mutation_scale=13,
            linewidth=weight,
            linestyle=(0, (4, 3)) if feedback else "solid",
            color=theme.neutral if feedback else theme.accent,
            alpha=0.85,
            shrinkA=26, shrinkB=26,
            zorder=2,
        )
        axes.add_patch(arrow)

    for name, (x, y) in positions.items():
        score = scores.get(name, float("nan"))
        normalised = (score - lo) / spread if np.isfinite(score) else 0.0
        size = 900 + 1500 * normalised
        axes.scatter(
            [x], [y], s=size,
            color=theme.accent if normalised > 0.66 else theme.neutral if normalised > 0.33 else theme.surface_alt,
            edgecolors=theme.border, linewidths=1.2, zorder=4,
        )
        label = name if len(name) <= 18 else name.replace(" (", "\n(")
        axes.text(
            x, y, label, ha="center", va="center", fontsize=7.5,
            color=theme.text, fontweight="bold", zorder=5,
        )
        if np.isfinite(score):
            axes.text(
                x, y - 0.19, f"{score:.2f}", ha="center", va="center",
                fontsize=7, color=theme.text_muted, zorder=5,
            )

    axes.set_xlim(-1.5, 1.5)
    axes.set_ylim(-1.5, 1.5)
    legend_handles = [
        Line2D([0], [0], color=theme.accent, linewidth=2, label="One-way influence"),
        Line2D([0], [0], color=theme.neutral, linewidth=2, linestyle="--",
               label="Two-way feedback"),
    ]
    legend = axes.legend(handles=legend_handles, fontsize=8, loc="lower center",
                         ncol=2, frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor(theme.surface_alt)
    legend.get_frame().set_edgecolor(theme.border)
    for text in legend.get_texts():
        text.set_color(theme.text)

    combined = list(notes or []) + [
        "Node size and colour follow the leadership score; arrow thickness "
        "follows the strength of the statistical evidence.",
    ]
    panel.finish(source, combined, None)


def draw_dual_axis(
    panel: ChartPanel,
    primary: pd.Series,
    secondary: pd.Series,
    *,
    primary_label: str = "",
    secondary_label: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    secondary_as_bars: bool = False,
) -> None:
    """Two series on independent y-axes -- price against arrivals, typically."""
    left_values = pd.to_numeric(primary, errors="coerce").dropna()
    right_values = pd.to_numeric(secondary, errors="coerce").dropna()
    if left_values.empty and right_values.empty:
        panel.show_unavailable("Neither series contains any observation.", source)
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme
    hover = HoverTracker(panel.canvas, axes, theme)

    if not left_values.empty:
        axes.plot(
            left_values.index, left_values.to_numpy(), color=theme.accent,
            linewidth=1.6, label=primary_label or "Primary", zorder=4,
        )
        hover.register_series(primary_label or "Primary", left_values, theme.accent)
    axes.set_ylabel(primary_label, color=theme.accent)
    axes.tick_params(axis="y", colors=theme.accent)

    twin = axes.twinx()
    if not right_values.empty:
        if secondary_as_bars:
            width = _bar_width(right_values.index)
            twin.bar(
                right_values.index, right_values.to_numpy(), width=width,
                color=theme.neutral, alpha=0.38, linewidth=0,
                label=secondary_label or "Secondary", zorder=2,
            )
        else:
            twin.plot(
                right_values.index, right_values.to_numpy(), color=theme.neutral,
                linewidth=1.3, alpha=0.9, label=secondary_label or "Secondary", zorder=3,
            )
    twin.set_ylabel(secondary_label, color=theme.neutral)
    twin.tick_params(axis="y", colors=theme.neutral, labelsize=8)
    twin.grid(False)
    twin.set_facecolor("none")
    for spine_name, spine in twin.spines.items():
        spine.set_visible(spine_name == "right")
        spine.set_color(theme.border)

    handles = axes.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axes.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    if handles:
        legend = axes.legend(handles, labels, fontsize=8, loc="upper left", framealpha=0.92)
        legend.get_frame().set_facecolor(theme.surface_alt)
        legend.get_frame().set_edgecolor(theme.border)
        for text in legend.get_texts():
            text.set_color(theme.text)

    _format_date_axis(axes, theme)
    panel.finish(source, notes, hover)


def _bar_width(index: pd.Index) -> float:
    """A bar width in date units that leaves a small gap between bars."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1.0
    spacing = pd.Series(index).diff().dt.days.median()
    if not np.isfinite(spacing) or spacing <= 0:
        return 1.0
    return float(spacing) * 0.8


def draw_stem(
    panel: ChartPanel,
    values: pd.Series,
    *,
    xlabel: str = "",
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
    band: float | None = None,
    highlight_index: Any = None,
) -> None:
    """Stem plot for lag profiles: cross-correlation, lagged impact, elasticity."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        panel.show_unavailable("There are no values to plot.", source)
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme

    colours = [
        theme.accent if highlight_index is not None and idx == highlight_index
        else theme.neutral
        for idx in numeric.index
    ]
    axes.vlines(numeric.index, 0, numeric.to_numpy(dtype=float), colors=colours, linewidth=2.0)
    axes.scatter(numeric.index, numeric.to_numpy(dtype=float), color=colours, s=22, zorder=4)
    axes.axhline(0, color=theme.border, linewidth=0.9)
    axes.axvline(0, color=theme.text_muted, linewidth=0.8, linestyle=":")

    if band is not None and np.isfinite(band):
        axes.axhspan(
            -abs(band), abs(band), color=theme.text_muted, alpha=0.15, zorder=0,
            label="95% significance band",
        )
        _style_legend(axes, theme)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    panel.finish(source, notes, None)


def draw_box_by_group(
    panel: ChartPanel,
    groups: dict[str, pd.Series],
    *,
    ylabel: str = "",
    source: str = "",
    notes: Sequence[str] | None = None,
) -> None:
    """Distribution of a value across categories (months, arrival buckets)."""
    usable = {
        name: pd.to_numeric(s, errors="coerce").dropna()
        for name, s in groups.items()
    }
    usable = {k: v for k, v in usable.items() if len(v) >= 2}
    if not usable:
        panel.show_unavailable(
            "No group has at least two observations, so distributions cannot "
            "be drawn.",
            source,
        )
        return

    figure = panel.begin()
    axes = figure.add_subplot(111)
    theme = panel.theme

    parts = axes.boxplot(
        [v.to_numpy(dtype=float) for v in usable.values()],
        tick_labels=list(usable.keys()),
        patch_artist=True,
        medianprops={"color": theme.accent, "linewidth": 1.6},
        whiskerprops={"color": theme.text_muted, "linewidth": 0.9},
        capprops={"color": theme.text_muted, "linewidth": 0.9},
        flierprops={
            "marker": "o", "markersize": 2.5,
            "markerfacecolor": theme.negative, "markeredgecolor": "none", "alpha": 0.6,
        },
    )
    for index, box in enumerate(parts["boxes"]):
        box.set_facecolor(_series_colour(theme, index))
        box.set_alpha(0.45)
        box.set_edgecolor(theme.border)

    axes.set_ylabel(ylabel)
    axes.tick_params(axis="x", rotation=45 if len(usable) > 8 else 0)
    for label in axes.get_xticklabels():
        label.set_fontsize(8)
        if len(usable) > 8:
            label.set_ha("right")
    panel.finish(source, notes, None)


def export_panels_to_pdf(panels: Sequence[ChartPanel], path: Path) -> int:
    """Write several chart panels into one multi-page PDF report.

    Returns the number of pages written.
    """
    written = 0
    with PdfPages(str(path)) as pdf:
        light = settings.THEMES["light"]
        for panel in panels:
            if not panel.figure.get_axes():
                continue
            original_face = panel.figure.patch.get_facecolor()
            original_axes = [a.get_facecolor() for a in panel.figure.get_axes()]
            try:
                style_figure(panel.figure, light)
                pdf.savefig(panel.figure, facecolor=light.surface, bbox_inches="tight")
                written += 1
            finally:
                panel.figure.patch.set_facecolor(original_face)
                for axes, colour in zip(panel.figure.get_axes(), original_axes):
                    axes.set_facecolor(colour)
                style_figure(panel.figure, panel.theme)
                panel.canvas.draw_idle()
    return written
