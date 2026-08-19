"""The PySide6 desktop shell: window, navigation, global filters and all pages.

Architecture
------------
``MainWindow`` owns the :class:`~chilli_desktop.preprocessing.DataService`, the
current :class:`~chilli_desktop.preprocessing.FilterState` and the active
theme, and hands all three to pages through a :class:`PageContext`.

Pages derive from :class:`BasePage` and are **lazily built**: a page's widgets
are created the first time it is shown, and rebuilt only when the filters
change. That keeps start-up to the cost of the workbook read plus one page.

Long computations (leadership ranking on daily data, the forecast model
sweep, the insight sweep) run on a :class:`QThreadPool` worker so the window
never blocks. Workers touch no Qt widgets: they return plain objects and the
page renders them on the main thread.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from . import analytics, charts, forecasting, insights, settings
from .data_loader import WorkbookData, build_data_dictionary, data_dictionary_markdown, load_workbook
from .preprocessing import DataService, FilterState, default_filters
from .settings import FORTNIGHT_FREQ, Theme
from .utils import (
    LOG,
    Result,
    describe_strength,
    ensure_dir,
    fmt_date,
    fmt_int,
    fmt_number,
    fmt_pct,
    fmt_pvalue,
)


# ==========================================================================
# Context passed to every page
# ==========================================================================


@dataclass
class PageContext:
    """Everything a page needs to render itself."""

    service: DataService
    filters: FilterState
    theme: Theme
    #: Report progress to the status bar: ``(message, percent)``.
    progress: Callable[[str, int], None]
    #: Submit a callable to the background thread pool.
    submit: Callable[..., None]


# ==========================================================================
# Background work
# ==========================================================================


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str, int)


class Worker(QtCore.QRunnable):
    """Runs one callable off the UI thread and emits the result.

    The callable receives a ``progress`` keyword argument if it accepts one,
    letting long analyses drive the status bar.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:  # pragma: no cover - exercised interactively
        try:
            if self.kwargs.pop("_wants_progress", False):
                self.kwargs["progress"] = lambda m, p: self.signals.progress.emit(m, int(p))
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Background task failed")
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(result)


# ==========================================================================
# Reusable widgets
# ==========================================================================


class PandasModel(QtCore.QAbstractTableModel):
    """Read-only table model over a DataFrame with sensible number formatting."""

    #: Column-name fragments that pin a column to two decimal places. Without
    #: this, the generic rule ("values below 1 get four decimals", which suits
    #: correlations) prints a percentage column as "0.0000" beside "59.62".
    TWO_DECIMAL_HINTS = ("%", "percent", "change", "ratio", "mape", "rmse", "mae")

    def __init__(self, frame: pd.DataFrame, theme: Theme, parent=None) -> None:
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._theme = theme
        self._show_index = not isinstance(self._frame.index, pd.RangeIndex)
        self._fixed_decimals = {
            index
            for index, name in enumerate(self._frame.columns)
            if any(hint in str(name).lower() for hint in self.TWO_DECIMAL_HINTS)
        }

    def rowCount(self, _parent=QtCore.QModelIndex()) -> int:
        return int(len(self._frame))

    def columnCount(self, _parent=QtCore.QModelIndex()) -> int:
        return int(self._frame.shape[1])

    @staticmethod
    def _format(value: Any, decimals: int | None = None) -> str:
        if value is None:
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        if isinstance(value, (pd.Timestamp,)):
            return "—" if pd.isna(value) else f"{value:%d %b %Y}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if not np.isfinite(number):
                return "—"
            if decimals is not None:
                return f"{number:,.{decimals}f}"
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

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._format(
                value, 2 if index.column() in self._fixed_decimals else None
            )
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, (bool, np.bool_)
            ):
                return int(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
            return int(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            # Colour signed numbers and pass/fail flags.
            column = str(self._frame.columns[index.column()]).lower()
            if isinstance(value, (bool, np.bool_)):
                return QtGui.QColor(
                    self._theme.positive if bool(value) else self._theme.text_muted
                )
            if isinstance(value, (float, np.floating, int, np.integer)) and np.isfinite(
                float(value)
            ):
                if any(k in column for k in ("change", "elasticity", "coefficient", "beta", "correlation", "r²", "r2")):
                    if float(value) > 0:
                        return QtGui.QColor(self._theme.positive)
                    if float(value) < 0:
                        return QtGui.QColor(self._theme.negative)
                if "p-value" in column and float(value) < settings.ANALYTICS.alpha:
                    return QtGui.QColor(self._theme.accent)
            if isinstance(value, str) and value.strip().upper() == "SELECTED":
                return QtGui.QColor(self._theme.accent)
        return None

    def headerData(
        self, section: int, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole
    ):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return str(self._frame.columns[section])
        if self._show_index:
            return self._format(self._frame.index[section])
        return str(section + 1)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame


class DataTable(QtWidgets.QWidget):
    """A titled table with provenance, notes and CSV export."""

    def __init__(
        self,
        title: str = "",
        theme: Theme | None = None,
        *,
        max_height: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.theme = theme or settings.THEMES[settings.DEFAULT_THEME]
        self._title = title
        self._frame = pd.DataFrame()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        header = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("tableTitle")
        self.title_label.setWordWrap(True)
        header.addWidget(self.title_label, 1)
        self.export_button = QtWidgets.QToolButton()
        self.export_button.setText("CSV")
        self.export_button.setToolTip("Export this table as CSV")
        self.export_button.clicked.connect(self.export_csv)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.view = QtWidgets.QTableView()
        self.view.setAlternatingRowColors(True)
        self.view.setSortingEnabled(False)
        self.view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.horizontalHeader().setHighlightSections(False)
        self.view.verticalHeader().setDefaultSectionSize(24)
        self.view.setWordWrap(False)
        if max_height:
            self.view.setMaximumHeight(max_height)
        layout.addWidget(self.view)

        self.message_label = QtWidgets.QLabel()
        self.message_label.setObjectName("unavailableMessage")
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

        self.caption_label = QtWidgets.QLabel()
        self.caption_label.setObjectName("sourceCaption")
        self.caption_label.setWordWrap(True)
        layout.addWidget(self.caption_label)

        self.apply_theme(self.theme)

    def set_frame(
        self,
        frame: pd.DataFrame,
        source: str = "",
        notes: Sequence[str] | None = None,
    ) -> None:
        if frame is None or frame.empty:
            self.show_unavailable("The table contains no rows.", source)
            return
        self._frame = frame
        self.view.setModel(PandasModel(frame, self.theme, self.view))
        self.view.resizeColumnsToContents()
        for column in range(frame.shape[1]):
            if self.view.columnWidth(column) > 320:
                self.view.setColumnWidth(column, 320)
        self.view.setVisible(True)
        self.message_label.setVisible(False)
        self.export_button.setEnabled(True)
        self._set_caption(source, notes)

    def show_unavailable(self, result_or_message: Any, source: str = "") -> None:
        if isinstance(result_or_message, Result):
            message = result_or_message.message()
            source = source or result_or_message.source
        else:
            message = str(result_or_message)
            if settings.DATA_UNAVAILABLE_MESSAGE not in message:
                message = f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{message}"
        self._frame = pd.DataFrame()
        self.view.setModel(None)
        self.view.setVisible(False)
        self.message_label.setText(message)
        self.message_label.setVisible(True)
        self.export_button.setEnabled(False)
        self._set_caption(source, None)

    def _set_caption(self, source: str, notes: Sequence[str] | None) -> None:
        parts: list[str] = [f"Source: {source}" if source else "Source: —"]
        for note in notes or []:
            if note:
                parts.append(f"•  {note}")
        self.caption_label.setText("\n".join(parts))

    def export_csv(self) -> None:
        if self._frame.empty:
            return
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in self._title)
        safe = safe.strip().replace(" ", "_") or "table"
        directory = ensure_dir(settings.EXPORT_DIR)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        suggested = str(directory / f"{safe}_{stamp}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table as CSV", suggested, "CSV file (*.csv)"
        )
        if not path:
            return
        try:
            self._frame.to_csv(path, index=True, encoding="utf-8-sig")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Export complete", f"Table saved to:\n{path}")

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.setStyleSheet(
            f"""
            QLabel#tableTitle {{ color: {theme.text}; font-size: 12px; font-weight: 600; }}
            QLabel#sourceCaption {{ color: {theme.text_muted}; font-size: 10px; font-style: italic; }}
            QLabel#unavailableMessage {{
                color: {theme.text_muted}; font-size: 12px;
                background-color: {theme.surface_alt};
                border: 1px dashed {theme.border};
                border-radius: 6px; padding: 18px;
            }}
            QToolButton {{
                color: {theme.text_muted}; background-color: {theme.surface_alt};
                border: 1px solid {theme.border}; border-radius: 4px;
                padding: 2px 8px; font-size: 10px;
            }}
            QToolButton:hover {{ color: {theme.text}; border-color: {theme.accent}; }}
            QTableView {{
                background-color: {theme.surface};
                alternate-background-color: {theme.surface_alt};
                color: {theme.text};
                gridline-color: {theme.grid};
                border: 1px solid {theme.border};
                border-radius: 6px;
                font-size: 11px;
                selection-background-color: {theme.accent_soft};
                selection-color: {theme.text};
            }}
            QHeaderView::section {{
                background-color: {theme.surface_alt};
                color: {theme.text_muted};
                border: none;
                border-right: 1px solid {theme.border};
                border-bottom: 1px solid {theme.border};
                padding: 5px 7px;
                font-size: 10px;
                font-weight: 600;
            }}
            QTableCornerButton::section {{
                background-color: {theme.surface_alt};
                border: none;
            }}
            """
        )
        if self.view.model() is not None and not self._frame.empty:
            self.view.setModel(PandasModel(self._frame, theme, self.view))
            self.view.resizeColumnsToContents()


class SummaryCard(QtWidgets.QFrame):
    """A metric card: label, big value, delta and a caption."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        delta: str = "",
        caption: str = "",
        theme: Theme | None = None,
        *,
        tone: str = "neutral",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.theme = theme or settings.THEMES[settings.DEFAULT_THEME]
        self._tone = tone
        self.setObjectName("summaryCard")
        self.setMinimumWidth(190)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)

        self.label_widget = QtWidgets.QLabel(label)
        self.label_widget.setObjectName("cardLabel")
        self.label_widget.setWordWrap(True)
        layout.addWidget(self.label_widget)

        self.value_widget = QtWidgets.QLabel(value)
        self.value_widget.setObjectName("cardValue")
        layout.addWidget(self.value_widget)

        self.delta_widget = QtWidgets.QLabel(delta)
        self.delta_widget.setObjectName("cardDelta")
        self.delta_widget.setVisible(bool(delta))
        layout.addWidget(self.delta_widget)

        self.caption_widget = QtWidgets.QLabel(caption)
        self.caption_widget.setObjectName("cardCaption")
        self.caption_widget.setWordWrap(True)
        self.caption_widget.setVisible(bool(caption))
        layout.addWidget(self.caption_widget)
        layout.addStretch(1)

        self.apply_theme(self.theme)

    def update_values(
        self, value: str, delta: str = "", caption: str = "", tone: str = "neutral"
    ) -> None:
        self.value_widget.setText(value)
        self.delta_widget.setText(delta)
        self.delta_widget.setVisible(bool(delta))
        self.caption_widget.setText(caption)
        self.caption_widget.setVisible(bool(caption))
        self._tone = tone
        self.apply_theme(self.theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        tone_colour = {
            "positive": theme.positive,
            "negative": theme.negative,
            "warning": theme.warning,
            "accent": theme.accent,
            "neutral": theme.text_muted,
        }.get(self._tone, theme.text_muted)
        self.setStyleSheet(
            f"""
            QFrame#summaryCard {{
                background-color: {theme.surface};
                border: 1px solid {theme.border};
                border-left: 3px solid {tone_colour};
                border-radius: 8px;
            }}
            QLabel#cardLabel {{
                color: {theme.text_muted}; font-size: 10px;
                font-weight: 600; text-transform: uppercase;
            }}
            QLabel#cardValue {{ color: {theme.text}; font-size: 21px; font-weight: 700; }}
            QLabel#cardDelta {{ color: {tone_colour}; font-size: 11px; font-weight: 600; }}
            QLabel#cardCaption {{ color: {theme.text_muted}; font-size: 10px; }}
            """
        )


class InsightCard(QtWidgets.QFrame):
    """Renders one :class:`~chilli_desktop.insights.Insight`."""

    TONES = {
        "strong": "positive",
        "moderate": "accent",
        "weak": "warning",
        "informational": "neutral",
        "data gap": "negative",
    }

    def __init__(self, insight: insights.Insight, theme: Theme, parent=None) -> None:
        super().__init__(parent)
        self.insight = insight
        self.theme = theme
        self.setObjectName("insightCard")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(5)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        self.badge = QtWidgets.QLabel(insight.strength.upper())
        self.badge.setObjectName("insightBadge")
        top.addWidget(self.badge)
        self.category = QtWidgets.QLabel(insight.category)
        self.category.setObjectName("insightCategory")
        top.addWidget(self.category)
        if insight.direction not in ("n/a", ""):
            self.direction = QtWidgets.QLabel(insight.direction.upper())
            self.direction.setObjectName("insightDirection")
            top.addWidget(self.direction)
        top.addStretch(1)
        layout.addLayout(top)

        self.headline = QtWidgets.QLabel(insight.headline)
        self.headline.setObjectName("insightHeadline")
        self.headline.setWordWrap(True)
        self.headline.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.headline)

        if insight.detail:
            self.detail = QtWidgets.QLabel(insight.detail)
            self.detail.setObjectName("insightDetail")
            self.detail.setWordWrap(True)
            self.detail.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            layout.addWidget(self.detail)

        self.source = QtWidgets.QLabel(f"Source: {insight.source or '—'}")
        self.source.setObjectName("sourceCaption")
        self.source.setWordWrap(True)
        layout.addWidget(self.source)

        if insight.evidence:
            self.evidence_button = QtWidgets.QToolButton()
            self.evidence_button.setText(f"Evidence ({len(insight.evidence)})")
            self.evidence_button.setCheckable(True)
            layout.addWidget(self.evidence_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            self.evidence_label = QtWidgets.QLabel(
                "\n".join(f"•  {e}" for e in insight.evidence)
            )
            self.evidence_label.setObjectName("notesCaption")
            self.evidence_label.setWordWrap(True)
            self.evidence_label.setVisible(False)
            layout.addWidget(self.evidence_label)
            self.evidence_button.toggled.connect(self.evidence_label.setVisible)

        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        tone = self.TONES.get(self.insight.strength, "neutral")
        colour = {
            "positive": theme.positive,
            "accent": theme.accent,
            "warning": theme.warning,
            "negative": theme.negative,
            "neutral": theme.text_muted,
        }[tone]
        direction_colour = {
            "bullish": theme.positive,
            "bearish": theme.negative,
        }.get(self.insight.direction, theme.text_muted)
        self.setStyleSheet(
            f"""
            QFrame#insightCard {{
                background-color: {theme.surface};
                border: 1px solid {theme.border};
                border-left: 3px solid {colour};
                border-radius: 8px;
            }}
            QLabel#insightBadge {{
                color: {colour}; font-size: 9px; font-weight: 700;
                border: 1px solid {colour}; border-radius: 3px; padding: 1px 5px;
            }}
            QLabel#insightCategory {{ color: {theme.text_muted}; font-size: 10px; font-weight: 600; }}
            QLabel#insightDirection {{
                color: {direction_colour}; font-size: 9px; font-weight: 700;
                border: 1px solid {direction_colour}; border-radius: 3px; padding: 1px 5px;
            }}
            QLabel#insightHeadline {{ color: {theme.text}; font-size: 13px; font-weight: 600; }}
            QLabel#insightDetail {{ color: {theme.text_muted}; font-size: 11px; }}
            QLabel#sourceCaption {{ color: {theme.text_muted}; font-size: 9px; font-style: italic; }}
            QLabel#notesCaption {{
                color: {theme.text_muted}; font-size: 10px;
                background-color: {theme.surface_alt};
                border: 1px solid {theme.border}; border-radius: 5px; padding: 7px;
            }}
            QToolButton {{
                color: {theme.text_muted}; background-color: {theme.surface_alt};
                border: 1px solid {theme.border}; border-radius: 4px;
                padding: 2px 8px; font-size: 10px;
            }}
            QToolButton:checked {{ color: {theme.accent}; border-color: {theme.accent}; }}
            """
        )


class InfoBox(QtWidgets.QLabel):
    """A bordered explanatory block: assumptions, methodology, caveats."""

    def __init__(
        self, text: str = "", theme: Theme | None = None, *, tone: str = "info", parent=None
    ) -> None:
        super().__init__(parent)
        self.theme = theme or settings.THEMES[settings.DEFAULT_THEME]
        self._tone = tone
        self.setObjectName("infoBox")
        self.setWordWrap(True)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setText(text)
        self.apply_theme(self.theme)

    def set_items(self, title: str, items: Sequence[str]) -> None:
        usable = [i for i in items if i]
        if not usable:
            self.setVisible(False)
            return
        self.setVisible(True)
        body = "\n".join(f"•  {i}" for i in usable)
        self.setText(f"{title}\n{body}" if title else body)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        colour = {
            "info": theme.neutral,
            "warning": theme.warning,
            "danger": theme.negative,
            "muted": theme.border,
        }.get(self._tone, theme.neutral)
        self.setStyleSheet(
            f"""
            QLabel#infoBox {{
                color: {theme.text_muted};
                background-color: {theme.surface_alt};
                border: 1px solid {theme.border};
                border-left: 3px solid {colour};
                border-radius: 6px;
                padding: 10px 12px;
                font-size: 11px;
            }}
            """
        )


class SectionHeader(QtWidgets.QWidget):
    """A titled section divider with an optional subtitle."""

    def __init__(self, title: str, subtitle: str = "", theme: Theme | None = None, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme or settings.THEMES[settings.DEFAULT_THEME]
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 2)
        layout.setSpacing(1)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)
        self.subtitle_label = QtWidgets.QLabel(subtitle)
        self.subtitle_label.setObjectName("sectionSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)
        self.apply_theme(self.theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.setStyleSheet(
            f"""
            QLabel#sectionTitle {{
                color: {theme.text}; font-size: 14px; font-weight: 700;
            }}
            QLabel#sectionSubtitle {{ color: {theme.text_muted}; font-size: 11px; }}
            """
        )


class SentimentGauge(QtWidgets.QWidget):
    """A horizontal bearish-to-bullish gauge drawn with QPainter."""

    def __init__(self, theme: Theme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self._score = float("nan")
        self._label = settings.DATA_UNAVAILABLE_MESSAGE
        self.setMinimumHeight(76)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def set_score(self, score: float, label: str) -> None:
        self._score = score
        self._label = label
        self.update()

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        width, height = self.width(), self.height()
        margin = 14
        track_height = 12
        track_y = height - 30
        track_width = width - 2 * margin

        gradient = QtGui.QLinearGradient(margin, 0, margin + track_width, 0)
        gradient.setColorAt(0.0, QtGui.QColor(theme.negative))
        gradient.setColorAt(0.5, QtGui.QColor(theme.text_muted))
        gradient.setColorAt(1.0, QtGui.QColor(theme.positive))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(
            QtCore.QRectF(margin, track_y, track_width, track_height), 6, 6
        )

        for fraction, text in ((0.0, "Bearish"), (0.5, "Neutral"), (1.0, "Bullish")):
            painter.setPen(QtGui.QColor(theme.text_muted))
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            x = margin + fraction * track_width
            alignment = (
                QtCore.Qt.AlignmentFlag.AlignLeft
                if fraction == 0
                else QtCore.Qt.AlignmentFlag.AlignRight
                if fraction == 1
                else QtCore.Qt.AlignmentFlag.AlignHCenter
            )
            painter.drawText(
                QtCore.QRectF(x - 40, track_y + track_height + 2, 80, 14),
                alignment,
                text,
            )

        if not np.isfinite(self._score):
            painter.setPen(QtGui.QColor(theme.text_muted))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                QtCore.QRectF(margin, 4, track_width, track_y - 8),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                settings.DATA_UNAVAILABLE_MESSAGE,
            )
            painter.end()
            return

        position = margin + (float(np.clip(self._score, -1, 1)) + 1) / 2 * track_width
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.text), 2))
        painter.setBrush(QtGui.QColor(theme.surface))
        painter.drawEllipse(QtCore.QPointF(position, track_y + track_height / 2), 8, 8)

        painter.setPen(QtGui.QColor(theme.text))
        font = painter.font()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(margin, 2, track_width, 24),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            f"{self._label}   ({self._score:+.2f})",
        )
        painter.end()


def card_row(cards: Sequence[QtWidgets.QWidget], stretch: bool = True) -> QtWidgets.QWidget:
    """Lay widgets out in a horizontal row that wraps gracefully."""
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for card in cards:
        layout.addWidget(card, 1)
    if stretch and len(cards) < 4:
        layout.addStretch(1)
    return container


def grid_of(widgets: Sequence[QtWidgets.QWidget], columns: int = 2) -> QtWidgets.QWidget:
    """Lay widgets out in a grid."""
    container = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, index // columns, index % columns)
    for column in range(columns):
        layout.setColumnStretch(column, 1)
    return container


def tone_for_change(value: float) -> str:
    if not np.isfinite(value):
        return "neutral"
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"


# ==========================================================================
# Page base
# ==========================================================================


class BasePage(QtWidgets.QScrollArea):
    """Base class for every dashboard page.

    Subclasses implement :meth:`build`, which populates ``self.body`` using the
    supplied :class:`PageContext`. The shell calls :meth:`render` when the page
    is first shown and whenever filters change.
    """

    title: str = ""
    subtitle: str = ""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.context: PageContext | None = None
        self._built_for: Any = None
        self._themed: list[QtWidgets.QWidget] = []
        #: Incremented on every rebuild. Background callbacks captured against
        #: an older generation are dropped -- see :meth:`bind`.
        self._generation = 0
        self._container: QtWidgets.QWidget
        self.body: QtWidgets.QVBoxLayout
        self._fresh_container()

    # -- lifecycle --------------------------------------------------------

    def render(self, context: PageContext, *, force: bool = False) -> None:
        """(Re)build the page if the filters or theme have changed."""
        signature = (context.filters.key(), context.filters.frequency, context.theme.name)
        if not force and signature == self._built_for and self.body.count():
            return
        self.context = context
        self.clear()
        try:
            self.build(context)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Page %s failed to build", type(self).__name__)
            box = InfoBox(
                f"This page could not be rendered.\n\n{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc(limit=3)}",
                context.theme,
                tone="danger",
            )
            self.body.addWidget(box)
        self.body.addStretch(1)
        # Settle the layout now rather than on the next paint, so a screenshot
        # or an immediate scroll sees final geometry.
        self.body.activate()
        self._container.adjustSize()
        self._built_for = signature

    def _fresh_container(self) -> None:
        """Install a brand-new content widget, discarding the previous one.

        Rebuilding into the *same* container leaves the scroll area holding
        stale geometry: on the rebuild after a theme switch the old children's
        sizes are still in effect, charts collapse to a sliver and labels
        overlap. ``QScrollArea.setWidget`` deletes the widget it replaces, so
        handing it a fresh container guarantees a clean layout pass every time.
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(18, 14, 18, 20)
        layout.setSpacing(12)
        self._container = container
        self.body = layout
        self.setWidget(container)  # deletes the previous container

    def clear(self) -> None:
        self._themed.clear()
        self._generation += 1
        self._fresh_container()

    def bind(self, method: Callable[..., None]) -> Callable[..., None]:
        """Wrap a render callback so it is safe to call from a worker.

        Rebuilding a page deletes its widgets outright. A worker started before
        the rebuild would otherwise deliver its result into deleted objects, so
        the callback is dropped if the page has been rebuilt since it was
        submitted, and a late ``RuntimeError`` from Qt is swallowed rather than
        surfaced as a crash.
        """
        generation = self._generation

        def wrapped(*args: Any, **kwargs: Any) -> None:
            if generation != self._generation:
                LOG.debug(
                    "Dropping a stale result for %s (generation %d, now %d)",
                    type(self).__name__, generation, self._generation,
                )
                return
            try:
                method(*args, **kwargs)
            except RuntimeError as exc:  # deleted underlying C++ object
                LOG.debug("Stale widget touched in %s: %s", type(self).__name__, exc)

        return wrapped

    def build(self, context: PageContext) -> None:  # pragma: no cover
        raise NotImplementedError

    def invalidate(self) -> None:
        self._built_for = None

    # -- helpers ----------------------------------------------------------

    def track(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Register a widget so it is re-themed on a theme switch."""
        self._themed.append(widget)
        return widget

    def add(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.body.addWidget(self.track(widget))
        return widget

    def section(self, title: str, subtitle: str = "") -> SectionHeader:
        assert self.context is not None
        header = SectionHeader(title, subtitle, self.context.theme)
        self.body.addWidget(self.track(header))
        return header

    def chart(self, title: str, height: int = 330) -> charts.ChartPanel:
        assert self.context is not None
        panel = charts.ChartPanel(title, theme=self.context.theme, height=height)
        self.body.addWidget(self.track(panel))
        return panel

    def new_chart(self, title: str, height: int = 300) -> charts.ChartPanel:
        """A chart panel that is *not* added to the layout (for grids)."""
        assert self.context is not None
        return self.track(
            charts.ChartPanel(title, theme=self.context.theme, height=height)
        )

    def table(self, title: str, max_height: int | None = None) -> DataTable:
        assert self.context is not None
        widget = DataTable(title, self.context.theme, max_height=max_height)
        self.body.addWidget(self.track(widget))
        return widget

    def new_table(self, title: str, max_height: int | None = None) -> DataTable:
        assert self.context is not None
        return self.track(DataTable(title, self.context.theme, max_height=max_height))

    def info(self, text: str = "", tone: str = "info") -> InfoBox:
        assert self.context is not None
        box = InfoBox(text, self.context.theme, tone=tone)
        self.body.addWidget(self.track(box))
        return box

    def filter_note(self) -> InfoBox:
        """Standard caption echoing the active filters and frequency."""
        assert self.context is not None
        frequency = settings.FORECAST.frequency_labels.get(
            self.context.filters.frequency, self.context.filters.frequency
        )
        return self.info(
            f"Sample: {self.context.filters.describe()}  "
            f"Analysis frequency: {frequency}.",
            tone="muted",
        )

    def apply_theme(self, theme: Theme) -> None:
        for widget in self._themed:
            if hasattr(widget, "apply_theme"):
                widget.apply_theme(theme)


# ==========================================================================
# Page 1 -- Executive Summary
# ==========================================================================


class ExecutiveSummaryPage(BasePage):
    title = "Executive Summary"
    subtitle = "Where the market stands, what is driving it, and where it is heading"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        # -- headline prices ---------------------------------------------
        self.section(
            "Latest prices",
            "Most recent quote per variety with change over several horizons.",
        )
        focus = service.focus_varieties()
        snapshots = insights.variety_snapshots(service, filters)
        focus_labels = set(focus.values())
        ordered = [s for s in snapshots if s.name in focus_labels] + [
            s for s in snapshots if s.name not in focus_labels
        ]

        if not ordered:
            self.info(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo variety price "
                "survived the current filters.",
                tone="danger",
            )
        else:
            cards: list[QtWidgets.QWidget] = []
            for snap in ordered[:4]:
                wow = snap.change_pct("Week on week")
                mom = snap.change_pct("Month on month")
                yoy = snap.change_pct("Year on year")
                cards.append(
                    SummaryCard(
                        snap.name,
                        fmt_number(snap.latest, 0),
                        f"WoW {fmt_pct(wow, 1, signed=True)}",
                        f"MoM {fmt_pct(mom, 1, signed=True)} · YoY {fmt_pct(yoy, 1, signed=True)}\n"
                        f"as at {fmt_date(snap.latest_date)}",
                        context.theme,
                        tone=tone_for_change(wow),
                    )
                )
            self.add(card_row(cards))
            self.info(
                "Prices are in the workbook's own unit (INR per quintal as "
                f"recorded on '{ordered[0].source}'). Changes compare the "
                "latest quote with the last quote on or before the target "
                "date; no value is interpolated.",
                tone="muted",
            )

            table = self.table("Change by horizon — all varieties", max_height=260)
            rows: list[dict[str, Any]] = []
            for snap in ordered:
                row: dict[str, Any] = {
                    "Variety": snap.name,
                    "Latest": snap.latest,
                    "As at": snap.latest_date,
                }
                for horizon in (
                    "Previous observation", "Week on week", "Fortnight",
                    "Month on month", "Quarter on quarter", "Year on year",
                ):
                    row[horizon] = snap.change_pct(horizon) * 100
                rows.append(row)
            table.set_frame(
                pd.DataFrame(rows).set_index("Variety"),
                ordered[0].source,
                ["Percentage change over each horizon."],
            )

        # -- sentiment ----------------------------------------------------
        primary = focus.get("Teja") or (ordered[0].name if ordered else "")
        self.section(
            "Market sentiment",
            f"Composite bullish/bearish reading for {primary or 'the lead variety'}, "
            "built from five workbook-derived components.",
        )
        gauge = SentimentGauge(context.theme)
        self.add(gauge)
        sentiment = insights.compute_sentiment(service, primary, filters) if primary else None
        if sentiment and sentiment.ok:
            value = sentiment.unwrap()
            gauge.set_score(value.score, value.label)
            component_frame = pd.DataFrame(
                [
                    {
                        "Component": c.name,
                        "Score": c.score,
                        "Reading": c.explanation,
                        "Source": c.source,
                    }
                    for c in value.components
                ]
            ).set_index("Component")
            self.table("Sentiment components", max_height=210).set_frame(
                component_frame, sentiment.source, value.notes
            )
        else:
            gauge.set_score(float("nan"), settings.DATA_UNAVAILABLE_MESSAGE)
            self.info(
                sentiment.message() if sentiment else settings.DATA_UNAVAILABLE_MESSAGE,
                tone="warning",
            )

        # -- price chart --------------------------------------------------
        self.section("Price history", "All varieties over the filtered window.")
        panel = self.chart("Guntur variety prices", height=340)
        variety_panel = service.variety_panel(filters.frequency, filters)
        if variety_panel:
            frame = variety_panel.unwrap()
            charts.draw_line(
                panel,
                {str(c): frame[c] for c in frame.columns},
                ylabel="INR per quintal",
                source=variety_panel.source,
                notes=[
                    settings.FORECAST.frequency_labels.get(
                        filters.frequency, filters.frequency
                    )
                    + " averages of the daily quotes.",
                ],
                highlight=primary,
            )
        else:
            panel.show_unavailable(variety_panel)

        # -- arrivals + exports ------------------------------------------
        self.section("Supply and trade", "Arrivals, offtake and export volume.")
        supply_cards: list[QtWidgets.QWidget] = []

        arrivals = service.guntur_arrivals()
        if arrivals:
            series = service.apply_filters(arrivals.unwrap(), filters, kind="arrivals")
            monthly = service.series_at(series, "ME", "sum")
            if len(monthly) >= 13:
                latest = float(monthly.iloc[-1])
                same_month = monthly[monthly.index.month == monthly.index[-1].month].iloc[:-1]
                norm = float(same_month.mean()) if not same_month.empty else float("nan")
                delta = (latest - norm) / norm if np.isfinite(norm) and norm else float("nan")
                supply_cards.append(
                    SummaryCard(
                        "Arrivals, latest month",
                        fmt_int(latest) + " bags",
                        f"{fmt_pct(delta, 0, signed=True)} vs same-month average",
                        f"{monthly.index[-1]:%b %Y}. Bag weight per sheet header: "
                        f"{service.market_bag_weight('Guntur') or '—'} kg.",
                        context.theme,
                        tone=tone_for_change(-delta if np.isfinite(delta) else float("nan")),
                    )
                )

        offtake = service.guntur_offtake()
        if offtake and arrivals:
            arr = service.series_at(
                service.apply_filters(arrivals.unwrap(), filters, kind="arrivals"), "ME", "sum"
            )
            off = service.series_at(
                service.apply_filters(offtake.unwrap(), filters), "ME", "sum"
            )
            joined = pd.concat([arr.rename("a"), off.rename("o")], axis=1).dropna()
            if not joined.empty:
                ratio = float(joined["o"].iloc[-1] / joined["a"].iloc[-1]) if joined["a"].iloc[-1] else float("nan")
                mean_ratio = float((joined["o"] / joined["a"]).mean())
                supply_cards.append(
                    SummaryCard(
                        "Offtake / arrivals",
                        fmt_pct(ratio, 0),
                        f"average {fmt_pct(mean_ratio, 0)}",
                        "Share of arriving material actually lifted. A falling "
                        "ratio means stock is accumulating in the mandi.",
                        context.theme,
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
                if not year_ago.empty and float(year_ago.iloc[-1])
                else float("nan")
            )
            supply_cards.append(
                SummaryCard(
                    "Exports, latest month",
                    fmt_number(latest, 0),
                    f"YoY {fmt_pct(delta, 0, signed=True)}",
                    f"{series.index[-1]:%b %Y}. Units are not stated on the "
                    "sheet, so this is comparable only against itself.",
                    context.theme,
                    tone=tone_for_change(delta),
                )
            )

        stock_use = service.balance_sheet_row("Stock to Use")
        if stock_use:
            values = stock_use.unwrap()
            latest = float(values.iloc[-1])
            mean = float(values.iloc[:-1].mean())
            projected = service.data.datasets["balance_sheet"].meta.get("projected_years", [])
            supply_cards.append(
                SummaryCard(
                    f"Stock-to-use {values.index[-1]}",
                    f"{latest:,.1f}%",
                    f"average {mean:,.1f}%",
                    (
                        "Workbook projection, not realised data."
                        if values.index[-1] in projected
                        else "Realised."
                    )
                    + " A thin buffer supports prices.",
                    context.theme,
                    tone="positive" if latest < mean else "negative",
                )
            )

        if supply_cards:
            self.add(card_row(supply_cards))
        else:
            self.info(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNone of the supply or "
                "trade series is available under the current filters.",
                tone="warning",
            )

        arrivals_chart = self.chart("Price against arrivals", height=300)
        if arrivals and variety_panel:
            price = service.series_at(
                service.apply_filters(
                    service.variety_series(primary).unwrap(), filters, kind="price"
                ),
                filters.frequency,
            )
            arr_series = service.series_at(
                service.apply_filters(arrivals.unwrap(), filters, kind="arrivals"),
                filters.frequency,
                "sum",
            )
            charts.draw_dual_axis(
                arrivals_chart,
                price,
                arr_series,
                primary_label=f"{primary} price (INR/quintal)",
                secondary_label="Arrivals (bags)",
                source=f"{variety_panel.source}; {arrivals.source}",
                notes=["Arrivals are summed within each period; price is averaged."],
                secondary_as_bars=True,
            )
        else:
            arrivals_chart.show_unavailable(
                arrivals if not arrivals else variety_panel
            )

        # -- forecast summary --------------------------------------------
        self.section(
            "Forecast summary",
            "Best-performing model per focus variety. The Forecast Center has "
            "the full model comparison and explanation.",
        )
        self.forecast_box = self.info(
            "Running the model sweep for the focus varieties…", tone="muted"
        )
        self.forecast_table = self.new_table("Forecast summary")
        self.body.addWidget(self.forecast_table)
        self.forecast_table.setVisible(False)

        targets = list(focus.values()) or ([primary] if primary else [])
        if targets:
            context.submit(
                self._compute_forecast_summary,
                service,
                filters,
                targets,
                on_done=self.bind(self._render_forecast_summary),
                on_fail=self.bind(self._forecast_failed),
            )
        else:
            self.forecast_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNeither focus variety "
                "is present in the workbook."
            )

        # -- key insights -------------------------------------------------
        self.section("Key insights", "The strongest findings the data supports.")
        self.insight_box = self.info("Generating insights…", tone="muted")
        self.insight_container = QtWidgets.QWidget()
        self.insight_layout = QtWidgets.QVBoxLayout(self.insight_container)
        self.insight_layout.setContentsMargins(0, 0, 0, 0)
        self.insight_layout.setSpacing(8)
        self.body.addWidget(self.insight_container)
        context.submit(
            insights.generate_all,
            service,
            filters,
            primary,
            wants_progress=True,
            on_done=self.bind(self._render_insights),
            on_fail=self.bind(self._insights_failed),
        )

    # -- background results ----------------------------------------------

    @staticmethod
    def _compute_forecast_summary(
        service: DataService, filters: FilterState, targets: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Fit the monthly model sweep for each focus variety (worker thread)."""
        freq = "ME"
        horizon = settings.FORECAST.horizons[freq]
        rows: list[dict[str, Any]] = []
        for variety in targets:
            result = service.variety_series(variety)
            if not result:
                rows.append({"Variety": variety, "Status": result.reason})
                continue
            raw = service.apply_filters(result.unwrap(), filters, kind="price")
            series = service.series_at(raw, freq)
            series.name = variety
            if len(series) < settings.FORECAST.min_obs_arima:
                rows.append(
                    {
                        "Variety": variety,
                        "Status": f"Only {len(series)} monthly observation(s) after filtering.",
                    }
                )
                continue
            exog = service.exogenous_matrix(freq)
            panel = service.variety_panel(freq, filters)
            comparison = forecasting.run_all_models(
                series,
                freq,
                horizon,
                target_name=variety,
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
                    "Variety": variety,
                    "Latest": latest,
                    "Model": best.label,
                    f"Forecast {best.forecast.index[-1]:%b %Y}": final,
                    "Change %": (final - latest) / latest * 100 if latest else float("nan"),
                    "Lower 95%": float(best.pred_lower.iloc[-1]),
                    "Upper 95%": float(best.pred_upper.iloc[-1]),
                    "Backtest MAPE %": best.metrics.mape,
                    "Directional accuracy %": best.metrics.directional_accuracy,
                    "Status": "OK",
                    "_source": best.source,
                    "_reason": comparison.selection_reason,
                }
            )
        return rows

    def _render_forecast_summary(self, rows: list[dict[str, Any]]) -> None:
        if not self.context:
            return
        if not rows:
            self.forecast_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo forecast could be produced."
            )
            return
        usable = [r for r in rows if r.get("Status") == "OK"]
        if not usable:
            self.forecast_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n"
                + "  ".join(str(r.get("Status", "")) for r in rows)
            )
            return

        source = usable[0].pop("_source", "")
        reasons = [r.pop("_reason", "") for r in usable]
        for row in rows:
            row.pop("_source", None)
            row.pop("_reason", None)
        frame = pd.DataFrame(usable).drop(columns=["Status"], errors="ignore")
        if "Variety" in frame.columns:
            frame = frame.set_index("Variety")
        self.forecast_table.set_frame(
            frame,
            source,
            [
                "Six months ahead on monthly data, using the model with the "
                "lowest backtest RMSE for each variety.",
                "Lower/Upper 95% is the backtest-calibrated prediction "
                "interval at the end of the horizon.",
            ],
        )
        self.forecast_table.setVisible(True)
        self.forecast_box.setText(
            "Model selection\n" + "\n".join(f"•  {r}" for r in reasons if r)
        )

    def _forecast_failed(self, message: str) -> None:
        self.forecast_box.setText(
            f"The forecast summary could not be produced.\n\n{message}"
        )

    def _render_insights(self, payload: tuple[list[insights.Insight], list[str]]) -> None:
        if not self.context:
            return
        found, failures = payload
        highlights = insights.executive_highlights(found, limit=6)
        if not highlights:
            self.insight_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo insight reached the "
                "moderate-or-better evidence threshold under these filters."
            )
            return
        self.insight_box.setText(
            f"{len(found)} finding(s) generated; the {len(highlights)} strongest "
            "are shown. The Automated Insights page lists all of them."
            + (f"  {len(failures)} generator(s) reported a problem." if failures else "")
        )
        for insight in highlights:
            card = InsightCard(insight, self.context.theme)
            self.insight_layout.addWidget(self.track(card))

    def _insights_failed(self, message: str) -> None:
        self.insight_box.setText(f"Insight generation failed.\n\n{message}")


# ==========================================================================
# Page 2 -- Market Overview
# ==========================================================================


class MarketOverviewPage(BasePage):
    title = "Market Overview"
    subtitle = "Prices, arrivals and coverage across every market in the workbook"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        self.section("Prices by market", "Teja basis, so all markets are comparable.")
        panel = self.chart("Teja price by market", height=340)
        price_panel = service.price_panel(filters.frequency, filters)
        if price_panel:
            frame = price_panel.unwrap()
            notes = [service.market_note(m) for m in frame.columns if service.market_note(m)]
            charts.draw_line(
                panel,
                {str(c): frame[c] for c in frame.columns},
                ylabel="INR per quintal",
                source=price_panel.source,
                notes=notes,
            )
        else:
            panel.show_unavailable(price_panel)

        self.section("Arrivals by market", "Volumes are in each sheet's own bag unit.")
        arrivals_panel_chart = self.chart("Arrivals by market", height=320)
        arrivals_panel = service.arrivals_panel(filters.frequency, filters)
        if arrivals_panel:
            frame = arrivals_panel.unwrap()
            notes = [
                f"{market}: 1 bag = {service.market_bag_weight(market):g} kg per sheet header."
                for market in frame.columns
                if service.market_bag_weight(market)
            ]
            notes.append(
                "Bag weights differ between markets (Guntur 45 kg, Warangal and "
                "Khammam 40 kg), so bag counts are not directly comparable "
                "across markets. The tonnage table below converts them."
            )
            charts.draw_line(
                arrivals_panel_chart,
                {str(c): frame[c] for c in frame.columns},
                ylabel="Bags per period",
                source=arrivals_panel.source,
                notes=notes,
            )
        else:
            arrivals_panel_chart.show_unavailable(arrivals_panel)

        tonnes_table = self.table("Arrivals in tonnes, by market and year", max_height=280)
        rows: dict[str, pd.Series] = {}
        sources: list[str] = []
        unavailable: list[str] = []
        for market in service.markets():
            converted = service.market_arrivals_tonnes(market)
            if not converted:
                unavailable.append(f"{market}: {converted.reason}")
                continue
            series = service.apply_filters(converted.unwrap(), filters, kind="arrivals")
            annual = series.resample("YE").sum()
            annual.index = annual.index.year
            rows[market] = annual
            if converted.source not in sources:
                sources.append(converted.source)
        if rows:
            tonnes_table.set_frame(
                pd.DataFrame(rows),
                "; ".join(sources),
                [
                    "Converted using each sheet's stated kilograms-per-bag. "
                    "Years are calendar years and partial years appear short."
                ]
                + unavailable,
            )
        else:
            tonnes_table.show_unavailable(
                "No market states a kilograms-per-bag conversion, so tonnage "
                "cannot be derived. " + " ".join(unavailable)
            )

        self.section(
            "Workbook coverage",
            "What each sheet contains and how far it reaches. Every analysis in "
            "this application is bounded by these spans.",
        )
        self.table("Dataset coverage", max_height=330).set_frame(
            service.coverage_table(),
            service.data.path.name,
            [
                f"Workbook read in {service.data.load_seconds:.2f}s at "
                f"{service.data.loaded_at:%d %b %Y %H:%M}.",
            ],
        )

        quality = service.data_quality_notes()
        if quality:
            self.info("", tone="warning").set_items(
                "Coverage limitations that affect the analyses in this application:",
                quality,
            )


# ==========================================================================
# Page 3 -- Price Analysis
# ==========================================================================


class PriceAnalysisPage(BasePage):
    title = "Price Analysis"
    subtitle = "Trend, rolling statistics, decomposition and time-series diagnostics"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        variety = self._target_variety(service, filters)
        if not variety:
            self.info(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nNo variety is available "
                "under the current filters.",
                tone="danger",
            )
            return

        result = service.variety_series(variety)
        raw = service.apply_filters(result.unwrap(), filters, kind="price")
        series = service.series_at(raw, filters.frequency)
        series.name = variety
        source = result.source
        window = settings.ANALYTICS.default_rolling_window

        partial = service.partial_last_period(raw, filters.frequency)
        if partial:
            self.info(partial, tone="warning")

        self.section(
            f"{variety} — price and moving averages",
            f"{settings.FORECAST.frequency_labels.get(filters.frequency, filters.frequency)} "
            f"series with a {window}-period simple and exponential average.",
        )
        rolling = analytics.rolling_statistics(series, window, source)
        panel = self.chart(f"{variety} price with rolling statistics", height=360)
        if rolling:
            frame = rolling.unwrap()
            charts.draw_line(
                panel,
                {
                    variety: frame["Value"],
                    f"Moving average ({window})": frame[f"Moving average ({window})"],
                    f"EMA ({window})": frame[f"EMA ({window})"],
                },
                ylabel="INR per quintal",
                source=source,
                notes=list(rolling.notes),
                highlight=variety,
                fill_between=(frame["Lower band (-2 sd)"], frame["Upper band (+2 sd)"]),
            )
            volatility_column = next(
                (c for c in frame.columns if c.startswith("Rolling volatility")), None
            )
            if volatility_column:
                vol_panel = self.chart("Rolling volatility (annualised)", height=250)
                charts.draw_line(
                    vol_panel,
                    {"Annualised volatility": frame[volatility_column] * 100},
                    ylabel="% annualised",
                    source=source,
                    notes=list(rolling.notes),
                )
        else:
            panel.show_unavailable(rolling)

        self.section("Comparison across varieties", "Select varieties in the sidebar filter.")
        compare_panel = self.chart("Variety comparison", height=330)
        variety_panel = service.variety_panel(filters.frequency, filters)
        if variety_panel:
            frame = variety_panel.unwrap()
            charts.draw_line(
                compare_panel,
                {str(c): frame[c] for c in frame.columns},
                ylabel="INR per quintal",
                source=variety_panel.source,
                highlight=variety,
            )
        else:
            compare_panel.show_unavailable(variety_panel)

        # Indexed comparison makes relative performance legible.
        indexed_panel = self.chart("Relative performance (first period = 100)", height=300)
        if variety_panel:
            frame = variety_panel.unwrap()
            normalised = {}
            for column in frame.columns:
                values = frame[column].dropna()
                if values.empty or values.iloc[0] == 0:
                    continue
                normalised[str(column)] = values / values.iloc[0] * 100
            if normalised:
                charts.draw_line(
                    indexed_panel,
                    normalised,
                    ylabel="Index (first period = 100)",
                    source=variety_panel.source,
                    notes=[
                        "Each series is rebased to 100 at its own first "
                        "observation in the filtered window, so the lines show "
                        "relative performance rather than absolute price."
                    ],
                    highlight=variety,
                )
            else:
                indexed_panel.show_unavailable(
                    "No variety has a usable first observation to rebase on."
                )
        else:
            indexed_panel.show_unavailable(variety_panel)

        self.section("Descriptive statistics and change")
        stats = analytics.descriptive_stats(series, source)
        changes = analytics.change_summary(raw, source)
        tables: list[QtWidgets.QWidget] = []
        stats_table = self.new_table("Descriptive statistics", max_height=340)
        if stats:
            values = stats.unwrap().drop(
                labels=["First observation", "Last observation"], errors="ignore"
            )
            stats_table.set_frame(
                values.to_frame("Value"), source, list(stats.notes)
            )
        else:
            stats_table.show_unavailable(stats)
        tables.append(stats_table)

        changes_table = self.new_table("Change by horizon", max_height=340)
        if changes:
            frame = changes.unwrap().copy()
            frame["Change %"] = frame["Change %"] * 100
            changes_table.set_frame(
                frame.set_index("Horizon"), source, list(changes.notes)
            )
        else:
            changes_table.show_unavailable(changes)
        tables.append(changes_table)
        self.add(grid_of(tables, 2))

        self.section(
            "Trend, seasonal and irregular components",
            "STL decomposition separates the persistent level from the "
            "repeating calendar pattern and the unexplained remainder.",
        )
        decomposition = analytics.decompose(series, filters.frequency, source=source)
        decomposition_panel = self.chart("Decomposition", height=460)
        if decomposition:
            charts.draw_decomposition(
                decomposition_panel,
                decomposition.unwrap(),
                source=source,
                notes=list(decomposition.notes),
            )
        else:
            decomposition_panel.show_unavailable(decomposition)

        self.section(
            "Time-series diagnostics",
            "These determine how the series must be modelled — see the "
            "Forecast Center for the models themselves.",
        )
        stationarity = analytics.stationarity_tests(series, source)
        stationarity_table = self.table("Stationarity tests", max_height=200)
        if stationarity:
            stationarity_table.set_frame(
                stationarity.unwrap(), source, list(stationarity.notes)
            )
        else:
            stationarity_table.show_unavailable(stationarity)

        autocorrelation = analytics.autocorrelation(series, 40, source)
        autocorrelation_panel = self.chart("Autocorrelation and partial autocorrelation", height=290)
        if autocorrelation:
            charts.draw_acf_pacf(
                autocorrelation_panel,
                autocorrelation.unwrap(),
                source=source,
                notes=list(autocorrelation.notes),
            )
        else:
            autocorrelation_panel.show_unavailable(autocorrelation)

        self.section("Outliers", "Flagged for review; never removed from any calculation.")
        outliers = analytics.zscore_and_outliers(series, window, source)
        outliers_table = self.table("Flagged observations", max_height=250)
        if outliers:
            frame = outliers.unwrap()
            flagged = frame[frame["Z outlier"] | frame["IQR outlier"]]
            if flagged.empty:
                outliers_table.show_unavailable(
                    "No observation in this window is flagged as an outlier by "
                    "either the z-score or the IQR rule. That is a result, not "
                    "a gap."
                )
                outliers_table.caption_label.setText(f"Source: {source}")
            else:
                outliers_table.set_frame(flagged, source, list(outliers.notes))
        else:
            outliers_table.show_unavailable(outliers)

    @staticmethod
    def _target_variety(service: DataService, filters: FilterState) -> str:
        if filters.varieties:
            resolved = service.resolve_variety(filters.varieties[0])
            if resolved:
                return resolved
        focus = service.focus_varieties()
        if focus:
            return next(iter(focus.values()))
        available = service.varieties()
        return available[0] if available else ""


# ==========================================================================
# Page 4 -- Arrival Analysis
# ==========================================================================


class ArrivalAnalysisPage(BasePage):
    title = "Arrival Analysis"
    subtitle = "How arrivals move prices: elasticity, thresholds, timing and seasonality"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        variety = PriceAnalysisPage._target_variety(service, filters)
        price_result = service.variety_series(variety) if variety else None
        arrivals_result = service.guntur_arrivals()

        if not price_result or not arrivals_result:
            self.info(
                (price_result.message() if price_result and not price_result else "")
                or (arrivals_result.message() if not arrivals_result else "")
                or settings.DATA_UNAVAILABLE_MESSAGE,
                tone="danger",
            )
            return

        source = f"{price_result.source}; {arrivals_result.source}"
        price = service.series_at(
            service.apply_filters(price_result.unwrap(), filters, kind="price"),
            filters.frequency,
        )
        arrivals = service.series_at(
            service.apply_filters(arrivals_result.unwrap(), filters, kind="arrivals"),
            filters.frequency,
            "sum",
        )
        offtake_result = service.guntur_offtake()

        self.info(
            f"Prices are {variety} averages; arrivals are Guntur mandi totals "
            f"summed within each {settings.FORECAST.frequency_labels.get(filters.frequency, filters.frequency).lower()} "
            f"period, in bags of "
            f"{service.market_bag_weight('Guntur') or '—'} kg as stated on the "
            "arrivals sheet.",
            tone="muted",
        )

        self.section("Price against arrivals")
        dual = self.chart("Price and arrivals", height=340)
        charts.draw_dual_axis(
            dual,
            price,
            arrivals,
            primary_label=f"{variety} price (INR/quintal)",
            secondary_label="Arrivals (bags)",
            source=source,
            secondary_as_bars=True,
        )

        if offtake_result:
            offtake = service.series_at(
                service.apply_filters(offtake_result.unwrap(), filters),
                filters.frequency,
                "sum",
            )
            flow = self.chart("Arrivals against offtake", height=280)
            charts.draw_line(
                flow,
                {"Arrivals": arrivals, "Offtake": offtake},
                ylabel="Bags per period",
                source=f"{arrivals_result.source}",
                notes=[
                    "Offtake is the quantity actually lifted. Arrivals "
                    "persistently above offtake means stock building in the mandi."
                ],
            )

        self.section(
            "Scatter and fitted relationship",
            "Colour shows when each observation occurred, which reveals whether "
            "the relationship has shifted over time.",
        )
        scatter = self.chart("Arrivals against price", height=340)
        charts.draw_scatter_with_fit(
            scatter,
            arrivals,
            price,
            xlabel="Arrivals (bags per period)",
            ylabel=f"{variety} price (INR/quintal)",
            source=source,
        )

        self.section(
            "Elasticity",
            "Estimated on log differences, so each coefficient reads as the "
            "percentage price response to a 1% change in arrivals.",
        )
        elasticity = analytics.elasticity(price, arrivals, source)
        elasticity_table = self.table("Elasticity by arrivals lag", max_height=220)
        if elasticity:
            elasticity_table.set_frame(
                elasticity.unwrap(), source, list(elasticity.notes)
            )
            stem = self.chart("Elasticity by lag", height=260)
            frame = elasticity.unwrap()
            significant = frame[frame["Significant"]]
            charts.draw_stem(
                stem,
                frame["Elasticity"],
                xlabel="Arrivals lag (periods)",
                ylabel="Elasticity (% price per % arrivals)",
                source=source,
                notes=list(elasticity.notes),
                highlight_index=(
                    int(significant["p-value"].idxmin()) if not significant.empty else None
                ),
            )
        else:
            elasticity_table.show_unavailable(elasticity)

        self.section(
            "Threshold effects",
            "Arrivals split into quintiles: at what level do prices start to "
            "come under pressure?",
        )
        thresholds = analytics.threshold_effects(price, arrivals, source=source)
        threshold_table = self.table("Price behaviour by arrivals bucket", max_height=230)
        if thresholds:
            frame = thresholds.unwrap()
            threshold_table.set_frame(frame, source, list(thresholds.notes))
            bucket_chart = self.chart("Next-period price change by arrivals bucket", height=280)
            labels = pd.Index(
                [
                    f"{row['Arrivals from']:,.0f}–{row['Arrivals to']:,.0f}"
                    for _, row in frame.iterrows()
                ],
                name="Arrivals range (bags)",
            )
            values = pd.Series(
                frame["Mean next-period change %"].to_numpy(), index=labels
            )
            charts.draw_bar(
                bucket_chart,
                values,
                ylabel="Mean next-period price change (%)",
                source=source,
                notes=list(thresholds.notes),
                reference=0.0,
                reference_label="No change",
                value_labels=True,
                value_format="{:+.2f}%",
            )
        else:
            threshold_table.show_unavailable(thresholds)

        self.section(
            "Timing of the arrivals effect",
            "Correlation between the change in arrivals and the change in price "
            "at successive lags.",
        )
        lagged = analytics.lagged_impact(price, arrivals, 8, source)
        lag_chart = self.chart("Lagged impact of arrivals on price change", height=280)
        if lagged:
            frame = lagged.unwrap()
            band = 1.96 / np.sqrt(float(frame["Observations"].max()))
            charts.draw_stem(
                lag_chart,
                frame["Correlation with price change"],
                xlabel="Arrivals lag (periods)",
                ylabel="Correlation with price change",
                source=source,
                notes=list(lagged.notes),
                band=band,
            )
            self.table("Lagged impact detail", max_height=230).set_frame(
                frame, source, list(lagged.notes)
            )
        else:
            lag_chart.show_unavailable(lagged)

        self.section(
            "Arrivals seasonality",
            "The monthly arrivals sheet, as a calendar grid and as a season "
            "classification derived from it.",
        )
        monthly = service.data.get("guntur_monthly_arrivals")
        calendar = self.chart("Monthly arrivals by year", height=360)
        if monthly is not None:
            charts.draw_calendar_heatmap(
                calendar,
                monthly.frame,
                source=monthly.sheet_name,
                notes=[
                    monthly.meta.get("primary_unit", ""),
                    monthly.meta.get("secondary_unit", ""),
                ],
                cbar_label="Bags",
            )
        else:
            calendar.show_unavailable(
                "The Guntur monthly arrivals sheet is not present in the workbook."
            )

        season = service.season_profile()
        season_table = self.table("Season classification by month", max_height=330)
        if season:
            frame = season.unwrap().set_index("month_name")
            frame.index.name = "Month"
            season_table.set_frame(frame, season.source, list(season.notes))
        else:
            season_table.show_unavailable(season)


# ==========================================================================
# Page 5 -- Market Integration
# ==========================================================================


class MarketIntegrationPage(BasePage):
    title = "Market Integration"
    subtitle = "Does Guntur drive Warangal and Khammam? Direction, strength and cointegration"

    #: Integration is assessed on daily data: a lead of one or two days is
    #: invisible once the series are averaged to weeks.
    ANALYSIS_FREQ = "D"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()
        self.info(
            "Lead-lag and causality on this page are computed on DAILY data "
            "regardless of the sidebar frequency: averaging to weeks or months "
            "would hide a one- or two-day lead, which is exactly what is being "
            "measured here. Cointegration is also run on daily levels.",
            tone="info",
        )

        panel_result = service.price_panel(self.ANALYSIS_FREQ, filters)
        if not panel_result:
            self.info(panel_result.message(), tone="danger")
            return
        frame = panel_result.unwrap()
        source = panel_result.source

        self.section("Market prices", "Teja basis across all available markets.")
        price_chart = self.chart("Teja price by market (daily)", height=330)
        charts.draw_line(
            price_chart,
            {str(c): frame[c] for c in frame.columns},
            ylabel="INR per quintal",
            source=source,
        )

        # Spread against the reference market makes convergence visible.
        reference = "Guntur" if "Guntur" in frame.columns else str(frame.columns[0])
        spreads = {
            f"{column} − {reference}": (frame[column] - frame[reference]).dropna()
            for column in frame.columns
            if column != reference
        }
        if spreads:
            spread_chart = self.chart(f"Spread to {reference}", height=300)
            charts.draw_line(
                spread_chart,
                spreads,
                ylabel="INR per quintal",
                source=source,
                notes=[
                    "A spread that oscillates around a stable level indicates "
                    "integrated markets. A spread that trends away indicates "
                    "divergence.",
                ],
            )

        self.section(
            "Influence structure",
            "Computed with Granger causality run in both directions for every "
            "pair. This can take a few seconds.",
        )
        self.network_panel = self.chart("Market influence diagram", height=420)
        self.influence_table = self.new_table("Pairwise direction of influence")
        self.body.addWidget(self.influence_table)
        self.leadership_table = self.new_table("Market leadership ranking")
        self.body.addWidget(self.leadership_table)
        self.leadership_box = self.info("Computing influence structure…", tone="muted")

        self.section(
            "Lead-lag timing",
            "Peak cross-correlation of the differenced series, in periods and "
            "in days.",
        )
        self.leadlag_table = self.new_table("Pairwise lead-lag")
        self.body.addWidget(self.leadlag_table)

        self.section(
            "Cointegration",
            "The formal test of market integration: cointegrated markets share "
            "a long-run equilibrium and their spreads mean-revert.",
        )
        self.johansen_table = self.new_table("Johansen system test")
        self.body.addWidget(self.johansen_table)
        self.pairwise_table = self.new_table("Engle-Granger pairwise test")
        self.body.addWidget(self.pairwise_table)
        self.cointegration_box = self.info("Testing for cointegration…", tone="muted")

        context.submit(
            self._compute,
            frame,
            self.ANALYSIS_FREQ,
            source,
            on_done=self.bind(self._render),
            on_fail=self.bind(self._failed),
        )

    @staticmethod
    def _compute(frame: pd.DataFrame, freq: str, source: str) -> dict[str, Any]:
        """Run the full integration battery on the worker thread."""
        return {
            "leadership": analytics.leadership_ranking(frame, freq, source, 20),
            "leadlag": analytics.lead_lag_matrix(frame, freq, 20, source),
            "cointegration": analytics.cointegration(frame.dropna(), source),
            "source": source,
        }

    def _render(self, payload: dict[str, Any]) -> None:
        if not self.context:
            return
        source = payload["source"]

        leadership = payload["leadership"]
        if leadership:
            table = leadership.unwrap()
            self.leadership_table.set_frame(table, source, list(leadership.notes))
            pairs = table.attrs.get("pairs")
            if pairs is not None and not pairs.empty:
                self.influence_table.set_frame(
                    pairs.drop(columns=["Note"], errors="ignore"), source
                )
                edges: list[tuple[str, str, float, str]] = []
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
                        strength = -np.log10(
                            max(min(float(row["A -> B best p"]), float(row["B -> A best p"])), 1e-12)
                        )
                        edges.append((start, end, float(strength), "feedback"))
                scores = {
                    str(row["Series"]): float(row["Leadership score"])
                    for _, row in table.iterrows()
                }
                charts.draw_influence_network(
                    self.network_panel,
                    list(scores.keys()),
                    edges,
                    source=source,
                    notes=list(leadership.notes),
                    node_scores=scores,
                )
            self.leadership_box.set_items("Reading of the influence structure:", leadership.notes)
        else:
            self.network_panel.show_unavailable(leadership)
            self.influence_table.show_unavailable(leadership)
            self.leadership_table.show_unavailable(leadership)
            self.leadership_box.setVisible(False)

        leadlag = payload["leadlag"]
        if leadlag:
            self.leadlag_table.set_frame(
                leadlag.unwrap().drop(columns=["Note"], errors="ignore"),
                source,
                list(leadlag.notes),
            )
        else:
            self.leadlag_table.show_unavailable(leadlag)

        cointegration = payload["cointegration"]
        if cointegration:
            data = cointegration.unwrap()
            johansen = data.get("johansen")
            if johansen is not None and not johansen.empty:
                self.johansen_table.set_frame(johansen, source, list(cointegration.notes))
            else:
                self.johansen_table.show_unavailable(
                    "The Johansen test could not be run on this panel."
                )
            pairwise = data.get("pairwise")
            if pairwise is not None and not pairwise.empty:
                self.pairwise_table.set_frame(pairwise, source)
            else:
                self.pairwise_table.show_unavailable(
                    "No pair had enough overlapping observations."
                )
            self.cointegration_box.set_items("Interpretation:", cointegration.notes)
        else:
            self.johansen_table.show_unavailable(cointegration)
            self.pairwise_table.show_unavailable(cointegration)
            self.cointegration_box.setText(cointegration.message())

    def _failed(self, message: str) -> None:
        self.leadership_box.setText(f"The integration analysis failed.\n\n{message}")
        self.cointegration_box.setVisible(False)


# ==========================================================================
# Page 6 -- Correlation Studio
# ==========================================================================


class CorrelationStudioPage(BasePage):
    title = "Correlation Studio"
    subtitle = "Pearson, Spearman, rolling, lag and cross-correlation"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        panel_result = service.variety_panel(filters.frequency, filters)
        if not panel_result:
            self.info(panel_result.message(), tone="danger")
            return
        frame = panel_result.unwrap()
        source = panel_result.source

        self.section(
            "Correlation matrices",
            "Recomputed from the daily price sheet at the selected frequency. "
            "Pearson measures linear co-movement; Spearman measures rank "
            "agreement and is robust to outliers.",
        )
        matrices: list[QtWidgets.QWidget] = []
        for method in ("pearson", "spearman"):
            result = analytics.correlation_matrix(frame, method, source)
            panel = self.new_chart(f"{method.title()} correlation", height=380)
            if result:
                charts.draw_heatmap(
                    panel,
                    result.unwrap(),
                    source=source,
                    notes=list(result.notes),
                    diverging=True,
                    vmin=-1,
                    vmax=1,
                    cbar_label="Correlation",
                )
            else:
                panel.show_unavailable(result)
            matrices.append(panel)
        self.add(grid_of(matrices, 2))

        workbook_matrix = service.workbook_variety_correlation()
        comparison_panel = self.chart(
            "The workbook's own correlation matrix, for comparison", height=380
        )
        if workbook_matrix:
            charts.draw_heatmap(
                comparison_panel,
                workbook_matrix.unwrap(),
                source=workbook_matrix.source,
                notes=list(workbook_matrix.notes)
                + [
                    "This matrix is read directly from the workbook and is not "
                    "recomputed. It covers more varieties than the daily price "
                    "sheet quotes, so some rows and columns have no counterpart "
                    "above.",
                ],
                diverging=True,
                vmin=-1,
                vmax=1,
                cbar_label="Correlation",
            )
        else:
            comparison_panel.show_unavailable(workbook_matrix)

        # Reconcile our numbers against the workbook's.
        if workbook_matrix:
            recomputed = analytics.correlation_matrix(frame, "pearson", source)
            if recomputed:
                ours = recomputed.unwrap()
                theirs = workbook_matrix.unwrap()
                rows: list[dict[str, Any]] = []
                for row_label in ours.index:
                    match_row = next(
                        (r for r in theirs.index if str(r).lower() == str(row_label).lower()), None
                    )
                    if match_row is None:
                        continue
                    for column_label in ours.columns:
                        if str(column_label) == str(row_label):
                            continue
                        match_col = next(
                            (c for c in theirs.columns if str(c).lower() == str(column_label).lower()),
                            None,
                        )
                        if match_col is None:
                            continue
                        mine = ours.loc[row_label, column_label]
                        yours = theirs.loc[match_row, match_col]
                        if pd.isna(mine) or pd.isna(yours):
                            continue
                        rows.append(
                            {
                                "Pair": f"{row_label} / {column_label}",
                                "This application": float(mine),
                                "Workbook": float(yours),
                                "Difference": float(mine) - float(yours),
                            }
                        )
                if rows:
                    reconciliation = (
                        pd.DataFrame(rows)
                        .drop_duplicates(subset="Pair")
                        .set_index("Pair")
                        .sort_values("Difference", key=abs, ascending=False)
                    )
                    self.table("Reconciliation against the workbook", max_height=280).set_frame(
                        reconciliation,
                        f"{source}; {workbook_matrix.source}",
                        [
                            "Differences are expected: this application "
                            f"correlates "
                            f"{settings.FORECAST.frequency_labels.get(filters.frequency, filters.frequency).lower()} "
                            "averages over the filtered window, while the "
                            "workbook's matrix was computed over its own "
                            "sample. Large gaps are worth investigating; small "
                            "ones confirm both are measuring the same thing.",
                            f"Median absolute difference: "
                            f"{reconciliation['Difference'].abs().median():.3f}.",
                        ],
                    )

        self.section(
            "Rolling correlation",
            "Whether a relationship holds through time, or breaks down.",
        )
        columns = list(frame.columns)
        pairs = [(columns[0], c) for c in columns[1:4]] if len(columns) > 1 else []
        markets = service.price_panel(filters.frequency, filters)
        if markets and markets.unwrap().shape[1] > 1:
            market_frame = markets.unwrap()
            market_columns = list(market_frame.columns)
            for other in market_columns[1:3]:
                pairs.append((market_columns[0], other))

        window = max(12, min(52, len(frame) // 6))
        rolling_charts: list[QtWidgets.QWidget] = []
        for left, right in pairs[:4]:
            left_series = (
                frame[left] if left in frame.columns else markets.unwrap()[left]
            )
            right_series = (
                frame[right] if right in frame.columns else markets.unwrap()[right]
            )
            result = analytics.rolling_correlation(left_series, right_series, window, source)
            panel = self.new_chart(f"{left} vs {right} — rolling correlation", height=280)
            if result:
                charts.draw_line(
                    panel,
                    {f"{window}-period correlation": result.unwrap()},
                    ylabel="Correlation",
                    source=source,
                    notes=list(result.notes),
                )
            else:
                panel.show_unavailable(result)
            rolling_charts.append(panel)
        if rolling_charts:
            self.add(grid_of(rolling_charts, 2))
        else:
            self.info(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nAt least two series are "
                "needed for a rolling correlation.",
                tone="warning",
            )

        self.section(
            "Cross-correlation and lag explorer",
            "Positive lag means the first series leads. Both series are "
            "differenced first, so a shared trend cannot manufacture a peak.",
        )
        cross_charts: list[QtWidgets.QWidget] = []
        for left, right in pairs[:4]:
            left_series = frame[left] if left in frame.columns else markets.unwrap()[left]
            right_series = frame[right] if right in frame.columns else markets.unwrap()[right]
            result = analytics.cross_correlation(
                left_series, right_series, 20, source, differenced=True
            )
            panel = self.new_chart(f"{left} → {right} — cross-correlation", height=290)
            if result:
                table = result.unwrap()
                peak = int(table["correlation"].abs().idxmax())
                charts.draw_stem(
                    panel,
                    table["correlation"],
                    xlabel=f"Lag ({left} leading →)",
                    ylabel="Correlation of changes",
                    source=source,
                    notes=list(result.notes),
                    band=float(table["upper_95"].iloc[0]),
                    highlight_index=peak,
                )
            else:
                panel.show_unavailable(result)
            cross_charts.append(panel)
        if cross_charts:
            self.add(grid_of(cross_charts, 2))

        self.section(
            "Multicollinearity among drivers",
            "Two drivers carrying the same information cannot be separately "
            "attributed in any regression.",
        )
        exog = service.exogenous_matrix("ME")
        vif_table = self.table("Variance inflation factors", max_height=200)
        if exog:
            vif = analytics.variance_inflation(exog.unwrap(), exog.source)
            if vif:
                vif_table.set_frame(vif.unwrap(), exog.source, list(vif.notes))
            else:
                vif_table.show_unavailable(vif)
        else:
            vif_table.show_unavailable(exog)


# ==========================================================================
# Page 7 -- Seasonality
# ==========================================================================


class SeasonalityPage(BasePage):
    title = "Seasonality"
    subtitle = "Monthly and weekly patterns, seasonal indices and the harvest calendar"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        variety = PriceAnalysisPage._target_variety(service, filters)
        result = service.variety_series(variety) if variety else None
        if not result:
            self.info(
                result.message() if result else settings.DATA_UNAVAILABLE_MESSAGE,
                tone="danger",
            )
            return
        series = service.apply_filters(result.unwrap(), filters, kind="price")
        source = result.source

        self.section(
            f"Monthly seasonal index — {variety}",
            "Month mean divided by the overall mean. Above 1.00 marks a "
            "seasonally firm month.",
        )
        indices = analytics.seasonal_indices(series, source)
        index_chart = self.chart("Seasonal index by month", height=300)
        if indices:
            table = indices.unwrap()
            charts.draw_bar(
                index_chart,
                table["Seasonal index"],
                ylabel="Seasonal index (1.00 = average)",
                source=source,
                notes=list(indices.notes),
                reference=1.0,
                reference_label="All-month average",
                colour_negative=False,
                value_labels=True,
                value_format="{:.3f}",
            )
            self.table("Seasonal statistics by month", max_height=330).set_frame(
                table, source, list(indices.notes)
            )
        else:
            index_chart.show_unavailable(indices)

        self.section(
            "Distribution by month",
            "Boxes show the spread of monthly averages across years, so a "
            "reliable seasonal month can be told from a volatile one.",
        )
        box_chart = self.chart("Monthly price distribution", height=320)
        monthly = service.series_at(series, "ME")
        if len(monthly) >= 24:
            groups = {
                settings.MONTH_ABBREVIATIONS[month - 1].title(): monthly[
                    monthly.index.month == month
                ]
                for month in range(1, 13)
            }
            charts.draw_box_by_group(
                box_chart,
                groups,
                ylabel="Monthly average price (INR/quintal)",
                source=source,
                notes=[
                    "Each box covers one calendar month across all years in the "
                    "filtered window. A narrow box means a dependable seasonal "
                    "level; a wide one means the month's outcome varies.",
                ],
            )
        else:
            box_chart.show_unavailable(
                f"Only {len(monthly)} monthly observation(s) after filtering; "
                "at least 24 are needed to show a distribution per month."
            )

        self.section(
            "Calendar grid",
            "Every month of every year. Vertical banding marks a seasonal "
            "pattern; horizontal banding marks a strong year effect.",
        )
        calendar = self.chart(f"{variety} monthly average price by year", height=380)
        if len(monthly) >= 12:
            grid = pd.DataFrame(
                {
                    "year": monthly.index.year,
                    "month": monthly.index.month,
                    "value": monthly.to_numpy(),
                }
            ).pivot_table(index="year", columns="month", values="value")
            grid.columns = [settings.MONTH_ABBREVIATIONS[m - 1].title() for m in grid.columns]
            grid.index.name = "Year"
            charts.draw_calendar_heatmap(
                calendar,
                grid,
                source=source,
                notes=[
                    "Monthly averages of the daily quotes. Blank cells are "
                    "months with no recorded trade.",
                ],
                cbar_label="INR per quintal",
            )
        else:
            calendar.show_unavailable(
                "At least twelve monthly observations are needed for a calendar grid."
            )

        self.section(
            "The workbook's own seasonality sheet",
            "Shown for comparison with the indices computed above.",
        )
        workbook_seasonality = service.workbook_seasonality()
        workbook_chart = self.chart("Workbook seasonality grid", height=360)
        if workbook_seasonality:
            dataset = workbook_seasonality.unwrap()
            charts.draw_calendar_heatmap(
                workbook_chart,
                dataset.frame,
                source=dataset.sheet_name,
                notes=[
                    "Read directly from the workbook, not recomputed.",
                    "Cells the workbook records as 'Closed' appear blank; they "
                    "are treated as missing observations, never as zero.",
                ],
                cbar_label="INR per quintal",
            )
            supplied = dataset.meta.get("workbook_supplied_rows")
            if supplied is not None and not supplied.empty:
                self.table(
                    "The workbook's own average and seasonality index rows",
                    max_height=140,
                ).set_frame(
                    supplied,
                    dataset.sheet_name,
                    [
                        "These are the workbook's figures. The chart above "
                        "compares them with this application's independent "
                        "calculation from the daily price sheet.",
                    ],
                )
                index_row = next(
                    (i for i in supplied.index if "season" in str(i).lower()), None
                )
                if index_row is not None and indices:
                    ours = indices.unwrap()["Seasonal index"]
                    theirs = pd.to_numeric(supplied.loc[index_row], errors="coerce")
                    comparison = pd.DataFrame(
                        {"This application": ours, "Workbook": theirs}
                    ).dropna()
                    if not comparison.empty:
                        comparison["Difference"] = (
                            comparison["This application"] - comparison["Workbook"]
                        )
                        compare_chart = self.chart(
                            "Seasonal index: this application against the workbook",
                            height=300,
                        )
                        charts.draw_grouped_bar(
                            compare_chart,
                            comparison[["This application", "Workbook"]],
                            ylabel="Seasonal index",
                            source=f"{source}; {dataset.sheet_name}",
                            notes=[
                                "Both series should broadly agree. Differences "
                                "arise because this application uses the "
                                "filtered daily price sheet while the workbook "
                                "used its own fixed sample.",
                                f"Median absolute difference: "
                                f"{comparison['Difference'].abs().median():.3f}.",
                            ],
                        )
            legend = dataset.meta.get("legend_text") or []
            if legend:
                self.info("", tone="muted").set_items(
                    "The sheet's own colour legend describes these bands:", legend
                )
        else:
            workbook_chart.show_unavailable(workbook_seasonality)

        self.section(
            "Harvest and lean seasons",
            "Derived by ranking calendar months on arrivals — the workbook's own "
            "harvest signature rather than an external crop calendar.",
        )
        season = service.season_profile()
        season_chart = self.chart("Mean arrivals by month", height=300)
        if season:
            table = season.unwrap()
            values = pd.Series(
                table["mean_arrivals"].to_numpy(),
                index=pd.Index(table["month_name"], name="Month"),
            )
            charts.draw_bar(
                season_chart,
                values,
                ylabel="Mean arrivals (bags)",
                source=season.source,
                notes=list(season.notes),
                reference=float(table["mean_arrivals"].mean()),
                reference_label="All-month average",
                colour_negative=False,
            )
            self.table("Season classification", max_height=330).set_frame(
                table.set_index("month_name").rename_axis("Month"),
                season.source,
                list(season.notes),
            )
        else:
            season_chart.show_unavailable(season)

        self.section("Weekly pattern")
        weekday = analytics.weekday_seasonality(series, source)
        weekday_table = self.table("Day-of-week statistics", max_height=230)
        if weekday:
            weekday_table.set_frame(weekday.unwrap(), source, list(weekday.notes))
        else:
            weekday_table.show_unavailable(weekday)

        self.info(
            f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\nFestival effects cannot be "
            "isolated. Diwali, Sankranti and comparable festivals move through "
            "the Gregorian calendar from year to year, so separating festival "
            "demand from harvest timing requires a festival-date calendar. The "
            "workbook contains none, so the monthly patterns above necessarily "
            "blend the two.",
            tone="warning",
        )


# ==========================================================================
# Page 8 -- Export Analysis
# ==========================================================================


class ExportAnalysisPage(BasePage):
    title = "Export Analysis"
    subtitle = "Export volume, its relationship with price, and the currency channel"

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters
        self.filter_note()

        exports_result = service.exports_monthly()
        if not exports_result:
            self.info(exports_result.message(), tone="danger")
            return
        exports = exports_result.unwrap()  # already month-end indexed
        matrix = service.exports_matrix()

        self.info(
            "The export sheet does not state its unit of measure. Values are "
            "used exactly as supplied and compared only against themselves; no "
            "conversion is applied and no cross-series arithmetic assumes a "
            "unit.",
            tone="warning",
        )

        self.section("Export volume over time")
        trend = self.chart("Monthly exports", height=320)
        charts.draw_line(
            trend,
            {"Exports": exports},
            ylabel="As supplied (unit not stated)",
            source=exports_result.source,
            notes=list(exports_result.notes),
        )

        if matrix:
            frame = matrix.unwrap()
            calendar = self.chart("Exports by month and year", height=360)
            charts.draw_calendar_heatmap(
                calendar,
                frame,
                source=matrix.source,
                notes=[
                    "Blank cells are months the workbook has not yet recorded.",
                ],
                cbar_label="As supplied",
            )

            annual = frame.sum(axis=1, min_count=1)
            complete = frame.notna().all(axis=1)
            annual_chart = self.chart("Annual export total", height=300)
            charts.draw_bar(
                annual_chart,
                annual,
                ylabel="As supplied (unit not stated)",
                source=matrix.source,
                notes=[
                    "Years with any missing month are hatched: their totals are "
                    "partial and must not be compared with complete years.",
                ],
                colour_negative=False,
                value_labels=True,
                projected=[str(y) for y in frame.index[~complete]],
            )

            seasonal_chart = self.chart("Average exports by calendar month", height=290)
            charts.draw_bar(
                seasonal_chart,
                frame.mean(axis=0),
                ylabel="Mean monthly exports",
                source=matrix.source,
                notes=["Averaged across every year present in the sheet."],
                reference=float(frame.mean(axis=0).mean()),
                reference_label="All-month average",
                colour_negative=False,
            )

        variety = PriceAnalysisPage._target_variety(service, filters)
        price_result = service.variety_series(variety) if variety else None
        if not price_result:
            self.info(
                "Export-price analysis needs a price series; none is available.",
                tone="warning",
            )
            return
        price = service.series_at(
            service.apply_filters(price_result.unwrap(), filters, kind="price"), "ME"
        )
        source = f"{exports_result.source}; {price_result.source}"

        self.section(
            "Exports against price",
            "Both on a monthly basis, the finest frequency the export sheet "
            "supports.",
        )
        dual = self.chart("Exports and price", height=330)
        charts.draw_dual_axis(
            dual,
            price,
            exports,
            primary_label=f"{variety} price (INR/quintal)",
            secondary_label="Exports (as supplied)",
            source=source,
            secondary_as_bars=True,
        )

        scatter = self.chart("Exports against price", height=330)
        charts.draw_scatter_with_fit(
            scatter,
            exports,
            price,
            xlabel="Monthly exports (as supplied)",
            ylabel=f"{variety} price (INR/quintal)",
            source=source,
        )

        self.section("Statistical relationship")
        pair = analytics.correlation_pair(price, exports, source)
        pair_table = self.table("Correlation", max_height=200)
        if pair:
            pair_table.set_frame(
                pair.unwrap().to_frame("Value"), source, list(pair.notes)
            )
        else:
            pair_table.show_unavailable(pair)

        cross = analytics.cross_correlation(exports, price, 12, source, differenced=True)
        cross_chart = self.chart("Cross-correlation: exports leading price", height=300)
        if cross:
            table = cross.unwrap()
            charts.draw_stem(
                cross_chart,
                table["correlation"],
                xlabel="Lag in months (positive = exports lead price)",
                ylabel="Correlation of month-on-month changes",
                source=source,
                notes=list(cross.notes),
                band=float(table["upper_95"].iloc[0]),
                highlight_index=int(table["correlation"].abs().idxmax()),
            )
        else:
            cross_chart.show_unavailable(cross)

        reading = analytics.lead_lag(
            exports, price, "Exports", f"{variety} price", "ME", 12, source
        )
        if reading:
            self.info("", tone="info").set_items(
                "Lead-lag reading:", [reading.unwrap().sentence()] + list(reading.notes)
            )

        rolling = analytics.rolling_correlation(price, exports, 24, source)
        rolling_chart = self.chart("24-month rolling correlation", height=290)
        if rolling:
            charts.draw_line(
                rolling_chart,
                {"Rolling correlation": rolling.unwrap()},
                ylabel="Correlation",
                source=source,
                notes=list(rolling.notes)
                + [
                    "A relationship that changes sign over time cannot be "
                    "relied on for positioning.",
                ],
            )
        else:
            rolling_chart.show_unavailable(rolling)

        granger = analytics.granger_causality(
            exports.rename("Exports"), price.rename(f"{variety} price"), 6, source
        )
        granger_table = self.table("Granger causality: exports → price", max_height=230)
        if granger:
            granger_table.set_frame(granger.unwrap(), source, list(granger.notes))
        else:
            granger_table.show_unavailable(granger)

        self.section(
            "Currency channel",
            "A weaker rupee raises the rupee proceeds of an export sale, which "
            "is the mechanism connecting the exchange rate to domestic prices.",
        )
        fx_result = service.usd_inr()
        fx_chart = self.chart("USD/INR and price", height=320)
        if fx_result:
            fx = service.series_at(fx_result.unwrap(), "ME")
            gaps = service.coverage_gaps(fx_result.unwrap())
            notes: list[str] = []
            if not gaps.empty:
                worst = gaps.iloc[0]
                notes.append(
                    f"The exchange-rate sheet has {len(gaps)} gap(s) over 45 "
                    f"days; the largest spans {int(worst['days'])} days "
                    f"({worst['from']} to {worst['to']}). The line is drawn "
                    "across the gap but no value exists inside it."
                )
            charts.draw_dual_axis(
                fx_chart,
                price,
                fx,
                primary_label=f"{variety} price (INR/quintal)",
                secondary_label="USD/INR",
                source=f"{price_result.source}; {fx_result.source}",
                notes=notes,
            )
            if not gaps.empty:
                self.table("Exchange-rate coverage gaps", max_height=160).set_frame(
                    gaps, fx_result.source,
                    ["Any statistic spanning these breaks joins disconnected periods."],
                )
            fx_pair = analytics.correlation_pair(price, fx, f"{price_result.source}; {fx_result.source}")
            if fx_pair:
                self.table("USD/INR against price", max_height=200).set_frame(
                    fx_pair.unwrap().to_frame("Value"),
                    f"{price_result.source}; {fx_result.source}",
                    list(fx_pair.notes)
                    + [
                        "Both series trend upward across this sample, so part of "
                        "this correlation is shared trend rather than a "
                        "mechanism. The Granger test below is the directional "
                        "check.",
                    ],
                )
            fx_granger = analytics.granger_causality(
                fx.rename("USD/INR"), price.rename(f"{variety} price"), 6,
                f"{price_result.source}; {fx_result.source}",
            )
            if fx_granger:
                self.table("Granger causality: USD/INR → price", max_height=230).set_frame(
                    fx_granger.unwrap(),
                    f"{price_result.source}; {fx_result.source}",
                    list(fx_granger.notes),
                )
        else:
            fx_chart.show_unavailable(fx_result)


# ==========================================================================
# Page 9 -- Balance Sheet
# ==========================================================================


class BalanceSheetPage(BasePage):
    title = "Balance Sheet"
    subtitle = "Production, supply, consumption, carry-forward and inventory"

    def build(self, context: PageContext) -> None:
        service = context.service

        balance = service.balance_sheet()
        if not balance:
            self.info(balance.message(), tone="danger")
        else:
            frame = balance.unwrap()
            dataset = service.data.datasets["balance_sheet"]
            projected = dataset.meta.get("projected_years", [])
            self.info(
                f"Units: {dataset.meta.get('unit_note', 'not stated')}. "
                + (
                    f"Year(s) {', '.join(str(y) for y in projected)} are marked "
                    "as expected in the workbook — these are the workbook's own "
                    "projections, not this application's forecasts, and are "
                    "hatched wherever they appear in a chart."
                    if projected
                    else "All years are realised data."
                ),
                tone="warning" if projected else "muted",
            )

            self.section("The balance sheet as supplied")
            self.table("Red chilli balance sheet", max_height=330).set_frame(
                frame, balance.source, list(balance.notes)
            )

            self.section(
                "Supply and demand build-up",
                "Opening stock plus production against consumption plus exports.",
            )
            supply_rows = [
                r for r in frame.index
                if any(k in str(r).lower() for k in ("openingstock", "opening stock", "production", "import"))
            ]
            demand_rows = [
                r for r in frame.index
                if any(k in str(r).lower() for k in ("consumption", "export"))
            ]
            if supply_rows:
                supply_chart = self.chart("Supply components by year", height=320)
                charts.draw_grouped_bar(
                    supply_chart,
                    frame.loc[supply_rows].T,
                    ylabel=dataset.meta.get("unit_note", ""),
                    source=balance.source,
                    notes=[
                        "Stacked to show total supply. "
                        + (
                            f"{', '.join(str(y) for y in projected)} is the "
                            "workbook's expectation."
                            if projected
                            else ""
                        )
                    ],
                    stacked=True,
                )
            if demand_rows:
                demand_chart = self.chart("Demand components by year", height=320)
                charts.draw_grouped_bar(
                    demand_chart,
                    frame.loc[demand_rows].T,
                    ylabel=dataset.meta.get("unit_note", ""),
                    source=balance.source,
                    notes=["Stacked to show total offtake."],
                    stacked=True,
                )

            self.section(
                "Stocks and the stock-to-use ratio",
                "The tightness measure: a thin buffer historically coincides "
                "with firmer, more volatile prices.",
            )
            for keyword, title, ylabel in (
                ("Ending Stock", "Ending stock by year", dataset.meta.get("unit_note", "")),
                ("Stock to Use", "Stock-to-use ratio by year", "%"),
            ):
                row = service.balance_sheet_row(keyword)
                panel = self.chart(title, height=290)
                if row:
                    values = row.unwrap()
                    charts.draw_bar(
                        panel,
                        values,
                        ylabel=ylabel,
                        source=row.source,
                        notes=list(row.notes),
                        reference=float(values.mean()),
                        reference_label="Sample average",
                        colour_negative=False,
                        value_labels=True,
                        value_format="{:,.2f}",
                        projected=[str(y) for y in projected],
                    )
                else:
                    panel.show_unavailable(row)

            # Relate the annual balance to the annual average price.
            price_result = service.variety_series("Teja")
            stock_use = service.balance_sheet_row("Stock to Use")
            if price_result and stock_use:
                annual_price = service.series_at(price_result.unwrap(), "YE")
                # Re-key by calendar year to join the year-indexed balance sheet.
                annual_price = pd.Series(
                    annual_price.to_numpy(),
                    index=pd.Index(annual_price.index.year, name="year"),
                )
                joined = pd.concat(
                    [
                        annual_price.rename("Annual average price"),
                        stock_use.unwrap().rename("Stock-to-use %"),
                    ],
                    axis=1,
                ).dropna()
                comparison = self.chart(
                    "Stock-to-use ratio against annual average price", height=320
                )
                if len(joined) >= 4:
                    charts.draw_scatter_with_fit(
                        comparison,
                        joined["Stock-to-use %"],
                        joined["Annual average price"],
                        xlabel="Stock-to-use ratio (%)",
                        ylabel="Annual average Teja price (INR/quintal)",
                        source=f"{balance.source}; {price_result.source}",
                        notes=[
                            f"Only {len(joined)} paired years are available. "
                            "That is far too few for a significance test or a "
                            "regression — read the fit as a directional "
                            "observation, not a finding.",
                        ],
                        colour_by_date=False,
                    )
                    self.table("Paired annual data", max_height=280).set_frame(
                        joined, f"{balance.source}; {price_result.source}",
                        ["Calendar-year average of the daily Teja quotes."],
                    )
                else:
                    comparison.show_unavailable(
                        f"Only {len(joined)} year(s) have both a balance-sheet "
                        "ratio and a price average."
                    )

        self.section(
            "Area, production and yield (APY)",
            "State-level crop statistics as supplied in the workbook.",
        )
        apy = service.apy()
        if apy:
            frame = apy.unwrap()
            dataset = service.data.datasets["apy"]
            projected = dataset.meta.get("projected_years", [])
            self.table("APY by state and year", max_height=360).set_frame(
                frame,
                apy.source,
                [
                    f"Year(s) marked expected in the workbook: "
                    f"{', '.join(str(y) for y in projected) or 'none'}.",
                    "The most recent expected year carries an area figure only; "
                    "production and yield are absent and are shown as blank "
                    "rather than estimated.",
                ],
            )
            national = dataset.meta.get("national_row")
            for metric, unit in (
                ("Production (MT)", "tonnes"),
                ("Area (Ha)", "hectares"),
                ("Yield (t/Ha)", "tonnes per hectare"),
            ):
                if metric not in frame.columns:
                    continue
                panel = self.chart(f"{metric} by year", height=300)
                pivot = frame.pivot_table(index="year", columns="state", values=metric)
                if national and national in pivot.columns:
                    charts.draw_bar(
                        panel,
                        pivot[national].dropna(),
                        ylabel=f"{metric} ({unit})",
                        source=apy.source,
                        notes=[
                            f"All-India total ('{national}' row). State detail "
                            "is in the table above.",
                        ],
                        colour_negative=False,
                        value_labels=False,
                        projected=[str(y) for y in projected],
                    )
                else:
                    states = [c for c in pivot.columns if c != national]
                    charts.draw_grouped_bar(
                        panel,
                        pivot[states],
                        ylabel=f"{metric} ({unit})",
                        source=apy.source,
                        stacked=True,
                    )
        else:
            self.info(apy.message(), tone="warning")

        self.section(
            "Cold storage stock",
            "Reported positions by state and market.",
        )
        cold = service.cold_storage()
        cold_table = self.table("Cold storage stock as reported", max_height=280)
        if cold:
            frame = cold.unwrap()
            dataset = service.data.datasets["cold_storage_stock"]
            cold_table.set_frame(
                frame,
                cold.source,
                list(cold.notes) + [dataset.meta.get("unit", "")],
            )
            self.info(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n"
                f"Inventory-versus-price analysis, delayed storage effects and "
                f"seasonal storage behaviour cannot be computed. The sheet holds "
                f"{len(frame)} reporting month(s) across {frame.shape[1]} "
                f"location(s), with at most "
                f"{max(dataset.meta.get('observations_per_location', {}).values(), default=0)} "
                f"observation(s) for any single location — against the "
                f"{settings.ANALYTICS.min_obs_correlation} minimum this "
                "application requires before reporting any correlation. The "
                "levels above are shown for reference only.\n\n"
                "The one storage relationship the workbook does support is the "
                "Khammam cold-storage price premium over fresh lots, which is "
                "computed from the two Khammam price sheets and reported on the "
                "Automated Insights page.",
                tone="warning",
            )

            cold_chart = self.chart("Reported stock by location", height=300)
            longest = max(
                dataset.meta.get("observations_per_location", {}).items(),
                key=lambda kv: kv[1],
                default=(None, 0),
            )
            if longest[0] and longest[1] >= 2:
                charts.draw_bar(
                    cold_chart,
                    frame[longest[0]].dropna(),
                    ylabel="Bags",
                    source=cold.source,
                    notes=[
                        f"'{longest[0]}' is the best-covered column with "
                        f"{longest[1]} observation(s). Shown as discrete "
                        "readings, not a time series, because the reporting is "
                        "too sparse to join into a line.",
                    ],
                    colour_negative=False,
                    value_labels=True,
                )
            else:
                cold_chart.show_unavailable(
                    "No location has even two observations, so nothing can be "
                    "plotted."
                )
        else:
            cold_table.show_unavailable(cold)


# ==========================================================================
# Page 10 -- Forecast Center
# ==========================================================================


class ForecastCenterPage(BasePage):
    title = "Forecast Center"
    subtitle = "Weekly, fortnightly and monthly projections to six months, with model comparison"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._comparison: forecasting.ModelComparison | None = None
        self._exog: pd.DataFrame | None = None

    def build(self, context: PageContext) -> None:
        service, filters = context.service, context.filters

        self.section(
            "Forecast controls",
            "Choose the target, the frequency and the horizon, then run the "
            "model sweep. Every applicable model is fitted, backtested and "
            "ranked; the winner is selected on out-of-sample error.",
        )
        controls = QtWidgets.QFrame()
        controls.setObjectName("controlPanel")
        grid = QtWidgets.QGridLayout(controls)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(9)

        varieties = service.varieties()
        focus = service.focus_varieties()
        default_variety = next(iter(focus.values()), varieties[0] if varieties else "")

        self.variety_combo = QtWidgets.QComboBox()
        self.variety_combo.addItems(varieties)
        if default_variety in varieties:
            self.variety_combo.setCurrentText(default_variety)

        self.freq_combo = QtWidgets.QComboBox()
        for alias in ("W", FORTNIGHT_FREQ, "ME"):
            self.freq_combo.addItem(
                settings.FORECAST.frequency_labels.get(alias, alias), alias
            )
        self.freq_combo.setCurrentIndex(2)

        self.horizon_spin = QtWidgets.QSpinBox()
        self.horizon_spin.setRange(1, 60)
        self.horizon_spin.setValue(settings.FORECAST.horizons["ME"])
        self.horizon_spin.setToolTip(
            "Number of periods ahead. The default reaches six months at the "
            "selected frequency."
        )

        self.history_spin = QtWidgets.QSpinBox()
        self.history_spin.setRange(0, 5000)
        self.history_spin.setValue(0)
        self.history_spin.setSpecialValueText("All available")
        self.history_spin.setToolTip(
            "Restrict training to the most recent N periods. 0 uses the full "
            "filtered history."
        )

        self.model_list = QtWidgets.QListWidget()
        self.model_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        for name in ("ARIMA", "SARIMA", "SARIMAX", "Holt-Winters", "VAR", "VECM"):
            item = QtWidgets.QListWidgetItem(name)
            self.model_list.addItem(item)
            item.setSelected(True)
        self.model_list.setMaximumHeight(112)
        self.model_list.setToolTip(
            "Models to attempt. Any that cannot be applied to the data are "
            "listed with the reason."
        )

        self.run_button = QtWidgets.QPushButton("Run model sweep")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._run)

        self.freq_combo.currentIndexChanged.connect(self._sync_horizon)

        for column, (label, widget) in enumerate(
            (
                ("Variety", self.variety_combo),
                ("Frequency", self.freq_combo),
                ("Horizon (periods)", self.horizon_spin),
                ("Training window", self.history_spin),
            )
        ):
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("controlLabel")
            grid.addWidget(caption, 0, column)
            grid.addWidget(widget, 1, column)
        models_label = QtWidgets.QLabel("Models")
        models_label.setObjectName("controlLabel")
        grid.addWidget(models_label, 0, 4)
        grid.addWidget(self.model_list, 1, 4, 2, 1)
        grid.addWidget(self.run_button, 2, 0, 1, 2)
        controls.setStyleSheet(
            f"""
            QFrame#controlPanel {{
                background-color: {context.theme.surface};
                border: 1px solid {context.theme.border};
                border-radius: 8px;
            }}
            QLabel#controlLabel {{
                color: {context.theme.text_muted}; font-size: 10px; font-weight: 600;
            }}
            """
        )
        self.body.addWidget(controls)

        self.status_box = self.info(
            "Press “Run model sweep” to fit and compare every applicable model. "
            "A monthly run takes a few seconds; weekly takes longer.",
            tone="muted",
        )

        self.forecast_panel = self.chart("Forecast", height=400)
        self.forecast_panel.show_unavailable(
            "No forecast has been run yet. Configure the controls above and "
            "press “Run model sweep”."
        )

        self.selection_box = self.info("", tone="info")
        self.selection_box.setVisible(False)

        self.comparison_table = self.new_table("Model comparison and backtest scores")
        self.body.addWidget(self.comparison_table)
        self.comparison_table.setVisible(False)

        self.forecast_table = self.new_table("Forecast table")
        self.body.addWidget(self.forecast_table)
        self.forecast_table.setVisible(False)

        self.skipped_box = self.info("", tone="warning")
        self.skipped_box.setVisible(False)

        self.explain_header = SectionHeader(
            "Forecast explanation",
            "Why the model projects what it projects, in plain language.",
            context.theme,
        )
        self.body.addWidget(self.track(self.explain_header))
        self.explain_header.setVisible(False)

        self.explain_box = self.info("", tone="info")
        self.explain_box.setVisible(False)

        self.assumptions_box = self.info("", tone="warning")
        self.assumptions_box.setVisible(False)

        self.components_panel = self.chart("Historical trend, seasonal and residual components", height=430)
        self.components_panel.setVisible(False)

        self.drivers_table = self.new_table("Driver attribution")
        self.body.addWidget(self.drivers_table)
        self.drivers_table.setVisible(False)

        self.vif_table = self.new_table("Driver collinearity (VIF)")
        self.body.addWidget(self.vif_table)
        self.vif_table.setVisible(False)

        self.stationarity_table = self.new_table("Stationarity diagnosis")
        self.body.addWidget(self.stationarity_table)
        self.stationarity_table.setVisible(False)

        self._sync_horizon()

    def _sync_horizon(self) -> None:
        alias = self.freq_combo.currentData()
        self.horizon_spin.setValue(settings.FORECAST.horizons.get(alias, 6))

    # -- run --------------------------------------------------------------

    def _run(self) -> None:
        if not self.context:
            return
        service, filters = self.context.service, self.context.filters
        variety = self.variety_combo.currentText()
        freq = self.freq_combo.currentData()
        horizon = int(self.horizon_spin.value())
        window = int(self.history_spin.value())
        models = [i.text() for i in self.model_list.selectedItems()] or None

        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        self.status_box.setText("Preparing the series…")

        self.context.submit(
            self._compute,
            service,
            filters,
            variety,
            freq,
            horizon,
            window,
            models,
            wants_progress=True,
            on_done=self.bind(self._render),
            on_fail=self.bind(self._failed),
            on_progress=self.bind(
                lambda message, percent: self.status_box.setText(
                    f"{message}  ({percent}%)"
                )
            ),
        )

    @staticmethod
    def _compute(
        service: DataService,
        filters: FilterState,
        variety: str,
        freq: str,
        horizon: int,
        window: int,
        models: Sequence[str] | None,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        """Fit the sweep and assemble the explanation (worker thread)."""
        result = service.variety_series(variety)
        if not result:
            return {"error": result.reason}
        raw = service.apply_filters(result.unwrap(), filters, kind="price")
        series = service.series_at(raw, freq)
        series.name = variety
        history_notes = [service.partial_last_period(raw, freq)]
        if window:
            if len(series) > window:
                history_notes.append(
                    f"Training restricted to the most recent {window} period(s) "
                    f"of {len(series)} available, as selected."
                )
            series = series.tail(window)

        if series.empty:
            return {"error": "The selected series is empty after filtering."}

        exog = service.exogenous_matrix(freq)
        panel = service.variety_panel(freq, filters)
        panel_frame = panel.unwrap() if panel else None
        if panel_frame is not None and variety in panel_frame.columns:
            # Put the target first and keep companions with enough overlap.
            companions = [
                c for c in panel_frame.columns
                if c != variety and panel_frame[c].notna().sum() >= len(series) * 0.6
            ][:3]
            panel_frame = panel_frame[[variety] + companions]

        comparison = forecasting.run_all_models(
            series,
            freq,
            horizon,
            target_name=variety,
            exog=exog.unwrap() if exog else None,
            panel=panel_frame,
            source=result.source,
            progress=progress,
            models=models,
            history_notes=history_notes,
        )
        explanation = (
            forecasting.explain(
                comparison.best, exog.unwrap() if exog else None, result.source
            )
            if comparison.best
            else None
        )
        return {
            "comparison": comparison,
            "explanation": explanation,
            "exog": exog.unwrap() if exog else None,
            "exog_source": exog.source if exog else "",
            "exog_reason": "" if exog else exog.reason,
            "variety": variety,
            "freq": freq,
        }

    def _render(self, payload: dict[str, Any]) -> None:
        self.run_button.setEnabled(True)
        self.run_button.setText("Run model sweep")
        if not self.context:
            return
        if payload.get("error"):
            self.status_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{payload['error']}"
            )
            return

        comparison: forecasting.ModelComparison = payload["comparison"]
        self._comparison = comparison
        variety = payload["variety"]
        freq = payload["freq"]
        frequency_label = settings.FORECAST.frequency_labels.get(freq, freq)

        self.comparison_table.setVisible(True)
        self.comparison_table.set_frame(
            comparison.comparison_table(),
            comparison.source,
            [
                f"Ranked by out-of-sample {settings.FORECAST.selection_metric} from "
                f"{settings.FORECAST.backtest_folds}-fold rolling-origin "
                "backtesting. Lower RMSE, MAE and MAPE are better; higher R² "
                "and directional accuracy are better.",
                "A negative R² means a flat line at the mean of the held-out "
                "window would have scored better than the model.",
            ],
        )

        if comparison.skipped:
            self.skipped_box.setVisible(True)
            self.skipped_box.set_items(
                "Models not applied to this series, and why:",
                [f"{name}: {reason}" for name, reason in comparison.skipped],
            )
        else:
            self.skipped_box.setVisible(False)

        best = comparison.best
        if best is None:
            self.status_box.setText(
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{comparison.selection_reason}"
            )
            self.forecast_panel.show_unavailable(comparison.selection_reason)
            for widget in (
                self.forecast_table, self.selection_box, self.explain_header,
                self.explain_box, self.assumptions_box, self.components_panel,
                self.drivers_table, self.vif_table, self.stationarity_table,
            ):
                widget.setVisible(False)
            return

        self.status_box.setText(
            f"{variety} — {frequency_label.lower()} forecast, "
            f"{len(best.forecast)} period(s) to "
            f"{best.forecast.index[-1]:%d %b %Y}. "
            f"{len(comparison.results)} model(s) fitted, "
            f"{len(comparison.skipped)} skipped."
        )

        charts.draw_forecast(
            self.forecast_panel,
            best.history,
            best.forecast,
            conf=(best.conf_lower, best.conf_upper),
            pred=(best.pred_lower, best.pred_upper),
            label=best.label,
            ylabel="INR per quintal",
            source=best.source,
            notes=best.notes,
            history_window=min(len(best.history), max(60, len(best.forecast) * 8)),
        )
        self.forecast_panel.set_title(
            f"{variety} — {frequency_label} forecast · {best.label}"
        )

        self.selection_box.setVisible(True)
        self.selection_box.setText(
            f"Model selection\n\n{comparison.selection_reason}"
        )

        self.forecast_table.setVisible(True)
        self.forecast_table.set_frame(
            best.table(),
            best.source,
            [
                f"Confidence interval is the model's analytic "
                f"{settings.FORECAST.confidence_level:.0%} band. Prediction "
                f"interval is the {settings.FORECAST.prediction_level:.0%} band "
                "after widening for the error the model actually made in "
                "backtesting — plan against the prediction interval, not the "
                "single line.",
                "All rows are projections. No row is historical data.",
            ],
        )

        explanation: forecasting.ForecastExplanation | None = payload.get("explanation")
        if explanation is None:
            return

        self.explain_header.setVisible(True)
        self.explain_box.setVisible(True)
        self.explain_box.set_items(
            explanation.headline, explanation.plain_language
        )

        self.assumptions_box.setVisible(True)
        self.assumptions_box.set_items(
            "Assumptions and caveats behind this forecast:", explanation.assumptions
        )

        if explanation.components:
            self.components_panel.setVisible(True)
            charts.draw_decomposition(
                self.components_panel,
                explanation.components.unwrap(),
                source=explanation.components.source,
                notes=list(explanation.components.notes),
            )
        else:
            self.components_panel.setVisible(True)
            self.components_panel.show_unavailable(explanation.components)

        if explanation.drivers:
            payload_drivers = explanation.drivers.unwrap()
            self.drivers_table.setVisible(True)
            self.drivers_table.set_frame(
                payload_drivers["coefficients"],
                payload.get("exog_source", ""),
                list(explanation.drivers.notes)
                + [
                    f"Model R² {payload_drivers['r_squared']:.3f} "
                    f"(adjusted {payload_drivers['adj_r_squared']:.3f}) on "
                    f"{payload_drivers['n_obs']} observations; "
                    f"overall F-test p={fmt_pvalue(payload_drivers['f_pvalue'])}.",
                ],
            )
            vif = payload_drivers.get("vif")
            if vif is not None and vif:
                self.vif_table.setVisible(True)
                self.vif_table.set_frame(
                    vif.unwrap(), payload.get("exog_source", ""), list(vif.notes)
                )
            else:
                self.vif_table.setVisible(True)
                self.vif_table.show_unavailable(
                    vif if vif is not None else "VIF was not computed."
                )
        else:
            self.drivers_table.setVisible(True)
            self.drivers_table.show_unavailable(explanation.drivers)

        if explanation.stationarity:
            self.stationarity_table.setVisible(True)
            self.stationarity_table.set_frame(
                explanation.stationarity.unwrap(),
                explanation.stationarity.source,
                list(explanation.stationarity.notes),
            )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.run_button.setText("Run model sweep")
        self.status_box.setText(f"The model sweep failed.\n\n{message}")


# ==========================================================================
# Page 11 -- Automated Insights
# ==========================================================================


class AutomatedInsightsPage(BasePage):
    title = "Automated Insights"
    subtitle = "Every finding the workbook supports, ranked by strength of evidence"

    def build(self, context: PageContext) -> None:
        self.filter_note()
        self.info(
            "Findings are generated from the workbook only. Each carries the "
            "statistic behind it and the sheet it came from. Items marked "
            "DATA GAP record questions the workbook cannot answer — they are "
            "findings too.",
            tone="info",
        )

        toolbar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label = QtWidgets.QLabel("Show:")
        label.setStyleSheet(f"color: {context.theme.text_muted}; font-size: 11px;")
        row.addWidget(label)
        self.strength_filter = QtWidgets.QComboBox()
        self.strength_filter.addItems(
            ["All findings", "Strong only", "Strong and moderate", "Data gaps only"]
        )
        self.strength_filter.setCurrentIndex(0)
        self.strength_filter.currentIndexChanged.connect(self._apply_filter)
        row.addWidget(self.strength_filter)
        self.category_filter = QtWidgets.QComboBox()
        self.category_filter.addItem("All categories")
        self.category_filter.currentIndexChanged.connect(self._apply_filter)
        row.addWidget(self.category_filter)
        row.addStretch(1)
        self.body.addWidget(toolbar)

        self.status_box = self.info("Generating insights…", tone="muted")

        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.body.addWidget(self.cards_container)

        self.failures_box = self.info("", tone="warning")
        self.failures_box.setVisible(False)

        self._insights: list[insights.Insight] = []
        variety = PriceAnalysisPage._target_variety(context.service, context.filters)
        context.submit(
            insights.generate_all,
            context.service,
            context.filters,
            variety,
            wants_progress=True,
            on_done=self.bind(self._render),
            on_fail=self.bind(self._failed),
            on_progress=self.bind(
                lambda message, percent: self.status_box.setText(
                    f"{message}  ({percent}%)"
                )
            ),
        )

    def _render(self, payload: tuple[list[insights.Insight], list[str]]) -> None:
        found, failures = payload
        self._insights = found
        counts: dict[str, int] = {}
        for insight in found:
            counts[insight.strength] = counts.get(insight.strength, 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in counts.items())
        self.status_box.setText(f"{len(found)} finding(s): {summary}.")

        categories = sorted({i.category for i in found})
        self.category_filter.blockSignals(True)
        self.category_filter.clear()
        self.category_filter.addItem("All categories")
        self.category_filter.addItems(categories)
        self.category_filter.blockSignals(False)

        if failures:
            self.failures_box.setVisible(True)
            self.failures_box.set_items(
                "Some generators could not complete:", failures
            )
        self._apply_filter()

    def _apply_filter(self) -> None:
        if not self.context:
            return
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        mode = self.strength_filter.currentText()
        allowed = {
            "All findings": None,
            "Strong only": {"strong"},
            "Strong and moderate": {"strong", "moderate"},
            "Data gaps only": {"data gap"},
        }[mode]
        category = self.category_filter.currentText()

        shown = 0
        for insight in self._insights:
            if allowed is not None and insight.strength not in allowed:
                continue
            if category not in ("All categories", "") and insight.category != category:
                continue
            self.cards_layout.addWidget(
                self.track(InsightCard(insight, self.context.theme))
            )
            shown += 1
        if shown == 0:
            self.cards_layout.addWidget(
                self.track(
                    InfoBox(
                        "No finding matches the current selection.",
                        self.context.theme,
                        tone="muted",
                    )
                )
            )

    def _failed(self, message: str) -> None:
        self.status_box.setText(f"Insight generation failed.\n\n{message}")


# ==========================================================================
# Page 12 -- Data Dictionary
# ==========================================================================


class DataDictionaryPage(BasePage):
    title = "Data Dictionary"
    subtitle = "Auto-generated from the workbook's sheets and columns"

    def build(self, context: PageContext) -> None:
        service = context.service
        data = service.data

        self.info(
            f"Workbook: {data.path}\n"
            f"Read at {data.loaded_at:%d %b %Y %H:%M:%S} in "
            f"{data.load_seconds:.2f}s.\n"
            f"Worksheets in file: {len(data.raw_shapes)} · mapped to analyses: "
            f"{len(data.datasets)} · unmapped: {len(data.unmapped_sheets)}.",
            tone="info",
        )

        export_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(export_row)
        row.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton("Export data dictionary as Markdown")
        button.setObjectName("primaryButton")
        button.clicked.connect(self._export_markdown)
        row.addWidget(button)
        row.addStretch(1)
        self.body.addWidget(export_row)

        self.section("Field-level dictionary")
        self.table("Data dictionary", max_height=520).set_frame(
            build_data_dictionary(data),
            data.path.name,
            [
                "Generated at runtime by inspecting the workbook. 'Populated' "
                "is the share of rows with a value; 'Role' is inferred from the "
                "column header.",
            ],
        )

        self.section("Dataset coverage")
        self.table("Coverage by dataset", max_height=340).set_frame(
            service.coverage_table(), data.path.name
        )

        self.section("Units and conversions read from the workbook")
        unit_rows: list[dict[str, Any]] = []
        for key, dataset in data.datasets.items():
            for label, weight in (dataset.meta.get("bag_weights_kg") or {}).items():
                unit_rows.append(
                    {
                        "Sheet": dataset.sheet_name,
                        "Field": label,
                        "Conversion": f"1 bag = {weight:g} kg",
                        "Read from": "column header text",
                    }
                )
            for note_key in ("primary_unit", "secondary_unit", "unit_note", "unit", "price_unit"):
                note = dataset.meta.get(note_key)
                if note:
                    unit_rows.append(
                        {
                            "Sheet": dataset.sheet_name,
                            "Field": note_key.replace("_", " "),
                            "Conversion": str(note),
                            "Read from": "sheet annotation",
                        }
                    )
        units_table = self.table("Units and conversions", max_height=340)
        if unit_rows:
            units_table.set_frame(
                pd.DataFrame(unit_rows),
                data.path.name,
                [
                    "No conversion is assumed anywhere in this application. "
                    "Where a sheet states no unit, quantities are reported in "
                    "the sheet's own terms and comparisons are confined to that "
                    "sheet.",
                ],
            )
        else:
            units_table.show_unavailable("No unit annotation was found in any sheet.")

        if data.unmapped_sheets:
            self.section("Unmapped worksheets")
            self.table("Present in the file but unused", max_height=200).set_frame(
                pd.DataFrame(
                    [
                        {
                            "Sheet": name,
                            "Rows": data.raw_shapes.get(name, (0, 0))[0],
                            "Columns": data.raw_shapes.get(name, (0, 0))[1],
                        }
                        for name in data.unmapped_sheets
                    ]
                ),
                data.path.name,
                ["No analysis in this application reads these sheets."],
            )

        quality = service.data_quality_notes()
        if quality:
            self.section("Data quality")
            self.info("", tone="warning").set_items(
                "Coverage limitations detected on load:", quality
            )
        if data.warnings:
            self.info("", tone="warning").set_items("Parse warnings:", data.warnings)

    def _export_markdown(self) -> None:
        if not self.context:
            return
        directory = ensure_dir(settings.DOCS_DIR)
        suggested = str(directory / "DATA_DICTIONARY.md")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data dictionary", suggested, "Markdown (*.md)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                data_dictionary_markdown(self.context.service.data), encoding="utf-8"
            )
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self, "Export complete", f"Data dictionary written to:\n{path}"
        )


# ==========================================================================
# The main window
# ==========================================================================

PAGE_CLASSES: dict[str, type[BasePage]] = {
    "executive": ExecutiveSummaryPage,
    "overview": MarketOverviewPage,
    "price": PriceAnalysisPage,
    "arrivals": ArrivalAnalysisPage,
    "integration": MarketIntegrationPage,
    "correlation": CorrelationStudioPage,
    "seasonality": SeasonalityPage,
    "exports": ExportAnalysisPage,
    "balance": BalanceSheetPage,
    "forecast": ForecastCenterPage,
    "insights": AutomatedInsightsPage,
    "dictionary": DataDictionaryPage,
}


class MainWindow(QtWidgets.QMainWindow):
    """The application shell."""

    def __init__(self, data: WorkbookData, theme_name: str = settings.DEFAULT_THEME) -> None:
        super().__init__()
        self.data = data
        self.service = DataService(data)
        self.theme = settings.THEMES.get(theme_name, settings.DARK_THEME)
        self.filters = default_filters(self.service)
        self.pool = QtCore.QThreadPool.globalInstance()
        self._active_workers: list[Worker] = []

        self.setWindowTitle(
            f"{settings.APP_NAME} {settings.APP_VERSION} — {data.path.name}"
        )
        self.setMinimumSize(settings.WINDOW_MIN_WIDTH, settings.WINDOW_MIN_HEIGHT)
        self.resize(settings.WINDOW_DEFAULT_WIDTH, settings.WINDOW_DEFAULT_HEIGHT)

        self._build_ui()
        self.apply_theme(self.theme)
        self._select_page("executive")

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = self._build_sidebar()
        outer.addWidget(self.sidebar)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.header = self._build_header()
        right_layout.addWidget(self.header)

        self.stack = QtWidgets.QStackedWidget()
        self.pages: dict[str, BasePage] = {}
        for key, _label, _icon in settings.NAV_ITEMS:
            page = PAGE_CLASSES[key]()
            self.pages[key] = page
            self.stack.addWidget(page)
        right_layout.addWidget(self.stack, 1)

        outer.addWidget(right, 1)
        self.setCentralWidget(central)

        # -- status bar ---------------------------------------------------
        status = self.statusBar()
        self.status_message = QtWidgets.QLabel()
        status.addWidget(self.status_message, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        status.addPermanentWidget(self.progress)
        self.workbook_label = QtWidgets.QLabel()
        status.addPermanentWidget(self.workbook_label)
        self._refresh_status()

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(settings.SIDEBAR_WIDTH)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand = QtWidgets.QWidget()
        brand_layout = QtWidgets.QVBoxLayout(brand)
        brand_layout.setContentsMargins(16, 16, 16, 12)
        brand_layout.setSpacing(1)
        title = QtWidgets.QLabel("🌶  CHILLI INTELLIGENCE")
        title.setObjectName("brandTitle")
        brand_layout.addWidget(title)
        subtitle = QtWidgets.QLabel(f"Desktop {settings.APP_VERSION} · {settings.ORG_NAME}")
        subtitle.setObjectName("brandSubtitle")
        brand_layout.addWidget(subtitle)
        layout.addWidget(brand)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 0, 8, 8)
        inner_layout.setSpacing(2)

        self.nav_buttons: dict[str, QtWidgets.QPushButton] = {}
        for key, label, icon in settings.NAV_ITEMS:
            button = QtWidgets.QPushButton(f"  {icon}   {label}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, k=key: self._select_page(k))
            inner_layout.addWidget(button)
            self.nav_buttons[key] = button

        inner_layout.addSpacing(10)
        filters_label = QtWidgets.QLabel("GLOBAL FILTERS")
        filters_label.setObjectName("sidebarSection")
        inner_layout.addWidget(filters_label)
        inner_layout.addWidget(self._build_filter_controls())
        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        footer = QtWidgets.QWidget()
        footer_layout = QtWidgets.QVBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 10)
        footer_layout.setSpacing(5)
        self.theme_button = QtWidgets.QPushButton()
        self.theme_button.setObjectName("secondaryButton")
        self.theme_button.clicked.connect(self.toggle_theme)
        footer_layout.addWidget(self.theme_button)
        reload_button = QtWidgets.QPushButton("↻  Reload workbook")
        reload_button.setObjectName("secondaryButton")
        reload_button.clicked.connect(self.reload_workbook)
        footer_layout.addWidget(reload_button)
        layout.addWidget(footer)
        return sidebar

    def _build_filter_controls(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(container)
        form.setContentsMargins(2, 4, 2, 4)
        form.setSpacing(7)

        start, end = self.service.full_date_span()
        span_start = start or pd.Timestamp("2014-01-01")
        span_end = end or pd.Timestamp.now()

        def caption(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setObjectName("filterLabel")
            return label

        form.addWidget(caption("Date range"))
        self.start_edit = QtWidgets.QDateEdit()
        self.start_edit.setCalendarPopup(True)
        self.start_edit.setDisplayFormat("dd MMM yyyy")
        self.start_edit.setDateRange(
            QtCore.QDate(span_start.year, span_start.month, span_start.day),
            QtCore.QDate(span_end.year, span_end.month, span_end.day),
        )
        self.start_edit.setDate(
            QtCore.QDate(span_start.year, span_start.month, span_start.day)
        )
        form.addWidget(self.start_edit)
        self.end_edit = QtWidgets.QDateEdit()
        self.end_edit.setCalendarPopup(True)
        self.end_edit.setDisplayFormat("dd MMM yyyy")
        self.end_edit.setDateRange(
            QtCore.QDate(span_start.year, span_start.month, span_start.day),
            QtCore.QDate(span_end.year, span_end.month, span_end.day),
        )
        self.end_edit.setDate(QtCore.QDate(span_end.year, span_end.month, span_end.day))
        form.addWidget(self.end_edit)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(4)
        for label, years in (("1Y", 1), ("3Y", 3), ("5Y", 5), ("All", 0)):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _c=False, y=years: self._apply_date_preset(y))
            preset_row.addWidget(button)
        form.addLayout(preset_row)

        form.addWidget(caption("Analysis frequency"))
        self.freq_combo = QtWidgets.QComboBox()
        for alias in ("D", "W", FORTNIGHT_FREQ, "ME"):
            label = {"D": "Daily"}.get(
                alias, settings.FORECAST.frequency_labels.get(alias, alias)
            )
            self.freq_combo.addItem(label, alias)
        self.freq_combo.setCurrentIndex(1)
        form.addWidget(self.freq_combo)

        form.addWidget(caption("Varieties (none = all)"))
        self.variety_list = QtWidgets.QListWidget()
        self.variety_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        for name in self.service.varieties():
            self.variety_list.addItem(QtWidgets.QListWidgetItem(name))
        self.variety_list.setMaximumHeight(112)
        form.addWidget(self.variety_list)

        form.addWidget(caption("Market"))
        self.market_combo = QtWidgets.QComboBox()
        self.market_combo.addItem("All markets", "")
        for market in self.service.markets():
            self.market_combo.addItem(market, market)
        form.addWidget(self.market_combo)

        form.addWidget(caption("Season (months)"))
        self.month_list = QtWidgets.QListWidget()
        self.month_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        for index, abbreviation in enumerate(settings.MONTH_ABBREVIATIONS, start=1):
            item = QtWidgets.QListWidgetItem(abbreviation.title())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            self.month_list.addItem(item)
        self.month_list.setMaximumHeight(96)
        form.addWidget(self.month_list)

        season_row = QtWidgets.QHBoxLayout()
        season_row.setSpacing(4)
        for label, tip in (("Peak", "Peak arrivals"), ("Lean", "Lean arrivals"), ("Clear", "")):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setObjectName("presetButton")
            button.setToolTip(
                f"Select the months the workbook's arrivals data classifies as "
                f"'{tip}'." if tip else "Clear the month selection."
            )
            button.clicked.connect(lambda _c=False, s=tip: self._apply_season_preset(s))
            season_row.addWidget(button)
        form.addLayout(season_row)

        form.addWidget(caption("Price range (INR/quintal)"))
        price_row = QtWidgets.QHBoxLayout()
        price_row.setSpacing(4)
        self.price_min = QtWidgets.QSpinBox()
        self.price_min.setRange(0, 1_000_000)
        self.price_min.setSingleStep(500)
        self.price_min.setSpecialValueText("min")
        self.price_max = QtWidgets.QSpinBox()
        self.price_max.setRange(0, 1_000_000)
        self.price_max.setSingleStep(500)
        self.price_max.setSpecialValueText("max")
        price_row.addWidget(self.price_min)
        price_row.addWidget(self.price_max)
        form.addLayout(price_row)

        form.addWidget(caption("Arrival range (bags)"))
        arrival_row = QtWidgets.QHBoxLayout()
        arrival_row.setSpacing(4)
        self.arrival_min = QtWidgets.QSpinBox()
        self.arrival_min.setRange(0, 100_000_000)
        self.arrival_min.setSingleStep(5000)
        self.arrival_min.setSpecialValueText("min")
        self.arrival_max = QtWidgets.QSpinBox()
        self.arrival_max.setRange(0, 100_000_000)
        self.arrival_max.setSingleStep(5000)
        self.arrival_max.setSpecialValueText("max")
        arrival_row.addWidget(self.arrival_min)
        arrival_row.addWidget(self.arrival_max)
        form.addLayout(arrival_row)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(5)
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.setObjectName("primaryButton")
        apply_button.clicked.connect(self.apply_filters)
        button_row.addWidget(apply_button)
        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setObjectName("secondaryButton")
        reset_button.clicked.connect(self.reset_filters)
        button_row.addWidget(reset_button)
        form.addLayout(button_row)
        return container

    def _build_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(10)

        text_column = QtWidgets.QVBoxLayout()
        text_column.setSpacing(1)
        self.page_title = QtWidgets.QLabel()
        self.page_title.setObjectName("pageTitle")
        text_column.addWidget(self.page_title)
        self.page_subtitle = QtWidgets.QLabel()
        self.page_subtitle.setObjectName("pageSubtitle")
        self.page_subtitle.setWordWrap(True)
        text_column.addWidget(self.page_subtitle)
        layout.addLayout(text_column, 1)

        self.latest_label = QtWidgets.QLabel()
        self.latest_label.setObjectName("headerMeta")
        self.latest_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.latest_label)
        return header

    # -- navigation and filters -------------------------------------------

    def _select_page(self, key: str) -> None:
        for name, button in self.nav_buttons.items():
            button.setChecked(name == key)
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        self.page_title.setText(page.title)
        self.page_subtitle.setText(page.subtitle)
        QtCore.QTimer.singleShot(0, lambda: self._render_page(page))

    def _render_page(self, page: BasePage) -> None:
        self.set_progress("Rendering…", 10)
        try:
            page.render(self._context())
        finally:
            self.set_progress("", 100)

    def _context(self) -> PageContext:
        return PageContext(
            service=self.service,
            filters=self.filters,
            theme=self.theme,
            progress=self.set_progress,
            submit=self.submit,
        )

    def _apply_date_preset(self, years: int) -> None:
        start, end = self.service.full_date_span()
        if end is None:
            return
        target = (
            pd.Timestamp(start) if years == 0 else pd.Timestamp(end) - pd.DateOffset(years=years)
        )
        if start is not None:
            target = max(pd.Timestamp(start), target)
        self.start_edit.setDate(QtCore.QDate(target.year, target.month, target.day))
        self.end_edit.setDate(QtCore.QDate(end.year, end.month, end.day))
        self.apply_filters()

    def _apply_season_preset(self, season: str) -> None:
        self.month_list.clearSelection()
        if not season:
            self.apply_filters()
            return
        profile = self.service.season_profile()
        if not profile:
            QtWidgets.QMessageBox.information(
                self,
                "Season presets unavailable",
                f"{settings.DATA_UNAVAILABLE_MESSAGE}\n\n{profile.reason}",
            )
            return
        table = profile.unwrap()
        wanted = set(table[table["season"] == season].index.tolist())
        for row in range(self.month_list.count()):
            item = self.month_list.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) in wanted:
                item.setSelected(True)
        self.apply_filters()

    def apply_filters(self) -> None:
        start_date = self.start_edit.date().toPython()
        end_date = self.end_edit.date().toPython()
        if start_date > end_date:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid date range",
                "The start date is after the end date. Please correct the range.",
            )
            return

        varieties = tuple(i.text() for i in self.variety_list.selectedItems())
        months = tuple(
            i.data(QtCore.Qt.ItemDataRole.UserRole) for i in self.month_list.selectedItems()
        )
        self.filters = FilterState(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            varieties=varieties,
            market=self.market_combo.currentData() or "",
            price_min=float(self.price_min.value()) or None,
            price_max=float(self.price_max.value()) or None,
            arrival_min=float(self.arrival_min.value()) or None,
            arrival_max=float(self.arrival_max.value()) or None,
            months=months,
            frequency=self.freq_combo.currentData(),
        )
        for page in self.pages.values():
            page.invalidate()
        self._refresh_status()
        self._render_page(self.stack.currentWidget())

    def reset_filters(self) -> None:
        start, end = self.service.full_date_span()
        if start is not None:
            self.start_edit.setDate(QtCore.QDate(start.year, start.month, start.day))
        if end is not None:
            self.end_edit.setDate(QtCore.QDate(end.year, end.month, end.day))
        self.variety_list.clearSelection()
        self.month_list.clearSelection()
        self.market_combo.setCurrentIndex(0)
        self.freq_combo.setCurrentIndex(1)
        for spin in (self.price_min, self.price_max, self.arrival_min, self.arrival_max):
            spin.setValue(0)
        self.apply_filters()

    # -- theming ----------------------------------------------------------

    def toggle_theme(self) -> None:
        self.apply_theme(
            settings.THEMES["light" if self.theme.is_dark else "dark"]
        )
        for page in self.pages.values():
            page.invalidate()
        self._render_page(self.stack.currentWidget())

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.theme_button.setText(
            "☀  Switch to light theme" if theme.is_dark else "🌙  Switch to dark theme"
        )
        self.setStyleSheet(self._stylesheet(theme))
        for page in self.pages.values():
            page.apply_theme(theme)

    @staticmethod
    def _stylesheet(theme: Theme) -> str:
        return f"""
        QMainWindow, QWidget {{
            background-color: {theme.window};
            color: {theme.text};
            font-family: "Segoe UI", "Inter", system-ui, sans-serif;
            font-size: 12px;
        }}
        QFrame#sidebar {{
            background-color: {theme.surface};
            border-right: 1px solid {theme.border};
        }}
        QLabel#brandTitle {{
            color: {theme.accent}; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;
        }}
        QLabel#brandSubtitle {{ color: {theme.text_muted}; font-size: 10px; }}
        QLabel#sidebarSection {{
            color: {theme.text_muted}; font-size: 9px; font-weight: 700;
            letter-spacing: 1px; padding: 6px 8px 2px 8px;
        }}
        QLabel#filterLabel {{ color: {theme.text_muted}; font-size: 10px; font-weight: 600; }}
        QPushButton#navButton {{
            background-color: transparent; color: {theme.text_muted};
            border: none; border-radius: 6px;
            padding: 8px 10px; text-align: left; font-size: 12px;
        }}
        QPushButton#navButton:hover {{
            background-color: {theme.surface_alt}; color: {theme.text};
        }}
        QPushButton#navButton:checked {{
            background-color: {theme.accent_soft}; color: {theme.accent}; font-weight: 700;
        }}
        QFrame#header {{
            background-color: {theme.surface};
            border-bottom: 1px solid {theme.border};
        }}
        QLabel#pageTitle {{ color: {theme.text}; font-size: 19px; font-weight: 800; }}
        QLabel#pageSubtitle {{ color: {theme.text_muted}; font-size: 11px; }}
        QLabel#headerMeta {{ color: {theme.text_muted}; font-size: 10px; }}
        QPushButton#primaryButton {{
            background-color: {theme.accent}; color: {'#111111' if not theme.is_dark or True else theme.text};
            border: none; border-radius: 6px; padding: 7px 14px;
            font-size: 11px; font-weight: 700;
        }}
        QPushButton#primaryButton:hover {{ background-color: {theme.warning}; }}
        QPushButton#primaryButton:disabled {{
            background-color: {theme.border}; color: {theme.text_muted};
        }}
        QPushButton#secondaryButton {{
            background-color: {theme.surface_alt}; color: {theme.text_muted};
            border: 1px solid {theme.border}; border-radius: 6px;
            padding: 6px 12px; font-size: 11px;
        }}
        QPushButton#secondaryButton:hover {{ color: {theme.text}; border-color: {theme.accent}; }}
        QToolButton#presetButton {{
            background-color: {theme.surface_alt}; color: {theme.text_muted};
            border: 1px solid {theme.border}; border-radius: 4px;
            padding: 3px 6px; font-size: 10px;
        }}
        QToolButton#presetButton:hover {{ color: {theme.accent}; border-color: {theme.accent}; }}
        QComboBox, QDateEdit, QSpinBox, QLineEdit {{
            background-color: {theme.surface_alt}; color: {theme.text};
            border: 1px solid {theme.border}; border-radius: 5px;
            padding: 4px 7px; font-size: 11px;
        }}
        QComboBox:hover, QDateEdit:hover, QSpinBox:hover {{ border-color: {theme.accent}; }}
        QComboBox QAbstractItemView {{
            background-color: {theme.surface}; color: {theme.text};
            border: 1px solid {theme.border};
            selection-background-color: {theme.accent_soft};
            selection-color: {theme.accent};
        }}
        QComboBox::drop-down {{ border: none; width: 16px; }}
        QListWidget {{
            background-color: {theme.surface_alt}; color: {theme.text};
            border: 1px solid {theme.border}; border-radius: 5px; font-size: 11px;
        }}
        QListWidget::item {{ padding: 3px 6px; }}
        QListWidget::item:selected {{
            background-color: {theme.accent_soft}; color: {theme.accent};
        }}
        QScrollArea {{ background-color: {theme.window}; border: none; }}
        QScrollBar:vertical {{
            background: {theme.window}; width: 11px; margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {theme.border}; border-radius: 5px; min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {theme.text_muted}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar:horizontal {{ background: {theme.window}; height: 11px; margin: 0; }}
        QScrollBar::handle:horizontal {{
            background: {theme.border}; border-radius: 5px; min-width: 28px;
        }}
        QStatusBar {{
            background-color: {theme.surface}; color: {theme.text_muted};
            border-top: 1px solid {theme.border}; font-size: 10px;
        }}
        QStatusBar::item {{ border: none; }}
        QProgressBar {{
            background-color: {theme.surface_alt}; border: 1px solid {theme.border};
            border-radius: 6px;
        }}
        QProgressBar::chunk {{ background-color: {theme.accent}; border-radius: 5px; }}
        QToolTip {{
            background-color: {theme.surface_alt}; color: {theme.text};
            border: 1px solid {theme.accent}; padding: 5px; font-size: 11px;
        }}
        QMessageBox {{ background-color: {theme.surface}; }}
        """

    # -- background work --------------------------------------------------

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_done: Callable[[Any], None] | None = None,
        on_fail: Callable[[str], None] | None = None,
        on_progress: Callable[[str, int], None] | None = None,
        wants_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run ``fn`` on the thread pool and deliver the result on the UI thread."""
        worker = Worker(fn, *args, _wants_progress=wants_progress, **kwargs)
        self._active_workers.append(worker)

        def cleanup() -> None:
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            if not self._active_workers:
                self.set_progress("", 100)

        def done(result: Any) -> None:
            try:
                if on_done:
                    on_done(result)
            except Exception:  # noqa: BLE001
                LOG.exception("Rendering a background result failed")
            finally:
                cleanup()

        def failed(message: str) -> None:
            try:
                if on_fail:
                    on_fail(message)
                else:
                    self.set_progress(f"Background task failed: {message}", 100)
            finally:
                cleanup()

        worker.signals.finished.connect(done)
        worker.signals.failed.connect(failed)
        worker.signals.progress.connect(
            on_progress if on_progress else lambda m, p: self.set_progress(m, p)
        )
        self.set_progress("Working…", 5)
        self.pool.start(worker)

    def set_progress(self, message: str, percent: int) -> None:
        if message:
            self.status_message.setText(message)
        else:
            self._refresh_status()
        if 0 <= percent < 100:
            self.progress.setVisible(True)
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
        else:
            self.progress.setVisible(False)

    def _refresh_status(self) -> None:
        latest = self.service.latest_observation_date()
        warnings = len(self.service.data_quality_notes())
        self.status_message.setText(
            f"Ready · {self.filters.describe()}"
            + (f" · {warnings} data-quality note(s)" if warnings else "")
        )
        self.workbook_label.setText(
            f"{self.data.path.name} · {len(self.data.datasets)}/"
            f"{len(self.data.raw_shapes)} sheets · read in "
            f"{self.data.load_seconds:.2f}s"
        )
        self.latest_label.setText(
            f"Latest observation: {fmt_date(latest)}\n"
            f"Workbook read {self.data.loaded_at:%d %b %Y %H:%M}"
        )

    # -- reload -----------------------------------------------------------

    def reload_workbook(self) -> None:
        """Re-read the workbook from disk and rebuild every page."""
        self.set_progress("Reloading the workbook…", 20)
        try:
            data = load_workbook(self.data.path, force_reload=True)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Reload failed")
            QtWidgets.QMessageBox.critical(
                self, "Reload failed", f"The workbook could not be re-read:\n\n{exc}"
            )
            self.set_progress("", 100)
            return
        self.data = data
        self.service = DataService(data)
        self.filters = default_filters(self.service)
        for page in self.pages.values():
            page.invalidate()
        self._refresh_status()
        self._render_page(self.stack.currentWidget())
        QtWidgets.QMessageBox.information(
            self,
            "Workbook reloaded",
            f"Re-read {len(data.datasets)} sheet(s) in {data.load_seconds:.2f}s.\n"
            + (
                f"{len(data.warnings)} warning(s) were reported."
                if data.warnings
                else "No warnings."
            ),
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt naming
        self.pool.clear()
        self.pool.waitForDone(2000)
        super().closeEvent(event)

