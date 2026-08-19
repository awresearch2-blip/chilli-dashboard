"""Application entry point.

Run from the project root:

    python -m chilli_desktop.main
    python chilli_desktop/main.py --workbook "path\\to\\workbook.xlsx" --theme light

The workbook is read before the window is shown, because there is nothing
meaningful to display without it. A read failure produces a dialog explaining
exactly which paths were tried, not a traceback.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

# Allow `python chilli_desktop/main.py` as well as `python -m chilli_desktop.main`
# by making sure the project root is importable either way.
if __package__ in (None, ""):  # pragma: no cover - script-mode bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Qt must know which binding to use before matplotlib picks a backend.
os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("QtAgg")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from chilli_desktop import settings  # noqa: E402
from chilli_desktop.data_loader import load_workbook  # noqa: E402
from chilli_desktop.ui import MainWindow  # noqa: E402
from chilli_desktop.utils import WorkbookError, configure_logging  # noqa: E402


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chilli-desktop",
        description=(
            f"{settings.APP_NAME} {settings.APP_VERSION} — an interactive "
            "desktop dashboard for the Indian red chilli market, driven "
            "entirely by the master Excel workbook."
        ),
    )
    parser.add_argument(
        "--workbook",
        default=None,
        help=(
            "Full path to the master workbook. When omitted, the standard "
            f"search order is used, and the {settings.WORKBOOK_ENV_VAR} "
            "environment variable takes priority within it."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=sorted(settings.THEMES),
        default=settings.DEFAULT_THEME,
        help="Colour theme to start in (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity (default: %(default)s).",
    )
    parser.add_argument(
        "--export-data-dictionary",
        metavar="PATH",
        default=None,
        help=(
            "Write the auto-generated data dictionary to PATH as Markdown and "
            "exit without opening the window."
        ),
    )
    return parser.parse_args(argv)


def install_exception_hook(logger: logging.Logger) -> None:
    """Report uncaught exceptions in a dialog rather than dying silently.

    Without this, an exception raised inside a Qt slot prints to a console the
    user may never see and leaves the window in an undefined state.
    """

    def hook(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        if QtWidgets.QApplication.instance() is not None:
            box = QtWidgets.QMessageBox()
            box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            box.setWindowTitle(f"{settings.APP_SHORT_NAME} — unexpected error")
            box.setText(f"{exc_type.__name__}: {exc_value}")
            box.setInformativeText(
                "The dashboard hit an unexpected error. The action was "
                "abandoned; the window should still be usable. Details have "
                "been written to the log file."
            )
            box.setDetailedText(detail)
            box.exec()

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    logger = configure_logging(getattr(logging, args.log_level))
    logger.info("Starting %s %s", settings.APP_NAME, settings.APP_VERSION)

    # -- headless mode: write the data dictionary and exit ----------------
    if args.export_data_dictionary:
        from chilli_desktop.data_loader import data_dictionary_markdown

        try:
            data = load_workbook(args.workbook)
        except WorkbookError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        target = Path(args.export_data_dictionary)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data_dictionary_markdown(data), encoding="utf-8")
        print(f"Data dictionary written to {target}")
        return 0

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(settings.APP_NAME)
    app.setApplicationVersion(settings.APP_VERSION)
    app.setOrganizationName(settings.ORG_NAME)
    app.setStyle("Fusion")
    install_exception_hook(logger)

    theme = settings.THEMES[args.theme]

    # A splash keeps the user informed during the workbook read, which is the
    # slowest part of start-up.
    splash_pixmap = QtGui.QPixmap(460, 150)
    splash_pixmap.fill(QtGui.QColor(theme.surface))
    painter = QtGui.QPainter(splash_pixmap)
    painter.setPen(QtGui.QColor(theme.accent))
    font = painter.font()
    font.setPointSize(15)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(
        splash_pixmap.rect().adjusted(0, -14, 0, -14),
        QtCore.Qt.AlignmentFlag.AlignCenter,
        "🌶  Chilli Intelligence Desktop",
    )
    painter.setPen(QtGui.QColor(theme.text_muted))
    font.setPointSize(9)
    font.setBold(False)
    painter.setFont(font)
    painter.drawText(
        splash_pixmap.rect().adjusted(0, 34, 0, 34),
        QtCore.Qt.AlignmentFlag.AlignCenter,
        "Reading the master workbook…",
    )
    painter.end()

    splash = QtWidgets.QSplashScreen(splash_pixmap)
    splash.show()
    app.processEvents()

    try:
        data = load_workbook(args.workbook)
    except WorkbookError as exc:
        splash.close()
        logger.error("Workbook could not be loaded: %s", exc)
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle(f"{settings.APP_SHORT_NAME} — workbook not found")
        box.setText("The master workbook could not be read.")
        box.setInformativeText(str(exc))
        box.exec()
        return 2
    except Exception as exc:  # noqa: BLE001
        splash.close()
        logger.exception("Unexpected failure while loading the workbook")
        QtWidgets.QMessageBox.critical(
            None,
            f"{settings.APP_SHORT_NAME} — startup failed",
            f"An unexpected error occurred while reading the workbook:\n\n{exc}",
        )
        return 3

    logger.info(
        "Workbook ready: %d sheet(s) parsed in %.2fs, %d warning(s)",
        len(data.datasets), data.load_seconds, len(data.warnings),
    )

    window = MainWindow(data, theme_name=args.theme)
    splash.finish(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
