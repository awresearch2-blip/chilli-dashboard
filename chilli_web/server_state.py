"""Process-wide shared state: the one workbook read and the one DataService.

The desktop application builds one :class:`DataService` per ``MainWindow`` and
keeps it for the life of the process. A Dash server is itself one long-lived
process handling many requests, so the same pattern applies directly: read the
workbook once at import time, hand every callback the same
:class:`~chilli_desktop.preprocessing.DataService` instance, and let its own
internal memoisation (already written for the desktop app) absorb repeat
requests for the same series.

This module intentionally has no Qt/Dash import -- it is pure backend glue,
reusable by any future front end.
"""

from __future__ import annotations

import threading

from chilli_desktop import settings
from chilli_desktop.data_loader import WorkbookData, load_workbook
from chilli_desktop.preprocessing import DataService, FilterState, default_filters
from chilli_desktop.utils import LOG, WorkbookError, configure_logging

configure_logging()

_lock = threading.Lock()
_service: DataService | None = None


def get_service(force_reload: bool = False) -> DataService:
    """Return the shared :class:`DataService`, creating it on first call.

    Reloading re-reads the workbook from disk and builds a fresh service, so
    every page picks up the new data on its next callback -- the web
    equivalent of the desktop app's "Reload workbook" button.
    """
    global _service
    with _lock:
        if _service is None or force_reload:
            data: WorkbookData = load_workbook(force_reload=force_reload)
            _service = DataService(data)
            LOG.info(
                "Web server data ready: %d sheet(s) in %.2fs, %d warning(s)",
                len(data.datasets), data.load_seconds, len(data.warnings),
            )
        return _service


def initial_filters() -> FilterState:
    """The filter state a fresh browser session should open with."""
    return default_filters(get_service())


def theme_for(name: str) -> settings.Theme:
    return settings.THEMES.get(name, settings.DARK_THEME)


try:
    get_service()
except WorkbookError:
    # Surfaced properly by app.py at start-up; importing this module must not
    # itself crash a `python -c "import chilli_web"` sanity check.
    LOG.exception("Workbook could not be loaded at import time")
