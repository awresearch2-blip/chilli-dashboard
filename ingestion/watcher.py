"""watchdog-based monitor for the mastersheet workbook.

Watches the containing folder (not the file handle) because Excel/OneDrive
save via a temp-file-then-rename dance, not a single in-place write. Ignores
`~$...` lock files, debounces bursts of write events into a single refresh,
and never lets a bad refresh kill the watch loop.
"""

import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils.logging_config import get_logger
from utils.paths import WORKBOOK_PATH

logger = get_logger("watcher")

DEFAULT_DEBOUNCE_SECONDS = 3.0


class WorkbookChangeHandler(FileSystemEventHandler):
    def __init__(self, target_path: Path, on_change, debounce_seconds: float):
        self.target_path = target_path.resolve()
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self._timer = None
        self._lock = threading.Lock()

    def _is_relevant(self, event_path: str) -> bool:
        p = Path(event_path)
        if p.name.startswith("~$"):
            return False
        try:
            return p.resolve() == self.target_path
        except OSError:
            return False

    def _schedule_refresh(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self):
        logger.info("Debounced workbook change detected -- triggering refresh")
        try:
            self.on_change()
        except Exception:
            logger.exception("Refresh callback raised an exception; watcher stays alive")

    def on_modified(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_refresh()

    def on_created(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_refresh()

    def on_moved(self, event):
        dest = getattr(event, "dest_path", None)
        if dest and not event.is_directory and self._is_relevant(dest):
            self._schedule_refresh()


def start_watching(
    on_change,
    workbook_path: Path = WORKBOOK_PATH,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
) -> Observer:
    workbook_path = Path(workbook_path)
    handler = WorkbookChangeHandler(workbook_path, on_change, debounce_seconds)
    observer = Observer()
    observer.schedule(handler, str(workbook_path.parent), recursive=False)
    observer.start()
    logger.info("Watching '%s' for changes (debounce=%.1fs)", workbook_path, debounce_seconds)
    return observer
