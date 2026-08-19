"""CLI entrypoint for the Phase 1 pipeline.

    python run_refresh.py --once           run a single refresh cycle and exit
    python run_refresh.py --watch          watch the workbook and refresh on every change
    python run_refresh.py --watch --interval 3600   also refresh on a fixed schedule as a fallback
"""

import argparse
import time

from ingestion.watcher import start_watching
from pipeline.refresh import run_refresh
from utils.logging_config import get_logger

logger = get_logger("cli")


def main():
    parser = argparse.ArgumentParser(description="Chilli Intelligence Platform -- Phase 1 refresh")
    parser.add_argument("--once", action="store_true", help="Run a single refresh cycle and exit")
    parser.add_argument("--watch", action="store_true", help="Watch the workbook and refresh automatically on change")
    parser.add_argument("--interval", type=float, default=None, help="Seconds between fallback refreshes while watching")
    args = parser.parse_args()

    if not args.once and not args.watch:
        args.once = True

    if args.once:
        result = run_refresh()
        print(result)
        return

    result = run_refresh()
    print(result)
    observer = start_watching(on_change=run_refresh)
    try:
        last_poll = time.time()
        while True:
            time.sleep(1)
            if args.interval and (time.time() - last_poll) >= args.interval:
                run_refresh()
                last_poll = time.time()
    except KeyboardInterrupt:
        logger.info("Stopping watcher (KeyboardInterrupt)")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
