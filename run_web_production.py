"""Production entry point for sharing the web dashboard beyond this machine.

    .venv\\Scripts\\python.exe run_web_production.py [--host 127.0.0.1] [--port 8060]

Unlike run_web.py (Flask's development server, used while building the app),
this serves the exact same Dash/Flask application through waitress -- a
production-grade, pure-Python WSGI server. That matters for anything reachable
outside this machine: Flask's dev server carries the Werkzeug interactive
debugger, which executes arbitrary Python from a browser if it ever shows an
error page. waitress has no such feature, prints no "development server,
do not use in production" warning, and is the appropriate thing to sit behind
a tunnel or reverse proxy.

The application logic, pages, callbacks and background-callback manager are
completely unchanged -- this file only swaps how the WSGI app is served.
"""

from __future__ import annotations

import argparse
import os

from waitress import serve

from chilli_desktop.utils import LOG
from chilli_web.app import app
from chilli_web import server_state


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    # Defaults fall back to HOST/PORT env vars so the same image/command works
    # unchanged on hosts that assign the port dynamically (Render, Railway, ...).
    parser = argparse.ArgumentParser(prog="chilli-web-production")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8060)))
    parser.add_argument("--threads", type=int, default=6, help="waitress worker threads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    server_state.get_service()  # load the workbook before accepting requests
    LOG.info(
        "Serving Chilli Intelligence Web via waitress on http://%s:%d (%d threads)",
        args.host, args.port, args.threads,
    )
    serve(app.server, host=args.host, port=args.port, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
