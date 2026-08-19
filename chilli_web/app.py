"""Dash application factory and entry point.

Run from the project root:

    .venv\\Scripts\\python.exe -m chilli_web.app

This is the browser-facing counterpart to ``chilli_desktop/main.py``. It
constructs one long-lived server process, reads the workbook once at start-up
through the exact same :func:`chilli_desktop.data_loader.load_workbook`, and
serves every page from the exact same :mod:`chilli_desktop.analytics` /
``forecasting`` / ``insights`` functions the desktop app calls -- only the
presentation layer (this package) differs.

Long-running analyses (the forecast sweep, the market-integration battery, the
automated-insights sweep) run as Dash *background callbacks* backed by
``diskcache`` -- a local, dependency-free stand-in for the desktop app's
``QThreadPool`` workers. No Redis or Celery is required.
"""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

import dash
import diskcache
from dash import DiskcacheManager, dcc, html
from flask import Response, request

from chilli_desktop import settings
from chilli_desktop.utils import LOG, WorkbookError

from . import assets_writer, layout_shell, server_state

PACKAGE_DIR = Path(__file__).resolve().parent

assets_writer.write_theme_css()

_cache = diskcache.Cache(str(PACKAGE_DIR / ".diskcache"))
background_callback_manager = DiskcacheManager(_cache)

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(PACKAGE_DIR / "pages"),
    assets_folder=str(PACKAGE_DIR / "assets"),
    background_callback_manager=background_callback_manager,
    suppress_callback_exceptions=True,
    title="Chilli Intelligence Web",
    update_title=None,
)
server = app.server  # exposed for a WSGI front end (gunicorn/waitress) later

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), layout_shell.app_shell(dash.page_container)]
)

# Optional HTTP Basic Auth gate -- opt-in via CHILLI_WEB_USERNAME/PASSWORD, so
# the app is unauthenticated by default (matching the desktop app, which has
# no login concept). Set both env vars before exposing the app publicly if
# the workbook contains data that shouldn't be open to anyone with the URL.
_auth_user = os.environ.get("CHILLI_WEB_USERNAME", "")
_auth_password = os.environ.get("CHILLI_WEB_PASSWORD", "")
if _auth_user and _auth_password:

    @server.before_request
    def _require_basic_auth():
        auth = request.authorization
        valid = bool(auth) and secrets.compare_digest(
            auth.username or "", _auth_user
        ) and secrets.compare_digest(auth.password or "", _auth_password)
        if not valid:
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Chilli Intelligence"'},
            )

    LOG.info("HTTP Basic Auth enabled for Chilli Intelligence Web")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    # Defaults fall back to HOST/PORT env vars so a launcher that assigns a
    # port dynamically (e.g. this project's own preview harness) can steer
    # the dev server without a hardcoded --port flag.
    parser = argparse.ArgumentParser(prog="chilli-web")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8060)))
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        server_state.get_service(force_reload=False)
    except WorkbookError as exc:
        LOG.error("Workbook could not be loaded: %s", exc)
        print(f"ERROR: {exc}")
        return 2
    LOG.info("Starting %s Web on http://%s:%d", settings.APP_NAME, args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
