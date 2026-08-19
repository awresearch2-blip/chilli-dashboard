"""`python run_web.py` entrypoint wrapper for the browser front end.

    .venv\\Scripts\\python.exe run_web.py

The desktop app's equivalent is run_dashboard.py; this launches the Dash
server on http://127.0.0.1:8060 instead of opening a native window.
"""

from chilli_web.app import main

if __name__ == "__main__":
    raise SystemExit(main())
