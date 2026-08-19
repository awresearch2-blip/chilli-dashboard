"""`streamlit run` entrypoint wrapper.

    .venv\\Scripts\\streamlit run run_dashboard.py
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("dashboard.app", run_name="__main__")
