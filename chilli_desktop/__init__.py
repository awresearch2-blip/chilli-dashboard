"""Chilli Intelligence Desktop -- an interactive PySide6 commodity dashboard.

Every figure the application shows is derived from the master Excel workbook
at runtime. See ``settings.py`` for configuration and ``README.md`` for setup.
"""

from .settings import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
__version__ = APP_VERSION
