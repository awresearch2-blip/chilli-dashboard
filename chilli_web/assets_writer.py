"""Writes the generated part of the stylesheet before the Dash app starts.

Dash serves every file under ``assets/`` automatically, but it has to exist on
disk *before* ``Dash(__name__)`` is constructed. The theme's colour values
live in :mod:`chilli_desktop.settings` (one source of truth, shared with the
desktop app); this just renders them to CSS custom properties so they never
need to be retyped here.
"""

from __future__ import annotations

from pathlib import Path

from . import theme as theme_mod

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def write_theme_css() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / "00_theme_vars.css"
    path.write_text(theme_mod.all_theme_css(), encoding="utf-8")
    return path
