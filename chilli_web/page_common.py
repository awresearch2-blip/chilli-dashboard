"""Small helpers shared by every page module, so each page file only contains
page-specific logic and not import/plumbing boilerplate.

``target_variety`` mirrors ``ui.PriceAnalysisPage._target_variety`` -- several
desktop pages (Price, Arrivals, Seasonality, Export) call that one static
method; the web pages share the same free function instead.
"""

from __future__ import annotations

from chilli_desktop import settings
from chilli_desktop.preprocessing import DataService, FilterState
from chilli_desktop.settings import Theme

from . import filters_io, server_state


def current(filters_data: dict | None, theme_name: str | None) -> tuple[DataService, FilterState, Theme]:
    service = server_state.get_service()
    filters = filters_io.from_dict(filters_data)
    theme = server_state.theme_for(theme_name or settings.DEFAULT_THEME)
    return service, filters, theme


def target_variety(service: DataService, filters: FilterState) -> str:
    if filters.varieties:
        resolved = service.resolve_variety(filters.varieties[0])
        if resolved:
            return resolved
    focus = service.focus_varieties()
    if focus:
        return next(iter(focus.values()))
    available = service.varieties()
    return available[0] if available else ""


def frequency_label(freq: str) -> str:
    return settings.FORECAST.frequency_labels.get(freq, freq)


def tone_for_change(value: float) -> str:
    import numpy as np

    if not np.isfinite(value):
        return "neutral"
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"

