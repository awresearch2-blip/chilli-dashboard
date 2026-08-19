"""Convert :class:`FilterState` to and from the JSON-safe dict a ``dcc.Store``
can hold.

A Dash store persists as JSON in the browser (or server-side with a session
backend), so the dataclass's ``pd.Timestamp`` and tuple fields need a plain
round-trip form. This is the only new concept the web front end introduces
that the desktop app didn't need -- Qt just kept the dataclass instance alive
on ``MainWindow``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from chilli_desktop.preprocessing import FilterState


def to_dict(filters: FilterState) -> dict[str, Any]:
    return {
        "start": None if filters.start is None else filters.start.isoformat(),
        "end": None if filters.end is None else filters.end.isoformat(),
        "varieties": list(filters.varieties),
        "market": filters.market,
        "price_min": filters.price_min,
        "price_max": filters.price_max,
        "arrival_min": filters.arrival_min,
        "arrival_max": filters.arrival_max,
        "months": list(filters.months),
        "frequency": filters.frequency,
        "horizon": filters.horizon,
    }


def from_dict(data: dict[str, Any] | None) -> FilterState:
    if not data:
        return FilterState()
    return FilterState(
        start=None if data.get("start") is None else pd.Timestamp(data["start"]),
        end=None if data.get("end") is None else pd.Timestamp(data["end"]),
        varieties=tuple(data.get("varieties") or ()),
        market=data.get("market") or "",
        price_min=data.get("price_min"),
        price_max=data.get("price_max"),
        arrival_min=data.get("arrival_min"),
        arrival_max=data.get("arrival_max"),
        months=tuple(data.get("months") or ()),
        frequency=data.get("frequency") or "W",
        horizon=data.get("horizon") or 0,
    )
