"""Reusable UI components. Unlike data_loader.py/charts.py, these call
Streamlit directly since they render.
"""

import streamlit as st


def caveat_banner(status: str, reason: str = None) -> None:
    """Renders the real reason from the underlying JSON -- never a blank
    section and never a fabricated number standing in for missing evidence.
    """
    if status == "insufficient_evidence":
        st.info(f"ℹ️ Insufficient evidence: {reason or 'not enough real data for this view yet.'}")
    else:
        st.error(f"⚠️ {reason or 'This section could not be computed.'}")


def render_or_caveat(data: dict, render_fn) -> None:
    """Calls render_fn(data) only when data's status is "ok"; otherwise
    shows an honest caveat banner with the real reason instead of a blank
    chart or a silently-skipped section."""
    if not data or data.get("status") != "ok":
        status = (data or {}).get("status", "insufficient_evidence")
        reason = (data or {}).get("reason")
        caveat_banner(status, reason)
        return
    render_fn(data)


def metric_row(items: list) -> None:
    """items: list of (label, value) or (label, value, delta) tuples."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        col.metric(label, value, delta)


def refresh_button() -> None:
    """The brief's required manual "Refresh" control. Synchronously calls
    the same run_refresh() the CLI/watcher use, then clears the cache so
    every view picks up the new data immediately."""
    from pipeline.refresh import run_refresh

    if st.sidebar.button("\U0001f504 Refresh Now", width="stretch"):
        with st.spinner("Refreshing -- can take up to ~2 minutes if new price data triggers model retraining..."):
            result = run_refresh()
        if result.get("status") == "ok":
            st.sidebar.success(f"Refreshed ({result['refresh_id']})")
        else:
            st.sidebar.error(f"Refresh failed: {result.get('error', 'unknown error')}")
        st.cache_data.clear()
        st.rerun()
