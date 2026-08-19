import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts, data_loader


def setup_function(_):
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# charts.py -- pure functions, must never raise on empty/None input
# ---------------------------------------------------------------------------

def test_time_series_chart_handles_empty_and_real_input():
    assert isinstance(charts.time_series_chart({}), go.Figure)
    assert isinstance(charts.time_series_chart(None), go.Figure)

    dates = pd.date_range("2024-01-01", periods=10)
    series = pd.Series(range(10), index=dates, dtype=float)
    fig = charts.time_series_chart({"Price": series})
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == list(series.values)


def test_forecast_fan_chart_handles_missing_point():
    assert isinstance(charts.forecast_fan_chart(None, "2024-01-01", None, None, None), go.Figure)

    dates = pd.date_range("2024-01-01", periods=10)
    history = pd.Series(range(10), index=dates, dtype=float)
    fig = charts.forecast_fan_chart(history, "2024-01-20", 15.0, 10.0, 20.0)
    assert len(fig.data) == 3  # history, forecast line, CI band


def test_seasonal_index_bar_normalizes_string_keys_from_json_roundtrip():
    index_by_month = json.loads(json.dumps({1: {"index": 80.0, "years_used": 5}, 7: {"index": 110.0, "years_used": 6}}))
    fig = charts.seasonal_index_bar(index_by_month)
    assert list(fig.data[0].x) == ["Jan", "Jul"]
    assert list(fig.data[0].y) == [80.0, 110.0]


def test_correlation_heatmap_builds_symmetric_matrix():
    pairwise = {"Teja vs 334": {"status": "ok", "r": 0.94}}
    fig = charts.correlation_heatmap(pairwise, ["Teja", "334"])
    z = fig.data[0].z
    assert z[0][0] == 1.0
    assert z[0][1] == 0.94
    assert z[1][0] == 0.94  # symmetric even though only one direction was in the input


def test_gauge_chart_handles_none():
    assert isinstance(charts.gauge_chart(None, "Test"), go.Figure)
    assert isinstance(charts.gauge_chart(75.0, "Test"), go.Figure)


def test_bar_chart_handles_empty():
    assert isinstance(charts.bar_chart([], []), go.Figure)
    assert isinstance(charts.bar_chart(["Jan"], [100]), go.Figure)


# ---------------------------------------------------------------------------
# data_loader.py -- cache keyed on mtime, honest insufficient_evidence on missing file
# ---------------------------------------------------------------------------

def test_load_analytical_reports_insufficient_evidence_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "ANALYTICAL_DIR", tmp_path)
    result = data_loader.load_analytical("does_not_exist")
    assert result["status"] == "insufficient_evidence"


def test_load_analytical_reads_real_content_and_invalidates_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "ANALYTICAL_DIR", tmp_path)
    path = tmp_path / "price_analysis.json"
    path.write_text(json.dumps({"result": {"status": "ok", "value": 1}}), encoding="utf-8")

    first = data_loader.load_analytical("price_analysis")
    assert first == {"status": "ok", "value": 1}

    # Simulate a refresh producing new content -- mtime must actually change
    # for the OS to report a different value; write different content and
    # nudge mtime forward explicitly to make the test deterministic.
    path.write_text(json.dumps({"result": {"status": "ok", "value": 2}}), encoding="utf-8")
    import os
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))

    second = data_loader.load_analytical("price_analysis")
    assert second == {"status": "ok", "value": 2}


def test_load_forecast_reports_insufficient_evidence_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "FORECASTS_DIR", tmp_path)
    result = data_loader.load_forecast("Teja")
    assert result["status"] == "insufficient_evidence"


def test_last_refresh_id_reports_never_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DATA_QUALITY_LATEST_PATH", tmp_path / "missing.json")
    assert data_loader.last_refresh_id() == "never"
