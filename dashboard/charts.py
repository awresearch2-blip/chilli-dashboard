"""Reusable Plotly figure builders. Plain functions, no Streamlit calls, so
they're directly unit-testable and reusable across views. Every builder is
defensive against missing/empty input (returns an annotated empty figure
rather than raising) -- but the *decision* of whether to show a chart or an
honest "insufficient evidence" caveat belongs to the view/component layer,
not silently swallowed here.
"""

import pandas as pd
import plotly.graph_objects as go

EMPTY_ANNOTATION = dict(text="No data available", showarrow=False, font=dict(size=14))


def _empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(annotations=[{**EMPTY_ANNOTATION, "text": message}], xaxis_visible=False, yaxis_visible=False)
    return fig


def time_series_chart(series_dict: dict, title: str = "", y_title: str = "") -> go.Figure:
    series_dict = {name: s for name, s in (series_dict or {}).items() if s is not None and not s.dropna().empty}
    if not series_dict:
        return _empty_figure()

    fig = go.Figure()
    for name, series in series_dict.items():
        s = series.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra>" + name + "</extra>"))
    fig.update_layout(title=title, yaxis_title=y_title, hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig


def forecast_fan_chart(history: pd.Series, target_date: str, point: float, lower, upper, title: str = "") -> go.Figure:
    if history is None or history.dropna().empty or point is None:
        return _empty_figure()

    hist = history.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History", hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>"))

    target = pd.Timestamp(target_date)
    last_date, last_value = hist.index[-1], float(hist.iloc[-1])
    fig.add_trace(go.Scatter(x=[last_date, target], y=[last_value, point], mode="lines+markers", name="Forecast", line=dict(dash="dash", color="orange")))

    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=[target, target], y=[lower, upper], mode="lines", name="Confidence interval",
                line=dict(color="orange", width=8), opacity=0.3, hovertemplate=f"Lower: {lower:,.2f}<br>Upper: {upper:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(title=title, hovermode="closest", legend=dict(orientation="h", y=1.1))
    return fig


def seasonal_index_bar(index_by_month: dict, title: str = "Seasonal Index (100 = neutral)") -> go.Figure:
    if not index_by_month:
        return _empty_figure()

    # JSON object keys are always strings once round-tripped through
    # json.dump/load (as every persisted analytics file is) -- normalize
    # here rather than depending on the caller's dict having int keys.
    normalized = {int(k): v for k, v in index_by_month.items()}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = sorted(normalized.keys())
    values = [normalized[m]["index"] for m in months]
    years_used = [normalized[m].get("years_used") for m in months]
    labels = [month_names[m - 1] for m in months]

    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=["#c0392b" if v is not None and v < 100 else "#27ae60" for v in values],
            hovertemplate="%{x}: index=%{y}<br>years_used=%{customdata}<extra></extra>", customdata=years_used,
        )
    )
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(title=title, yaxis_title="Index")
    return fig


def correlation_heatmap(pairwise: dict, varieties: list, title: str = "Pairwise Correlation") -> go.Figure:
    if not pairwise or not varieties:
        return _empty_figure()

    n = len(varieties)
    z = [[1.0] * n for _ in range(n)]
    for i, a in enumerate(varieties):
        for j, b in enumerate(varieties):
            if i == j:
                continue
            entry = pairwise.get(f"{a} vs {b}") or pairwise.get(f"{b} vs {a}")
            if entry and entry.get("status") == "ok":
                z[i][j] = entry["r"]

    fig = go.Figure(go.Heatmap(z=z, x=varieties, y=varieties, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title=title)
    return fig


def dual_axis_chart(series_a: pd.Series, series_b: pd.Series, name_a: str, name_b: str, title: str = "") -> go.Figure:
    a = series_a.dropna() if series_a is not None else pd.Series(dtype=float)
    b = series_b.dropna() if series_b is not None else pd.Series(dtype=float)
    if a.empty and b.empty:
        return _empty_figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.index, y=a.values, name=name_a, mode="lines", yaxis="y1"))
    fig.add_trace(go.Scatter(x=b.index, y=b.values, name=name_b, mode="lines", yaxis="y2"))
    fig.update_layout(
        title=title,
        yaxis=dict(title=name_a),
        yaxis2=dict(title=name_b, overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def bar_chart(categories: list, values: list, title: str = "", y_title: str = "") -> go.Figure:
    if not categories or not values:
        return _empty_figure()
    fig = go.Figure(go.Bar(x=categories, y=values, hovertemplate="%{x}: %{y:,.2f}<extra></extra>"))
    fig.update_layout(title=title, yaxis_title=y_title)
    return fig


def gauge_chart(value, title: str = "", max_value: float = 100) -> go.Figure:
    if value is None:
        return _empty_figure()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={"axis": {"range": [0, max_value]}, "bar": {"color": "#2980b9"}},
        )
    )
    return fig
