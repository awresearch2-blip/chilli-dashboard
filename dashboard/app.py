"""Entrypoint: page config, sidebar navigation + global filters, dispatch to
one view module per section. The dashboard is a pure read layer over
data/clean, data/analytical, and data/forecasts -- see components.py's
refresh_button() for the one place it writes anything (by calling the same
run_refresh() the CLI/watcher use).
"""

import datetime as dt

import streamlit as st

from dashboard.components import refresh_button
from dashboard.data_loader import last_refresh_id
from dashboard.views import composite_scores, forecasts, price_arrivals, seasonality, trade_supply, variety_market

st.set_page_config(page_title="Chilli Intelligence Platform", layout="wide")

VIEWS = {
    "Price, Arrivals & Price-vs-Arrivals": price_arrivals,
    "Seasonality": seasonality,
    "Variety & Market Comparison": variety_market,
    "Export, Balance Sheet, FX & Cold Storage": trade_supply,
    "Composite Scores": composite_scores,
    "Forecasts": forecasts,
}

st.sidebar.title("\U0001f336️ Chilli Intelligence Platform")
st.sidebar.caption(f"Last refresh: {last_refresh_id()}")
refresh_button()

st.sidebar.divider()
section = st.sidebar.radio("Section", list(VIEWS.keys()))

st.sidebar.divider()
st.sidebar.subheader("Filters")
variety = st.sidebar.selectbox("Variety", ["Teja", "334"], help="LCA334 is labeled '334' in the workbook.")
market = st.sidebar.selectbox("Market (Market Comparison view only)", ["Guntur", "Warangal", "Khammam"])

date_preset = st.sidebar.radio("Date range", ["Last 2 years", "Full history"], horizontal=True)
if date_preset == "Last 2 years":
    date_range = (dt.date.today() - dt.timedelta(days=730), dt.date.today())
else:
    date_range = (dt.date(2014, 1, 1), dt.date.today())

horizon = st.sidebar.selectbox("Forecast horizon (days, Forecasts view only)", [7, 14, 30, 60, 90, 120, 180], index=2)

VIEWS[section].render(variety=variety, market=market, date_range=date_range, horizon=horizon)
