"""Chilli Intelligence Web -- a browser-native front end for the same backend
that powers the desktop application.

Everything that computes a number lives in ``chilli_desktop`` (data loading,
preprocessing, statistics, forecasting, automated insights) and is imported
here unchanged. Only the presentation layer differs: the desktop app renders
Qt widgets and matplotlib-in-Qt canvases; this package renders Dash/Plotly
components serving HTML over HTTP so the identical analysis can be reached
from a browser and shared as a URL.

No market data, statistic or forecast value is computed twice. If a number
disagrees between the two applications, that is a bug -- both read from the
same :class:`chilli_desktop.preprocessing.DataService`.
"""
