"""XGBoost, direct multi-horizon: a separate model per (variety, horizon)
trained on the no-leakage feature matrix from feature_engineering.py.
"Direct" (one model per horizon) rather than recursive (chaining one-step
forecasts) avoids compounding one-step errors over long horizons.

No confidence interval is computed here -- forecast_engine.py attaches an
empirical band from out-of-sample backtest residuals uniformly to every
model that doesn't have a native one, rather than duplicating that logic
per model.
"""

import pandas as pd
from xgboost import XGBRegressor

from forecasting.feature_engineering import FEATURE_COLUMNS, add_target

MIN_TRAIN_ROWS = 200
MODEL_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)


def fit(features_df: pd.DataFrame, price_series: pd.Series, horizon_days: int):
    labeled = add_target(features_df, price_series, horizon_days)
    training_rows = labeled.dropna(subset=FEATURE_COLUMNS + ["target"])
    if len(training_rows) < MIN_TRAIN_ROWS:
        return None

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(training_rows[FEATURE_COLUMNS], training_rows["target"])
    return {"model": model, "features_df": features_df}


def predict(fitted_state, origin_date):
    if fitted_state is None:
        return None
    features_df = fitted_state["features_df"]
    origin_date = pd.Timestamp(origin_date)

    available = features_df.index[features_df.index <= origin_date]
    if len(available) == 0:
        return None
    row = features_df.loc[[available[-1]], FEATURE_COLUMNS]
    if row.isna().any(axis=1).iloc[0]:
        return None

    point = float(fitted_state["model"].predict(row)[0])
    return {"point": point, "lower": None, "upper": None}
