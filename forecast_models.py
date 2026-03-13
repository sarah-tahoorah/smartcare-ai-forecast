import warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def _naive_forecast(train: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    last = train.iloc[-1]
    pred = pd.Series([last] * horizon)
    residuals = train.diff().dropna()
    sigma = residuals.std() if len(residuals) else 1.0
    ci_low = pred - 1.96 * sigma
    ci_high = pred + 1.96 * sigma
    return pred, ci_low, ci_high


def _moving_avg_forecast(train: pd.Series, horizon: int, window: int = 7) -> Tuple[pd.Series, pd.Series, pd.Series]:
    avg = train.iloc[-window:].mean()
    pred = pd.Series([avg] * horizon)
    residuals = train - train.rolling(window).mean()
    sigma = residuals.dropna().std() if len(residuals.dropna()) else 1.0
    ci_low = pred - 1.96 * sigma
    ci_high = pred + 1.96 * sigma
    return pred, ci_low, ci_high


def _arima_forecast(train: pd.Series, horizon: int, order=(1, 1, 1)):
    model = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return pred, ci.iloc[:, 0], ci.iloc[:, 1]


def _sarima_forecast(train: pd.Series, horizon: int, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return pred, ci.iloc[:, 0], ci.iloc[:, 1]


def _exp_smoothing_forecast(train: pd.Series, horizon: int):
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
    res = model.fit(optimized=True)
    pred = res.forecast(horizon)
    residuals = train - res.fittedvalues
    sigma = residuals.dropna().std() if len(residuals.dropna()) else 1.0
    ci_low = pred - 1.96 * sigma
    ci_high = pred + 1.96 * sigma
    return pred, ci_low, ci_high


def _train_ml_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=8)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("Unsupported ML model")
    model.fit(X, y)
    return model


def _recursive_ml_forecast(model, last_row: pd.DataFrame, horizon: int, feature_cols: list) -> pd.Series:
    preds = []
    current = last_row.copy()

    care_history = [
        current["care_lag_14"].iloc[0],
        current["care_lag_7"].iloc[0],
        current["care_lag_1"].iloc[0],
        current["children_in_care"].iloc[0],
    ]
    pressure_history = [
        current["pressure_lag_14"].iloc[0],
        current["pressure_lag_7"].iloc[0],
        current["pressure_lag_1"].iloc[0],
    ]

    for _ in range(horizon):
        yhat = model.predict(current[feature_cols])[0]
        preds.append(yhat)

        # Update lagged care values from history
        care_history = care_history[1:] + [yhat]
        current["care_lag_14"] = care_history[0]
        current["care_lag_7"] = care_history[1]
        current["care_lag_1"] = care_history[2]

        # Keep net pressure stable in recursion
        pressure_history = pressure_history[1:] + [pressure_history[-1]]
        current["pressure_lag_14"] = pressure_history[0]
        current["pressure_lag_7"] = pressure_history[1]
        current["pressure_lag_1"] = pressure_history[2]

        # Rolling features update (approx)
        current["care_roll_7"] = (current["care_roll_7"] * 6 + yhat) / 7
        current["care_roll_14"] = (current["care_roll_14"] * 13 + yhat) / 14

        # Calendar features advance by one day
        current["day_of_week"] = (current["day_of_week"] + 1) % 7
        if int(current["day_of_week"].iloc[0]) == 0:
            current["month"] = (current["month"] % 12) + 1

    return pd.Series(preds)


def forecast(model_name: str, df: pd.DataFrame, horizon: int, feature_cols: list) -> Dict[str, pd.Series]:
    series = df["children_in_care"].copy()

    if model_name == "Naive":
        pred, lo, hi = _naive_forecast(series, horizon)
    elif model_name == "Moving Average":
        pred, lo, hi = _moving_avg_forecast(series, horizon)
    elif model_name == "ARIMA":
        pred, lo, hi = _arima_forecast(series, horizon)
    elif model_name == "SARIMA":
        pred, lo, hi = _sarima_forecast(series, horizon)
    elif model_name == "Exponential Smoothing":
        pred, lo, hi = _exp_smoothing_forecast(series, horizon)
    elif model_name in ["Random Forest", "Gradient Boosting"]:
        clean = df.dropna()
        X = clean[feature_cols]
        y = clean["children_in_care"]
        model = _train_ml_model(model_name, X, y)
        last_row = clean.iloc[[-1]].copy()
        pred = _recursive_ml_forecast(model, last_row, horizon, feature_cols)
        residuals = y - model.predict(X)
        sigma = residuals.std() if len(residuals) else 1.0
        lo = pred - 1.96 * sigma
        hi = pred + 1.96 * sigma
    else:
        raise ValueError("Unknown model")

    return {"pred": pred, "lo": lo, "hi": hi}


def quick_baseline_mae(df: pd.DataFrame, horizon: int = 14) -> float:
    series = df["children_in_care"].copy()
    if len(series) < horizon + 10:
        return float("nan")
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    pred, _, _ = _naive_forecast(train, horizon)
    return mean_absolute_error(test, pred)
