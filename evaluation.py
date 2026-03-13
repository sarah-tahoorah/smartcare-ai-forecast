from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecast_models import (
    _naive_forecast,
    _moving_avg_forecast,
    _arima_forecast,
    _sarima_forecast,
    _exp_smoothing_forecast,
    _train_ml_model,
    _recursive_ml_forecast,
)


def _mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_models(df: pd.DataFrame, horizon: int, feature_cols: List[str]) -> pd.DataFrame:
    series = df["children_in_care"].copy()
    split_idx = len(series) - horizon
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    results = []

    def add_result(name, pred):
        mae = mean_absolute_error(test, pred)
        # Some sklearn versions don't support squared= parameter; compute RMSE manually.
        mse = mean_squared_error(test, pred)
        rmse = float(np.sqrt(mse))
        mape = _mape(test, pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

    pred, _, _ = _naive_forecast(train, horizon)
    add_result("Naive", pred)

    pred, _, _ = _moving_avg_forecast(train, horizon)
    add_result("Moving Average", pred)

    pred, _, _ = _arima_forecast(train, horizon)
    add_result("ARIMA", pred)

    pred, _, _ = _sarima_forecast(train, horizon)
    add_result("SARIMA", pred)

    pred, _, _ = _exp_smoothing_forecast(train, horizon)
    add_result("Exponential Smoothing", pred)

    # ML models
    clean = df.dropna()
    split_idx_ml = len(clean) - horizon
    train_ml = clean.iloc[:split_idx_ml]
    test_ml = clean.iloc[split_idx_ml:]

    X_train = train_ml[feature_cols]
    y_train = train_ml["children_in_care"]
    X_test = test_ml[feature_cols]
    y_test = test_ml["children_in_care"]

    for name in ["Random Forest", "Gradient Boosting"]:
        model = _train_ml_model(name, X_train, y_train)
        # recursive forecast using last row of training
        last_row = train_ml.iloc[[-1]].copy()
        pred = _recursive_ml_forecast(model, last_row, horizon, feature_cols)
        add_result(name, pred)

    return pd.DataFrame(results).sort_values("RMSE")
