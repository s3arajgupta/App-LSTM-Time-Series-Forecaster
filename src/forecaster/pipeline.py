"""End-to-end pipeline connecting preprocessing, LSTM modeling, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.forecaster.config import CONFIG, ForecastConfig
from src.forecaster.evaluator import compute_forecast_metrics
from src.forecaster.lstm_model import LSTMForecaster
from src.forecaster.preprocessing import (
    TimeSeriesScaler,
    df_to_X_y,
    generate_future_dates,
    load_time_series_data,
)


def run_forecast_pipeline(
    config: Optional[ForecastConfig] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Execute complete end-to-end time-series forecasting pipeline.

    1. Ingests historical time-series data.
    2. Scales sequence into normalized domain [0, 1].
    3. Transforms into sliding window pairs (X, y).
    4. Trains the LSTM neural network.
    5. Reconstructs in-sample train/test predictions.
    6. Autoregressively projects N future steps with confidence bounds.
    7. Inverse-transforms all series to original unit domain.
    8. Exports 'combined_results.csv' and 'future_results.csv'.
    """
    cfg = config or CONFIG
    df = load_time_series_data(cfg.data_path, target_col=cfg.target_column)

    # Scale target series
    scaler = TimeSeriesScaler(feature_range=(0.0, 1.0))
    series_scaled = scaler.fit_transform(df[cfg.target_column].values)

    # Create sliding windows
    W = cfg.window_size
    X_all, y_all = df_to_X_y(series_scaled, window_size=W)

    split_idx = int(len(X_all) * cfg.train_split)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    # Train LSTM
    model = LSTMForecaster(
        window_size=W,
        hidden_units=cfg.hidden_units,
        dense_units=cfg.dense_units,
    )
    model.fit(X_train, y_train, epochs=cfg.epochs)

    # In-sample predictions
    preds_scaled = model.predict(X_all)
    preds_inv = scaler.inverse_transform(preds_scaled)
    actuals_inv = scaler.inverse_transform(y_all)

    # Compute evaluation metrics
    metrics = compute_forecast_metrics(actuals_inv, preds_inv)

    # Historical dates corresponding to window predictions
    history_dates = df["Date"].iloc[W:].reset_index(drop=True)

    train_results = pd.DataFrame({
        "Date": history_dates.dt.strftime("%Y-%m-%d"),
        "Train Predictions": np.round(preds_inv, 2),
        "Actuals": np.round(actuals_inv, 2),
    })

    # Future multi-step autoregressive roll-forward
    recent_window = X_all[-1]
    fut_scaled, fut_lower_scaled, fut_upper_scaled = model.forecast_future(
        recent_window, num_steps=cfg.future_steps
    )

    future_dates = generate_future_dates(df["Date"].iloc[-1], num_steps=cfg.future_steps)
    future_preds_inv = np.round(scaler.inverse_transform(fut_scaled), 2)
    future_lower_inv = np.round(scaler.inverse_transform(fut_lower_scaled), 2)
    future_upper_inv = np.round(scaler.inverse_transform(fut_upper_scaled), 2)

    future_results = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predictions": future_preds_inv,
        "Lower_Bound": future_lower_inv,
        "Upper_Bound": future_upper_inv,
    })

    # Build combined_results.csv exactly as expected by your_app.py
    # Includes historical train predictions and actuals, with future predictions appended
    combined_results = pd.concat([train_results, future_results[["Date", "Predictions"]]], ignore_index=True)

    if save_outputs:
        combined_path = cfg.combined_results_path
        future_path = cfg.future_results_path
        combined_results.to_csv(combined_path, index=False)
        future_results.to_csv(future_path, index=False)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "train_results": train_results,
        "future_results": future_results,
        "combined_results": combined_results,
        "history_df": df,
    }
