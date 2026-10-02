"""Evaluation metrics for time-series forecasting models."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_forecast_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive time-series evaluation metrics.

    Args:
        actual: 1D array of ground truth series.
        pred: 1D array of predicted series.

    Returns:
        Dictionary of RMSE, MAE, MAPE, R2, and Directional Accuracy.
    """
    y_true = np.asarray(actual).flatten()
    y_pred = np.asarray(pred).flatten()

    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # Avoid zero division in MAPE
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs(diff[non_zero] / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    # R-squared
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(diff ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

    # Directional Accuracy (trend prediction)
    if len(y_true) > 1:
        true_direction = np.sign(y_true[1:] - y_true[:-1])
        pred_direction = np.sign(y_pred[1:] - y_true[:-1])
        dir_acc = float(np.mean(true_direction == pred_direction) * 100.0)
    else:
        dir_acc = 100.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 2),
    }
