"""Unit tests for time-series forecasting evaluation metrics."""

import numpy as np
import pytest

from pathlib import Path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forecaster.evaluator import compute_forecast_metrics


def test_perfect_prediction_metrics():
    y_true = np.array([100.0, 105.0, 110.0, 108.0, 112.0])
    metrics = compute_forecast_metrics(y_true, y_true)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["directional_accuracy"] == 100.0


def test_compute_metrics_with_error():
    y_true = np.array([100.0, 110.0, 120.0])
    y_pred = np.array([102.0, 108.0, 122.0])

    metrics = compute_forecast_metrics(y_true, y_pred)
    assert metrics["rmse"] == 2.0
    assert metrics["mae"] == 2.0
    assert metrics["mape"] > 0.0
    assert metrics["r2"] > 0.90
    assert "directional_accuracy" in metrics
