"""Unit tests for LSTM forward pass, training, and autoregressive forecasting."""

import numpy as np
import pytest

from pathlib import Path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forecaster.lstm_model import LSTMForecaster, NumpyLSTMForecaster


def test_numpy_lstm_initialization_and_forward():
    model = NumpyLSTMForecaster(window_size=10, hidden_units=32, dense_units=4)
    sample_seq = np.random.randn(10, 1).astype(np.float32)

    pred = model.forward_single(sample_seq)
    assert isinstance(pred, float)
    assert not np.isnan(pred)


def test_numpy_lstm_batch_predict():
    model = NumpyLSTMForecaster(window_size=10, hidden_units=32, dense_units=4)
    batch = np.random.randn(5, 10, 1).astype(np.float32)

    preds = model.predict(batch)
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()


def test_numpy_lstm_fit():
    # Synthetic trending series
    X = np.zeros((30, 8, 1), dtype=np.float32)
    y = np.zeros((30,), dtype=np.float32)
    for i in range(30):
        X[i, :, 0] = np.linspace(i, i + 8, 8)
        y[i] = i + 9

    model = NumpyLSTMForecaster(window_size=8, hidden_units=16, dense_units=4)
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert preds.shape == (30,)
    assert np.corrcoef(y, preds)[0, 1] > 0.5


def test_autoregressive_future_forecast_bounds():
    model = NumpyLSTMForecaster(window_size=10, hidden_units=32, dense_units=4)
    recent = np.ones((10, 1), dtype=np.float32)

    preds, lowers, uppers = model.forecast_future(recent, num_steps=15)
    assert len(preds) == 15
    assert len(lowers) == 15
    assert len(uppers) == 15

    # Check that upper >= pred >= lower
    assert (uppers >= preds).all()
    assert (preds >= lowers).all()

    # Check uncertainty cone expands over time
    margins = uppers - lowers
    assert margins[-1] > margins[0]


def test_unified_lstm_forecaster_wrapper():
    wrapper = LSTMForecaster(window_size=5, hidden_units=16, dense_units=4)
    X = np.random.randn(10, 5, 1).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)

    wrapper.fit(X, y)
    preds = wrapper.predict(X)
    assert len(preds) == 10

    fut_preds, l, u = wrapper.forecast_future(X[-1], num_steps=5)
    assert len(fut_preds) == 5
