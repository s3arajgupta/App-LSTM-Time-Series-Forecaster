"""Unit tests for time-series scaling, sliding window transformation, and date generation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forecaster.preprocessing import (
    TimeSeriesScaler,
    df_to_X_y,
    generate_future_dates,
    load_time_series_data,
)


def test_time_series_scaler():
    raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = TimeSeriesScaler()
    scaled = scaler.fit_transform(raw)

    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    assert len(scaled) == 5

    # Inverse transform identity
    inv = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(inv, raw, rtol=1e-5)


def test_df_to_X_y():
    series = np.arange(1, 21) # 20 elements
    window_size = 5

    X, y = df_to_X_y(series, window_size=window_size)
    assert X.shape == (15, 5, 1)
    assert y.shape == (15,)

    # First pair: input [1, 2, 3, 4, 5], target 6
    np.testing.assert_array_equal(X[0, :, 0], [1, 2, 3, 4, 5])
    assert y[0] == 6

    # Last pair: input [15, 16, 17, 18, 19], target 20
    np.testing.assert_array_equal(X[-1, :, 0], [15, 16, 17, 18, 19])
    assert y[-1] == 20


def test_df_to_X_y_invalid_length():
    short_series = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        df_to_X_y(short_series, window_size=5)


def test_load_time_series_data():
    df = load_time_series_data(BASE_DIR / "datafile.csv", target_col="Open")
    assert not df.empty
    assert "Date" in df.columns
    assert "Open" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    # Verify chronological ascending order
    assert df["Date"].is_monotonic_increasing


def test_generate_future_dates():
    last_date = pd.Timestamp("2023-10-07")
    fut_dates = generate_future_dates(last_date, num_steps=10)

    assert len(fut_dates) == 10
    assert fut_dates[0] == pd.Timestamp("2023-10-08")
    assert fut_dates[-1] == pd.Timestamp("2023-10-17")
