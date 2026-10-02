"""Data loading, normalization, and sliding window sequence generation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesScaler:
    """Min-Max normalizer for single-feature time-series sequences."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]
        self.data_min_: float = 0.0
        self.data_max_: float = 1.0

    def fit(self, data: np.ndarray | pd.Series) -> "TimeSeriesScaler":
        arr = np.asarray(data).flatten()
        self.data_min_ = float(np.min(arr))
        self.data_max_ = float(np.max(arr))
        if self.data_max_ == self.data_min_:
            self.data_max_ += 1e-7
        return self

    def transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(data)
        scale = (self.max_val - self.min_val) / (self.data_max_ - self.data_min_)
        return (arr - self.data_min_) * scale + self.min_val

    def fit_transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(data)
        scale = (self.max_val - self.min_val) / (self.data_max_ - self.data_min_)
        return (arr - self.min_val) / scale + self.data_min_


def load_time_series_data(file_path: str | Path, target_col: str = "Open") -> pd.DataFrame:
    """Load and parse time series CSV dataset with chronological ordering."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Time-series data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {list(df.columns)}")

    return df


def df_to_X_y(series: np.ndarray | pd.Series, window_size: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D sequence into sliding window input-output pairs (X, y).

    Args:
        series: 1D array or series of time series observations.
        window_size: Number of past time steps to condition on.

    Returns:
        X: 3D array of shape (N - window_size, window_size, 1).
        y: 1D array of shape (N - window_size,).
    """
    arr = np.asarray(series).flatten()
    n_samples = len(arr) - window_size
    if n_samples <= 0:
        raise ValueError(f"Series length ({len(arr)}) must be greater than window size ({window_size}).")

    X = np.empty((n_samples, window_size, 1), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        X[i, :, 0] = arr[i : i + window_size]
        y[i] = arr[i + window_size]

    return X, y


def generate_future_dates(last_date: pd.Timestamp | str, num_steps: int = 30) -> pd.DatetimeIndex:
    """Generate daily forecast timestamps starting from the day after last_date."""
    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=num_steps, freq="D")
