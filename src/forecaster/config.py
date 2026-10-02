"""Configuration parameters for LSTM time-series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ForecastConfig:
    """Hyperparameters and file settings for LSTM time-series modeling."""

    def __init__(
        self,
        window_size: int = 15,
        hidden_units: int = 64,
        dense_units: int = 8,
        future_steps: int = 30,
        train_split: float = 0.8,
        learning_rate: float = 0.001,
        epochs: int = 50,
        target_column: str = "Open",
        data_path: Optional[str | Path] = None,
    ):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.future_steps = future_steps
        self.train_split = train_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.target_column = target_column

        base_dir = Path(__file__).parent.parent.parent
        self.data_path = Path(data_path) if data_path else base_dir / "datafile.csv"
        self.combined_results_path = base_dir / "combined_results.csv"
        self.future_results_path = base_dir / "future_results.csv"


CONFIG = ForecastConfig()
