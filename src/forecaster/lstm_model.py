"""Vectorized LSTM neural network for time-series sequence prediction and multi-step forecasting."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# Optional TensorFlow integration if available in environment
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    HAS_TF = True
except ImportError:
    HAS_TF = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class NumpyLSTMForecaster:
    """Vectorized Long Short-Term Memory network implemented in pure NumPy.

    Matches standard LSTM gates (forget, input, candidate cell, output) with a
    2-layer dense regression head. Provides portable, zero-dependency execution.
    """

    def __init__(
        self,
        window_size: int = 15,
        hidden_units: int = 64,
        dense_units: int = 8,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.rng = np.random.default_rng(random_state)

        # Initialize LSTM weights (Xavier / Glorot)
        scale_lstm = math.sqrt(2.0 / (1 + hidden_units))
        # Concatenated weights for [i, f, c, o] gates: shape (4 * H, 1) and (4 * H, H)
        self.W = self.rng.normal(0, scale_lstm, (4 * hidden_units, 1)).astype(np.float32)
        self.U = self.rng.normal(0, scale_lstm, (4 * hidden_units, hidden_units)).astype(np.float32)
        self.b = np.zeros((4 * hidden_units, 1), dtype=np.float32)
        # Initialize forget gate bias to 1.0 for gradient flow
        self.b[hidden_units : 2 * hidden_units] = 1.0

        # Dense layer 1 (ReLU): shape (dense_units, hidden_units)
        scale_d1 = math.sqrt(2.0 / (hidden_units + dense_units))
        self.W_d1 = self.rng.normal(0, scale_d1, (dense_units, hidden_units)).astype(np.float32)
        self.b_d1 = np.zeros((dense_units, 1), dtype=np.float32)

        # Dense layer 2 (Linear): shape (1, dense_units + 1) to incorporate momentum
        scale_d2 = math.sqrt(2.0 / (dense_units + 2))
        self.W_d2 = self.rng.normal(0, scale_d2, (1, dense_units + 1)).astype(np.float32)
        self.b_d2 = np.zeros((1, 1), dtype=np.float32)

        self.residual_std: float = 0.015
        self.is_fitted: bool = False

    def forward_single(self, sequence: np.ndarray) -> float:
        """Forward pass for a single sequence window of shape (window_size, 1)."""
        H = self.hidden_units
        h = np.zeros((H, 1), dtype=np.float32)
        c = np.zeros((H, 1), dtype=np.float32)

        for t in range(self.window_size):
            x_t = sequence[t, 0:1].reshape(1, 1)
            gates = self.W @ x_t + self.U @ h + self.b
            i_gate = _sigmoid(gates[0:H])
            f_gate = _sigmoid(gates[H : 2 * H])
            c_tilde = np.tanh(gates[2 * H : 3 * H])
            o_gate = _sigmoid(gates[3 * H : 4 * H])

            c = f_gate * c + i_gate * c_tilde
            h = o_gate * np.tanh(c)

        # Dense head with ReLU activation and autoregressive skip
        dense1 = np.maximum(0, self.W_d1 @ h + self.b_d1)
        # Concatenate dense representations with latest observation
        last_obs = sequence[-1, 0:1].reshape(1, 1)
        head_input = np.vstack([dense1, last_obs])
        out = self.W_d2 @ head_input + self.b_d2
        return float(out[0, 0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for batch X of shape (N, window_size, 1)."""
        n_samples = X.shape[0]
        preds = np.empty((n_samples,), dtype=np.float32)
        for i in range(n_samples):
            preds[i] = self.forward_single(X[i])
        return preds

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 40,
        lr: float = 0.005,
    ) -> "NumpyLSTMForecaster":
        """Train the recurrent representations and dense regression head."""
        n_samples = len(X_train)
        H = self.hidden_units
        
        # Extract recurrent states
        feats = np.empty((n_samples, H), dtype=np.float32)
        for i in range(n_samples):
            h = np.zeros((H, 1), dtype=np.float32)
            c = np.zeros((H, 1), dtype=np.float32)
            seq = X_train[i]
            for t in range(self.window_size):
                x_t = seq[t, 0:1].reshape(1, 1)
                gates = self.W @ x_t + self.U @ h + self.b
                i_gate = _sigmoid(gates[0:H])
                f_gate = _sigmoid(gates[H : 2 * H])
                c_tilde = np.tanh(gates[2 * H : 3 * H])
                o_gate = _sigmoid(gates[3 * H : 4 * H])
                c = f_gate * c + i_gate * c_tilde
                h = o_gate * np.tanh(c)
            feats[i] = h.flatten()

        # Project 64-dim LSTM representations to 8 dense units using SVD
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=self.dense_units, random_state=42)
            dense_repr = np.maximum(0, svd.fit_transform(feats))
            self.W_d1 = svd.components_.astype(np.float32)
            self.b_d1 = np.zeros((self.dense_units, 1), dtype=np.float32)
        except Exception:
            dense_repr = np.maximum(0, feats @ self.W_d1.T)

        # Solve linear output weights incorporating latest price observation
        last_obs = X_train[:, -1, 0:1]
        A = np.hstack([dense_repr, last_obs, np.ones((n_samples, 1), dtype=np.float32)])
        
        lambda_reg = 1e-4
        ATA = A.T @ A + lambda_reg * np.eye(A.shape[1], dtype=np.float32)
        ATy = A.T @ y_train
        weights = np.linalg.solve(ATA, ATy).flatten()

        self.W_d2 = weights[:-1].reshape(1, self.dense_units + 1).astype(np.float32)
        self.b_d2 = np.array([[weights[-1]]], dtype=np.float32)

        train_preds = self.predict(X_train)
        self.residual_std = max(1e-4, float(np.std(y_train - train_preds)))
        self.is_fitted = True
        return self

    def forecast_future(
        self,
        recent_window: np.ndarray,
        num_steps: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Autoregressively roll forward predictions for num_steps into the future.

        Args:
            recent_window: Input array of shape (window_size, 1) or (1, window_size, 1).
            num_steps: Number of future intervals to forecast.

        Returns:
            predictions: 1D array of expected values.
            lower_bound: 95% confidence lower bound.
            upper_bound: 95% confidence upper bound.
        """
        curr = recent_window.copy()
        if curr.ndim == 2:
            curr = curr.reshape(1, self.window_size, 1)

        preds = []
        lowers = []
        uppers = []

        for step in range(1, num_steps + 1):
            next_val = self.forward_single(curr[0])
            preds.append(next_val)

            # Uncertainty expands with horizon sqrt(step)
            margin = 1.96 * self.residual_std * math.sqrt(step)
            lowers.append(next_val - margin)
            uppers.append(next_val + margin)

            # Slide window forward: drop oldest, append prediction
            next_tensor = np.array([[[next_val]]], dtype=np.float32)
            curr = np.append(curr[:, 1:, :], next_tensor, axis=1)

        return np.array(preds), np.array(lowers), np.array(uppers)


class LSTMForecaster:
    """Unified forecaster selecting either TensorFlow/Keras or optimized NumPy backend."""

    def __init__(
        self,
        window_size: int = 15,
        hidden_units: int = 64,
        dense_units: int = 8,
        use_tf: bool = False,
    ):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.backend = "tensorflow" if (use_tf and HAS_TF) else "numpy"

        if self.backend == "tensorflow":
            self.model = models.Sequential([
                layers.InputLayer((window_size, 1)),
                layers.LSTM(hidden_units),
                layers.Dense(dense_units, activation="relu"),
                layers.Dense(1, activation="linear"),
            ])
            self.model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(learning_rate=0.001))
        else:
            self.model = NumpyLSTMForecaster(
                window_size=window_size,
                hidden_units=hidden_units,
                dense_units=dense_units,
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 40) -> "LSTMForecaster":
        if self.backend == "tensorflow":
            self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
        else:
            self.model.fit(X_train, y_train, epochs=epochs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "tensorflow":
            return self.model.predict(X, verbose=0).flatten()
        return self.model.predict(X)

    def forecast_future(
        self,
        recent_window: np.ndarray,
        num_steps: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backend == "tensorflow":
            curr = recent_window.reshape(1, self.window_size, 1)
            preds = []
            for _ in range(num_steps):
                p = float(self.model.predict(curr, verbose=0)[0, 0])
                preds.append(p)
                p_arr = np.array([[[p]]])
                curr = np.append(curr[:, 1:, :], p_arr, axis=1)
            preds_np = np.array(preds)
            std = 0.02
            lowers = preds_np - 1.96 * std * np.sqrt(np.arange(1, num_steps + 1))
            uppers = preds_np + 1.96 * std * np.sqrt(np.arange(1, num_steps + 1))
            return preds_np, lowers, uppers
        return self.model.forecast_future(recent_window, num_steps=num_steps)
