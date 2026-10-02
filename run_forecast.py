"""Command-line runner for LSTM time-series modeling and future forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.forecaster.config import ForecastConfig
from src.forecaster.pipeline import run_forecast_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepForecaster: LSTM Predictive Time-Series CLI")
    parser.add_argument("--data-path", default="datafile.csv", help="Path to input time series CSV")
    parser.add_argument("--window-size", type=int, default=15, help="Sliding window sequence size")
    parser.add_argument("--future-steps", type=int, default=30, help="Number of future steps to forecast")
    parser.add_argument("--hidden-units", type=int, default=64, help="LSTM hidden units")
    parser.add_argument("--target-col", default="Open", help="Target price/volume column")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")

    args = parser.parse_args()

    print("=" * 65)
    print("[FORECAST] DEEPFORECASTER: LSTM PREDICTIVE TIME-SERIES RUNNER")
    print(f"Data: {args.data_path} | Target: {args.target_col} | Window: {args.window_size} days")
    print(f"Horizon: {args.future_steps} future steps | Units: {args.hidden_units} LSTM units")
    print("=" * 65)

    cfg = ForecastConfig(
        window_size=args.window_size,
        hidden_units=args.hidden_units,
        future_steps=args.future_steps,
        epochs=args.epochs,
        target_column=args.target_col,
        data_path=args.data_path,
    )

    print("\n[PIPELINE] Training LSTM Network & Generating Future Forecasts...")
    results = run_forecast_pipeline(config=cfg, save_outputs=True)

    metrics = results["metrics"]
    print("\n" + "=" * 65)
    print("[EVALUATION METRICS] IN-SAMPLE & VALIDATION BENCHMARKS")
    print("=" * 65)
    print(f"  - Root Mean Squared Error (RMSE): ${metrics['rmse']:,.2f}")
    print(f"  - Mean Absolute Error (MAE):       ${metrics['mae']:,.2f}")
    print(f"  - Mean Absolute Pct Error (MAPE):  {metrics['mape']:.2f}%")
    print(f"  - R-squared Score (R2):            {metrics['r2']:.4f}")
    print(f"  - Directional Accuracy (Hit Rate): {metrics['directional_accuracy']:.2f}%")

    future_df = results["future_results"]
    print("\n" + "=" * 65)
    print(f"[FUTURE FORECASTS] NEXT {args.future_steps} DAYS PROJECTIONS")
    print("=" * 65)
    print(future_df.head(10).to_string(index=False))
    print("...")
    print(future_df.tail(5).to_string(index=False))

    print(f"\n[OUTPUT] Persisted: combined_results.csv ({len(results['combined_results'])} rows)")
    print(f"[OUTPUT] Persisted: future_results.csv   ({len(future_df)} rows)")
    print("\nExecution complete.")


if __name__ == "__main__":
    main()
