"""Flask + Dash Web Application for Time-Series LSTM Forecasting."""

from pathlib import Path
import sys

# Add root directory to sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.forecaster.pipeline import run_forecast_pipeline

# Ensure results CSV files exist; generate on the fly if missing
combined_path = BASE_DIR / "combined_results.csv"
future_path = BASE_DIR / "future_results.csv"

if not combined_path.exists() or not future_path.exists():
    print("[INIT] combined_results.csv or future_results.csv missing. Generating via LSTM pipeline...")
    run_forecast_pipeline()

combined_results = pd.read_csv(combined_path)
future_results = pd.read_csv(future_path)

try:
    from flask import Flask
    from dash import Dash, dcc, html
    import plotly.graph_objs as go
    HAS_DASH = True
except ImportError:
    HAS_DASH = False


if HAS_DASH:
    # Initialize Flask server and Dash app
    server = Flask(__name__)
    app = Dash(__name__, server=server)

    app.layout = html.Div(
        style={"backgroundColor": "#0f172a", "minHeight": "100vh", "padding": "2rem", "fontFamily": "sans-serif"},
        children=[
            html.H1(
                "DeepForecaster: Bitcoin LSTM Time-Series Dashboard",
                style={"color": "#38bdf8", "textAlign": "center", "marginBottom": "0.5rem"},
            ),
            html.P(
                "Historical actuals, in-sample LSTM reconstructions, and 30-day future projections.",
                style={"color": "#94a3b8", "textAlign": "center", "marginBottom": "2rem"},
            ),
            dcc.Graph(
                id="predictions-plot",
                figure={
                    "data": [
                        go.Scatter(
                            x=combined_results["Date"],
                            y=combined_results["Actuals"],
                            mode="lines",
                            name="Actual Bitcoin Price",
                            line=dict(color="#38bdf8", width=2),
                        ),
                        go.Scatter(
                            x=combined_results["Date"],
                            y=combined_results["Train Predictions"],
                            mode="lines",
                            name="LSTM In-Sample Prediction",
                            line=dict(color="#22c55e", width=1.5, dash="dot"),
                        ),
                        go.Scatter(
                            x=future_results["Date"],
                            y=future_results["Predictions"],
                            mode="lines",
                            name="30-Day Future Forecast",
                            line=dict(color="#f59e0b", width=2.5),
                        ),
                        go.Scatter(
                            x=future_results["Date"],
                            y=future_results.get("Upper_Bound", future_results["Predictions"]),
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="none",
                        ),
                        go.Scatter(
                            x=future_results["Date"],
                            y=future_results.get("Lower_Bound", future_results["Predictions"]),
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(245, 158, 11, 0.15)",
                            line=dict(width=0),
                            name="95% Confidence Band",
                        ),
                    ],
                    "layout": go.Layout(
                        title="Bitcoin (BTC-USD) LSTM Predictions vs. Actuals",
                        template="plotly_dark",
                        paper_bgcolor="#1e293b",
                        plot_bgcolor="#1e293b",
                        xaxis={"title": "Date", "gridcolor": "#334155"},
                        yaxis={"title": "USD ($)", "gridcolor": "#334155"},
                        margin={"l": 60, "r": 20, "t": 60, "b": 60},
                        legend={"x": 0.02, "y": 0.98},
                        hovermode="x unified",
                    ),
                },
            ),
        ],
    )

    if __name__ == "__main__":
        print("Starting Dash + Flask server on http://0.0.0.0:8050...")
        app.run_server(host="0.0.0.0", port=8050, debug=False)

else:
    print("[INFO] Flask/Dash not installed in this environment.")
    print("[INFO] Use 'streamlit run streamlit_app.py' or install Dash via 'pip install flask dash'.")