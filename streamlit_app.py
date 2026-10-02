"""DeepForecaster: Interactive Streamlit Studio for LSTM Time-Series Forecasting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add root directory to sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.forecaster.config import ForecastConfig
from src.forecaster.pipeline import run_forecast_pipeline


st.set_page_config(
    page_title="DeepForecaster | LSTM Time-Series Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .metric-badge-green {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">DeepForecaster: LSTM Time-Series Studio</div>', unsafe_allow_html=True)
st.markdown(
    """
    <span class="metric-badge">Recurrent Neural Network (LSTM)</span>
    <span class="metric-badge">Sliding Window (W=15)</span>
    <span class="metric-badge-green">Autoregressive Multi-Step Roll-Forward</span>
    <span class="metric-badge">95% Confidence Bounds</span>
    <span class="metric-badge">Dual-Engine Serving (Streamlit + Flask)</span>
    """,
    unsafe_allow_html=True,
)
st.caption("Deep predictive modeling of financial asset trajectories using Long Short-Term Memory networks and expanding uncertainty cones.")


@st.cache_data
def get_forecast_data(window_size: int, future_steps: int, hidden_units: int):
    cfg = ForecastConfig(
        window_size=window_size,
        future_steps=future_steps,
        hidden_units=hidden_units,
    )
    return run_forecast_pipeline(config=cfg, save_outputs=True)


# Sidebar controls
with st.sidebar:
    st.header("⚙️ Forecast Hyperparameters")
    w_size = st.slider("Sliding Window Size (Days)", 5, 30, 15, 1)
    f_steps = st.slider("Future Forecast Horizon (Days)", 7, 60, 30, 1)
    h_units = st.select_slider("LSTM Hidden Units", options=[32, 64, 128], value=64)

    st.markdown("---")
    if st.button("🚀 Re-compute Forecasts", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Data Source")
    st.write("📁 `datafile.csv` (Bitcoin Historical Daily)")
    st.write("Asset: **Bitcoin (BTC-USD)**")

# Run pipeline
with st.spinner("Computing LSTM forecasts..."):
    results = get_forecast_data(w_size, f_steps, h_units)

metrics = results["metrics"]
train_df = results["train_results"]
future_df = results["future_results"]
history_df = results["history_df"]

last_actual = train_df["Actuals"].iloc[-1]
final_forecast = future_df["Predictions"].iloc[-1]
pct_change = ((final_forecast - last_actual) / last_actual) * 100.0

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forecast & Scenario Studio",
    "🧠 LSTM Architecture & Math",
    "📊 Backtesting & Error Metrics",
    "🚀 Serving & Export Center",
])

# ----------------- TAB 1: FORECAST STUDIO -----------------
with tab1:
    st.subheader("Predictive Trajectory with Uncertainty Envelopes")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Observed Price", f"${last_actual:,.2f}")
    with col2:
        st.metric(
            f"{f_steps}-Day Target Forecast",
            f"${final_forecast:,.2f}",
            delta=f"{pct_change:+.2f}%",
        )
    with col3:
        st.metric("Model RMSE (USD)", f"${metrics['rmse']:,.2f}", delta="Low Variance")
    with col4:
        st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%", delta="Trend Alignment")

    # Main Interactive Plotly Forecast Chart
    fig = go.Figure()

    # Actuals
    fig.add_trace(
        go.Scatter(
            x=train_df["Date"],
            y=train_df["Actuals"],
            mode="lines",
            name="Actual BTC Price",
            line=dict(color="#38bdf8", width=2),
        )
    )

    # In-Sample LSTM Fits
    fig.add_trace(
        go.Scatter(
            x=train_df["Date"],
            y=train_df["Train Predictions"],
            mode="lines",
            name="In-Sample LSTM Fit",
            line=dict(color="#22c55e", width=1.5, dash="dot"),
        )
    )

    # Upper Confidence Bound
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Upper_Bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Lower Confidence Bound + Fill
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Lower_Bound"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.18)",
            line=dict(width=0),
            name="95% Confidence Envelope",
        )
    )

    # Future Forecast Line
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Predictions"],
            mode="lines+markers",
            name="Autoregressive Roll-Forward",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        title="Bitcoin (BTC-USD): Historical Actuals, LSTM In-Sample Fit, and Future Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        height=520,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Future Trajectory Breakdown")
    st.dataframe(future_df, use_container_width=True, height=220)

# ----------------- TAB 2: LSTM ARCHITECTURE & MATH -----------------
with tab2:
    st.subheader("Long Short-Term Memory (LSTM) Mathematical Formulations")

    st.markdown(
        r"""
        The LSTM neural cell overcomes the vanishing gradient problem of recurrent neural networks through gated state transitions:
        
        $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{(Forget Gate)}$$
        $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{(Input Gate)}$$
        $$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad \text{(Candidate Cell State)}$$
        $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State Memory Update)}$$
        $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{(Output Gate)}$$
        $$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden Recurrent Representation)}$$
        
        The final hidden state $h_T \in \mathbb{R}^{64}$ passes through a 2-layer regression head:
        $$\hat{y} = W_{d2} \, \text{ReLU}(W_{d1} h_T + b_{d1}) + b_{d2}$$
        """
    )

    st.markdown("### 🔄 Autoregressive Multi-Step Roll-Forward Mechanism")
    st.markdown(
        """
        ```mermaid
        graph LR
            W1["Window t: [x_{t-14}, ..., x_t]"] -->|LSTM Forward| P1["Forecast x_{t+1}"]
            P1 -->|Append & Slide Window| W2["Window t+1: [x_{t-13}, ..., x_{t+1}]"]
            W2 -->|LSTM Forward| P2["Forecast x_{t+2}"]
            P2 -->|Append & Slide Window| W3["Window t+2: [x_{t-12}, ..., x_{t+2}]"]
            W3 -->|Iterate N Steps| PF["Future Trajectory: [x_{t+1}, ..., x_{t+N}]"]

            style W1 fill:#1e293b,stroke:#38bdf8,color:#fff
            style P1 fill:#0f766e,stroke:#2dd4bf,color:#fff
            style W2 fill:#1e293b,stroke:#38bdf8,color:#fff
            style P2 fill:#0f766e,stroke:#2dd4bf,color:#fff
            style W3 fill:#1e293b,stroke:#38bdf8,color:#fff
            style PF fill:#b45309,stroke:#f59e0b,color:#fff
        ```
        """
    )

# ----------------- TAB 3: BACKTESTING & ERROR METRICS -----------------
with tab3:
    st.subheader("Model Evaluation & Residual Diagnostics")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        metrics_df = pd.DataFrame([
            {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"${metrics['rmse']:,.2f}", "Interpretation": "Average magnitude of error in USD"},
            {"Metric": "Mean Absolute Error (MAE)", "Value": f"${metrics['mae']:,.2f}", "Interpretation": "Robust average absolute dollar discrepancy"},
            {"Metric": "Mean Absolute Percentage Error (MAPE)", "Value": f"{metrics['mape']:.2f}%", "Interpretation": "Relative percentage error against asset price"},
            {"Metric": "R-squared Coefficient (R2)", "Value": f"{metrics['r2']:.4f}", "Interpretation": "Variance explained by recurrent representations"},
            {"Metric": "Directional Accuracy (Hit Rate)", "Value": f"{metrics['directional_accuracy']:.2f}%", "Interpretation": "% of days predicting correct up/down direction"},
        ])
        st.table(metrics_df)

    with col_m2:
        # Residuals Plot
        residuals = train_df["Actuals"] - train_df["Train Predictions"]
        fig_res = px.histogram(
            residuals,
            nbins=35,
            title="Residual Error Distribution (Actual - Fit)",
            template="plotly_dark",
            labels={"value": "Residual (USD)"},
            color_discrete_sequence=["#38bdf8"],
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Actual vs Predicted Scatter
    fig_scatter = px.scatter(
        train_df,
        x="Actuals",
        y="Train Predictions",
        title="Actual Price vs. Model Fitted Price",
        template="plotly_dark",
        labels={"Actuals": "Ground Truth BTC Price ($)", "Train Predictions": "LSTM Predicted Price ($)"},
        trendline="ols",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 4: SERVING & EXPORT -----------------
with tab4:
    st.subheader("Model Serving & Data Export Center")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.markdown("**Combined Results (History + Forecast)**")
        st.caption("Matches legacy `your_app.py` and container artifact specifications.")
        st.download_button(
            label="⬇️ Download combined_results.csv",
            data=results["combined_results"].to_csv(index=False),
            file_name="combined_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        st.markdown("**Future Forecasts with 95% Bounds**")
        st.caption("Includes date, expected forecast, lower bound, and upper bound.")
        st.download_button(
            label="⬇️ Download future_results.csv",
            data=future_df.to_csv(index=False),
            file_name="future_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🐳 Docker & Flask/Dash Serving Specifications")
    st.code(
        """
# Build Docker image
docker build -t deepforecaster:latest .

# Run container on port 8050
docker run -p 8050:8050 deepforecaster:latest

# Access Flask / Dash Web Application
http://localhost:8050
        """,
        language="bash",
    )
