FROM python:3.11-slim

LABEL maintainer="DeepForecaster Team"
LABEL description="Predictive Time-Series Analysis with LSTM and Dual-Engine Serving"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application assets and source code
COPY datafile.csv .
COPY combined_results.csv .
COPY future_results.csv .
COPY your_app.py .
COPY run_forecast.py .
COPY app.py .
COPY streamlit_app.py .
COPY src/ /app/src/

# Expose Dash (8050) and Streamlit (8501)
EXPOSE 8050
EXPOSE 8501

# Default: Run Flask + Dash application
CMD ["python", "your_app.py"]