"""Unit tests for the end-to-end forecasting pipeline and CSV output contracts."""

from pathlib import Path
import pandas as pd
import pytest

import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forecaster.config import ForecastConfig
from src.forecaster.pipeline import run_forecast_pipeline


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("pipeline_test")
    cfg = ForecastConfig(
        window_size=10,
        future_steps=15,
        hidden_units=32,
        data_path=BASE_DIR / "datafile.csv",
    )
    cfg.combined_results_path = tmp_dir / "combined_results.csv"
    cfg.future_results_path = tmp_dir / "future_results.csv"

    res = run_forecast_pipeline(config=cfg, save_outputs=True)
    return res, cfg


def test_pipeline_execution(pipeline_output):
    res, cfg = pipeline_output
    assert res is not None
    assert "metrics" in res
    assert res["metrics"]["r2"] > 0.80
    assert res["metrics"]["rmse"] < 2000.0


def test_combined_results_schema(pipeline_output):
    res, cfg = pipeline_output
    combined_csv = cfg.combined_results_path
    assert combined_csv.exists()

    df = pd.read_csv(combined_csv)
    assert "Date" in df.columns
    assert "Train Predictions" in df.columns
    assert "Actuals" in df.columns
    assert "Predictions" in df.columns
    assert len(df) > 300


def test_future_results_schema(pipeline_output):
    res, cfg = pipeline_output
    future_csv = cfg.future_results_path
    assert future_csv.exists()

    df = pd.read_csv(future_csv)
    assert len(df) == 15
    assert set(["Date", "Predictions", "Lower_Bound", "Upper_Bound"]).issubset(df.columns)
    assert not df["Predictions"].isna().any()
    assert (df["Upper_Bound"] >= df["Predictions"]).all()
    assert (df["Predictions"] >= df["Lower_Bound"]).all()
