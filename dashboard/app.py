"""Streamlit dashboard for the Prefect-based churn platform."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline.config import CONFIG

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")
st.title("Churn Analytics Dashboard (Prefect)")

log_file = CONFIG.logs_dir / "pipeline.log"
metadata_file = CONFIG.reports_dir / "run_metadata.json"
model_metadata_file = CONFIG.reports_dir / "model_run_metadata.json"
metrics_file = CONFIG.reports_dir / "metrics.json"

st.header("Pipeline Logs")
if log_file.exists():
    st.code(log_file.read_text(), language="text")
else:
    st.info("No logs yet. Run the Prefect flows to generate logs.")

st.header("Run Metadata")
if metadata_file.exists():
    history = json.loads(metadata_file.read_text())
    st.json(history[-5:] if isinstance(history, list) else history)
else:
    st.warning("Metadata file not found. Run the data pipeline.")

st.header("Model Metrics")
if metrics_file.exists():
    metrics = json.loads(metrics_file.read_text())
    st.dataframe(pd.DataFrame(metrics).T)
else:
    st.warning("Metrics not available. Run the ML pipeline.")

st.header("EDA Figures")
figures_dir = CONFIG.reports_dir / "figures"
if figures_dir.exists():
    for image_path in sorted(figures_dir.glob("*.png")):
        st.subheader(image_path.stem.replace("_", " ").title())
        st.image(str(image_path))
else:
    st.info("EDA figures will appear after the data pipeline runs.")

st.header("Model Runs History")
if model_metadata_file.exists():
    history = json.loads(model_metadata_file.read_text())
    st.json(history[-5:] if isinstance(history, list) else history)
else:
    st.info("Model runs metadata not yet generated.")
