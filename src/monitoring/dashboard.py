"""Streamlit dashboard for observing pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_pipeline.metadata_store import fetch_recent_runs

st.set_page_config(page_title="Churn Platform Dashboard", layout="wide")
st.title("Churn Analytics Platform · Monitoring")

st.sidebar.header("Controls")
pipeline_filter = st.sidebar.selectbox(
    "Pipeline", ["all", "data-pipeline", "ml-pipeline"], index=0
)
limit = st.sidebar.slider("Recent Runs", 5, 50, 20)

pipeline_name = None if pipeline_filter == "all" else pipeline_filter
runs = fetch_recent_runs(pipeline_name, limit)
st.subheader("Pipeline Runs")
if runs:
    st.dataframe(pd.DataFrame(runs))
else:
    st.info("No pipeline runs recorded yet.")

st.subheader("Latest Model Metrics")
metrics_root = Path("reports/ml")
if metrics_root.exists():
    for model_dir in metrics_root.iterdir():
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            st.markdown(f"**{model_dir.name}**")
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            st.json(metrics)
else:
    st.warning("Model metrics not available. Run the ML pipeline.")

st.subheader("Recent Logs")
log_file = Path("logs/pipeline.log")
if log_file.exists():
    tail_lines = log_file.read_text(encoding="utf-8").splitlines()[-200:]
    st.code("\n".join(tail_lines))
else:
    st.info("Log file not found.")

