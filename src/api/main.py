"""FastAPI service exposing pipeline details."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from src.ml.modeling import METRICS_PATH
from src.pipeline.config import CONFIG

app = FastAPI(title="Churn Analytics API", version="1.0.0")

METADATA_PATH = CONFIG.reports_dir / "run_metadata.json"
MODEL_METADATA_PATH = CONFIG.reports_dir / "model_run_metadata.json"


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return json.loads(path.read_text())


@app.get("/api/details")
def pipeline_details() -> Dict[str, Any]:
    history = _load_json(METADATA_PATH)
    latest = history[-1] if isinstance(history, list) else history
    details = {
        "last_run": latest.get("last_run"),
        "raw_path": latest.get("raw_path"),
        "processed_rows": latest.get("processed_rows"),
        "processed_columns": latest.get("processed_columns"),
        "eda_artifacts": latest.get("eda_artifacts", {}),
    }
    return details


@app.get("/api/metrics")
def model_metrics() -> Dict[str, Any]:
    metrics = _load_json(METRICS_PATH)
    return metrics


@app.get("/api/model-runs")
def model_runs() -> Dict[str, Any]:
    return {"history": _load_json(MODEL_METADATA_PATH)}


@app.get("/api/artifacts")
def artifact_listing() -> Dict[str, Any]:
    figures = sorted(str(p) for p in (CONFIG.reports_dir / "figures").glob("*.png"))
    models = sorted(str(p) for p in CONFIG.artifacts_dir.glob("*.joblib"))
    return {"figures": figures, "models": models}
