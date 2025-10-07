"""FastAPI entrypoint for the churn analytics platform."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException

from src.api.schemas import (
    PipelineRun,
    PipelineTriggerRequest,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import ModelService
from src.common.logging_utils import get_logger
from src.common.paths import REPORTS_DIR, ROOT_DIR
from src.data_pipeline.flow import build_data_pipeline_flow
from src.data_pipeline.metadata_store import fetch_recent_runs
from src.ml_pipeline.flow import build_ml_pipeline_flow

logger = get_logger("api.main")

app = FastAPI(
    title="Churn Analytics Platform",
    version="1.0.0",
    description="Cloud-ready data and ML pipelines orchestrated with Prefect.",
)

router = APIRouter(prefix="/api")


@app.on_event("startup")
async def startup_event() -> None:
    """Warm up artifacts on startup."""
    try:
        ModelService.get_instance()
    except FileNotFoundError as exc:
        logger.warning("Artifacts not ready at startup", extra={"error": str(exc)})


@router.get("/health")
async def health() -> dict:
    """Basic health probe."""
    return {"status": "ok"}


@router.get("/pipelines", response_model=List[PipelineRun])
async def list_pipeline_runs(
    pipeline_name: Optional[str] = None, limit: int = 10
) -> List[PipelineRun]:
    """Fetch recent pipeline runs from metadata store."""
    runs = fetch_recent_runs(pipeline_name, limit)
    return [PipelineRun(**run) for run in runs]


async def _execute_flow(flow_callable) -> None:
    """Execute Prefect flow in background thread."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, flow_callable)


@router.post("/pipelines/run")
async def trigger_pipeline(payload: PipelineTriggerRequest) -> dict:
    """Trigger data or ML pipeline asynchronously."""
    if payload.pipeline_name == "data-pipeline":
        asyncio.create_task(_execute_flow(build_data_pipeline_flow))
    else:
        asyncio.create_task(_execute_flow(build_ml_pipeline_flow))
    return {"status": "scheduled", "pipeline": payload.pipeline_name}


@router.get("/metrics/model")
async def get_model_metrics() -> dict:
    """Return metrics of the current champion model."""
    metrics_path = REPORTS_DIR / "ml"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics unavailable.")

    aggregated = {}
    for model_dir in metrics_path.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            aggregated[model_dir.name] = json.loads(metrics_file.read_text(encoding="utf-8"))
    if not aggregated:
        raise HTTPException(status_code=404, detail="No metrics found.")
    return aggregated


@router.get("/reports/eda")
async def list_eda_artifacts() -> dict:
    """Return generated EDA artifacts for dashboards."""
    eda_dir = REPORTS_DIR / "eda"
    if not eda_dir.exists():
        raise HTTPException(status_code=404, detail="EDA artifacts not found.")
    assets = [
        path.relative_to(REPORTS_DIR).as_posix()
        for path in eda_dir.rglob("*")
        if path.is_file()
    ]
    return {"artifacts": assets}


@router.get("/app/details")
async def application_details() -> dict:
    """Expose key application metadata (at least four details)."""
    details = {}

    runs = fetch_recent_runs(limit=1)
    details["latest_pipeline_run"] = runs[0] if runs else {}

    deployments = []
    for deployment_file in (ROOT_DIR / "deployments").glob("*.yaml"):
        deployments.append(
            {
                "name": deployment_file.stem,
                "path": deployment_file.relative_to(ROOT_DIR).as_posix(),
            }
        )
    details["deployments"] = deployments

    model_metadata_path = Path("artifacts/ml_pipeline/model_metadata.json")
    if model_metadata_path.exists():
        details["model_metadata"] = json.loads(model_metadata_path.read_text(encoding="utf-8"))

    data_quality = {
        "summary_stats": "reports/data_quality/summary_statistics.parquet",
        "missing_report": "reports/data_quality/missing_values.parquet",
        "schema": "reports/data_quality/schema.json",
    }
    details["data_quality_artifacts"] = data_quality

    return details


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Score a single customer record."""
    service = ModelService.get_instance()
    try:
        result = service.predict(request.dict(by_alias=True))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictionResponse(customerID=request.customer_id, **result)


@router.post("/model/reload")
async def reload_model() -> dict:
    """Reload artifacts after retraining without restarting the service."""
    service = ModelService.get_instance()
    service.reload()
    return {"status": "reloaded"}


app.include_router(router)

