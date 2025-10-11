"""Prefect flow orchestrating the ML pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.ml.modeling import METRICS_PATH, train_models
from src.pipeline.config import CONFIG
from src.pipeline.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_METADATA_PATH = CONFIG.reports_dir / "model_run_metadata.json"


@task(name="load_processed_dataset")
def load_processed_dataset_task() -> pd.DataFrame:
    processed_path = CONFIG.processed_path
    if not processed_path.exists():
        raise FileNotFoundError(
            "Processed dataset missing. Run the data pipeline before the ML pipeline."
        )
    df = pd.read_parquet(processed_path)
    logger.info("Processed dataset loaded", extra={"shape": df.shape})
    return df


@task(name="train_models")
def train_models_task(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return train_models(df)


@task(name="log_model_metadata")
def log_model_metadata_task(metrics: dict[str, dict[str, float]]) -> None:
    metadata_entry = {
        "run_time": datetime.utcnow().isoformat(),
        "metrics_path": str(METRICS_PATH),
        "metrics": metrics,
        "models": {
            "logistic_regression": str((CONFIG.artifacts_dir / "logistic_regression.joblib")),
            "gradient_boosting": str((CONFIG.artifacts_dir / "gradient_boosting.joblib")),
        },
    }
    if MODEL_METADATA_PATH.exists():
        try:
            history = json.loads(MODEL_METADATA_PATH.read_text())
            if isinstance(history, list):
                history.append(metadata_entry)
            else:
                history = [history, metadata_entry]
        except json.JSONDecodeError:
            history = [metadata_entry]
    else:
        history = [metadata_entry]
    MODEL_METADATA_PATH.write_text(json.dumps(history[-20:], indent=2))
    logger.info("Model metadata written", extra={"path": str(MODEL_METADATA_PATH)})


@flow(name="ml-pipeline")
def ml_pipeline_flow() -> None:
    df = load_processed_dataset_task()
    metrics = train_models_task(df)
    log_model_metadata_task(metrics)


if __name__ == "__main__":
    ml_pipeline_flow()
