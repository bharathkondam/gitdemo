"""Prefect flow orchestrating the data pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.pipeline.config import CONFIG
from src.pipeline.ingestion import download_dataset, load_dataset
from src.pipeline.preprocessing import preprocess_dataset
from src.pipeline.eda import run_eda
from src.pipeline.logging_utils import get_logger

logger = get_logger(__name__)

METADATA_PATH = CONFIG.reports_dir / "run_metadata.json"


@task(name="download_dataset")
def download_dataset_task() -> str:
    path = download_dataset()
    return str(path)


@task(name="preprocess_dataset")
def preprocess_dataset_task(raw_path: str) -> str:
    df = load_dataset(Path(raw_path))
    result = preprocess_dataset(df)
    return str(result.processed_path)


@task(name="run_eda")
def run_eda_task(processed_path: str) -> dict[str, str]:
    df = pd.read_parquet(processed_path)
    artifacts = run_eda(df)
    return {key: str(path) for key, path in artifacts.items()}


@task(name="update_metadata")
def update_metadata_task(raw_path: str, processed_path: str, eda_artifacts: dict[str, str]) -> None:
    metadata = {
        "last_run": datetime.utcnow().isoformat(),
        "raw_path": raw_path,
        "processed_path": processed_path,
        "eda_artifacts": eda_artifacts,
    }
    if Path(processed_path).exists():
        df = pd.read_parquet(processed_path)
        metadata["processed_rows"] = int(len(df))
        metadata["processed_columns"] = int(len(df.columns))
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists():
        try:
            history = json.loads(METADATA_PATH.read_text())
            if isinstance(history, list):
                history.append(metadata)
            else:
                history = [history, metadata]
        except json.JSONDecodeError:
            history = [metadata]
    else:
        history = [metadata]
    METADATA_PATH.write_text(json.dumps(history[-20:], indent=2))
    logger.info("Metadata updated", extra={"metadata_path": str(METADATA_PATH)})


@flow(name="data-pipeline")
def data_pipeline_flow() -> None:
    raw_path = download_dataset_task()
    processed_path = preprocess_dataset_task(raw_path)
    eda_artifacts = run_eda_task(processed_path)
    update_metadata_task(raw_path, processed_path, eda_artifacts)


if __name__ == "__main__":
    data_pipeline_flow()
