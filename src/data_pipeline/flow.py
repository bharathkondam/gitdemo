"""Prefect orchestration for the data pipeline."""

from __future__ import annotations

from typing import Dict

from prefect import flow
from prefect.context import get_run_context

from src.common.config import get_settings
from src.common.logging_utils import configure_logging, get_logger
from .metadata_store import record_pipeline_completion, record_pipeline_start
from .tasks import (
    checksum_dataset,
    ingest_dataset,
    load_raw_dataset,
    preprocess_dataset,
    run_eda_task,
)

logger = get_logger("data_pipeline.flow")


@flow(
    name="data-pipeline",
    retries=0,
    timeout_seconds=900,
    log_prints=False,
)
def build_data_pipeline_flow() -> Dict[str, str]:
    """Execute the end-to-end data engineering workflow."""
    configure_logging()
    settings = get_settings()
    context = get_run_context()
    run_id_obj = getattr(getattr(context, "flow_run", None), "id", "local-run")
    run_id = str(run_id_obj)
    checksum = None

    record_pipeline_start("data-pipeline", run_id)
    logger.info(
        "Starting data pipeline",
        extra={"run_id": run_id, "interval_seconds": settings.prefect_deployment_interval_seconds},
    )

    try:
        dataset_path = ingest_dataset()
        checksum = checksum_dataset(dataset_path)
        raw_df = load_raw_dataset(dataset_path)
        processed_df, metadata, dq_artifacts = preprocess_dataset(raw_df)
        eda_artifacts = run_eda_task(raw_df, metadata["target_column"])

        artifacts = {
            "checksum": checksum,
            "processed_dataset": dq_artifacts.processed_path.as_posix(),
            "summary_stats": dq_artifacts.summary_path.as_posix(),
            "missing_report": dq_artifacts.missing_path.as_posix(),
            "schema_report": dq_artifacts.schema_path.as_posix(),
            "eda_artifacts": {key: value.as_posix() for key, value in eda_artifacts.items()},
        }

        logger.info(
            "Data pipeline completed",
            extra={"records": len(processed_df), "processed_path": artifacts["processed_dataset"]},
        )
        record_pipeline_completion(run_id, "success", checksum, artifacts)
        return artifacts
    except Exception:
        logger.exception("Data pipeline failed", extra={"run_id": run_id})
        record_pipeline_completion(run_id, "failed", checksum, {})
        raise


if __name__ == "__main__":
    build_data_pipeline_flow()
