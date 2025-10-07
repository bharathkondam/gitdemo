"""Prefect flow for the ML pipeline."""

from __future__ import annotations

from typing import Dict

from prefect import flow
from prefect.context import get_run_context

from src.common.logging_utils import configure_logging, get_logger
from src.data_pipeline.metadata_store import record_pipeline_completion, record_pipeline_start
from .tasks import (
    evaluate_models,
    load_processed_data,
    optimize_gradient_boosting,
    persist_champion,
    split_dataset,
    train_models,
)

logger = get_logger("ml_pipeline.flow")


@flow(
    name="ml-pipeline",
    retries=0,
    timeout_seconds=1200,
    log_prints=False,
)
def build_ml_pipeline_flow() -> Dict[str, any]:
    """Run the end-to-end ML workflow."""
    configure_logging("ml_pipeline.log")
    context = get_run_context()
    run_id_obj = getattr(getattr(context, "flow_run", None), "id", "local-ml-run")
    run_id = str(run_id_obj)

    record_pipeline_start("ml-pipeline", run_id)
    logger.info("Starting ML pipeline", extra={"run_id": run_id})

    try:
        X, y = load_processed_data()
        splits = split_dataset(X, y)
        gb_params = optimize_gradient_boosting(splits["X_train"], splits["y_train"])
        models = train_models(splits, gb_params)
        evaluations = evaluate_models(models, splits)
        artifacts = persist_champion(models, evaluations)
        record_pipeline_completion(run_id, "success", None, artifacts)
        logger.info(
            "ML pipeline completed",
            extra={
                "champion_model": artifacts["champion_model"],
                "metrics": artifacts["evaluations"][artifacts["champion_model"]]["metrics"],
            },
        )
        return artifacts
    except Exception:
        logger.exception("ML pipeline failed", extra={"run_id": run_id})
        record_pipeline_completion(run_id, "failed", None, {})
        raise


if __name__ == "__main__":
    build_ml_pipeline_flow()
