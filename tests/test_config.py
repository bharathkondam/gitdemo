from pathlib import Path

from src.pipeline.config import CONFIG


def test_directories_exist():
    assert CONFIG.raw_dir.exists()
    assert CONFIG.processed_dir.exists()
    assert CONFIG.reports_dir.exists()
    assert CONFIG.logs_dir.exists()
    assert CONFIG.artifacts_dir.exists()


def test_prefect_flows_importable():
    from flows.data_pipeline_flow import data_pipeline_flow  # noqa: F401
    from flows.ml_pipeline_flow import ml_pipeline_flow  # noqa: F401

    assert callable(data_pipeline_flow)
    assert callable(ml_pipeline_flow)
