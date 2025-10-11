"""Configuration values shared across the Prefect pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_url: str = (
        "https://raw.githubusercontent.com/IBM/"
        "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    dataset_name: str = "telco_customer_churn.csv"

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")
    artifacts_dir: Path = Path("artifacts/models")

    target_column: str = "Churn"
    id_column: str = "customerID"

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.dataset_name

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / "telco_churn_processed.parquet"


CONFIG = PipelineConfig()

for directory in (
    CONFIG.raw_dir,
    CONFIG.processed_dir,
    CONFIG.reports_dir,
    CONFIG.logs_dir,
    CONFIG.artifacts_dir,
):
    directory.mkdir(parents=True, exist_ok=True)
