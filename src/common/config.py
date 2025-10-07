"""Centralized runtime configuration using Pydantic BaseSettings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings sourced from env vars or .env file."""

    project_name: str = Field(default="churn-analytics-platform")

    dataset_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        )
    )
    dataset_name: str = Field(default="telco_customer_churn.csv")

    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("logs"))
    reports_dir: Path = Field(default=Path("reports"))
    artifacts_dir: Path = Field(default=Path("artifacts"))

    prefect_deployment_interval_seconds: int = Field(default=120)

    metadata_db_url: str = Field(default="sqlite:///metadata.db")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("data_dir", "logs_dir", "reports_dir", "artifacts_dir", pre=True)
    def _ensure_path(cls, value: Optional[str]) -> Path:  # noqa: D401
        """Convert raw values into Path objects."""
        if isinstance(value, Path):
            return value
        return Path(value)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
