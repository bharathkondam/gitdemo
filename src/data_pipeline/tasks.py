"""Prefect tasks implementing the data pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import requests
from prefect import task
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import get_settings
from src.common.logging_utils import get_logger
from src.common.paths import ARTIFACT_DIR, DATA_DIR, REPORTS_DIR
from .eda import run_eda
from .preprocessing import PreprocessArtifacts, build_transformers, generate_data_quality_reports

logger = get_logger("data_pipeline.tasks")


@task(name="ingest_dataset", retries=2, retry_delay_seconds=10, log_prints=False)
def ingest_dataset() -> Path:
    """Download the dataset if it does not exist locally."""
    settings = get_settings()
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / settings.dataset_name

    if destination.exists():
        logger.info("Dataset already present", extra={"path": str(destination)})
        return destination

    candidate_urls = [settings.dataset_url]
    candidate_urls.append(
        "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    last_error: Exception | None = None
    for url in candidate_urls:
        try:
            logger.info("Attempting dataset download", extra={"url": url})
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            destination.write_bytes(response.content)
            logger.info(
                "Dataset ingested", extra={"bytes": len(response.content), "url": url}
            )
            return destination
        except Exception as exc:  # noqa: BLE001 - capture HTTP and connection errors
            last_error = exc
            logger.warning("Dataset download failed", extra={"url": url, "error": str(exc)})

    raise RuntimeError(
        "Failed to download churn dataset from configured sources. "
        f"Verify network access or manually place the file at {destination}. "
        f"Last error: {last_error}"
    )


@task(name="load_raw_dataset")
def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load CSV dataset into pandas."""
    df = pd.read_csv(path)
    logger.info("Dataset loaded", extra={"records": len(df), "columns": list(df.columns)})
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-specific cleaning rules."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int64")
    df = df.drop_duplicates(subset=["customerID"])
    return df


@task(name="preprocess_dataset")
def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any], PreprocessArtifacts]:
    """Impute, scale, encode, and persist processed data."""
    df = _clean(df)
    numeric_imputer, categorical_imputer, scaler, encoder, metadata = build_transformers(df)

    numeric_cols = metadata["numeric_features"]
    categorical_cols = metadata["categorical_features"]

    numeric_values = pd.DataFrame(
        numeric_imputer.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )
    categorical_values = pd.DataFrame(
        categorical_imputer.fit_transform(df[categorical_cols]),
        columns=categorical_cols,
        index=df.index,
    )

    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(numeric_values), columns=numeric_cols, index=df.index
    )

    encoded_array = encoder.fit_transform(categorical_values)
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    target = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    processed = pd.concat(
        [df["customerID"], scaled_numeric, encoded_df, target.rename("Churn")], axis=1
    )

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "telco_churn_processed.parquet"
    processed.to_parquet(processed_path)

    transforms_dir = ARTIFACT_DIR / "data_pipeline"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(numeric_imputer, transforms_dir / "numeric_imputer.joblib")
    joblib.dump(categorical_imputer, transforms_dir / "categorical_imputer.joblib")
    joblib.dump(scaler, transforms_dir / "numeric_scaler.joblib")
    joblib.dump(encoder, transforms_dir / "categorical_encoder.joblib")

    metadata_path = transforms_dir / "feature_metadata.json"
    metadata["encoded_features"] = encoded_cols.tolist()
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(
        "Processed dataset generated",
        extra={
            "processed_path": str(processed_path),
            "records": len(processed),
            "features": processed.shape[1] - 2,
        },
    )

    dq_artifacts = generate_data_quality_reports(df, REPORTS_DIR / "data_quality")
    dq_artifacts.processed_path = processed_path
    return processed, metadata, dq_artifacts


@task(name="run_eda")
def run_eda_task(df: pd.DataFrame, target_column: str) -> Dict[str, Path]:
    """Execute EDA routine and return artifact paths."""
    artifacts = run_eda(df, target_column, REPORTS_DIR / "eda")
    logger.info(
        "EDA artifacts created",
        extra={"artifacts": {key: str(value) for key, value in artifacts.items()}},
    )
    return artifacts


@task(name="checksum_dataset")
def checksum_dataset(path: Path) -> str:
    """Return SHA256 checksum for reproducibility."""
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    logger.info("Dataset checksum computed", extra={"checksum": checksum})
    return checksum
