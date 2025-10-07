"""Reusable preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.logging_utils import get_logger

logger = get_logger("data_pipeline.preprocessing")


@dataclass
class PreprocessArtifacts:
    """Paths of preprocessing outputs."""

    summary_path: Path
    missing_path: Path
    schema_path: Path
    processed_path: Path


def build_transformers(df: pd.DataFrame) -> Tuple[SimpleImputer, SimpleImputer, StandardScaler, OneHotEncoder, Dict[str, list]]:
    """Identify feature groups and return fitted transformers."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [
        col for col in df.columns if df[col].dtype == "object" and col not in {"customerID", "Churn"}
    ]
    logger.info(
        "Feature groups inferred",
        extra={"numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
    )

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    metadata = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_column": "Churn",
    }
    return numeric_imputer, categorical_imputer, scaler, encoder, metadata


def generate_data_quality_reports(df: pd.DataFrame, output_dir: Path) -> PreprocessArtifacts:
    """Produce summary statistics, missing value report, and schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_statistics.parquet"
    missing_path = output_dir / "missing_values.parquet"
    schema_path = output_dir / "schema.json"

    df.describe(include="all").transpose().to_parquet(summary_path)
    df.isna().sum().rename("missing").to_frame().to_parquet(missing_path)
    df.dtypes.apply(lambda x: x.name).to_json(schema_path)

    processed_path = output_dir / "processed.parquet"
    return PreprocessArtifacts(
        summary_path=summary_path,
        missing_path=missing_path,
        schema_path=schema_path,
        processed_path=processed_path,
    )

