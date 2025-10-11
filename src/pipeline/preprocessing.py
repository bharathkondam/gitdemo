"""Preprocessing utilities for churn dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    processed_path: Path
    summary_path: Path
    missing_path: Path
    dtypes_path: Path
    transformer_path: Path


def _generate_reports(df: pd.DataFrame) -> Tuple[Path, Path, Path]:
    reports_dir = CONFIG.reports_dir
    summary_path = reports_dir / "summary_statistics.csv"
    df.describe(include="all").transpose().to_csv(summary_path)

    missing_path = reports_dir / "missing_values.csv"
    df.isna().sum().to_frame("missing_count").to_csv(missing_path)

    dtypes_path = reports_dir / "data_types.csv"
    df.dtypes.to_frame("dtype").to_csv(dtypes_path)

    logger.info(
        "Generated summary reports",
        extra={
            "summary_path": str(summary_path),
            "missing_path": str(missing_path),
            "dtypes_path": str(dtypes_path),
        },
    )
    return summary_path, missing_path, dtypes_path


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col not in {CONFIG.id_column, CONFIG.target_column}
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    logger.info(
        "Preprocessor built",
        extra={"numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
    )
    return preprocessor, numeric_cols, categorical_cols


def preprocess_dataset(df: pd.DataFrame) -> PreprocessingResult:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop_duplicates(subset=[CONFIG.id_column], inplace=True)

    summary_path, missing_path, dtypes_path = _generate_reports(df)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df)

    features = df.drop(columns=[CONFIG.id_column, CONFIG.target_column])
    target = df[CONFIG.target_column].map({"Yes": 1, "No": 0})

    transformed = preprocessor.fit_transform(features)
    feature_names = preprocessor.get_feature_names_out()

    processed = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    processed[CONFIG.target_column] = target.values

    processed_path = CONFIG.processed_path
    processed.to_parquet(processed_path, index=False)

    transformer_path = CONFIG.reports_dir / "preprocessor.pkl"
    pd.to_pickle(preprocessor, transformer_path)

    logger.info(
        "Dataset preprocessed",
        extra={"processed_path": str(processed_path), "rows": len(processed)},
    )
    return PreprocessingResult(
        processed_path=processed_path,
        summary_path=summary_path,
        missing_path=missing_path,
        dtypes_path=dtypes_path,
        transformer_path=transformer_path,
    )
