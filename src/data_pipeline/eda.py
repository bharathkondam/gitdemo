"""Automated EDA helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from ydata_profiling import ProfileReport

from src.common.logging_utils import get_logger

logger = get_logger("data_pipeline.eda")


def profiling_report(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate an HTML profile report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "profiling_report.html"
    ProfileReport(df, title="Telco Churn Profiling", minimal=True).to_file(report_path)
    logger.info("Profiling report generated", extra={"path": str(report_path)})
    return report_path


def correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=180)
    plt.close()
    logger.info("Correlation heatmap saved", extra={"path": str(heatmap_path)})
    return heatmap_path


def numerical_histograms(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate histograms for each numeric column."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}
    for column in df.columns:
        if not is_numeric_dtype(df[column]):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        fig.tight_layout()
        path = output_dir / f"hist_{column}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        artifacts[column] = path
    return artifacts


def categorical_bars(
    df: pd.DataFrame, target_col: str, output_dir: Path, max_categories: int = 20
) -> Dict[str, Path]:
    """Bar plots of categorical distributions grouped by target."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}
    for column in df.columns:
        if is_numeric_dtype(df[column]) or column == target_col:
            continue
        if df[column].nunique() > max_categories:
            logger.info("Skipping categorical plot", extra={"column": column})
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x=column, hue=target_col, ax=ax)
        ax.set_title(f"{column} distribution by {target_col}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = output_dir / f"{column}_counts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        artifacts[column] = path
    return artifacts


def run_eda(df: pd.DataFrame, target_col: Optional[str], output_dir: Path) -> Dict[str, Path]:
    """Execute the EDA workflow and return artifact paths."""
    artifacts: Dict[str, Path] = {}
    artifacts["profile_report"] = profiling_report(df, output_dir)
    artifacts["correlation_heatmap"] = correlation_heatmap(df, output_dir)
    artifacts.update(numerical_histograms(df, output_dir / "histograms"))
    if target_col:
        artifacts.update(categorical_bars(df, target_col, output_dir / "categorical"))
    return artifacts

