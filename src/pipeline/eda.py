"""Exploratory data analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .logging_utils import get_logger

logger = get_logger(__name__)


FIGURES_DIR = CONFIG.reports_dir / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame) -> Path:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
    ax.set_title("Correlation Heatmap")
    path = _save_fig(fig, "correlation_heatmap.png")
    logger.info("Correlation heatmap created", extra={"path": str(path)})
    return path


def plot_tenure(df: pd.DataFrame) -> Path:
    if "tenure" not in df.columns:
        return Path()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["tenure"], bins=20, ax=ax)
    ax.set_title("Tenure Distribution")
    path = _save_fig(fig, "tenure_distribution.png")
    logger.info("Tenure histogram saved", extra={"path": str(path)})
    return path


def plot_categorical(df: pd.DataFrame, column: str) -> Path:
    if column not in df.columns:
        return Path()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x=column, hue=CONFIG.target_column, ax=ax)
    ax.set_title(f"{column} Distribution by Churn")
    plt.xticks(rotation=30, ha="right")
    path = _save_fig(fig, f"{column.lower()}_distribution.png")
    logger.info("Categorical plot created", extra={"column": column, "path": str(path)})
    return path


def feature_importance(df: pd.DataFrame) -> Path:
    features = df.drop(columns=[CONFIG.target_column])
    target = df[CONFIG.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, stratify=target, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)
    importances = pd.Series(clf.feature_importances_, index=features.columns)
    top = importances.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("RandomForest Feature Importance")
    path = _save_fig(fig, "feature_importance.png")

    report_path = CONFIG.reports_dir / "classification_report.csv"
    pd.Series({"accuracy": clf.score(X_test, y_test)}).to_csv(report_path)
    logger.info(
        "Feature importance computed",
        extra={"importance_path": str(path), "report_path": str(report_path)},
    )
    return path


def run_eda(df: pd.DataFrame) -> Dict[str, Path]:
    artifacts: Dict[str, Path] = {}
    artifacts["correlation"] = plot_correlation(df)
    tenure_path = plot_tenure(df)
    if tenure_path:
        artifacts["tenure"] = tenure_path
    for column in ["InternetService", "Contract", "PaymentMethod"]:
        cat_path = plot_categorical(df, column)
        if cat_path:
            artifacts[column.lower()] = cat_path
    artifacts["feature_importance"] = feature_importance(df)
    logger.info("EDA complete", extra={"artifacts": {k: str(v) for k, v in artifacts.items()}})
    return artifacts
