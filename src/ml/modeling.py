"""Machine learning pipeline components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from src.pipeline.config import CONFIG
from src.pipeline.logging_utils import get_logger

logger = get_logger(__name__)


METRICS_PATH = CONFIG.reports_dir / "metrics.json"
MODEL_DIR = CONFIG.artifacts_dir
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_models(processed_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    X = processed_df.drop(columns=[CONFIG.target_column])
    y = processed_df[CONFIG.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        logger.info("Model trained", extra={"model": name, "metrics": metrics[name]})

    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics
