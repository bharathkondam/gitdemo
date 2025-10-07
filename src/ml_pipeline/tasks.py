"""Prefect tasks for ML operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from matplotlib import pyplot as plt
from prefect import task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from src.common.logging_utils import get_logger
from src.common.paths import ARTIFACT_DIR, REPORTS_DIR

logger = get_logger("ml_pipeline.tasks")


@task(name="load_processed_data")
def load_processed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed dataset produced by the data pipeline."""
    dataset_path = Path("data/processed/telco_churn_processed.parquet")
    if not dataset_path.exists():
        raise FileNotFoundError("Processed dataset missing. Run the data pipeline first.")
    df = pd.read_parquet(dataset_path)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    logger.info(
        "Processed dataset loaded",
        extra={"records": len(df), "feature_dim": X.shape[1]},
    )
    return X, y


@task(name="split_dataset")
def split_dataset(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3
) -> Dict[str, pd.DataFrame]:
    """Split dataset into train/validation/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    logger.info(
        "Dataset split",
        extra={"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@task(name="optimize_gradient_boosting")
def optimize_gradient_boosting(X: pd.DataFrame, y: pd.Series, n_trials: int = 10) -> Dict[str, float]:
    """Use Optuna to tune Gradient Boosting hyperparameters."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingClassifier(random_state=42, **params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            probas = model.predict_proba(X_valid)[:, 1]
            scores.append(roc_auc_score(y_valid, probas))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(
        "Gradient Boosting optimization complete",
        extra={"best_params": study.best_params, "best_value": study.best_value},
    )

    reports_dir = REPORTS_DIR / "ml" / "optuna"
    reports_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(reports_dir / "gb_trials.csv", index=False)

    return study.best_params


@task(name="train_models")
def train_models(
    splits: Dict[str, pd.DataFrame], gb_params: Dict[str, float]
) -> Dict[str, any]:
    """Train Logistic Regression and Gradient Boosting models."""
    X_train = pd.concat([splits["X_train"], splits["X_val"]])
    y_train = pd.concat([splits["y_train"], splits["y_val"]])

    logistic_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    logistic_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(random_state=42, **gb_params)
    gb_model.fit(X_train, y_train)

    logger.info("Models trained", extra={"models": ["LogisticRegression", "GradientBoosting"]})
    return {"logistic": logistic_model, "gradient_boosting": gb_model}


def _evaluate(model, X_test, y_test, model_name: str, reports_dir: Path) -> Dict[str, any]:
    """Evaluate a trained model and write artifacts."""
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    metrics.update({"precision": precision, "recall": recall, "f1": f1})

    model_dir = reports_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "classification_report.txt").write_text(
        classification_report(y_test, preds, target_names=["No Churn", "Churn"])
    )
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax)
    fig.tight_layout()
    cm_path = model_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=180)
    plt.close(fig)

    roc = RocCurveDisplay.from_predictions(y_test, probas)
    roc.figure_.set_size_inches(6, 5)
    roc_path = model_dir / "roc_curve.png"
    roc.figure_.savefig(roc_path, dpi=180)
    plt.close(roc.figure_)

    shap_path = ""
    try:
        shap_sample = shap.sample(X_test, 200, random_state=42)
        explainer = shap.Explainer(model.predict_proba, shap_sample)
        shap_values = explainer(shap_sample)
        shap.summary_plot(shap_values, shap_sample, show=False)
        shap_path = model_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP summary failed", extra={"model": model_name, "error": str(exc)})

    return {
        "metrics": metrics,
        "confusion_matrix": cm_path.as_posix(),
        "roc_curve": roc_path.as_posix(),
        "shap_summary": shap_path.as_posix() if shap_path else "",
    }


@task(name="evaluate_models")
def evaluate_models(models: Dict[str, any], splits: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Evaluate both models and capture artifacts."""
    reports_dir = REPORTS_DIR / "ml"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, model in models.items():
        results[name] = _evaluate(model, splits["X_test"], splits["y_test"], name, reports_dir)
    logger.info("Model evaluation complete", extra={"models": list(results.keys())})
    return results


@task(name="persist_champion")
def persist_champion(models: Dict[str, any], evaluations: Dict[str, any]) -> Dict[str, any]:
    """Persist the best-performing model by ROC AUC."""
    best_name = max(evaluations, key=lambda name: evaluations[name]["metrics"]["roc_auc"])
    best_model = models[best_name]

    model_dir = ARTIFACT_DIR / "ml_pipeline"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{best_name}_model.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "champion_model": best_name,
        "champion_path": model_path.as_posix(),
        "evaluations": evaluations,
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Champion model persisted", extra={"model": best_name, "path": str(model_path)})
    return metadata

