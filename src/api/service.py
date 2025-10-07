"""Model serving utilities for the API layer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger("api.service")


class ModelService:
    """Singleton service encapsulating inference logic."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.artifact_root = Path("artifacts")
        self.data_artifacts = self.artifact_root / "data_pipeline"
        self.ml_artifacts = self.artifact_root / "ml_pipeline"

        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoder = None

        self._load_assets()

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _load_assets(self) -> None:
        """Load preprocessing artifacts and champion model."""
        metadata_path = self.data_artifacts / "feature_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Feature metadata missing. Run the data pipeline first.")

        model_metadata_path = self.ml_artifacts / "model_metadata.json"
        if not model_metadata_path.exists():
            raise FileNotFoundError("Model metadata missing. Run the ML pipeline first.")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
        champion_path = Path(model_metadata["champion_path"])
        if not champion_path.exists():
            raise FileNotFoundError(f"Champion model not found at {champion_path}")

        self.numeric_imputer = joblib.load(self.data_artifacts / "numeric_imputer.joblib")
        self.categorical_imputer = joblib.load(self.data_artifacts / "categorical_imputer.joblib")
        self.scaler = joblib.load(self.data_artifacts / "numeric_scaler.joblib")
        self.encoder = joblib.load(self.data_artifacts / "categorical_encoder.joblib")
        self.model = joblib.load(champion_path)

        logger.info(
            "Artifacts loaded",
            extra={"champion_model": model_metadata["champion_model"], "path": str(champion_path)},
        )

    def reload(self) -> None:
        """Reload assets after retraining."""
        with self._lock:
            self._load_assets()

    def _preprocess(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Transform request payload into model-ready features."""
        numeric_features = self.metadata.get("numeric_features", [])
        categorical_features = self.metadata.get("categorical_features", [])

        df = pd.DataFrame([payload])
        numeric_df = df[numeric_features].astype(float)
        categorical_df = df[categorical_features].astype(str)

        numeric_imputed = pd.DataFrame(
            self.numeric_imputer.transform(numeric_df),
            columns=numeric_features,
        )
        scaled_numeric = pd.DataFrame(
            self.scaler.transform(numeric_imputed),
            columns=numeric_features,
        )

        categorical_imputed = self.categorical_imputer.transform(categorical_df)
        encoded_array = self.encoder.transform(categorical_imputed)
        encoded_cols = self.encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

        features = pd.concat([scaled_numeric, encoded_df], axis=1)
        return features

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate churn probability and label."""
        features = self._preprocess(payload)
        probability = float(self.model.predict_proba(features)[:, 1][0])
        label = "Churn" if probability >= 0.5 else "No Churn"
        return {"probability": probability, "label": label}

