"""Data ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import CONFIG
from .logging_utils import get_logger

logger = get_logger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Download dataset from the configured URL if not present."""

    destination = CONFIG.raw_path
    if destination.exists() and not force:
        logger.info("Dataset already present", extra={"path": str(destination)})
        return destination

    response = requests.get(CONFIG.dataset_url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logger.info(
        "Dataset downloaded",
        extra={"bytes": len(response.content), "path": str(destination)},
    )
    return destination


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw dataset into a DataFrame."""

    data_path = path or CONFIG.raw_path
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found; run download_dataset first.")
    df = pd.read_csv(data_path)
    logger.info("Dataset loaded", extra={"shape": df.shape})
    return df
