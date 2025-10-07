"""Convenience path helpers and directory creation."""

from __future__ import annotations

from pathlib import Path

from .config import get_settings

settings = get_settings()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / settings.data_dir
LOGS_DIR = ROOT_DIR / settings.logs_dir
REPORTS_DIR = ROOT_DIR / settings.reports_dir
ARTIFACT_DIR = ROOT_DIR / settings.artifacts_dir

for directory in (DATA_DIR, LOGS_DIR, REPORTS_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

