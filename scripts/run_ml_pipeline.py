"""Run the Prefect ML pipeline locally."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline.flow import build_ml_pipeline_flow


def ensure_offline_defaults() -> None:
    """Apply local defaults only when Prefect API settings are absent."""

    if os.getenv("PREFECT_API_MODE") or os.getenv("PREFECT_API_URL"):
        return

    os.environ.setdefault("PREFECT_API_MODE", "OFFLINE")
    os.environ.setdefault("PREFECT_API_URL", "")
    os.environ.setdefault("PREFECT_SERVER_API_HOST", "127.0.0.1")
    os.environ.setdefault("PREFECT_SERVER_API_PORT", "43210")


def main() -> None:
    ensure_offline_defaults()
    build_ml_pipeline_flow()


if __name__ == "__main__":
    main()
