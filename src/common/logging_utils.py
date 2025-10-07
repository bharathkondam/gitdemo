"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import get_settings
from .paths import LOGS_DIR


class JsonFormatter(logging.Formatter):
    """Render log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "args",
                    "asctime",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                }
            }
        )
        return json.dumps(payload)


def configure_logging(log_file: str = "pipeline.log") -> logging.Logger:
    """Create a JSON logger writing to stdout and a file."""
    settings = get_settings()
    logs_path = LOGS_DIR / log_file
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(settings.project_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = JsonFormatter()

    file_handler = logging.FileHandler(logs_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("Logger configured", extra={"log_file": str(logs_path)})
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with shared handlers."""
    parent = logging.getLogger(get_settings().project_name)
    if not parent.handlers:
        configure_logging()
    return parent.getChild(name)

