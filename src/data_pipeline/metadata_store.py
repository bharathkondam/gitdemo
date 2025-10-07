"""Metadata persistence for pipeline observability."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.common.config import get_settings
from src.common.paths import ARTIFACT_DIR

Base = declarative_base()


class PipelineRun(Base):
    """ORM model for tracking pipeline runs."""

    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_name = Column(String(64), nullable=False)
    run_id = Column(String(64), nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(32), nullable=False, default="running")
    checksum = Column(String(128), nullable=True)
    artifacts = Column(Text, nullable=True)


def _get_engine():
    settings = get_settings()
    db_url = settings.metadata_db_url
    if db_url.startswith("sqlite:///"):
        db_file = ARTIFACT_DIR / db_url.replace("sqlite:///", "")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_file}", echo=False, future=True)
    else:
        engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return engine


SessionLocal = sessionmaker(bind=_get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope():
    """Context manager for transactional sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def record_pipeline_start(pipeline_name: str, run_id: str) -> None:
    """Insert a record for a pipeline run start."""
    run_id = str(run_id)
    with session_scope() as session:
        session.add(
            PipelineRun(
                pipeline_name=pipeline_name,
                run_id=run_id,
                started_at=datetime.utcnow(),
                status="running",
            )
        )


def record_pipeline_completion(
    run_id: str,
    status: str,
    checksum: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> None:
    """Update a run record on completion."""
    run_id = str(run_id)
    with session_scope() as session:
        run = session.query(PipelineRun).filter(PipelineRun.run_id == run_id).one_or_none()
        if not run:
            return
        run.completed_at = datetime.utcnow()
        run.status = status
        run.checksum = checksum
        run.artifacts = json.dumps(artifacts or {})
        session.add(run)


def fetch_recent_runs(pipeline_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent runs for dashboards/APIs."""
    with session_scope() as session:
        query = session.query(PipelineRun)
        if pipeline_name:
            query = query.filter(PipelineRun.pipeline_name == pipeline_name)
        records = query.order_by(PipelineRun.started_at.desc()).limit(limit).all()

        results: List[Dict[str, Any]] = []
        for record in records:
            results.append(
                {
                    "id": record.id,
                    "pipeline_name": record.pipeline_name,
                    "run_id": record.run_id,
                    "status": record.status,
                    "started_at": record.started_at.isoformat() if record.started_at else None,
                    "completed_at": (
                        record.completed_at.isoformat() if record.completed_at else None
                    ),
                    "checksum": record.checksum,
                    "artifacts": json.loads(record.artifacts or "{}"),
                }
            )
        return results
