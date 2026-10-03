"""Records every job run, including failures, for the Status page and alerts."""

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Engine

from radar.db import session_scope
from radar.db.models import JobRun


@dataclass
class RunHandle:
    id: int
    stats: dict = field(default_factory=dict)


@contextmanager
def job_run(engine: Engine, job: str, params: dict | None = None) -> Iterator[RunHandle]:
    """Log the job as running, then as ok with its stats or failed with the traceback."""
    with session_scope(engine) as session:
        run = JobRun(job=job, params=params, status="running")
        session.add(run)
        session.flush()
        handle = RunHandle(id=run.id)

    try:
        yield handle
    except BaseException:
        _finish(engine, handle, "failed", traceback.format_exc())
        raise
    _finish(engine, handle, "ok", None)


def _finish(engine: Engine, handle: RunHandle, status: str, error: str | None) -> None:
    with session_scope(engine) as session:
        run = session.get(JobRun, handle.id)
        run.status = status
        run.stats = handle.stats
        run.error = error
        run.finished_at = datetime.now(UTC)
