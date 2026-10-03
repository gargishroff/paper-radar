"""Tests against a real Postgres + pgvector database (skipped if none is reachable)."""

import dataclasses
from datetime import date, timedelta
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import func, select

from radar.db import session_scope
from radar.db.models import ApiUsage, JobRun, Paper, PaperEmbedding
from radar.embed.cohere_embed import CohereEmbedder
from radar.embed.pipeline import embed_missing
from radar.ingest.oaipmh import OAIClient, parse_records
from radar.ingest.store import upsert_papers
from radar.jobs.daily import run_daily
from radar.jobs.runs import job_run
from radar.usage import BudgetExceeded, UsageTracker
from tests.conftest import FIXTURES, FakeEmbedder, oai_transport

DAY = date(2026, 9, 24)


def _count(engine, model) -> int:
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_daily_job_is_idempotent(engine, settings):
    oai = OAIClient(settings, http=httpx.Client(transport=oai_transport()))
    embedder = FakeEmbedder()

    first = run_daily(engine, settings, DAY, oai, embedder)
    assert first["inserted"] == 4
    assert first["announced"] == 1  # only 2609.21583 is a new submission
    assert first["embedded"] == 1 and first["embed_calls"] == 1

    with session_scope(engine) as s:
        fresh = s.get(Paper, "2609.21583")
        assert fresh.announced_on == DAY and fresh.primary_cat == "cs.NE"
        assert s.get(Paper, "2009.13987").announced_on is None

    second = run_daily(engine, settings, DAY, oai, embedder)
    assert second["inserted"] == second["updated"] == 0
    assert second["unchanged"] == 4
    assert second["embed_calls"] == 0
    assert embedder.calls == 1

    with session_scope(engine) as s:
        runs = s.scalars(select(JobRun).order_by(JobRun.id)).all()
    assert [r.status for r in runs] == ["ok", "ok"]
    assert runs[1].stats["unchanged"] == 4


def _fixture_records():
    records, _ = parse_records((FIXTURES / "oai_page2.xml").read_text())
    return records


def test_changed_abstract_drops_vector_and_reembeds(engine):
    records = _fixture_records()
    with session_scope(engine) as s:
        upsert_papers(s, records, source="oai")
    embed_missing(engine, FakeEmbedder())
    assert _count(engine, PaperEmbedding) == 3

    revised = dataclasses.replace(
        records[0],
        abstract="A rewritten abstract.",
        latest_version=2,
        version_at=records[0].version_at + timedelta(days=1),
    )
    with session_scope(engine) as s:
        stats = upsert_papers(s, [revised], source="oai")
    assert stats.updated == 1 and stats.reembed_ids == [revised.arxiv_id]
    assert _count(engine, PaperEmbedding) == 2

    assert embed_missing(engine, FakeEmbedder())["embedded"] == 1
    assert _count(engine, PaperEmbedding) == 3


def test_older_snapshot_does_not_overwrite(engine):
    [record, *_] = _fixture_records()
    with session_scope(engine) as s:
        upsert_papers(s, [record], source="oai")
    stale = dataclasses.replace(
        record, title="Old title", version_at=record.version_at - timedelta(days=30)
    )
    with session_scope(engine) as s:
        stats = upsert_papers(s, [stale], source="kaggle")
        assert stats.unchanged == 1
        assert s.get(Paper, record.arxiv_id).title == record.title


def test_embed_batches_respect_max_batch(engine):
    with session_scope(engine) as s:
        upsert_papers(s, _fixture_records(), source="oai")
    embedder = FakeEmbedder()  # max_batch = 2
    stats = embed_missing(engine, embedder)
    assert stats == {"pending": 3, "embedded": 3, "embed_calls": 2}


def test_budget_guard(engine, settings):
    tracker = UsageTracker(engine, settings, today=DAY)  # budget 10, soft limit 8
    tracker.record("embed", "m", calls=8, units=100)
    with pytest.raises(BudgetExceeded):
        tracker.check(essential=False)
    tracker.check(essential=True)
    tracker.record("rerank", "m", calls=2)
    with pytest.raises(BudgetExceeded):
        tracker.check(essential=True)

    # Last month's calls don't count against this month.
    next_month = UsageTracker(engine, settings, today=date(2026, 10, 1))
    assert next_month.calls_this_month() == 0

    tracker.record("embed", "m", calls=1, units=5)
    with session_scope(engine) as s:
        row = s.get(ApiUsage, (DAY, "embed", "m"))
        assert (row.calls, row.units) == (9, 105)


class _FakeCohere:
    def __init__(self):
        self.requests = []

    def embed(self, **kwargs):
        self.requests.append(kwargs)
        n = len(kwargs["texts"])
        return SimpleNamespace(
            embeddings=SimpleNamespace(float_=[[0.1] * 1024 for _ in range(n)]),
            meta=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=42)),
        )


def test_cohere_embedder_logs_usage_and_requests_1024_dims(engine, settings):
    tracker = UsageTracker(engine, settings, today=DAY)
    fake = _FakeCohere()
    embedder = CohereEmbedder(settings, tracker, client=fake)

    vectors = embedder.embed(["a", "b"], "search_document")
    assert len(vectors) == 2 and len(vectors[0]) == 1024
    assert fake.requests[0]["output_dimension"] == 1024
    assert fake.requests[0]["model"] == "embed-v4.0"
    with session_scope(engine) as s:
        row = s.get(ApiUsage, (DAY, "embed", "embed-v4.0"))
        assert (row.calls, row.units) == (1, 42)

    tracker.record("embed", "embed-v4.0", calls=9)
    with pytest.raises(BudgetExceeded):
        embedder.embed(["c"], "search_document", essential=True)
    assert len(fake.requests) == 1  # refused before calling the API


def test_failed_job_is_recorded(engine):
    with pytest.raises(ValueError), job_run(engine, "boom", {"x": 1}):
        raise ValueError("kaput")
    with session_scope(engine) as s:
        run = s.scalars(select(JobRun)).one()
        assert run.status == "failed" and "kaput" in run.error and run.finished_at


def test_rate_limited_call_waits_retries_and_is_recorded(engine, settings, monkeypatch):
    from cohere.errors import TooManyRequestsError

    sleeps = []
    monkeypatch.setattr("radar.cohere_calls.time.sleep", sleeps.append)

    class Flaky(_FakeCohere):
        def embed(self, **kwargs):
            if not self.requests:
                self.requests.append(kwargs)
                raise TooManyRequestsError(body=None)
            return super().embed(**kwargs)

    tracker = UsageTracker(engine, settings, today=DAY)
    fake = Flaky()
    embedder = CohereEmbedder(settings, tracker, client=fake)
    assert len(embedder.embed(["a"], "search_document")) == 1
    assert sleeps == [60]
    assert fake.requests[-1]["request_options"] == {"max_retries": 0}
    with session_scope(engine) as s:
        assert s.get(ApiUsage, (DAY, "embed_rejected", "embed-v4.0")).calls == 1
        assert s.get(ApiUsage, (DAY, "embed", "embed-v4.0")).calls == 1


def test_scheduler_catches_up_on_missed_days(engine):
    from radar.jobs.scheduler import pending_days

    today = date(2026, 9, 28)
    assert pending_days(engine, today) == [date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27)]
    with job_run(engine, "daily", {"date": "2026-09-25"}):
        pass
    with job_run(engine, "daily", {"date": "2026-09-27"}):
        pass
    # 25th done; 26th missed; yesterday (27th) is always re-run, which is safe.
    assert pending_days(engine, today) == [date(2026, 9, 26), date(2026, 9, 27)]
