"""Daily job: harvest a datestamp, embed new papers, build each user's feed and email it."""

import argparse
import json
import logging
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import Engine, select

from radar.config import Settings, get_settings
from radar.db import make_engine, session_scope
from radar.db.models import Paper, User
from radar.email.digest import send_digest
from radar.embed.base import Embedder
from radar.embed.pipeline import embed_missing
from radar.ingest.oaipmh import OAIClient, harvest_day, is_new_submission
from radar.ingest.records import PaperRecord
from radar.ingest.store import upsert_papers
from radar.jobs.runs import job_run
from radar.rank.feed import build_feed
from radar.services import Services, make_services

log = logging.getLogger(__name__)


def run_daily(
    engine: Engine,
    settings: Settings,
    day: date,
    oai: OAIClient,
    embedder: Embedder | None,
) -> dict:
    with job_run(engine, "daily", {"date": day.isoformat(), "embed": embedder is not None}) as run:
        records = harvest_day(oai, settings.oai_sets, day)
        wanted = [r for r in records.values() if not settings.categories.isdisjoint(r.categories)]

        def announced_on(record: PaperRecord) -> date | None:
            return day if is_new_submission(record, day) else None

        with session_scope(engine) as session:
            upsert = upsert_papers(session, wanted, source="oai", announced_on=announced_on)
        run.stats = {"harvested": len(records), **upsert.as_dict()}

        if embedder is not None:
            with session_scope(engine) as session:
                todays = session.scalars(
                    select(Paper.arxiv_id).where(Paper.announced_on == day)
                ).all()
            # The day's new papers are essential: the feed cannot be built without them.
            run.stats |= embed_missing(
                engine, embedder, set(todays) | set(upsert.reembed_ids), essential=True
            )
    return run.stats


def run_feeds(
    engine: Engine, settings: Settings, day: date, svc: Services, email: bool = True
) -> dict:
    """Build (and email) the feed for `day` for every user."""
    with session_scope(engine) as session:
        user_ids = session.scalars(select(User.id).order_by(User.id)).all()
    out = {}
    for user_id in user_ids:
        with job_run(engine, "feed", {"date": day.isoformat(), "user": user_id}) as run:
            feed_id = build_feed(
                engine, settings, user_id, day, svc.embedder.model, svc.reranker, svc.chat
            )
            run.stats = {"feed_id": feed_id}
            if feed_id is not None and email:
                run.stats["digest"] = send_digest(engine, settings, feed_id)
        out[user_id] = run.stats
    return out


def run_all(
    engine: Engine,
    settings: Settings,
    day: date,
    oai: OAIClient,
    svc: Services | None,
    email: bool = True,
) -> dict:
    """Ingest, embed, then feeds and digests. With `svc=None`, ingest only."""
    stats = {"ingest": run_daily(engine, settings, day, oai, svc.embedder if svc else None)}
    if svc is not None:
        stats["feeds"] = run_feeds(engine, settings, day, svc, email)
    return stats


def bootstrap(
    engine: Engine,
    settings: Settings,
    oai: OAIClient,
    svc: Services | None,
    days: int = 3,
    lookback: int = 10,
) -> list[date]:
    """Ingest the last `days` announcement days, skipping weekends and holidays."""
    done: list[date] = []
    today = datetime.now(UTC).date()
    for back in range(1, lookback + 1):
        day = today - timedelta(days=back)
        if run_daily(engine, settings, day, oai, svc.embedder if svc else None)["announced"]:
            done.append(day)
        if len(done) >= days:
            break
    return done


def has_papers(engine: Engine) -> bool:
    with session_scope(engine) as session:
        return session.scalar(select(Paper.arxiv_id).limit(1)) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest, store and embed one day of arXiv.")
    parser.add_argument("--date", type=date.fromisoformat, help="OAI datestamp; default yesterday")
    parser.add_argument(
        "--no-embed", action="store_true", help="harvest only: no embedding, feeds or email"
    )
    parser.add_argument("--no-email", action="store_true", help="build feeds but send nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    engine = make_engine()
    day = args.date or (datetime.now(UTC).date() - timedelta(days=1))

    svc = None if args.no_embed else make_services(engine, settings)
    stats = run_all(engine, settings, day, OAIClient(settings), svc, email=not args.no_email)
    print(json.dumps(stats, default=str))


if __name__ == "__main__":
    main()
