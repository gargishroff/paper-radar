"""The worker process: runs the daily job every day at DAILY_HOUR in TIMEZONE."""

import argparse
import logging
import traceback
from datetime import UTC, date, datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import Engine, select

from radar.config import Settings, get_settings
from radar.db import make_engine, session_scope
from radar.db.models import JobRun, User
from radar.email.digest import send_alert
from radar.ingest.oaipmh import OAIClient
from radar.jobs.daily import bootstrap, has_papers, run_all
from radar.services import make_services

log = logging.getLogger(__name__)
CATCH_UP_DAYS = 3


def pending_days(engine: Engine, today: date) -> list[date]:
    """Yesterday plus earlier days in the catch-up window without a successful ingest."""
    window = [today - timedelta(days=d) for d in range(CATCH_UP_DAYS, 0, -1)]
    with session_scope(engine) as session:
        done = {
            r.params.get("date")
            for r in session.scalars(
                select(JobRun).where(JobRun.job == "daily", JobRun.status == "ok")
            )
            if r.params
        }
    return [d for d in window if d.isoformat() not in done or d == window[-1]]


def tick(engine: Engine, settings: Settings) -> None:
    svc = make_services(engine, settings)
    oai = OAIClient(settings)
    for day in pending_days(engine, datetime.now(UTC).date()):
        try:
            stats = run_all(engine, settings, day, oai, svc)
            log.info("daily run for %s: %s", day, stats)
        except Exception:
            log.exception("daily run for %s failed", day)
            with session_scope(engine) as session:
                to = session.scalar(select(User.email).where(User.email.is_not(None)))
            send_alert(settings, to, f"daily run for {day} failed", traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperRadar scheduler")
    parser.add_argument("--once", action="store_true", help="run one pass now and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    engine = make_engine()
    if not has_papers(engine):
        log.info("empty database: ingesting the last 3 announcement days")
        bootstrap(engine, settings, OAIClient(settings), make_services(engine, settings))
    # Catch up at start too, so a server started after DAILY_HOUR still gets today's feed.
    tick(engine, settings)
    if args.once:
        return
    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        tick,
        CronTrigger(hour=settings.daily_hour, minute=0, timezone=settings.timezone),
        args=[engine, settings],
        misfire_grace_time=3600,
        coalesce=True,
    )
    log.info("scheduler started: daily at %02d:00 %s", settings.daily_hour, settings.timezone)
    scheduler.start()


if __name__ == "__main__":
    main()
