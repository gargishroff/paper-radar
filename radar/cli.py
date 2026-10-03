"""Command-line tools for setting up and inspecting PaperRadar."""

import argparse
import logging
from datetime import date

from sqlalchemy import select

from radar.config import get_settings
from radar.db import make_engine, session_scope
from radar.db.models import FeedItem, Interest, Paper, ProfilePaper, User
from radar.ingest.oaipmh import OAIClient
from radar.jobs.daily import bootstrap
from radar.profile.build import (
    add_profile_papers,
    add_seed_interest,
    add_text_interest,
    rebuild_interests,
)
from radar.rank.feed import build_feed
from radar.services import make_services


def _user_id(session, user_id: int | None) -> int:
    if user_id is not None:
        return user_id
    uid = session.scalar(select(User.id).order_by(User.id))
    if uid is None:
        raise SystemExit("no user yet: run `python -m radar.cli onboard` first")
    return uid


def parse_text_interest(text: str) -> tuple[str, str]:
    """'Name: description' -> (name, description); without a colon the text is both."""
    name, sep, description = text.partition(":")
    return (name.strip(), description.strip()) if sep else (text.strip()[:60], text.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperRadar command-line tools")
    parser.add_argument("--user", type=int, help="user id (default: the first user)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("onboard", help="create a user and their interests")
    p.add_argument("--name", required=True, help="as it appears on your papers")
    p.add_argument("--email")
    p.add_argument("--s2", help="Semantic Scholar author ID: your papers become the profile")
    p.add_argument("--seed", nargs="*", default=[], help="arXiv IDs or URLs of papers you like")
    p.add_argument("--text", action="append", default=[], help="'Name: description' for a topic")

    p = sub.add_parser("bootstrap", help="ingest and embed the last few announcement days")
    p.add_argument("--days", type=int, default=3)

    sub.add_parser("interests", help="list interests")
    sub.add_parser("rebuild-interests", help="recluster papers not held by pinned interests")

    p = sub.add_parser("feed", help="build a day's feed")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--rebuild", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    engine = make_engine()
    svc = make_services(engine, settings)
    model = svc.embedder.model

    if args.cmd == "bootstrap":
        days = bootstrap(engine, settings, OAIClient(settings), svc, days=args.days)
        print("ingested:", ", ".join(d.isoformat() for d in days) or "nothing")
        return

    if args.cmd == "onboard":
        with session_scope(engine) as session:
            user = User(name=args.name, email=args.email, s2_author_id=args.s2)
            session.add(user)
            session.flush()
            uid = user.id
        if args.s2:
            ids = svc.fetcher.author_arxiv_ids(args.s2)
            print(f"Semantic Scholar: {len(ids)} papers on arXiv")
            add_profile_papers(engine, svc.fetcher, svc.embedder, uid, ids, "authored")
            rebuild_interests(engine, svc.chat, settings, uid, model)
        if args.seed:
            add_seed_interest(engine, svc.fetcher, svc.embedder, svc.chat, settings, uid, args.seed)
        for text in args.text:
            add_text_interest(engine, svc.embedder, settings, uid, *parse_text_interest(text))
        print(f"created user {uid}")
        args.cmd, args.user = "interests", uid

    if args.cmd == "rebuild-interests":
        with session_scope(engine) as session:
            uid = _user_id(session, args.user)
        rebuild_interests(engine, svc.chat, settings, uid, model)
        args.cmd = "interests"

    if args.cmd == "interests":
        with session_scope(engine) as session:
            uid = _user_id(session, args.user)
            for i in session.scalars(
                select(Interest).where(Interest.user_id == uid).order_by(Interest.weight.desc())
            ):
                n = len(
                    session.scalars(
                        select(ProfilePaper.arxiv_id).where(ProfilePaper.interest_id == i.id)
                    ).all()
                )
                pin = " [pinned]" if i.pinned else ""
                print(f"{i.id:>4}  {i.weight:5.2f}  {n:>3} papers  {i.name}{pin}")
                print(f"{'':>6}{i.description}")

    if args.cmd == "feed":
        with session_scope(engine) as session:
            uid = _user_id(session, args.user)
        feed_id = build_feed(
            engine, settings, uid, args.date, model, svc.reranker, svc.chat, args.rebuild
        )
        if feed_id is None:
            print("no feed (no interests, or no papers that day)")
            return
        with session_scope(engine) as session:
            items = session.execute(
                select(FeedItem, Paper)
                .join(Paper, Paper.arxiv_id == FeedItem.arxiv_id)
                .where(FeedItem.feed_id == feed_id)
                .order_by(FeedItem.rank)
            ).all()
            for item, paper in items:
                print(f"{item.rank + 1:>2}. [{item.slot}] {paper.title}  ({paper.arxiv_id})")
                print(f"      TL;DR: {item.tldr}")
                print(f"      Why:   {item.why}")


if __name__ == "__main__":
    main()
