"""Backfill historical papers from the Kaggle arXiv metadata snapshot."""

import argparse
import json
import logging
from collections.abc import Iterator
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

from radar.config import get_settings
from radar.db import make_engine, session_scope
from radar.ingest.records import PaperRecord, clean_text, latex_to_unicode, split_authors
from radar.ingest.store import UpsertStats, upsert_papers
from radar.jobs.runs import job_run

log = logging.getLogger(__name__)

BATCH_SIZE = 2000


def parse_line(line: str) -> PaperRecord:
    obj = json.loads(line)
    versions = sorted(
        (int(v["version"].lstrip("v")), parsedate_to_datetime(v["created"]))
        for v in obj["versions"]
    )
    if obj.get("authors_parsed"):
        # [keyname, forenames, suffix] -> "Forenames Keyname Suffix"
        authors = [
            latex_to_unicode(clean_text(" ".join((a[1], a[0], *a[2:]))))
            for a in obj["authors_parsed"]
            if a[0]
        ]
    else:
        authors = split_authors(obj.get("authors") or "")
    return PaperRecord(
        arxiv_id=obj["id"],
        title=clean_text(obj["title"]),
        abstract=clean_text(obj["abstract"]),
        authors=authors,
        categories=obj["categories"].split(),
        published_at=versions[0][1],
        version_at=versions[-1][1],
        latest_version=versions[-1][0],
        comments=clean_text(obj.get("comments")) or None,
        license=obj.get("license"),
    )


def iter_snapshot(
    path: Path, categories: set[str], start: date, end: date
) -> Iterator[PaperRecord]:
    """Records with any category in `categories` and a v1 date in [start, end]."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            # Cheap pre-filter before JSON parsing: most lines are in other categories.
            if not any(c in line for c in categories):
                continue
            record = parse_line(line)
            if categories.isdisjoint(record.categories):
                continue
            if start <= record.published_at.date() <= end:
                yield record


def backfill(path: Path, start: date, end: date, categories: set[str]) -> UpsertStats:
    engine = make_engine()
    total = UpsertStats()
    params = {"file": str(path), "from": start.isoformat(), "to": end.isoformat()}
    with job_run(engine, "kaggle_backfill", params) as run:
        batch: list[PaperRecord] = []

        def flush() -> None:
            with session_scope(engine) as session:
                stats = upsert_papers(session, batch, source="kaggle")
            for k, v in stats.as_dict().items():
                setattr(total, k, getattr(total, k) + v)
            log.info("backfill progress: %s", total.as_dict())
            batch.clear()

        for record in iter_snapshot(path, categories, start, end):
            batch.append(record)
            if len(batch) >= BATCH_SIZE:
                flush()
        if batch:
            flush()
        run.stats = total.as_dict()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--categories", nargs="*", help="default: the configured OAI sets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    categories = set(args.categories) if args.categories else get_settings().categories
    stats = backfill(args.file, args.start, args.end, categories)
    print(json.dumps(stats.as_dict()))


if __name__ == "__main__":
    main()
