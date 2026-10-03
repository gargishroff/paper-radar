"""Write paper records to Postgres. Upserts are keyed by arXiv ID, so re-running is harmless."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from radar.db.models import Paper, PaperEmbedding
from radar.ingest.records import PaperRecord


@dataclass
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    announced: int = 0  # inserted rows marked as that day's new submissions
    # Papers whose title or abstract changed; their stale vectors were deleted.
    reembed_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        counts = {k: v for k, v in vars(self).items() if isinstance(v, int)}
        return counts | {"reembed": len(self.reembed_ids)}


def _values(record: PaperRecord) -> dict:
    return {
        "title": record.title,
        "abstract": record.abstract,
        "authors": record.authors,
        "categories": record.categories,
        "primary_cat": record.primary_cat,
        "version_at": record.version_at,
        "latest_version": record.latest_version,
        "comments": record.comments,
        "license": record.license,
        "has_code": record.has_code,
    }


def upsert_papers(
    session: Session,
    records: Iterable[PaperRecord],
    source: str,
    announced_on: Callable[[PaperRecord], date | None] = lambda _: None,
) -> UpsertStats:
    """Insert unseen papers and refresh changed ones."""
    records = list(records)
    stats = UpsertStats()
    if not records:
        return stats

    existing = {
        p.arxiv_id: p
        for p in session.scalars(
            select(Paper).where(Paper.arxiv_id.in_([r.arxiv_id for r in records]))
        )
    }
    for record in records:
        values = _values(record)
        paper = existing.get(record.arxiv_id)
        if paper is None:
            day = announced_on(record)
            session.add(
                Paper(
                    arxiv_id=record.arxiv_id,
                    published_at=record.published_at,
                    announced_on=day,
                    source=source,
                    **values,
                )
            )
            stats.inserted += 1
            stats.announced += day is not None
        elif record.version_at < paper.version_at:
            # An older snapshot (e.g. a stale Kaggle dump) never overwrites newer metadata.
            stats.unchanged += 1
        elif any(getattr(paper, k) != v for k, v in values.items()):
            if (paper.title, paper.abstract) != (record.title, record.abstract):
                stats.reembed_ids.append(paper.arxiv_id)
            for k, v in values.items():
                setattr(paper, k, v)
            stats.updated += 1
        else:
            stats.unchanged += 1
    if stats.reembed_ids:
        session.execute(
            delete(PaperEmbedding).where(PaperEmbedding.arxiv_id.in_(stats.reembed_ids))
        )
    session.flush()
    return stats
