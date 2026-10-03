"""Embed every paper that has no vector yet for the embedder's model."""

import logging
import math
from collections.abc import Collection

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert

from radar.db import session_scope
from radar.db.models import Paper, PaperEmbedding
from radar.embed.base import Embedder, paper_text

log = logging.getLogger(__name__)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def embed_missing(
    engine: Engine,
    embedder: Embedder,
    arxiv_ids: Collection[str] | None = None,
    essential: bool = True,
) -> dict[str, int]:
    """Embed papers lacking a vector for `embedder.model`, optionally only those in `arxiv_ids`."""
    with session_scope(engine) as session:
        query = (
            select(Paper.arxiv_id, Paper.title, Paper.abstract)
            .outerjoin(
                PaperEmbedding,
                (PaperEmbedding.arxiv_id == Paper.arxiv_id)
                & (PaperEmbedding.model == embedder.model),
            )
            .where(PaperEmbedding.arxiv_id.is_(None))
            .order_by(Paper.arxiv_id)
        )
        if arxiv_ids is not None:
            query = query.where(Paper.arxiv_id.in_(list(arxiv_ids)))
        pending = session.execute(query).all()

    embedded = calls = 0
    for start in range(0, len(pending), embedder.max_batch):
        batch = pending[start : start + embedder.max_batch]
        vectors = embedder.embed(
            [paper_text(p.title, p.abstract) for p in batch], "search_document", essential
        )
        calls += 1
        with session_scope(engine) as session:
            session.execute(
                insert(PaperEmbedding)
                .values(
                    [
                        {"arxiv_id": p.arxiv_id, "model": embedder.model, "embedding": normalize(v)}
                        for p, v in zip(batch, vectors, strict=True)
                    ]
                )
                .on_conflict_do_nothing()
            )
        embedded += len(batch)
        log.info("embedded %d/%d", embedded, len(pending))
    return {"pending": len(pending), "embedded": embedded, "embed_calls": calls}
