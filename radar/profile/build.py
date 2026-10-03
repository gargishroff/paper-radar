"""Build and maintain a user's interests from their profile papers."""

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import Engine, delete, select, update
from sqlalchemy.dialects.postgresql import insert

from radar.config import Settings
from radar.db import session_scope
from radar.db.models import Interest, Paper, PaperEmbedding, ProfilePaper
from radar.embed.base import Embedder
from radar.embed.pipeline import embed_missing, normalize
from radar.ingest.by_id import PaperFetcher, ensure_papers
from radar.llm import ChatModel
from radar.profile.cluster import choose_clusters, medoid, normalize_weights, recency_weights
from radar.usage import BudgetExceeded

log = logging.getLogger(__name__)


def add_profile_papers(
    engine: Engine,
    fetcher: PaperFetcher,
    embedder: Embedder,
    user_id: int,
    arxiv_ids: Iterable[str],
    source: str,
    interest_id: int | None = None,
) -> list[str]:
    """Fetch and embed the papers, then add them to the profile. Returns the added IDs."""
    found = ensure_papers(engine, fetcher, arxiv_ids)
    if not found:
        return []
    embed_missing(engine, embedder, found, essential=True)
    with session_scope(engine) as session:
        session.execute(
            insert(ProfilePaper)
            .values(
                [
                    {
                        "user_id": user_id,
                        "arxiv_id": a,
                        "source": source,
                        "interest_id": interest_id,
                    }
                    for a in found
                ]
            )
            .on_conflict_do_nothing()
        )
    return sorted(found)


def _profile_rows(session, user_id: int, model: str, where=None):
    query = (
        select(ProfilePaper, Paper, PaperEmbedding.embedding)
        .join(Paper, Paper.arxiv_id == ProfilePaper.arxiv_id)
        .join(
            PaperEmbedding,
            (PaperEmbedding.arxiv_id == ProfilePaper.arxiv_id) & (PaperEmbedding.model == model),
        )
        .where(ProfilePaper.user_id == user_id)
    )
    if where is not None:
        query = query.where(where)
    return session.execute(query).all()


def _weights(rows, tau_days: float, now: datetime) -> np.ndarray:
    """Recency weights: authored papers age from publication, others from when they were added."""
    ages = np.array(
        [
            ((now - (paper.published_at if pp.source == "authored" else pp.added_at)).days)
            for pp, paper, _ in rows
        ],
        dtype=float,
    )
    return recency_weights(ages, tau_days)


def interest_vectors(session, user_id: int, model: str) -> list[tuple[Interest, np.ndarray]]:
    """Each interest's retrieval vector: its medoid's embedding, else its description's."""
    out = []
    for interest in session.scalars(
        select(Interest).where(Interest.user_id == user_id).order_by(Interest.id)
    ):
        vec = None
        if interest.medoid_arxiv_id:
            vec = session.scalar(
                select(PaperEmbedding.embedding).where(
                    PaperEmbedding.arxiv_id == interest.medoid_arxiv_id,
                    PaperEmbedding.model == model,
                )
            )
        if vec is None and interest.desc_embedding is not None:
            vec = interest.desc_embedding
        if vec is not None:
            out.append((interest, np.asarray(vec, dtype=float)))
    return out


def rebuild_interests(
    engine: Engine,
    chat: ChatModel | None,
    settings: Settings,
    user_id: int,
    model: str,
    now: datetime | None = None,
) -> list[int]:
    """Recluster all profile papers not held by a pinned interest into fresh interests."""
    now = now or datetime.now(UTC)
    with session_scope(engine) as session:
        pinned_ids = session.scalars(
            select(Interest.id).where(Interest.user_id == user_id, Interest.pinned)
        ).all()
        session.execute(
            delete(Interest).where(Interest.user_id == user_id, Interest.pinned.is_(False))
        )
        rows = _profile_rows(
            session,
            user_id,
            model,
            where=ProfilePaper.interest_id.is_(None)
            | ProfilePaper.interest_id.not_in(pinned_ids or [-1]),
        )
        if not rows:
            _refresh_weights(session, user_id, model, settings, now)
            return []

        x = np.array([np.asarray(e, dtype=float) for _, _, e in rows])
        w = _weights(rows, settings.recency_tau_days, now)
        k_max = max(1, settings.max_interests - len(pinned_ids))
        clustering = choose_clusters(x, w, k_max)

        clusters = [
            [rows[i][1] for i in np.flatnonzero(clustering.labels == c)]
            for c in range(len(clustering.medoids))
        ]
        names = describe_clusters(chat, clusters)
        new_ids = []
        for c, (name, description) in enumerate(names):
            interest = Interest(
                user_id=user_id,
                name=name,
                description=description,
                medoid_arxiv_id=rows[clustering.medoids[c]][1].arxiv_id,
                weight=float(clustering.mass[c]),
                pinned=False,
            )
            session.add(interest)
            session.flush()
            new_ids.append(interest.id)
            members = [rows[i][0].arxiv_id for i in np.flatnonzero(clustering.labels == c)]
            session.execute(
                update(ProfilePaper)
                .where(ProfilePaper.user_id == user_id, ProfilePaper.arxiv_id.in_(members))
                .values(interest_id=interest.id)
            )
        _refresh_weights(session, user_id, model, settings, now)
        return new_ids


def assign_unplaced(
    engine: Engine, settings: Settings, user_id: int, model: str, now: datetime | None = None
) -> int:
    """Attach profile papers with no interest to the nearest interest; refresh weights."""
    now = now or datetime.now(UTC)
    with session_scope(engine) as session:
        vectors = interest_vectors(session, user_id, model)
        rows = _profile_rows(session, user_id, model, where=ProfilePaper.interest_id.is_(None))
        if vectors and rows:
            centres = np.array([v for _, v in vectors])
            for pp, _, emb in rows:
                nearest = int(np.argmax(centres @ np.asarray(emb, dtype=float)))
                pp.interest_id = vectors[nearest][0].id
        _refresh_weights(session, user_id, model, settings, now)
        return len(rows) if vectors else 0


def _refresh_weights(session, user_id: int, model: str, settings: Settings, now: datetime):
    """pi_c proportional to the recency-weighted mass of each interest's papers."""
    session.flush()
    interests = session.scalars(select(Interest).where(Interest.user_id == user_id)).all()
    if not interests:
        return
    rows = _profile_rows(session, user_id, model, where=ProfilePaper.interest_id.is_not(None))
    w = _weights(rows, settings.recency_tau_days, now) if rows else np.array([])
    mass = {i.id: 0.0 for i in interests}
    for (pp, _, _), wi in zip(rows, w, strict=True):
        mass[pp.interest_id] = mass.get(pp.interest_id, 0.0) + float(wi)
    nonzero = [m for m in mass.values() if m > 0]
    fill = float(np.median(nonzero)) if nonzero else 1.0
    raw = np.array([mass[i.id] or fill for i in interests])
    for interest, pi in zip(interests, normalize_weights(raw), strict=True):
        interest.weight = float(pi)


_DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "interests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "integer"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["cluster", "name", "description"],
            },
        }
    },
    "required": ["interests"],
}


def describe_clusters(
    chat: ChatModel | None, clusters: Sequence[Sequence[Paper]]
) -> list[tuple[str, str]]:
    """A short name and a 1–2 sentence description per cluster, in one Command A call."""
    fallback = [(c[0].title[:60], f"Research related to: {c[0].title}") for c in clusters]
    if chat is None:
        return fallback
    listing = []
    for i, papers in enumerate(clusters):
        titles = "\n".join(f"  - {p.title}" for p in papers[:10])
        listing.append(f"Cluster {i} ({len(papers)} papers):\n{titles}")
    system = (
        "You name a researcher's research interests. Each cluster is a group of papers the "
        "researcher wrote or liked. For each cluster give a short name (2-6 words) and a "
        "description of at most 30 words, phrased as a request for new papers "
        "('Papers on ...'). Describe the underlying research topic broadly enough to match "
        "future work on it: name the problems and methods, not the specific papers, models "
        "or datasets in the titles."
    )
    try:
        reply = chat.chat_json(system, "\n\n".join(listing), _DESCRIBE_SCHEMA)
    except BudgetExceeded as exc:
        log.warning("describing clusters skipped: %s", exc)
        return fallback
    out = list(fallback)
    for item in reply.get("interests", []):
        c = item.get("cluster")
        if isinstance(c, int) and 0 <= c < len(out) and item.get("name"):
            out[c] = (item["name"].strip(), item.get("description", "").strip() or out[c][1])
    return out


def add_seed_interest(
    engine: Engine,
    fetcher: PaperFetcher,
    embedder: Embedder,
    chat: ChatModel | None,
    settings: Settings,
    user_id: int,
    arxiv_ids: Iterable[str],
) -> int | None:
    """A pinned interest built from a handful of seed papers."""
    with session_scope(engine) as session:
        interest = Interest(user_id=user_id, name="…", description="…", weight=0.0, pinned=True)
        session.add(interest)
        session.flush()
        interest_id = interest.id
    added = add_profile_papers(
        engine, fetcher, embedder, user_id, arxiv_ids, "seed", interest_id=interest_id
    )
    now = datetime.now(UTC)
    with session_scope(engine) as session:
        interest = session.get(Interest, interest_id)
        rows = _profile_rows(
            session, user_id, embedder.model, where=ProfilePaper.interest_id == interest_id
        )
        if not added or not rows:
            session.delete(interest)
            return None
        x = np.array([np.asarray(e, dtype=float) for _, _, e in rows])
        w = _weights(rows, settings.recency_tau_days, now)
        interest.medoid_arxiv_id = rows[medoid(x, w, np.ones(len(rows), bool))][1].arxiv_id
        [(interest.name, interest.description)] = describe_clusters(chat, [[r[1] for r in rows]])
        _refresh_weights(session, user_id, embedder.model, settings, now)
    return interest_id


def query_text(name: str, description: str) -> str:
    """What an interest searches with."""
    return f"{name}: {description}"


def add_text_interest(
    engine: Engine,
    embedder: Embedder,
    settings: Settings,
    user_id: int,
    name: str,
    description: str,
) -> int:
    """A pinned interest from a free-text description, for topics with no papers yet."""
    [vec] = embedder.embed([query_text(name, description)], "search_query", essential=True)
    with session_scope(engine) as session:
        interest = Interest(
            user_id=user_id,
            name=name,
            description=description,
            desc_embedding=normalize(vec),
            weight=0.0,
            pinned=True,
        )
        session.add(interest)
        session.flush()
        _refresh_weights(session, user_id, embedder.model, settings, datetime.now(UTC))
        return interest.id
