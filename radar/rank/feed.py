"""Build one user's feed for one day: candidates → Rerank 4 → slate → explanations."""

import logging
from datetime import date

import numpy as np
from sqlalchemy import Engine, delete, select

from radar.config import Settings
from radar.db import session_scope
from radar.db.models import Feed, FeedItem, Paper, PaperEmbedding, ProfilePaper, User
from radar.embed.base import paper_text
from radar.explain.explain import ExplainInput, explain
from radar.llm import ChatModel, Reranker
from radar.profile.build import assign_unplaced, interest_vectors, query_text
from radar.rank.candidates import generate
from radar.rank.slate import Scored, build_slate
from radar.usage import BudgetExceeded

log = logging.getLogger(__name__)

MODEL_VERSION = "untrained-v1"


def _name_key(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").split())


def build_feed(
    engine: Engine,
    settings: Settings,
    user_id: int,
    day: date,
    embed_model: str,
    reranker: Reranker | None,
    chat: ChatModel | None,
    rebuild: bool = False,
) -> int | None:
    """Create (or with `rebuild`, replace) the feed for `day`."""
    with session_scope(engine) as session:
        existing = session.scalar(
            select(Feed.id).where(Feed.user_id == user_id, Feed.feed_date == day)
        )
        if existing and not rebuild:
            return existing
        if existing:
            session.execute(delete(Feed).where(Feed.id == existing))

    assign_unplaced(engine, settings, user_id, embed_model)

    with session_scope(engine) as session:
        user = session.get(User, user_id)
        interests = interest_vectors(session, user_id, embed_model)
        if not interests:
            log.info("user %s has no interests yet", user_id)
            return None
        profile_ids = set(
            session.scalars(select(ProfilePaper.arxiv_id).where(ProfilePaper.user_id == user_id))
        )
        rows = session.execute(
            select(Paper, PaperEmbedding.embedding)
            .join(PaperEmbedding, PaperEmbedding.arxiv_id == Paper.arxiv_id)
            .where(Paper.announced_on == day, PaperEmbedding.model == embed_model)
            .order_by(Paper.arxiv_id)
        ).all()
        rows = [r for r in rows if r[0].arxiv_id not in profile_ids]
        if not rows:
            log.info("no papers announced on %s", day)
            return None
        papers = [r[0] for r in rows]
        x = np.array([np.asarray(r[1], dtype=float) for r in rows])
        coauthors = _coauthors(session, user_id, user.name)

        # Profile vectors, for the explanation step's nearest-neighbour lookup.
        profile = session.execute(
            select(Paper.arxiv_id, Paper.title, PaperEmbedding.embedding, ProfilePaper.interest_id)
            .join(ProfilePaper, ProfilePaper.arxiv_id == Paper.arxiv_id)
            .join(PaperEmbedding, PaperEmbedding.arxiv_id == Paper.arxiv_id)
            .where(ProfilePaper.user_id == user_id, PaperEmbedding.model == embed_model)
        ).all()

    weights = np.array([i.weight for i, _ in interests])
    centres = np.array([v for _, v in interests])
    rng = np.random.default_rng(int(day.strftime("%Y%m%d")) * 1000 + user_id)
    cands = generate(
        x,
        centres,
        settings.candidates_per_interest,
        settings.explore_band,
        settings.explore_pool_size,
        rng,
    )

    # Stage 2: Rerank 4 per interest, with the interest description as the query.
    rerank_scores: dict[tuple[int, int], float] = {}
    rerank_calls = 0
    reranked = reranker is not None
    for c, (interest, _) in enumerate(interests):
        idx = cands.per_interest[c]
        if not reranked:
            break
        try:
            scores = reranker.rerank(
                query_text(interest.name, interest.description),
                [paper_text(papers[i].title, papers[i].abstract) for i in idx],
            )
        except BudgetExceeded as exc:
            log.warning("rerank skipped, using cosine similarity: %s", exc)
            reranked = False
            break
        rerank_calls += 1
        for i, s in zip(idx, scores, strict=True):
            rerank_scores[(int(i), c)] = s

    def score(i: int, c: int) -> float:
        return rerank_scores[(i, c)] if reranked else float(cands.sims[i, c])

    best: dict[int, Scored] = {}
    for c in range(len(interests)):
        for i in cands.per_interest[c]:
            s = Scored(int(i), c, score(int(i), c))
            if int(i) not in best or s.score > best[int(i)].score:
                best[int(i)] = s

    rule = []
    for i, paper in enumerate(papers):
        if coauthors & {_name_key(a) for a in paper.authors}:
            c = int(np.argmax(cands.sims[i]))
            rule.append(best.get(i) or Scored(i, c, float(cands.sims[i, c])))

    slate = build_slate(
        list(best.values()),
        weights,
        x,
        settings.feed_size,
        [int(i) for i in cands.explore],
        settings.explore_slots,
        rule,
        settings.mmr_lambda,
    )

    # Explanations, grounded in the matched interest and its two most similar papers.
    inputs = []
    for slot in slate:
        paper = papers[slot.index]
        c = slot.interest if slot.interest is not None else int(np.argmax(cands.sims[slot.index]))
        interest = interests[c][0]
        members = [p for p in profile if p.interest_id == interest.id]
        m_x = np.array([np.asarray(p.embedding, dtype=float) for p in members])
        near = np.argsort(-(m_x @ x[slot.index]))[:2] if len(members) else []
        inputs.append(
            ExplainInput(
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                [(members[j].arxiv_id, members[j].title) for j in near],
                interest=(interest.name, interest.description),
                explore=slot.kind == "explore",
            )
        )
    explanations = explain(chat, inputs)

    stats = {
        "papers": len(papers),
        "candidates": len(best),
        "explore_pool": len(cands.explore),
        "rule": len(rule),
        "reranked": reranked,
        "rerank_calls": rerank_calls,
        "grounded_explanations": sum(e.grounded for e in explanations.values()),
    }
    with session_scope(engine) as session:
        feed = Feed(user_id=user_id, feed_date=day, model_version=MODEL_VERSION, stats=stats)
        session.add(feed)
        session.flush()
        for rank, slot in enumerate(slate):
            paper = papers[slot.index]
            ex = explanations[paper.arxiv_id]
            interest_id = interests[slot.interest][0].id if slot.interest is not None else None
            c_best = int(np.argmax(cands.sims[slot.index]))
            session.add(
                FeedItem(
                    feed_id=feed.id,
                    arxiv_id=paper.arxiv_id,
                    rank=rank,
                    slot=slot.kind,
                    interest_id=interest_id,
                    scores={
                        "cosine": round(float(cands.sims[slot.index].max()), 4),
                        "rerank": (
                            round(rerank_scores[(slot.index, slot.interest)], 4)
                            if (slot.index, slot.interest) in rerank_scores
                            else None
                        ),
                        "nearest_interest": interests[c_best][0].id,
                        "grounded": ex.grounded,
                    },
                    tldr=ex.tldr,
                    why=ex.why,
                )
            )
        log.info("feed %s for %s: %s", feed.id, day, stats)
        return feed.id


def _coauthors(session, user_id: int, own_name: str) -> set[str]:
    """Names on the user's authored papers, minus their own."""
    lists = session.scalars(
        select(Paper.authors)
        .join(ProfilePaper, ProfilePaper.arxiv_id == Paper.arxiv_id)
        .where(ProfilePaper.user_id == user_id, ProfilePaper.source == "authored")
    ).all()
    names = {_name_key(a) for authors in lists for a in authors}
    return names - {_name_key(own_name)}
