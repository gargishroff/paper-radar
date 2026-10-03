"""End-to-end product tests on a real database with fake Cohere components."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from radar.config import EMBED_DIM
from radar.db import session_scope
from radar.db.models import Feed, FeedItem, Interest, ProfilePaper, User
from radar.email.digest import send_digest
from radar.embed.pipeline import embed_missing
from radar.feedback import sign
from radar.ingest.records import PaperRecord
from radar.ingest.store import upsert_papers
from radar.profile.build import add_text_interest, rebuild_interests
from radar.rank.feed import build_feed
from tests.conftest import FakeEmbedder

DAY = date(2026, 9, 24)


class TopicEmbedder(FakeEmbedder):
    """Texts mentioning 'alpha' or 'beta' embed near one of two directions."""

    max_batch = 96

    def embed(self, texts, input_type, essential=False):
        self.calls += 1
        rng = np.random.default_rng(len(texts))
        out = []
        for t in texts:
            v = 0.2 * rng.standard_normal(EMBED_DIM)
            v[0] += 3 if "alpha" in t else 0
            v[1] += 3 if "beta" in t else 0
            out.append((v / np.linalg.norm(v)).tolist())
        return out


class FakeReranker:
    model = "fake-rerank"

    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents, essential=False):
        self.calls += 1
        word = "alpha" if "alpha" in query else "beta"
        return [0.9 if word in d else 0.1 for d in documents]


class FakeChat:
    model = "fake-chat"

    def chat_json(self, system, user, schema, essential=False):
        if "interests" in schema["properties"]:
            n = user.count("Cluster ")
            return {
                "interests": [
                    {"cluster": i, "name": f"Topic {i}", "description": f"Papers on alpha {i}"}
                    for i in range(n)
                ]
            }
        ids = [
            line.split("[")[1].split("]")[0] for line in user.splitlines() if "NEW PAPER" in line
        ]
        return {
            "papers": [
                {"arxiv_id": a, "tldr": f"tldr {a}", "why": "because", "relates_to": "interest"}
                for a in ids
            ]
        }


def _record(
    arxiv_id: str, title: str, published: datetime, authors: tuple[str, ...] = ("Someone Else",)
) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"{title}. More text.",
        authors=list(authors),
        categories=["cs.LG"],
        published_at=published,
        version_at=published,
        latest_version=1,
        comments=None,
        license=None,
    )


@pytest.fixture
def world(engine, settings):
    """A user with 8 authored papers on two topics, and 60 new papers announced on DAY."""
    embedder = TopicEmbedder()
    old = datetime(2025, 6, 1, tzinfo=UTC)
    me = ("Ada Lovelace", "Co Author")
    authored = [_record(f"2506.{i:05d}", f"alpha study {i}", old, me) for i in range(4)]
    authored += [_record(f"2506.{i:05d}", f"beta study {i}", old, me) for i in range(4, 8)]
    new_time = datetime(2026, 9, 22, tzinfo=UTC)
    new = [_record(f"2609.{i:05d}", f"alpha new {i}", new_time) for i in range(20)]
    new += [_record(f"2609.{i:05d}", f"beta new {i}", new_time) for i in range(20, 40)]
    new += [_record(f"2609.{i:05d}", f"gamma other {i}", new_time) for i in range(40, 59)]
    new += [_record("2609.00099", "alpha coauthored work", new_time, ("Co Author", "X Y"))]

    with session_scope(engine) as s:
        upsert_papers(s, authored, source="oai")
        upsert_papers(s, new, source="oai", announced_on=lambda _: DAY)
        user = User(name="Ada Lovelace", email="ada@example.org")
        s.add(user)
        s.flush()
        uid = user.id
        s.add_all(
            ProfilePaper(user_id=uid, arxiv_id=r.arxiv_id, source="authored") for r in authored
        )
    embed_missing(engine, embedder)
    rebuild_interests(engine, FakeChat(), settings, uid, embedder.model)
    return {"uid": uid, "embedder": embedder}


def test_onboarding_finds_two_interests(engine, world):
    with session_scope(engine) as s:
        interests = s.scalars(select(Interest).where(Interest.user_id == world["uid"])).all()
        assert len(interests) == 2
        assert sum(i.weight for i in interests) == pytest.approx(1.0)
        members = s.scalars(select(ProfilePaper.interest_id)).all()
        assert None not in members


def test_feed_build_end_to_end(engine, settings, world):
    reranker = FakeReranker()
    feed_id = build_feed(
        engine, settings, world["uid"], DAY, world["embedder"].model, reranker, FakeChat()
    )
    with session_scope(engine) as s:
        feed = s.get(Feed, feed_id)
        items = s.scalars(
            select(FeedItem).where(FeedItem.feed_id == feed_id).order_by(FeedItem.rank)
        ).all()
        assert len(items) == settings.feed_size
        assert feed.stats["reranked"] and reranker.calls == 2
        slots = [i.slot for i in items]
        assert slots.count("explore") == 2
        assert slots[0] == "rule" and items[0].arxiv_id == "2609.00099"  # co-author paper
        main_titles = [i.arxiv_id for i in items if i.slot == "main"]
        # Main slots come from the two topics, never from the unrelated "gamma" papers.
        assert all(int(a.split(".")[1]) < 40 for a in main_titles)
        assert all(i.tldr and i.why for i in items)
        assert feed.stats["grounded_explanations"] == settings.feed_size - 2

    # Building again returns the same feed; no new reranker calls.
    assert (
        build_feed(
            engine, settings, world["uid"], DAY, world["embedder"].model, reranker, FakeChat()
        )
        == feed_id
    )
    assert reranker.calls == 2


def test_feed_without_reranker_uses_cosine(engine, settings, world):
    feed_id = build_feed(engine, settings, world["uid"], DAY, world["embedder"].model, None, None)
    with session_scope(engine) as s:
        assert not s.get(Feed, feed_id).stats["reranked"]
        assert len(s.scalars(select(FeedItem).where(FeedItem.feed_id == feed_id)).all()) == 12


def test_no_feed_on_a_day_without_papers(engine, settings, world):
    day = DAY + timedelta(days=2)
    assert build_feed(engine, settings, world["uid"], day, "fake-embed", None, None) is None


def test_text_interest_gets_median_weight(engine, settings, world):
    add_text_interest(engine, world["embedder"], settings, world["uid"], "Gamma", "Papers on gamma")
    with session_scope(engine) as s:
        weights = sorted(i.weight for i in s.scalars(select(Interest)))
    assert len(weights) == 3 and sum(weights) == pytest.approx(1.0)


def test_web_app_flow(engine, settings, world, tmp_path):
    from radar.app.main import create_app
    from radar.services import Services

    settings.digest_dir = str(tmp_path)
    feed_id = build_feed(
        engine, settings, world["uid"], DAY, world["embedder"].model, FakeReranker(), FakeChat()
    )
    fake = Services(
        embedder=world["embedder"], reranker=FakeReranker(), chat=FakeChat(), fetcher=None
    )
    client = TestClient(create_app(engine, settings, services=lambda: fake))

    page = client.get("/today")
    assert page.status_code == 200 and "tldr 2609" in page.text

    with session_scope(engine) as s:
        first = s.scalar(select(FeedItem.arxiv_id).where(FeedItem.feed_id == feed_id))
    assert "on-up" in client.post(f"/rate/{first}/up?feed_id={feed_id}").text
    assert "Saved" in client.post(f"/rate/{first}/save?feed_id={feed_id}").text
    assert first in client.get("/library").text
    assert f"eprint = {{{first}}}" in client.get("/library.bib").text
    with session_scope(engine) as s:
        assert s.get(ProfilePaper, (world["uid"], first)) is not None

    # A 👎 via a signed email link removes the like from the profile.
    token = sign(settings.secret_key, world["uid"], first, "down", feed_id)
    assert client.get(f"/r/{token}").status_code == 200
    with session_scope(engine) as s:
        assert s.get(ProfilePaper, (world["uid"], first)) is None
    assert client.get(f"/r/{token}x").status_code == 400

    assert "alpha" in client.get("/search?q=alpha").text
    for path in ("/interests", "/status", "/healthz"):
        assert client.get(path).status_code == 200
    assert client.post("/rate/9999.99999/up").status_code == 404

    # The digest is written once to the digest folder (no SMTP configured).
    assert send_digest(engine, settings, feed_id) is not None
    assert send_digest(engine, settings, feed_id) is None
    html = next(tmp_path.iterdir()).read_text()
    assert "/r/" in html and "tldr 2609" in html
