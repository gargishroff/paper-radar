"""ORM models: ingestion, embeddings, operations, users, profiles, feeds, feedback."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from radar.config import EMBED_DIM


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSONB)
    categories: Mapped[list[str]] = mapped_column(ARRAY(String(32)))
    primary_cat: Mapped[str] = mapped_column(String(32))
    # Date of v1: when the paper first appeared.
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Date of the latest version we have seen.
    version_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latest_version: Mapped[int] = mapped_column(Integer)
    # Feed day of a new submission; NULL for revisions and backfill.
    announced_on: Mapped[date | None] = mapped_column(Date, index=True)
    comments: Mapped[str | None] = mapped_column(Text)
    has_code: Mapped[bool] = mapped_column(Boolean, default=False)
    license: Mapped[str | None] = mapped_column(Text)
    type_probs: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("source IN ('oai', 'kaggle', 'arxiv_api')", name="papers_source_check"),
        Index("ix_papers_published_at", "published_at"),
        Index("ix_papers_categories", "categories", postgresql_using="gin"),
    )


class PaperEmbedding(Base):
    __tablename__ = "paper_embeddings"

    arxiv_id: Mapped[str] = mapped_column(
        ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_paper_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ApiUsage(Base):
    __tablename__ = "api_usage"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(32), primary_key=True)
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    # Billed units reported by the API (input tokens for embed, search units for rerank).
    units: Mapped[int] = mapped_column(Integer, default=0)


class JobRun(Base):
    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="running")
    params: Mapped[dict | None] = mapped_column(JSONB)
    stats: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("status IN ('running', 'ok', 'failed')", name="job_runs_status_check"),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)
    s2_author_id: Mapped[str | None] = mapped_column(String(32))
    settings: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Interest(Base):
    """One topic a user follows: a cluster of profile papers, or a free-text description."""

    __tablename__ = "interests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(Text)
    # Editable in the UI and used as the Rerank 4 query.
    description: Mapped[str] = mapped_column(Text)
    # Retrieval vector for text-only interests, which have no medoid paper.
    desc_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    medoid_arxiv_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.arxiv_id", ondelete="SET NULL")
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    # Pinned interests (edited or created by hand) survive "rebuild interests".
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProfilePaper(Base):
    __tablename__ = "profile_papers"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    arxiv_id: Mapped[str] = mapped_column(
        ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    source: Mapped[str] = mapped_column(String(16))
    interest_id: Mapped[int | None] = mapped_column(
        ForeignKey("interests.id", ondelete="SET NULL"), index=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "source IN ('authored', 'seed', 'liked', 'saved')", name="profile_papers_source_check"
        ),
    )


class Feed(Base):
    __tablename__ = "feeds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    feed_date: Mapped[date] = mapped_column(Date)
    model_version: Mapped[str] = mapped_column(String(32))
    stats: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "feed_date", name="feeds_user_date_key"),)


class FeedItem(Base):
    __tablename__ = "feed_items"

    feed_id: Mapped[int] = mapped_column(
        ForeignKey("feeds.id", ondelete="CASCADE"), primary_key=True
    )
    arxiv_id: Mapped[str] = mapped_column(
        ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(Integer)
    slot: Mapped[str] = mapped_column(String(16))
    interest_id: Mapped[int | None] = mapped_column(ForeignKey("interests.id", ondelete="SET NULL"))
    scores: Mapped[dict] = mapped_column(JSONB)
    tldr: Mapped[str | None] = mapped_column(Text)
    why: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("slot IN ('main', 'explore', 'rule')", name="feed_items_slot_check"),
    )


class Feedback(Base):
    """Append-only event log; a paper's current rating is its latest event."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    arxiv_id: Mapped[str] = mapped_column(ForeignKey("papers.arxiv_id", ondelete="CASCADE"))
    feed_id: Mapped[int | None] = mapped_column(ForeignKey("feeds.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(16))
    via: Mapped[str] = mapped_column(String(8), default="web")
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "action IN ('up', 'down', 'clear', 'save', 'unsave', 'click_abs', 'click_pdf')",
            name="feedback_action_check",
        ),
        CheckConstraint("via IN ('web', 'email')", name="feedback_via_check"),
        Index("ix_feedback_user_paper", "user_id", "arxiv_id", "at"),
    )
