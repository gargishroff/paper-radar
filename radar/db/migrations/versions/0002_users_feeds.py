"""Users, profiles, interests, feeds and feedback.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27
"""

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    # Papers fetched by ID for onboarding (seed papers, a Semantic Scholar author's papers).
    op.drop_constraint("papers_source_check", "papers")
    op.create_check_constraint(
        "papers_source_check", "papers", "source IN ('oai', 'kaggle', 'arxiv_api')"
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("s2_author_id", sa.String(32), nullable=True),
        sa.Column("settings", postgresql.JSONB(), nullable=True),
        _created_at(),
    )

    op.create_table(
        "interests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("desc_embedding", pgvector.sqlalchemy.Vector(1024), nullable=True),
        sa.Column(
            "medoid_arxiv_id",
            sa.String(32),
            sa.ForeignKey("papers.arxiv_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        _created_at(),
    )
    op.create_index("ix_interests_user_id", "interests", ["user_id"])

    op.create_table(
        "profile_papers",
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "arxiv_id",
            sa.String(32),
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column(
            "interest_id",
            sa.Integer(),
            sa.ForeignKey("interests.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "added_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "source IN ('authored', 'seed', 'liked', 'saved')", name="profile_papers_source_check"
        ),
    )
    op.create_index("ix_profile_papers_interest_id", "profile_papers", ["interest_id"])

    op.create_table(
        "feeds",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("feed_date", sa.Date(), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("stats", postgresql.JSONB(), nullable=True),
        _created_at(),
        sa.UniqueConstraint("user_id", "feed_date", name="feeds_user_date_key"),
    )

    op.create_table(
        "feed_items",
        sa.Column(
            "feed_id", sa.Integer(), sa.ForeignKey("feeds.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "arxiv_id",
            sa.String(32),
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("slot", sa.String(16), nullable=False),
        sa.Column(
            "interest_id",
            sa.Integer(),
            sa.ForeignKey("interests.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("tldr", sa.Text(), nullable=True),
        sa.Column("why", sa.Text(), nullable=True),
        sa.CheckConstraint("slot IN ('main', 'explore', 'rule')", name="feed_items_slot_check"),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "arxiv_id",
            sa.String(32),
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "feed_id", sa.Integer(), sa.ForeignKey("feeds.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("via", sa.String(8), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "action IN ('up', 'down', 'clear', 'save', 'unsave', 'click_abs', 'click_pdf')",
            name="feedback_action_check",
        ),
        sa.CheckConstraint("via IN ('web', 'email')", name="feedback_via_check"),
    )
    op.create_index("ix_feedback_user_paper", "feedback", ["user_id", "arxiv_id", "at"])


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("feed_items")
    op.drop_table("feeds")
    op.drop_table("profile_papers")
    op.drop_table("interests")
    op.drop_table("users")
    op.drop_constraint("papers_source_check", "papers")
    op.create_check_constraint("papers_source_check", "papers", "source IN ('oai', 'kaggle')")
