"""Ingestion, embeddings and operations tables.

Revision ID: 0001
Revises:
Create Date: 2026-09-27
"""

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "papers",
        sa.Column("arxiv_id", sa.String(32), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("authors", postgresql.JSONB(), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.String(32)), nullable=False),
        sa.Column("primary_cat", sa.String(32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latest_version", sa.Integer(), nullable=False),
        sa.Column("announced_on", sa.Date(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("has_code", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("type_probs", postgresql.JSONB(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("source IN ('oai', 'kaggle')", name="papers_source_check"),
    )
    op.create_index("ix_papers_announced_on", "papers", ["announced_on"])
    op.create_index("ix_papers_published_at", "papers", ["published_at"])
    op.create_index("ix_papers_categories", "papers", ["categories"], postgresql_using="gin")

    op.create_table(
        "paper_embeddings",
        sa.Column(
            "arxiv_id",
            sa.String(32),
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model", sa.String(64), primary_key=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.execute(
        "CREATE INDEX ix_paper_embeddings_hnsw ON paper_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "api_usage",
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("endpoint", sa.String(32), primary_key=True),
        sa.Column("model", sa.String(64), primary_key=True),
        sa.Column("calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("units", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "job_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job", sa.String(64), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("stats", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint("status IN ('running', 'ok', 'failed')", name="job_runs_status_check"),
    )


def downgrade() -> None:
    op.drop_table("job_runs")
    op.drop_table("api_usage")
    op.drop_table("paper_embeddings")
    op.drop_table("papers")
