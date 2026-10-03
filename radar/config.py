"""Runtime settings, read from environment variables or a local `.env` file."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fixed so pgvector can build an HNSW index on the column.
EMBED_DIM = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://localhost/paperradar"

    # arXiv asks harvesters to identify themselves; the contact goes in the User-Agent.
    arxiv_contact_email: str = ""
    oai_base_url: str = "https://oaipmh.arxiv.org/oai"
    oai_delay_seconds: float = 3.0
    # OAI-PMH set specs (group:archive:subject). Cross-lists are included by arXiv.
    oai_sets: list[str] = ["cs:cs:LG", "cs:cs:CL", "cs:cs:AI", "cs:cs:CV", "stat:stat:ML"]

    cohere_api_key: str = ""
    embed_model: str = "embed-v4.0"
    embed_batch_size: int = 96  # Cohere's per-call maximum for texts
    # Measured trial-key limit: a 4th call of 96 abstracts within a minute gets a 429.
    embed_inputs_per_minute: int = 2000
    embed_calls_per_minute: int = 3

    # JSON-schema output on Command A+ needs thinking on; command-a-03-2025 is cheaper.
    chat_model: str = "command-a-plus-05-2026"
    chat_calls_per_minute: int = 20
    rerank_model: str = "rerank-v4.0-fast"
    rerank_calls_per_minute: int = 10

    # Feed shape.
    feed_size: int = 12
    explore_slots: int = 2
    candidates_per_interest: int = 50
    explore_band: tuple[int, int] = (50, 300)
    explore_pool_size: int = 20
    mmr_lambda: float = 0.7
    max_interests: int = 6
    recency_tau_days: float = 365.0

    # Web app and email digest.
    app_base_url: str = "http://localhost:8000"
    # Signs the rating links in emails. Set a long random value in production.
    secret_key: str = "dev-secret-change-me"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    # Where digests are written when SMTP is not configured.
    digest_dir: str = "data/digests"

    # Scheduler: run the daily job at this local hour.
    daily_hour: int = 7
    timezone: str = "UTC"

    semantic_scholar_api_key: str = ""

    # Trial keys allow ~1,000 calls/month across all endpoints.
    monthly_call_budget: int = 1000
    # Past this fraction of the budget, only calls marked essential are allowed.
    nonessential_budget_fraction: float = 0.8

    @field_validator("embed_batch_size")
    @classmethod
    def _batch_in_range(cls, v: int) -> int:
        if not 1 <= v <= 96:
            raise ValueError("embed_batch_size must be between 1 and 96")
        return v

    @property
    def categories(self) -> set[str]:
        """arXiv categories implied by the OAI sets, e.g. 'cs:cs:LG' -> 'cs.LG'."""
        return {set_spec_to_category(s) for s in self.oai_sets}


def set_spec_to_category(set_spec: str) -> str:
    _group, archive, subject = set_spec.split(":")
    return f"{archive}.{subject}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
