import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from radar.config import EMBED_DIM, Settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://localhost/paperradar_test"
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        oai_delay_seconds=0,
        cohere_api_key="test",
        monthly_call_budget=10,
    )


@pytest.fixture
def engine():
    """A freshly migrated test database. Skips the test if Postgres is unreachable."""
    eng = create_engine(TEST_DATABASE_URL)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip(f"no test database at {TEST_DATABASE_URL}")

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "radar/db/migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


def oai_transport(requests: list[httpx.Request] | None = None) -> httpx.MockTransport:
    """Serves the two fixture pages: page 1 for a fresh query, page 2 for the resumption token."""
    page1 = (FIXTURES / "oai_page1.xml").read_text()
    page2 = (FIXTURES / "oai_page2.xml").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        token = request.url.params.get("resumptionToken")
        return httpx.Response(200, text=page2 if token == "tok123" else page1)

    return httpx.MockTransport(handler)


class FakeEmbedder:
    """Deterministic pseudo-embeddings from a hash of the text; counts calls."""

    model = "fake-embed"
    max_batch = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str], input_type: str, essential: bool = False):
        self.calls += 1
        out = []
        for t in texts:
            seed = hashlib.sha256(t.encode()).digest()
            v = [((seed[i % 32] + i) % 17) - 8.0 for i in range(EMBED_DIM)]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out
