"""Builds the Cohere-backed components shared by the CLI, jobs and web app."""

from dataclasses import dataclass

from sqlalchemy import Engine

from radar.config import Settings
from radar.embed.cohere_embed import CohereEmbedder
from radar.ingest.by_id import PaperFetcher
from radar.llm import CohereChat, CohereReranker
from radar.usage import UsageTracker


@dataclass
class Services:
    embedder: CohereEmbedder
    reranker: CohereReranker
    chat: CohereChat
    fetcher: PaperFetcher


def make_services(engine: Engine, settings: Settings) -> Services:
    tracker = UsageTracker(engine, settings)
    return Services(
        embedder=CohereEmbedder(settings, tracker),
        reranker=CohereReranker(settings, tracker),
        chat=CohereChat(settings, tracker),
        fetcher=PaperFetcher(settings),
    )
