"""Rerank 4 and Command A behind small interfaces, so the feed pipeline can run with fakes."""

import json
from collections.abc import Sequence
from typing import Protocol

import cohere

from radar.cohere_calls import NO_SDK_RETRIES, RateLimiter, guarded_call
from radar.config import Settings
from radar.usage import UsageTracker


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: Sequence[str], essential: bool = False) -> list[float]:
        """Relevance score in [0, 1] for each document, in input order."""
        ...


class ChatModel(Protocol):
    model: str

    def chat_json(self, system: str, user: str, schema: dict, essential: bool = False) -> dict:
        """One chat turn whose reply is a JSON object matching `schema`."""
        ...


def _client(settings: Settings, client: cohere.ClientV2 | None) -> cohere.ClientV2:
    if client is None and not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is not set")
    return client or cohere.ClientV2(api_key=settings.cohere_api_key)


class CohereReranker:
    def __init__(
        self, settings: Settings, tracker: UsageTracker, client: cohere.ClientV2 | None = None
    ):
        self._client = _client(settings, client)
        self._tracker = tracker
        self._limiter = RateLimiter(settings.rerank_calls_per_minute)
        self.model = settings.rerank_model

    def rerank(self, query: str, documents: Sequence[str], essential: bool = False) -> list[float]:
        if not documents:
            return []
        response = guarded_call(
            lambda: self._client.rerank(
                model=self.model,
                query=query,
                documents=list(documents),
                top_n=len(documents),
                request_options=NO_SDK_RETRIES,
            ),
            tracker=self._tracker,
            limiter=self._limiter,
            endpoint="rerank",
            model=self.model,
            essential=essential,
            billed_units=lambda r: int(
                (r.meta and r.meta.billed_units and r.meta.billed_units.search_units) or 0
            ),
        )
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


class CohereChat:
    def __init__(
        self, settings: Settings, tracker: UsageTracker, client: cohere.ClientV2 | None = None
    ):
        self._client = _client(settings, client)
        self._tracker = tracker
        self._limiter = RateLimiter(settings.chat_calls_per_minute)
        self.model = settings.chat_model

    def chat_json(self, system: str, user: str, schema: dict, essential: bool = False) -> dict:
        response = guarded_call(
            lambda: self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object", "json_schema": schema},
                temperature=0.2,
                request_options=NO_SDK_RETRIES,
            ),
            tracker=self._tracker,
            limiter=self._limiter,
            endpoint="chat",
            model=self.model,
            essential=essential,
            billed_units=lambda r: int(
                (r.usage and r.usage.billed_units and r.usage.billed_units.output_tokens) or 0
            ),
        )
        # Reasoning models may put a "thinking" item before the answer; take the text item.
        text = next(c.text for c in response.message.content if getattr(c, "type", "") == "text")
        return json.loads(text)
