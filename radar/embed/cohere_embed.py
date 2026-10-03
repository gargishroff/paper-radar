"""Cohere Embed 4 behind the `Embedder` interface."""

from collections.abc import Sequence

import cohere

from radar.cohere_calls import NO_SDK_RETRIES, RateLimiter, guarded_call
from radar.config import EMBED_DIM, Settings
from radar.embed.base import InputType
from radar.usage import UsageTracker


class CohereEmbedder:
    def __init__(
        self,
        settings: Settings,
        tracker: UsageTracker,
        client: cohere.ClientV2 | None = None,
    ):
        if client is None and not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self._client = client or cohere.ClientV2(api_key=settings.cohere_api_key)
        self._tracker = tracker
        self._limiter = RateLimiter(
            settings.embed_calls_per_minute, settings.embed_inputs_per_minute
        )
        self.model = settings.embed_model
        self.max_batch = settings.embed_batch_size

    def embed(
        self, texts: Sequence[str], input_type: InputType, essential: bool = False
    ) -> list[list[float]]:
        if len(texts) > self.max_batch:
            raise ValueError(f"batch of {len(texts)} exceeds max {self.max_batch}")
        response = guarded_call(
            lambda: self._client.embed(
                model=self.model,
                texts=list(texts),
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=EMBED_DIM,
                request_options=NO_SDK_RETRIES,
            ),
            tracker=self._tracker,
            limiter=self._limiter,
            endpoint="embed",
            model=self.model,
            essential=essential,
            billed_units=lambda r: int(
                (r.meta and r.meta.billed_units and r.meta.billed_units.input_tokens) or 0
            ),
            n_inputs=len(texts),
        )
        return [list(v) for v in response.embeddings.float_]
