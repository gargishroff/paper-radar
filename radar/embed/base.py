from collections.abc import Sequence
from typing import Literal, Protocol

InputType = Literal["search_document", "search_query"]


class Embedder(Protocol):
    """Anything that turns texts into vectors. Tests use a fake; the product uses Cohere."""

    model: str
    max_batch: int

    def embed(
        self, texts: Sequence[str], input_type: InputType, essential: bool = False
    ) -> list[list[float]]: ...


def paper_text(title: str, abstract: str) -> str:
    """The text we embed for a paper: title, newline, abstract."""
    return f"{title}\n{abstract}"
