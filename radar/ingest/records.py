"""The source-independent paper record that every ingester produces."""

import re
from dataclasses import dataclass
from datetime import datetime

from pylatexenc.latex2text import LatexNodes2Text

_WHITESPACE = re.compile(r"\s+")
_CODE_HOSTS = re.compile(
    r"(github\.com|gitlab\.com|huggingface\.co|bitbucket\.org)/", re.IGNORECASE
)


@dataclass(frozen=True)
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]  # primary category first, as arXiv lists it
    published_at: datetime  # v1 date
    version_at: datetime  # latest version date
    latest_version: int
    comments: str | None
    license: str | None

    @property
    def primary_cat(self) -> str:
        return self.categories[0]

    @property
    def has_code(self) -> bool:
        return bool(_CODE_HOSTS.search(self.abstract) or _CODE_HOSTS.search(self.comments or ""))


def clean_text(text: str | None) -> str:
    """arXiv metadata wraps lines and double-spaces; collapse all runs of whitespace."""
    return _WHITESPACE.sub(" ", text or "").strip()


_LATEX = LatexNodes2Text()


def latex_to_unicode(text: str) -> str:
    """arXiv author strings use LaTeX accents: 'Pawe{\\l} M\\k{a}ka' -> 'Paweł Mąka'."""
    if "\\" not in text and "{" not in text:
        return text
    try:
        return clean_text(_LATEX.latex_to_text(text))
    except Exception:  # noqa: BLE001 - malformed LaTeX must never stop ingestion; keep raw text
        return text


def split_authors(authors: str) -> list[str]:
    """Split arXiv's free-text author string ("A, B and C") into names."""
    text = clean_text(re.sub(r"\([^)]*\)", "", authors))
    parts = re.split(r",\s*|\s+and\s+", text)
    return [
        latex_to_unicode(p.strip())
        for p in parts
        if p.strip() and p.strip().lower() not in {"et al.", "et al"}
    ]
