"""Fetches papers by arXiv ID or Semantic Scholar author for onboarding."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime

import httpx
from sqlalchemy import Engine, select

from radar.config import Settings
from radar.db import session_scope
from radar.db.models import Paper
from radar.ingest.oaipmh import OAIClient
from radar.ingest.records import PaperRecord, clean_text, latex_to_unicode
from radar.ingest.store import upsert_papers

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# New-style (2401.01234) and old-style (cs/0112017) identifiers, with optional version suffix.
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


def normalize_arxiv_id(text: str) -> str | None:
    """'https://arxiv.org/abs/2401.01234v2', 'arXiv:2401.01234' -> '2401.01234'."""
    match = _ARXIV_ID.search(text.strip())
    return match.group(1) if match else None


def parse_atom(xml_text: str) -> list[PaperRecord]:
    root = ET.fromstring(xml_text)
    records = []
    for entry in root.iterfind("a:entry", ATOM):
        abs_url = entry.findtext("a:id", default="", namespaces=ATOM)
        match = re.search(r"abs/(.+?)v(\d+)$", abs_url)
        if not match:  # the API returns an error entry for unknown IDs
            continue
        primary = entry.find("arxiv:primary_category", ATOM)
        others = [c.get("term") for c in entry.iterfind("a:category", ATOM)]
        primary_term = primary.get("term") if primary is not None else others[0]
        categories = [primary_term] + [c for c in others if c != primary_term]
        records.append(
            PaperRecord(
                arxiv_id=match.group(1),
                title=clean_text(entry.findtext("a:title", namespaces=ATOM)),
                abstract=clean_text(entry.findtext("a:summary", namespaces=ATOM)),
                authors=[
                    latex_to_unicode(clean_text(a.findtext("a:name", namespaces=ATOM)))
                    for a in entry.iterfind("a:author", ATOM)
                ],
                categories=categories,
                published_at=datetime.fromisoformat(entry.findtext("a:published", namespaces=ATOM)),
                version_at=datetime.fromisoformat(entry.findtext("a:updated", namespaces=ATOM)),
                latest_version=int(match.group(2)),
                comments=clean_text(entry.findtext("arxiv:comment", namespaces=ATOM)) or None,
                license=None,
            )
        )
    return records


class FetchError(RuntimeError):
    pass


class PaperFetcher:
    """Uses arXiv's batch export API, falling back to OAI-PMH when it returns 406s."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        delay: float = 3.0,
        oai: OAIClient | None = None,
    ):
        agent = "PaperRadar/0.1"
        if settings.arxiv_contact_email:
            agent += f" (mailto:{settings.arxiv_contact_email})"
        # arXiv asks API clients to identify themselves.
        self._http = http or httpx.Client(
            timeout=60, follow_redirects=True, headers={"User-Agent": agent}
        )
        self._s2_headers = (
            {"x-api-key": settings.semantic_scholar_api_key}
            if settings.semantic_scholar_api_key
            else {}
        )
        self._delay = delay
        self._oai = oai or OAIClient(settings)

    def fetch_arxiv(self, arxiv_ids: Iterable[str]) -> list[PaperRecord]:
        ids = sorted(set(arxiv_ids))
        records: list[PaperRecord] = []
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            if len(batch) <= 10:
                # A few papers are as quick over OAI as waiting out one failed API request.
                records.extend(r for r in map(self._oai.get_record, batch) if r is not None)
                continue
            try:
                response = self._get(
                    ARXIV_API,
                    {"id_list": ",".join(batch), "max_results": str(len(batch))},
                    attempts=3,
                )
                records.extend(parse_atom(response.text))
                time.sleep(self._delay)
            except FetchError:
                log.warning("arXiv export API unavailable; fetching %d papers over OAI", len(batch))
                records.extend(r for r in map(self._oai.get_record, batch) if r is not None)
        return records

    def author_arxiv_ids(self, s2_author_id: str) -> list[str]:
        """arXiv IDs of every paper by a Semantic Scholar author that is on arXiv."""
        ids: list[str] = []
        offset: int | None = 0
        while offset is not None:
            data = self._get(
                f"{S2_API}/author/{s2_author_id}/papers",
                {"fields": "externalIds", "limit": "1000", "offset": str(offset)},
                headers=self._s2_headers,
            ).json()
            ids += [
                p["externalIds"]["ArXiv"]
                for p in data.get("data", [])
                if (p.get("externalIds") or {}).get("ArXiv")
            ]
            offset = data.get("next")
        return ids

    def _get(
        self, url: str, params: dict, headers: dict | None = None, attempts: int = 6
    ) -> httpx.Response:
        for attempt in range(attempts):
            response = self._http.get(url, params=params, headers=headers)
            # arXiv's API intermittently answers 406 to requests it accepts moments later.
            if response.status_code in (406, 429, 503):
                wait = 5 * 2**attempt
                log.info("%s returned %s; retrying in %ss", url, response.status_code, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        raise FetchError(f"gave up on {url}")


def ensure_papers(engine: Engine, fetcher: PaperFetcher, arxiv_ids: Iterable[str]) -> set[str]:
    """Make sure these papers are in the database; return the IDs that exist afterwards."""
    wanted = {i for i in (normalize_arxiv_id(x) for x in arxiv_ids) if i}
    with session_scope(engine) as session:
        have = set(session.scalars(select(Paper.arxiv_id).where(Paper.arxiv_id.in_(wanted))))
    missing = wanted - have
    if missing:
        records = fetcher.fetch_arxiv(missing)
        with session_scope(engine) as session:
            upsert_papers(session, records, source="arxiv_api")
        have |= {r.arxiv_id for r in records}
        if len(have) < len(wanted):
            log.warning("not found on arXiv: %s", sorted(wanted - have))
    return have
