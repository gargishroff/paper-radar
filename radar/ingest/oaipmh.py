"""Harvest arXiv metadata over OAI-PMH (https://oaipmh.arxiv.org/oai)."""

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date
from email.utils import parsedate_to_datetime

import httpx

from radar.config import Settings
from radar.ingest.records import PaperRecord, clean_text, split_authors

log = logging.getLogger(__name__)

NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}

# Older v1 submissions are revisions or late metadata edits, not new papers.
NEW_SUBMISSION_MAX_AGE_DAYS = 7


class OAIError(RuntimeError):
    pass


def parse_records(xml_text: str) -> tuple[list[PaperRecord], str | None]:
    """Parse one ListRecords page into records plus the resumption token (None when done)."""
    root = ET.fromstring(xml_text)

    error = root.find("oai:error", NS)
    if error is not None:
        if error.get("code") in ("noRecordsMatch", "idDoesNotExist"):
            return [], None
        raise OAIError(f"{error.get('code')}: {error.text}")

    records = []
    # ListRecords pages and single GetRecord responses both wrap <record> elements.
    for rec in root.iterfind(".//oai:record", NS):
        header = rec.find("oai:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue
        meta = rec.find("oai:metadata/raw:arXivRaw", NS)
        if meta is None:
            continue
        records.append(_parse_raw(meta))

    token_el = root.find("oai:ListRecords/oai:resumptionToken", NS)
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return records, token


def _parse_raw(meta: ET.Element) -> PaperRecord:
    def text(tag: str) -> str | None:
        el = meta.find(f"raw:{tag}", NS)
        return el.text if el is not None else None

    versions = sorted(
        (
            int(v.get("version", "v1").lstrip("v")),
            parsedate_to_datetime(v.findtext("raw:date", namespaces=NS)),
        )
        for v in meta.iterfind("raw:version", NS)
    )
    if not versions:
        raise OAIError(f"record {text('id')} has no version history")

    return PaperRecord(
        arxiv_id=clean_text(text("id")),
        title=clean_text(text("title")),
        abstract=clean_text(text("abstract")),
        authors=split_authors(text("authors") or ""),
        categories=clean_text(text("categories")).split(),
        published_at=versions[0][1],
        version_at=versions[-1][1],
        latest_version=versions[-1][0],
        comments=clean_text(text("comments")) or None,
        license=clean_text(text("license")) or None,
    )


def is_new_submission(record: PaperRecord, datestamp: date) -> bool:
    """True if the record is a first version submitted shortly before `datestamp`."""
    age_days = (datestamp - record.published_at.date()).days
    return record.latest_version == 1 and 0 <= age_days <= NEW_SUBMISSION_MAX_AGE_DAYS


class OAIClient:
    """A polite OAI-PMH client: fixed delay between requests, honours 503 Retry-After."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None, max_retries: int = 5):
        agent = "PaperRadar/0.1"
        if settings.arxiv_contact_email:
            agent += f" (mailto:{settings.arxiv_contact_email})"
        self._http = http or httpx.Client(timeout=120, headers={"User-Agent": agent})
        self._base_url = settings.oai_base_url
        self._delay = settings.oai_delay_seconds
        self._max_retries = max_retries
        self._last_request = 0.0

    def list_records(self, set_spec: str, day: date) -> Iterator[PaperRecord]:
        """All records in `set_spec` whose OAI datestamp is `day`."""
        params: dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": "arXivRaw",
            "set": set_spec,
            "from": day.isoformat(),
            "until": day.isoformat(),
        }
        while True:
            records, token = parse_records(self._get(params))
            yield from records
            if not token:
                return
            params = {"verb": "ListRecords", "resumptionToken": token}

    def get_record(self, arxiv_id: str) -> PaperRecord | None:
        """One paper by arXiv ID, or None if arXiv does not know it."""
        records, _ = parse_records(
            self._get(
                {
                    "verb": "GetRecord",
                    "metadataPrefix": "arXivRaw",
                    "identifier": f"oai:arXiv.org:{arxiv_id}",
                }
            )
        )
        return records[0] if records else None

    def _get(self, params: dict[str, str]) -> str:
        for attempt in range(self._max_retries):
            wait = self._delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

            response = self._http.get(self._base_url, params=params)
            if response.status_code == 503:
                retry_after = int(response.headers.get("Retry-After", "10"))
                log.info("OAI 503, retrying in %ss (attempt %d)", retry_after, attempt + 1)
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.text
        raise OAIError(f"gave up after {self._max_retries} attempts: {params}")


def harvest_day(client: OAIClient, set_specs: list[str], day: date) -> dict[str, PaperRecord]:
    """Union of records across sets for one datestamp, deduplicated by arXiv ID."""
    by_id: dict[str, PaperRecord] = {}
    for set_spec in set_specs:
        n = 0
        for record in client.list_records(set_spec, day):
            by_id[record.arxiv_id] = record
            n += 1
        log.info("set %s: %d records for %s", set_spec, n, day)
    return by_id
