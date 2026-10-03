from datetime import date

import httpx
import pytest

from radar.ingest.oaipmh import (
    OAIClient,
    OAIError,
    harvest_day,
    is_new_submission,
    parse_records,
)
from tests.conftest import FIXTURES, oai_transport

DAY = date(2026, 9, 24)


def test_parse_revision_of_old_paper():
    records, token = parse_records((FIXTURES / "oai_page1.xml").read_text())
    assert token == "tok123"
    [rec] = records
    assert rec.arxiv_id == "2009.13987"
    assert rec.title == "Random Polytope Descriptors"
    assert rec.authors == ["Michael Joswig", "Marek Kaluba", "Lukas Ruff"]
    assert rec.categories == ["cs.LG", "math.MG", "stat.ML"]
    assert rec.primary_cat == "cs.LG"
    assert rec.latest_version == 3
    assert rec.published_at.date() == date(2020, 9, 29)
    assert rec.version_at.date() == date(2026, 9, 23)
    assert "\n" not in rec.abstract and "  " not in rec.abstract
    assert not is_new_submission(rec, DAY)


def test_parse_last_page_and_new_submission_detection():
    records, token = parse_records((FIXTURES / "oai_page2.xml").read_text())
    assert token is None  # an empty resumptionToken marks the last page
    by_id = {r.arxiv_id: r for r in records}

    fresh = by_id["2609.21583"]
    assert fresh.authors == ["Aidin Attar", "Michele Rossi"]
    assert fresh.primary_cat == "cs.NE"  # cross-listed into cs.LG
    assert is_new_submission(fresh, DAY)

    # v1-only papers from earlier months whose metadata was edited: not new.
    assert not is_new_submission(by_id["2602.18262"], DAY)
    assert not is_new_submission(by_id["2607.08406"], DAY)


def test_no_records_match_is_empty_not_error():
    xml = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
    <error code="noRecordsMatch">nothing</error></OAI-PMH>"""
    assert parse_records(xml) == ([], None)


def test_other_oai_errors_raise():
    xml = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
    <error code="badArgument">bad</error></OAI-PMH>"""
    with pytest.raises(OAIError, match="badArgument"):
        parse_records(xml)


def test_client_follows_resumption_token(settings):
    seen: list[httpx.Request] = []
    client = OAIClient(settings, http=httpx.Client(transport=oai_transport(seen)))
    records = list(client.list_records("cs:cs:LG", DAY))

    assert [r.arxiv_id for r in records] == ["2009.13987", "2602.18262", "2607.08406", "2609.21583"]
    first, second = (dict(r.url.params) for r in seen)
    assert first["set"] == "cs:cs:LG" and first["from"] == first["until"] == "2026-09-24"
    # Per the OAI-PMH spec, a resumption request carries only the verb and the token.
    assert second == {"verb": "ListRecords", "resumptionToken": "tok123"}


def test_client_retries_503_with_retry_after(settings):
    calls = {"n": 0}
    page = (FIXTURES / "oai_page2.xml").read_text()

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200, text=page)

    client = OAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(list(client.list_records("cs:cs:LG", DAY))) == 3
    assert calls["n"] == 2


def test_harvest_day_dedupes_across_sets(settings):
    client = OAIClient(settings, http=httpx.Client(transport=oai_transport()))
    records = harvest_day(client, ["cs:cs:LG", "stat:stat:ML"], DAY)
    assert len(records) == 4


def test_fetcher_falls_back_to_oai_when_export_api_fails(settings, monkeypatch):
    from radar.ingest.by_id import PaperFetcher

    monkeypatch.setattr("radar.ingest.by_id.time.sleep", lambda s: None)
    export_calls = []

    def export(request):
        export_calls.append(request)
        return httpx.Response(406)

    getrecord = (FIXTURES / "oai_page1.xml").read_text()
    oai = OAIClient(
        settings,
        http=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text=getrecord))
        ),
    )
    fetcher = PaperFetcher(
        settings, http=httpx.Client(transport=httpx.MockTransport(export)), oai=oai
    )
    # A few papers go straight to OAI.
    assert [r.arxiv_id for r in fetcher.fetch_arxiv(["2009.13987"])] == ["2009.13987"]
    assert export_calls == []
    # A larger batch tries the export API first and falls back after 3 failed attempts.
    records = fetcher.fetch_arxiv([f"2009.{i:05d}" for i in range(11)])
    assert len(records) == 11
    assert len(export_calls) == 3
