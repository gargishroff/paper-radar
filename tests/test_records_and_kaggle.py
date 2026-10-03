import json
from datetime import date

from radar.config import Settings
from radar.ingest.kaggle_backfill import iter_snapshot, parse_line
from radar.ingest.records import PaperRecord, clean_text, split_authors


def test_split_authors_handles_and_and_affiliations():
    assert split_authors("A. Smith (MIT), B. Jones and\n  C. Lee") == [
        "A. Smith",
        "B. Jones",
        "C. Lee",
    ]
    assert split_authors("Solo Author") == ["Solo Author"]
    assert split_authors("X Y, et al.") == ["X Y"]


def test_clean_text():
    assert clean_text("  a\n  b\t c ") == "a b c"
    assert clean_text(None) == ""


def _record(abstract: str, comments: str | None = None) -> PaperRecord:
    t = parse_line(_kaggle_line()).published_at
    return PaperRecord("1", "t", abstract, [], ["cs.LG"], t, t, 1, comments, None)


def test_has_code_checks_abstract_and_comments():
    assert _record("Code at https://github.com/foo/bar.").has_code
    assert _record("No code.", "Data: huggingface.co/datasets/x").has_code
    assert not _record("No code here.").has_code


def test_category_names_from_oai_sets():
    s = Settings(_env_file=None, oai_sets=["cs:cs:LG", "stat:stat:ML"])
    assert s.categories == {"cs.LG", "stat.ML"}


def _kaggle_line(**overrides) -> str:
    obj = {
        "id": "2401.00001",
        "authors": "Ann Lee and Bo Chen",
        "title": "A  Title\n  Over Lines",
        "comments": "12 pages",
        "categories": "cs.CL cs.LG",
        "license": None,
        "abstract": "  An abstract.\n",
        "versions": [
            {"version": "v2", "created": "Tue, 9 Jan 2024 10:00:00 GMT"},
            {"version": "v1", "created": "Mon, 1 Jan 2024 09:00:00 GMT"},
        ],
        "authors_parsed": [["Lee", "Ann", ""], ["Chen", "Bo", "Jr"]],
    }
    obj.update(overrides)
    return json.dumps(obj)


def test_parse_kaggle_line():
    rec = parse_line(_kaggle_line())
    assert rec.title == "A Title Over Lines"
    assert rec.abstract == "An abstract."
    assert rec.authors == ["Ann Lee", "Bo Chen Jr"]
    assert rec.primary_cat == "cs.CL"
    assert rec.published_at.date() == date(2024, 1, 1)  # v1, even though listed second
    assert rec.latest_version == 2


def test_iter_snapshot_filters_category_and_window(tmp_path):
    lines = [
        _kaggle_line(),
        _kaggle_line(id="2401.00002", categories="math.AG"),  # wrong category
        _kaggle_line(
            id="2301.00003",
            versions=[{"version": "v1", "created": "Mon, 2 Jan 2023 09:00:00 GMT"}],
        ),  # outside window
    ]
    path = tmp_path / "snap.json"
    path.write_text("\n".join(lines) + "\n")
    got = list(iter_snapshot(path, {"cs.LG"}, date(2023, 12, 1), date(2024, 1, 31)))
    assert [r.arxiv_id for r in got] == ["2401.00001"]
