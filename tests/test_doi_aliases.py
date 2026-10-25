import json
from pathlib import Path

import pandas
import pytest

from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)


def _source():
    return json.loads(DEFAULT_DOI_ALIASES_PATH.read_text(encoding="utf-8"))


def test_walkington_alias_matches_exact_raw_scholar_row() -> None:
    reviewed = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    key = ("CGSMDGwAAAAJ", "CGSMDGwAAAAJ:SP6oXDckpogC")
    aliases = {key: reviewed[key]}
    alias = aliases[key]
    assert alias.doi == "10.4230/lipics.approx-random.2019.8"
    frame = pandas.DataFrame(
        [
            {
                "source_type": "scholar",
                "scholar_id": key[0],
                "author_pub_id": key[1],
                "title": alias.observed_title,
                "year": alias.observed_year,
                "source_doi": "",
            }
        ]
    )
    validate_doi_aliases(frame, aliases, {})


def test_trusted_upload_uses_producer_doi_alias_manifest() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    assert (
        "--reviewed-doi-aliases source/data/curated/scholar-doi-aliases.json"
        in workflow
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_type": "curated_verified"}, "must match one raw row"),
        ({"title": "Different paper"}, "observed title or year changed"),
        ({"year": 2019}, "observed title or year changed"),
        ({"source_doi": "10.1000/conflict"}, "conflicts with a source DOI"),
    ],
)
def test_alias_blocks_stale_or_conflicting_source(change, message) -> None:
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    (scholar_id, author_pub_id), alias = next(iter(aliases.items()))
    row = {
        "source_type": "scholar",
        "scholar_id": scholar_id,
        "author_pub_id": author_pub_id,
        "title": alias.observed_title,
        "year": alias.observed_year,
        "source_doi": "",
        **change,
    }
    with pytest.raises(ValueError, match=message):
        validate_doi_aliases(pandas.DataFrame([row]), aliases, {})


def test_alias_cannot_restore_reviewed_exclusion() -> None:
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    key, alias = next(iter(aliases.items()))
    frame = pandas.DataFrame(
        [
            {
                "source_type": "scholar",
                "scholar_id": key[0],
                "author_pub_id": key[1],
                "title": alias.observed_title,
                "year": alias.observed_year,
                "source_doi": "",
            }
        ]
    )
    with pytest.raises(ValueError, match="reviewed exclusion"):
        validate_doi_aliases(frame, aliases, {key: object()})


def test_alias_with_missing_observed_year_requires_missing_raw_year(tmp_path) -> None:
    source = _source()
    source["aliases"] = [{**source["aliases"][0], "observed_year": None}]
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    aliases = load_doi_aliases(path)
    (scholar_id, author_pub_id), alias = next(iter(aliases.items()))
    row = {
        "source_type": "scholar",
        "scholar_id": scholar_id,
        "author_pub_id": author_pub_id,
        "title": alias.observed_title,
        "year": pandas.NA,
        "source_doi": "",
    }
    validate_doi_aliases(pandas.DataFrame([row]), aliases, {})
    with pytest.raises(ValueError, match="observed title or year changed"):
        validate_doi_aliases(pandas.DataFrame([{**row, "year": 2025}]), aliases, {})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"scholar_id": "other"}, "invalid observation identity"),
        ({"published_doi": "not-a-doi"}, "valid published DOI"),
        ({"preprint_doi": "10.48550/arXiv.1812.02841"}, "preprint-only fields"),
        ({"observed_title": ""}, "exact observed title and year"),
        ({"observed_year": True}, "exact observed title and year"),
        ({"observed_year": ""}, "exact observed title and year"),
        ({"relationship": "same_title"}, "supported DOI relationship"),
        ({"evidence_urls": ["https://arxiv.org/abs/1812.02841"]}, "distinct HTTPS"),
        (
            {"evidence_urls": ["http://example.org", "https://example.org"]},
            "distinct HTTPS",
        ),
        ({"verified_at": "yesterday"}, "YYYY-MM-DD"),
    ],
)
def test_loader_rejects_invalid_alias_fields(tmp_path, change, message) -> None:
    source = _source()
    source["aliases"][0].update(change)
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_doi_aliases(path)


def test_loader_rejects_structure_and_duplicate(tmp_path) -> None:
    source = _source()
    path = tmp_path / "aliases.json"
    for invalid, message in (
        ({"schema_version": 2, "aliases": []}, "Unsupported"),
        ({"schema_version": 1, "aliases": {}}, "must be a list"),
        ({"schema_version": 1, "aliases": [None]}, "must be an object"),
        ({"schema_version": 1, "aliases": source["aliases"] * 2}, "Duplicate"),
    ):
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_doi_aliases(path)


@pytest.mark.parametrize(
    ("scholar_id", "author_pub_id", "doi"),
    [
        (
            "8AqWr_4AAAAJ",
            "8AqWr_4AAAAJ:XvxMoLDsR5gC",
            "10.48550/arxiv.2112.04569",
        ),
        (
            "JbSQsXAAAAAJ",
            "JbSQsXAAAAAJ:2osOgNQ5qMEC",
            "10.48550/arxiv.1804.08406",
        ),
    ],
)
def test_physics_arxiv_preprint_doi_matches_exact_saved_row(
    scholar_id, author_pub_id, doi
) -> None:
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    key = (scholar_id, author_pub_id)
    alias = aliases[key]
    assert alias.relationship == "arxiv_preprint_doi"
    assert alias.doi == doi
    assert alias.observed_venue is not None
    row = {
        "source_type": "scholar",
        "scholar_id": scholar_id,
        "author_pub_id": author_pub_id,
        "title": alias.observed_title,
        "year": alias.observed_year,
        "venue": alias.observed_venue,
        "source_doi": "",
    }
    validate_doi_aliases(pandas.DataFrame([row]), {key: alias}, {})
    with pytest.raises(ValueError, match="observed venue changed"):
        validate_doi_aliases(
            pandas.DataFrame([{**row, "venue": "Different arXiv record"}]),
            {key: alias},
            {},
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"preprint_doi": "10.22323/1.396.0098"}, "valid arXiv preprint DOI"),
        ({"preprint_doi": "10.48550/arXiv.1804.08406"}, "exact observed venue"),
        ({"published_doi": "10.22323/1.396.0098"}, "must not be a published-work DOI"),
        ({"observed_venue": "Proceedings of Science"}, "exact observed venue"),
        (
            {
                "evidence_urls": [
                    "https://arxiv.org/abs/1804.08406",
                    "https://www.cmu.edu/physics/",
                ]
            },
            "matching arXiv record",
        ),
    ],
)
def test_arxiv_preprint_doi_rejects_wrong_work_or_version(
    tmp_path, change, message
) -> None:
    source = _source()
    source["aliases"] = [
        next(
            alias
            for alias in source["aliases"]
            if alias["author_pub_id"] == "8AqWr_4AAAAJ:XvxMoLDsR5gC"
        )
    ]
    source["aliases"][0].update(change)
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_doi_aliases(path)
