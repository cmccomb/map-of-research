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
        ({"observed_title": ""}, "exact observed title and year"),
        ({"observed_year": True}, "exact observed title and year"),
        ({"observed_year": ""}, "exact observed title and year"),
        ({"relationship": "same_title"}, "supported version relationship"),
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


def _gordon_alias_entry() -> dict:
    return {
        "scholar_id": "n6flaAsAAAAJ",
        "author_pub_id": "n6flaAsAAAAJ:M3NEmzRMIkIC",
        "observed_title": "Exchange processes",
        "observed_year": 2006,
        "published_doi": "10.1007/1-4020-3500-4_18",
        "relationship": "version_of_record",
        "evidence_urls": [
            "https://link.springer.com/chapter/10.1007/1-4020-3500-4_18",
            "https://link.springer.com/book/10.1007/1-4020-3500-4",
        ],
        "verified_at": "2026-10-25",
        "allow_short_title_region_override": True,
    }


def _gordon_source_row() -> dict:
    return {
        "source_type": "scholar",
        "scholar_id": "n6flaAsAAAAJ",
        "author_pub_id": "n6flaAsAAAAJ:M3NEmzRMIkIC",
        "title": "Exchange processes",
        "year": 2006,
        "authors": "GS Rule, TK Hitchens",
        "venue": "Fundamentals of Protein NMR Spectroscopy, 403-430, 2006",
        "source_doi": "",
        "source_url": (
            "https://scholar.google.com/citations?view_op=view_citation"
            "&user=n6flaAsAAAAJ&citation_for_view=n6flaAsAAAAJ:M3NEmzRMIkIC"
        ),
    }


def test_short_title_override_requires_exact_scholar_citation_and_springer_evidence(
    tmp_path,
) -> None:
    entry = _gordon_alias_entry()
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"schema_version": 1, "aliases": [entry]}))
    aliases = load_doi_aliases(path)
    assert aliases[
        (entry["scholar_id"], entry["author_pub_id"])
    ].allow_short_title_region_override
    validate_doi_aliases(pandas.DataFrame([_gordon_source_row()]), aliases, {})

    for field, value in (
        ("title", "Different chapter"),
        ("year", 2007),
        ("source_url", "https://scholar.google.com/citations?user=n6flaAsAAAAJ"),
        ("authors", ""),
        ("venue", ""),
    ):
        with pytest.raises(ValueError, match="observed title|short-title override"):
            validate_doi_aliases(
                pandas.DataFrame([{**_gordon_source_row(), field: value}]),
                aliases,
                {},
            )

    for field, value in (
        ("allow_short_title_region_override", "true"),
        ("observed_year", None),
        ("observed_title", "A long enough title to pass an ordinary region guard"),
        (
            "evidence_urls",
            [
                "https://example.org/chapter/10.1007/1-4020-3500-4_18",
                "https://link.springer.com/book/10.1007/1-4020-3500-4",
            ],
        ),
        ("published_doi", "10.1007/not-the-chapter"),
    ):
        path.write_text(
            json.dumps({"schema_version": 1, "aliases": [{**entry, field: value}]})
        )
        with pytest.raises(ValueError, match="short-title override"):
            load_doi_aliases(path)
