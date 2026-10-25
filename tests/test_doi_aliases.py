import json
from pathlib import Path

import numpy
import pandas
import pytest

import map_of_research.keywords as keywords
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


def test_murphy_spatial_package_exact_alias_survives_weak_region() -> None:
    key = ("qQLlBH4AAAAJ", "qQLlBH4AAAAJ:RMgMIBzvq-4C")
    alias = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)[key]
    assert alias.doi == "10.1515/jib-2022-0054"
    assert alias.observed_title == (
        "SBML level 3 package: spatial processes, version 1, release 1"
    )
    assert alias.observed_year == 2023
    assert "https://sbml.org/documents/specifications/level-3/version-1/spatial/" in (
        alias.evidence_urls
    )
    raw = pandas.DataFrame(
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
    validate_doi_aliases(raw, {key: alias}, {})

    size = keywords.MINIMUM_AUDIT_REGION_SIZE
    rows = []
    for index in range(size):
        embedding = numpy.zeros(size, dtype=numpy.float32)
        embedding[index] = 1.0
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": embedding.tolist(),
                "year": None,
                "citation_count": 0,
                "title": f"Fragment {index}",
                "authors": "",
                "venue": "",
                "doi": "",
                "source_urls": [],
            }
        )
    rows[0].update(
        {
            "year": alias.observed_year,
            "citation_count": 12,
            "title": alias.observed_title,
            "authors": "JC Schaff, A Lakshminarayana, RF Murphy, FT Bergmann, et al.",
            "venue": "Journal of Integrative Bioinformatics 20 (1), 20220054",
            "doi": alias.doi,
            "source_urls": list(alias.evidence_urls),
        }
    )
    audited = keywords.exclude_low_information_regions(pandas.DataFrame(rows))
    assert bool(audited.loc[0, "map_eligible"])
    assert bool(audited.loc[0, "map_region_documented_override"])
    assert not bool(audited.loc[0, "map_region_outlier"])
    assert audited["map_eligible"].sum() == 1


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
