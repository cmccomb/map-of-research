"""The two Pego preprints and Iyer's journal row are one reviewed work."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)
from map_of_research.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/math_iyer_pego_version_rows.json"
SOURCE_SHA256 = "9bf9570577618313c333155836187e800c55e50526950d98650decc08321e364"
TARGET_ID = "author-site:person-gautamiyer-3894621b:c60cc76f0b3ffff7dc82"
SCHOLAR_IDS = (
    "neJ3NIUAAAAJ:VL0QpB8kHFEC",
    "neJ3NIUAAAAJ:NJ774b8OgUMC",
)
DOI = "10.4310/cms.2020.v18.n7.a2"


def _rows():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_snapshot_sha256"] == SOURCE_SHA256
    return pandas.DataFrame(fixture["rows"])


def _aliases():
    all_aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    selected = {
        key: alias for key, alias in all_aliases.items() if key[1] in SCHOLAR_IDS
    }
    assert {key[1] for key in selected} == set(SCHOLAR_IDS)
    return selected


def test_exact_three_record_version_family_preserves_raw_observations() -> None:
    raw = _rows()
    aliases = _aliases()
    validate_doi_aliases(raw, aliases, {})
    registry = load_registry()
    before = build_dataset_tables(raw, registry)
    after = build_dataset_tables(raw, registry, doi_aliases=aliases)
    assert len(before["works"]) == 3
    assert len(after["works"]) == 1
    assert len(after["profile_publications"]) == len(after["authorships"]) == 3
    retained = after["profile_publications"].set_index("author_pub_id")
    for _, observed in raw.iterrows():
        normalized = retained.loc[observed["author_pub_id"]]
        for field in (
            "source_type",
            "source_record_json",
            "title",
            "year",
            "authors",
            "venue",
            "citation",
            "citation_count",
            "source_url",
            "source_doi",
            "source_native_id",
        ):
            assert normalized[field] == observed[field]
    assert retained["work_id"].nunique() == 1
    assert {retained.loc[key, "year"] for key in SCHOLAR_IDS} == {2018}
    assert retained.loc[SCHOLAR_IDS[0], "citation_count"] == 1
    assert retained.loc[SCHOLAR_IDS[1], "citation_count"] == 0
    for key, alias in aliases.items():
        assert alias.doi == DOI
        assert alias.relationship == "preprint_of"
        assert alias.target_observation is not None
        assert alias.target_observation.author_pub_id == TARGET_ID
        assert retained.loc[key[1], "work_match_method"] == "reviewed_version_alias"
    work = after["works"].iloc[0]
    assert work["title"] == "Anomalous diffusion in comb-shaped domains and graphs"
    assert work["year"] == 2020
    assert work["venue"] == "Communications in Mathematical Sciences"
    assert work["doi"] == DOI
    assert work["observation_count"] == 3
    assert work["citation_count"] == 1
    assert set(work["author_pub_ids"]) == {*SCHOLAR_IDS, TARGET_ID}
    assert set(work["person_ids"]) == {
        "person-gautamiyer-3894621b",
        "person-robertpego-707a4eaf",
    }
    assert work["title_variants"] == [
        "Anomalous diffusion in comb-shaped domains and graphs",
        "Anomalous diffusion in one and two dimensional combs",
        "Anomalous diffusion in one and two dimensional combs,",
    ]
    assert work["year_variants"] == [2018, 2020]


@pytest.mark.parametrize(
    "review_change",
    [
        {"doi": "10.1000/wrong"},
        {"publisher_title": "Another paper"},
        {"publisher_year": 2019},
        {"publisher_registry_url": "https://example.org/wrong"},
    ],
)
def test_published_display_needs_exact_target_review(review_change) -> None:
    raw = _rows()
    key = ("neJ3NIUAAAAJ", SCHOLAR_IDS[0])
    alias = _aliases()[key]
    target = alias.target_observation
    assert target is not None
    target_row = raw.loc[raw["author_pub_id"] == TARGET_ID].iloc[0]
    source_record = json.loads(target_row["source_record_json"])
    source_record["doi_review"].update(review_change)
    changed_record = json.dumps(source_record, sort_keys=True, separators=(",", ":"))
    changed = replace(
        alias,
        target_observation=replace(
            target,
            source_record_sha256=hashlib.sha256(
                changed_record.encode("utf-8")
            ).hexdigest(),
        ),
    )
    raw.loc[raw["author_pub_id"] == TARGET_ID, "source_record_json"] = changed_record
    with pytest.raises(ValueError, match="published display changed"):
        validate_doi_aliases(raw, {key: changed}, {})


@pytest.mark.parametrize(
    "published_display",
    [
        "not-an-object",
        {
            "title": "",
            "year": 2020,
            "venue": "CMS",
            "registry_url": "https://example.org",
        },
        {
            "title": "Paper",
            "year": "2020",
            "venue": "CMS",
            "registry_url": "https://example.org",
        },
        {
            "title": "Paper",
            "year": 2020,
            "venue": "",
            "registry_url": "https://example.org",
        },
        {
            "title": "Paper",
            "year": 2020,
            "venue": "CMS",
            "registry_url": "http://example.org",
        },
        {
            "title": "Paper",
            "year": 2020,
            "venue": "CMS",
            "registry_url": "https://example.org",
        },
    ],
)
def test_loader_rejects_unpinned_published_display(tmp_path, published_display) -> None:
    source = json.loads(DEFAULT_DOI_ALIASES_PATH.read_text(encoding="utf-8"))
    entry = next(
        item for item in source["aliases"] if item["author_pub_id"] == SCHOLAR_IDS[0]
    )
    entry = copy.deepcopy(entry)
    entry["target_observation"]["published_display"] = published_display
    source["aliases"] = [entry]
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="published display"):
        load_doi_aliases(path)


def test_conflicting_displays_or_missing_target_cannot_build_work() -> None:
    raw = _rows()
    aliases = _aliases()
    registry = load_registry()
    key = ("neJ3NIUAAAAJ", SCHOLAR_IDS[1])
    target = aliases[key].target_observation
    assert target is not None and target.published_display is not None
    changed_target = replace(
        target,
        published_display=replace(target.published_display, year=2019),
    )
    conflicting = {
        **aliases,
        key: replace(aliases[key], target_observation=changed_target),
    }
    with pytest.raises(ValueError, match="Conflicting reviewed published displays"):
        build_dataset_tables(raw, registry, doi_aliases=conflicting)
    scholar_only = raw.loc[raw["source_type"] == "scholar"]
    with pytest.raises(ValueError, match="published display target missing"):
        build_dataset_tables(scholar_only, registry, doi_aliases=aliases)
