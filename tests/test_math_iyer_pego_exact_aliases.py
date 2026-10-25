"""Source-pinned Iyer/Pego aliases join only six reviewed research works."""

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
FIXTURE = ROOT / "tests/fixtures/math_iyer_pego_exact_rows.json"
REVIEWED_IDS = {
    "neJ3NIUAAAAJ:ML0RJ9NH7IQC",
    "neJ3NIUAAAAJ:J-pR_7NvFogC",
    "neJ3NIUAAAAJ:LPZeul_q3PIC",
    "neJ3NIUAAAAJ:2osOgNQ5qMEC",
    "neJ3NIUAAAAJ:EUQCXRtRnyEC",
    "neJ3NIUAAAAJ:qjMakFHDy7sC",
}


def _fixture_frame():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_snapshot_sha256"] == (
        "9bf9570577618313c333155836187e800c55e50526950d98650decc08321e364"
    )
    return pandas.DataFrame(fixture["rows"])


def _iyer_aliases():
    all_aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    selected = {
        key: value
        for key, value in all_aliases.items()
        if value.target_observation is not None
    }
    assert {key[1] for key in selected} == REVIEWED_IDS
    assert not any(
        key[1].endswith(suffix)
        for key in selected
        for suffix in (":Tiz5es2fbqcC", ":VL0QpB8kHFEC", ":NJ774b8OgUMC")
    )
    return selected


def test_six_source_pinned_aliases_join_only_exact_iyer_pego_pairs() -> None:
    raw = _fixture_frame()
    aliases = _iyer_aliases()
    validate_doi_aliases(raw, aliases, {})
    registry = load_registry()
    before = build_dataset_tables(raw, registry)
    after = build_dataset_tables(raw, registry, doi_aliases=aliases)
    assert (
        len(before["profile_publications"]) == len(after["profile_publications"]) == 12
    )
    assert len(before["works"]) == 12
    assert len(after["works"]) == 6
    assert len(after["authorships"]) == 12
    retained = after["profile_publications"]
    for _, row in raw.iterrows():
        normalized = retained.loc[
            retained["author_pub_id"] == row["author_pub_id"]
        ].iloc[0]
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
            assert normalized[field] == row[field]
    for key, alias in aliases.items():
        target = alias.target_observation
        assert target is not None
        scholar = retained.loc[retained["author_pub_id"] == key[1]].iloc[0]
        author = retained.loc[retained["author_pub_id"] == target.author_pub_id].iloc[0]
        assert scholar["work_id"] == author["work_id"]
        assert scholar["work_match_method"] == "reviewed_version_alias"
        assert author["work_match_method"] == "doi"
        assert scholar["doi"] == author["doi"] == alias.doi
        assert author["observation_id"] == target.observation_id
        work = (
            after["works"].loc[after["works"]["work_id"] == scholar["work_id"]].iloc[0]
        )
        assert work["observation_count"] == 2
        assert set(work["person_ids"]) == {
            "person-gautamiyer-3894621b",
            "person-robertpego-707a4eaf",
        }
        assert work["citation_count"] == scholar["citation_count"]
        if key[1].endswith(":2osOgNQ5qMEC"):
            assert scholar["year"] == 2013
            assert work["year"] == 2015
            assert work["venue"] == "Commun. Math. Sci. (2015)."
        if key[1].endswith(":EUQCXRtRnyEC"):
            assert scholar["venue"] == ""
            assert work["venue"] == "Ann. Appl. Probab. (2015)."


@pytest.mark.parametrize(
    ("row_kind", "field", "replacement", "message"),
    [
        ("scholar", "source_record_json", "{}", "source record changed"),
        ("target", "person_id", "person-other", "one exact target row"),
        ("target", "title", "Different paper", "target identity changed"),
        ("target", "year", 2000, "target identity changed"),
        ("target", "source_doi", "10.1000/wrong", "target identity changed"),
        ("target", "source_record_json", "{}", "target source record changed"),
    ],
)
def test_pinned_alias_rejects_changed_source(
    row_kind, field, replacement, message
) -> None:
    aliases = _iyer_aliases()
    key = ("neJ3NIUAAAAJ", "neJ3NIUAAAAJ:ML0RJ9NH7IQC")
    alias = aliases[key]
    target = alias.target_observation
    assert target is not None
    raw = _fixture_frame()
    pair = raw.loc[raw["author_pub_id"].isin((key[1], target.author_pub_id))].copy()
    identity = key[1] if row_kind == "scholar" else target.author_pub_id
    pair.loc[pair["author_pub_id"] == identity, field] = replacement
    with pytest.raises(ValueError, match=message):
        validate_doi_aliases(pair, {key: alias}, {})


def test_pinned_alias_rejects_missing_or_duplicate_target() -> None:
    aliases = _iyer_aliases()
    key = ("neJ3NIUAAAAJ", "neJ3NIUAAAAJ:ML0RJ9NH7IQC")
    alias = aliases[key]
    target = alias.target_observation
    assert target is not None
    raw = _fixture_frame()
    pair = raw.loc[raw["author_pub_id"].isin((key[1], target.author_pub_id))]
    scholar_only = pair.loc[pair["source_type"] == "scholar"]
    target_only = pair.loc[pair["source_type"] == "author_site"]
    for rows in (scholar_only, pandas.concat((pair, target_only))):
        with pytest.raises(ValueError, match="one exact target row"):
            validate_doi_aliases(rows, {key: alias}, {})


def test_pinned_alias_rejects_changed_capture(tmp_path) -> None:
    aliases = _iyer_aliases()
    key = ("neJ3NIUAAAAJ", "neJ3NIUAAAAJ:ML0RJ9NH7IQC")
    alias = aliases[key]
    target = alias.target_observation
    assert target is not None
    capture = tmp_path / target.capture_path.name
    capture.write_text('{"changed":true}', encoding="utf-8")
    changed = replace(alias, target_observation=replace(target, capture_path=capture))
    raw = _fixture_frame()
    pair = raw.loc[raw["author_pub_id"].isin((key[1], target.author_pub_id))]
    with pytest.raises(ValueError, match="target capture changed"):
        validate_doi_aliases(pair, {key: changed}, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_source_record_sha256", "bad"),
        ("target_observation", None),
        ("target_observation", "invalid"),
        ("target_observation.person_id", "person-other"),
        ("target_observation.author_pub_id", "author-site:person-other:wrong"),
        ("target_observation.observation_id", "observation-wrong"),
        ("target_observation.observed_title", ""),
        ("target_observation.observed_year", "2018"),
        ("target_observation.source_doi", "10.1000/wrong"),
        ("target_observation.source_record_sha256", "bad"),
        ("target_observation.capture_path", "../wrong.json"),
        ("target_observation.capture_sha256", "bad"),
    ],
)
def test_pinned_alias_loader_rejects_invalid_target(tmp_path, field, value) -> None:
    source = json.loads(DEFAULT_DOI_ALIASES_PATH.read_text(encoding="utf-8"))
    entry = next(
        item for item in source["aliases"] if item["author_pub_id"] in REVIEWED_IDS
    )
    source["aliases"] = [copy.deepcopy(entry)]
    if field.startswith("target_observation."):
        source["aliases"][0]["target_observation"][field.split(".")[1]] = value
    else:
        source["aliases"][0][field] = value
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Scholar|target"):
        load_doi_aliases(path)


def test_pinned_raw_record_hashes_match_exact_fixture() -> None:
    raw = _fixture_frame().set_index("author_pub_id")
    for key, alias in _iyer_aliases().items():
        target = alias.target_observation
        assert target is not None
        scholar_hash = hashlib.sha256(
            raw.loc[key[1], "source_record_json"].encode("utf-8")
        ).hexdigest()
        target_hash = hashlib.sha256(
            raw.loc[target.author_pub_id, "source_record_json"].encode("utf-8")
        ).hexdigest()
        assert scholar_hash == alias.observed_source_record_sha256
        assert target_hash == target.source_record_sha256
