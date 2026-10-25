"""The whole Neumayer bibliography is atomic and cannot enter ordinary upload."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research.neumayer_source import load_neumayer_candidate_rows
from map_of_research.registry import load_registry
from map_of_research.snapshot import build_snapshot, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_reviews/robin-neumayer-2026-10-25"


def _registry():
    return load_registry(
        ROOT / "registry/people.csv",
        ROOT / "registry/memberships.csv",
        ROOT / "registry/departments.csv",
    )


def test_neumayer_candidate_retains_entire_bibliography_and_source_types() -> None:
    rows = load_neumayer_candidate_rows(SOURCE, _registry())
    assert len(rows) == len({row["author_pub_id"] for row in rows}) == 31
    assert sum(row["source_type"] == "author_site" for row in rows) == 30
    assert sum(row["source_type"] == "researcher_cv" for row in rows) == 1
    assert sum(bool(row["source_doi"]) for row in rows) == 18
    assert sum(row["year"] is None for row in rows) == 2
    assert all(row["scholar_id"] == row["fetched_at_utc"] == "" for row in rows)
    assert all(row["source_review_state"] == "source_only_release_held" for row in rows)
    assert all(not row["source_collaboration_eligible"] for row in rows)
    assert all(not row["source_citation_count_known"] for row in rows)
    assert rows[29]["year"] is None
    assert rows[27]["year"] is None
    assert rows[30]["source_type"] == "researcher_cv"
    source_record = json.loads(rows[30]["source_record_json"])
    assert source_record["source_kind"] == "researcher_cv"
    assert source_record["release_state"] == "source_only_release_held"
    assert "year_basis" in source_record
    assert "row_hold" in source_record


def test_neumayer_candidate_snapshot_validates_only_with_explicit_source(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "candidate.parquet"
    manifest = tmp_path / "candidate.manifest.json"
    built = build_snapshot(
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        cache_dir=tmp_path / "empty-cache",
        neumayer_source_path=SOURCE,
        snapshot_path=snapshot,
        manifest_path=manifest,
    )
    assert built["snapshot_schema_version"] == 7
    assert built["neumayer_candidate"]["publisher_allowed"] is False
    frame, _ = validate_snapshot(
        snapshot,
        manifest,
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        neumayer_source_path=SOURCE,
    )
    bibliography = frame.loc[
        frame["source_type"].isin({"author_site", "researcher_cv"})
    ]
    assert len(bibliography) == 31
    assert bibliography["year"].isna().sum() == 2
    assert bibliography["person_id"].nunique() == 1
    with pytest.raises(ValueError, match="schema is unsupported"):
        validate_snapshot(
            snapshot,
            manifest,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
        )
    modified = json.loads(manifest.read_text())
    modified["neumayer_candidate"]["publisher_allowed"] = True
    manifest.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="release hold changed"):
        validate_snapshot(
            snapshot,
            manifest,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
            neumayer_source_path=SOURCE,
        )
    changed = pandas.read_parquet(snapshot)
    cv_index = changed.index[changed["source_type"] == "researcher_cv"][0]
    changed.at[cv_index, "title"] = "Incorrect replacement title"
    changed.to_parquet(snapshot, index=False)
    rebuilt_manifest = dict(built)
    rebuilt_manifest["snapshot_sha256"] = hashlib.sha256(
        snapshot.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(rebuilt_manifest))
    with pytest.raises(ValueError, match="differs from pinned source rows"):
        validate_snapshot(
            snapshot,
            manifest,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
            neumayer_source_path=SOURCE,
        )


def test_neumayer_candidate_rejects_partial_batch(tmp_path: Path) -> None:
    rows = load_neumayer_candidate_rows(SOURCE, _registry())
    frame = pandas.DataFrame(rows[:-1])
    from map_of_research.snapshot import _validate_frame

    with pytest.raises(ValueError, match="whole-person candidate"):
        _validate_frame(frame, allow_neumayer_candidate=True)


def test_neumayer_candidate_rejects_substituted_source_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "candidate.parquet"
    manifest = tmp_path / "candidate.manifest.json"
    built = build_snapshot(
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        cache_dir=tmp_path / "empty-cache",
        neumayer_source_path=SOURCE,
        snapshot_path=snapshot,
        manifest_path=manifest,
    )
    frame = pandas.read_parquet(snapshot)
    index = frame.index[frame["source_type"] == "researcher_cv"][0]
    frame.at[index, "author_pub_id"] = "substituted-source-id"
    frame.to_parquet(snapshot, index=False)
    built["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(built))
    with pytest.raises(ValueError, match="differs from pinned source rows"):
        validate_snapshot(
            snapshot,
            manifest,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
            neumayer_source_path=SOURCE,
        )


def test_neumayer_candidate_rejects_wrong_registry_identity() -> None:
    registry = _registry()
    people = tuple(
        replace(person, scholar_id="wrong-id")
        if person.person_id == "person-robinneumayer-a5f54a8c"
        else person
        for person in registry.people
    )
    with pytest.raises(ValueError, match="unresolved Math registry identity"):
        load_neumayer_candidate_rows(SOURCE, replace(registry, people=people))
    memberships = tuple(
        membership
        for membership in registry.memberships
        if membership.person_id != "person-robinneumayer-a5f54a8c"
    )
    with pytest.raises(ValueError, match="evidenced Math membership"):
        load_neumayer_candidate_rows(SOURCE, replace(registry, memberships=memberships))


def test_neumayer_candidate_rejects_missing_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import map_of_research.neumayer_source as adapter

    rows = adapter.runpy.run_path(str(SOURCE / "load_staged.py"))[
        "load_staged_neumayer_rows"
    ](SOURCE)
    monkeypatch.setattr(
        adapter.runpy,
        "run_path",
        lambda _path: {"load_staged_neumayer_rows": lambda _source: rows[:-1]},
    )
    with pytest.raises(ValueError, match="whole-person source coverage changed"):
        load_neumayer_candidate_rows(SOURCE, _registry())


def test_standard_snapshot_rejects_unrequested_candidate_marker(tmp_path: Path) -> None:
    snapshot = tmp_path / "standard.parquet"
    manifest = tmp_path / "standard.manifest.json"
    build_snapshot(
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        cache_dir=tmp_path / "empty-cache",
        snapshot_path=snapshot,
        manifest_path=manifest,
    )
    marked = json.loads(manifest.read_text())
    marked["neumayer_candidate"] = {"publisher_allowed": False}
    manifest.write_text(json.dumps(marked))
    with pytest.raises(ValueError, match="explicit validation input"):
        validate_snapshot(
            snapshot,
            manifest,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
        )
