"""The Iyer fallback is a whole author-owned source, not a Scholar profile."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pandas
import pytest

from map_of_research.author_site import IYER_PERSON_ID, load_author_site_rows
from map_of_research.dataset import build_dataset_tables
from map_of_research.registry import load_registry
from map_of_research.snapshot import _validate_frame, build_snapshot, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data/author_sites/gautam-iyer-2026-10-25.json"


def _registry():
    return load_registry(
        ROOT / "registry/people.csv",
        ROOT / "registry/memberships.csv",
        ROOT / "registry/departments.csv",
    )


def test_whole_iyer_capture_retains_raw_and_reviewed_metadata() -> None:
    rows = load_author_site_rows(CAPTURE, _registry())
    assert len(rows) == len({row["author_pub_id"] for row in rows}) == 47
    assert sum(bool(row["source_doi"]) for row in rows) == 20
    assert sum(not bool(row["venue"]) for row in rows) == 7
    assert all(row["source_type"] == "author_site" for row in rows)
    assert all(row["scholar_id"] == row["fetched_at_utc"] == "" for row in rows)
    assert all(row["source_citation_count_known"] is False for row in rows)
    assert all(row["source_native_id"] for row in rows)
    comb = rows[21]
    assert comb["year"] == 2020
    assert comb["source_doi"] == "10.4310/cms.2020.v18.n7.a2"
    raw = json.loads(comb["source_record_json"])
    assert raw["author_site_row"]["year_as_listed"] == "2018"
    assert raw["author_site_row"]["title_as_listed"] == (
        "Anomalous diffusion in one and two dimensional combs,"
    )
    assert rows[43]["source_output_type"] == "thesis"


def test_author_site_doi_mining_and_thesis_edge_are_blocked() -> None:
    registry = _registry()
    rows = load_author_site_rows(CAPTURE, registry)
    for row in rows:
        row["embedding"] = [0.0, 1.0]
    tables = build_dataset_tables(pandas.DataFrame(rows), registry)
    publications = tables["profile_publications"]
    assert len(publications) == len(tables["works"]) == 47
    publisher_link_with_unreviewed_doi = publications.loc[
        publications["author_pub_id"] == rows[9]["author_pub_id"]
    ].iloc[0]
    assert (
        "10.1137/24M168115X" in publisher_link_with_unreviewed_doi["source_record_json"]
    )
    assert publisher_link_with_unreviewed_doi["doi"] == ""
    assert publisher_link_with_unreviewed_doi["work_match_method"] == (
        "source_native_id"
    )
    thesis = publications.loc[
        publications["author_pub_id"] == rows[43]["author_pub_id"]
    ].iloc[0]
    assert thesis["publication_type"] == "thesis"
    assert not thesis["collaboration_eligible"]


def test_author_site_without_reviewed_doi_requires_native_id() -> None:
    registry = _registry()
    rows = load_author_site_rows(CAPTURE, registry)
    rows[0]["source_native_id"] = ""
    with pytest.raises(ValueError, match="lacks its source-native ID"):
        build_dataset_tables(pandas.DataFrame(rows), registry)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_citation_count_known", None, "citation provenance is missing"),
        ("scholar_id", "claimed", "cannot claim Scholar"),
        ("source_citation_count_known", True, "cannot claim Scholar"),
    ],
)
def test_author_site_snapshot_rejects_false_provenance(
    field: str, value: object, message: str
) -> None:
    frame = pandas.DataFrame(load_author_site_rows(CAPTURE, _registry()))
    if value is None:
        frame = frame.drop(columns=[field])
    else:
        frame.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        _validate_frame(frame)


def test_author_site_snapshot_is_explicit_and_source_bound(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "preview.parquet"
    manifest_path = tmp_path / "preview.manifest.json"
    manifest = build_snapshot(
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        cache_dir=tmp_path / "empty-cache",
        author_site_path=CAPTURE,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
    assert manifest["author_site_rows"] == 47
    assert manifest["curated_work_rows"] == 28
    frame, _ = validate_snapshot(
        snapshot_path,
        manifest_path,
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        author_site_path=CAPTURE,
    )
    assert len(frame) == 75
    with pytest.raises(ValueError, match="Author-site capture does not match"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
        )

    # A valid checksum cannot turn a partial profile into an accepted capture.
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_manifest["author_site_rows"] = 46
    manifest_path.write_text(json.dumps(saved_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="author-site row count"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
            author_site_path=CAPTURE,
        )
    frame = frame.drop(index=frame.index[frame["source_type"] == "author_site"][0])
    frame.to_parquet(snapshot_path, index=False, engine="pyarrow")
    saved_manifest["publication_rows"] = 74
    saved_manifest["snapshot_sha256"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(saved_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete Iyer author-site profile"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
            author_site_path=CAPTURE,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("scope", "scope changed"),
        ("html", "capture file changed"),
        ("csv", "capture file changed"),
        ("audit", "capture file changed"),
        ("doi", "DOI source review changed"),
        ("comb", "comb-paper title review is missing"),
        ("url", "Invalid author-site URL"),
        ("time", "capture time lacks timezone"),
        ("selector", "All Publications section changed"),
        ("audit_rows", "bibliography or DOI review is incomplete"),
        ("output_type", "output-type review changed"),
        ("source_row", "source row changed"),
        ("empty_title", "bibliographic row is incomplete"),
        ("duplicate_title", "title/year is duplicated"),
        ("duplicate_manuscript", "manuscript identity is duplicated"),
        ("doi_coverage", "DOI review coverage changed"),
    ],
)
def test_author_site_rejects_changed_capture_or_review(
    tmp_path: Path, change: str, message: str
) -> None:
    data_dir = tmp_path / "data/author_sites"
    review_dir = tmp_path / "registry/reviews"
    data_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    source_manifest = json.loads(CAPTURE.read_text(encoding="utf-8"))
    manifest_path = data_dir / CAPTURE.name
    html_path = data_dir / source_manifest["html_file"]
    csv_path = tmp_path / source_manifest["review_csv_path"]
    audit_path = tmp_path / source_manifest["doi_audit_csv_path"]
    for source, destination in (
        (CAPTURE.parent / source_manifest["html_file"], html_path),
        (ROOT / source_manifest["review_csv_path"], csv_path),
        (ROOT / source_manifest["doi_audit_csv_path"], audit_path),
    ):
        shutil.copyfile(source, destination)
    if change == "scope":
        source_manifest["expected_entry_count"] = 46
    elif change == "html":
        html_path.write_bytes(html_path.read_bytes() + b"changed")
    elif change == "csv":
        csv_path.write_bytes(csv_path.read_bytes() + b"changed")
    elif change == "audit":
        audit_path.write_bytes(audit_path.read_bytes() + b"changed")
    elif change == "doi":
        source_manifest["doi_reviews"][0]["publisher_year"] = 1900
    elif change == "comb":
        next(
            item for item in source_manifest["doi_reviews"] if item["row_index"] == 22
        )["additional_evidence_urls"] = []
    elif change == "url":
        source_manifest["source_page_url"] = (
            "http://www.math.cmu.edu/~gautam/sj/index.html"
        )
    elif change == "time":
        source_manifest["captured_at_utc"] = "2026-10-25T09:48:40"
    elif change == "selector":
        source_manifest["source_selector"] = "div#other > ul > li"
    elif change == "output_type":
        source_manifest["publication_type_overrides"][0]["publication_type"] = (
            "publication"
        )
    elif change == "audit_rows":
        with audit_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            audited_rows = list(reader)
            audited_fields = reader.fieldnames
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=audited_fields)
            writer.writeheader()
            writer.writerows(audited_rows[:-1])
        source_manifest["doi_audit_csv_sha256"] = hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest()
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            extracted_rows = list(reader)
            extracted_fields = reader.fieldnames
        if change == "source_row":
            extracted_rows[0]["person_id"] = "someone_else"
        elif change == "empty_title":
            extracted_rows[0]["title_as_listed"] = ""
        elif change == "duplicate_title":
            extracted_rows[1]["title_as_listed"] = extracted_rows[0]["title_as_listed"]
            extracted_rows[1]["year_as_listed"] = extracted_rows[0]["year_as_listed"]
        elif change == "duplicate_manuscript":
            extracted_rows[1]["author_manuscript_url_as_linked"] = extracted_rows[0][
                "author_manuscript_url_as_linked"
            ]
        elif change == "doi_coverage":
            extracted_rows[0]["doi_as_linked"] = "10.1000/unreviewed"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=extracted_fields)
            writer.writeheader()
            writer.writerows(extracted_rows)
        source_manifest["review_csv_sha256"] = hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_author_site_rows(manifest_path, _registry())


def test_author_site_requires_verified_registry_identity() -> None:
    registry = _registry()
    registry = dataclasses.replace(
        registry,
        people=tuple(
            person for person in registry.people if person.person_id != IYER_PERSON_ID
        ),
    )
    with pytest.raises(ValueError, match="lacks a verified faculty identity"):
        load_author_site_rows(CAPTURE, registry)


def test_publisher_holds_author_site_until_public_release_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from map_of_research import publisher

    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: _registry())
    monkeypatch.setattr(
        publisher, "implicit_primary_baseline_for_people", lambda *args: {}
    )
    monkeypatch.setattr(
        publisher,
        "validate_included_scholar_primary_homes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            pandas.DataFrame({"source_type": ["author_site"]}),
            {},
        ),
    )
    with pytest.raises(ValueError, match="exact reviewed release gate"):
        publisher.publish_snapshot(
            Path("snapshot.parquet"),
            Path("manifest.json"),
            hf_token="unused",
            prior_revision="bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd",
            atomic_intent_sink=lambda intent: None,
        )
