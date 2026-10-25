"""The Anderson source stays complete and cannot be mistaken for Scholar data."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pandas
import pytest

from map_of_research.anderson_author_site import PERSON_ID, _url
from map_of_research.author_site import load_author_site_rows
from map_of_research.dataset import build_dataset_tables
from map_of_research.registry import load_registry
from map_of_research.snapshot import build_snapshot, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data/author_sites/theresa-anderson-2026-10-25.json"


def _registry():
    return load_registry(
        ROOT / "registry/people.csv",
        ROOT / "registry/memberships.csv",
        ROOT / "registry/departments.csv",
    )


def _copy_capture(tmp_path: Path) -> Path:
    copied = tmp_path / "data/author_sites"
    copied.mkdir(parents=True)
    for source in CAPTURE.parent.glob("theresa-anderson-2026-10-25.*"):
        shutil.copy2(source, copied / source.name)
    return copied / CAPTURE.name


def _rebind_hash(manifest_path: Path, key: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest_path.parent / manifest[f"{key}_file"]
    manifest[f"{key}_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _change_csv_row(
    manifest_path: Path, key: str, index: int, field: str, value: str
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = manifest_path.parent / manifest[f"{key}_file"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[index][field] = value
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rebind_hash(manifest_path, key)


def test_anderson_full_displayed_source_and_genre_holds() -> None:
    registry = _registry()
    rows = load_author_site_rows(CAPTURE, registry)
    assert len(rows) == len({row["source_native_id"] for row in rows}) == 42
    assert {row["person_id"] for row in rows} == {PERSON_ID}
    assert {row["source_type"] for row in rows} == {"author_site"}
    assert all(row["scholar_id"] == row["fetched_at_utc"] == "" for row in rows)
    assert all(row["source_citation_count_known"] is False for row in rows)
    assert all(row["source_doi"] == "" for row in rows)
    assert all(row["external_coauthors"] == [] for row in rows)
    assert [row["year"] for row in rows[:5]] == [None] * 5
    assert rows[29]["year"] == 2020
    assert [rows[index - 1]["source_output_type"] for index in (5, 11, 37)] == [
        "appendix",
        "personal_essay",
        "thesis",
    ]
    assert "Kim, Elena" not in rows[4]["authors"]
    assert "Kim, Elena" in rows[4]["title"]
    assert "different bases" in rows[22]["title"]
    raw = json.loads(rows[22]["source_record_json"])
    assert "number theoretic approach" in raw["cv_crosswalk_row"]["cv_citation"]
    assert raw["review_row"]["metadata_review_status"] == (
        "publication_metadata_review"
    )

    for row in rows:
        row["embedding"] = [0.0, 1.0]
    tables = build_dataset_tables(pandas.DataFrame(rows), registry)
    assert len(tables["profile_publications"]) == len(tables["works"]) == 42
    assert len(tables["authorships"]) == 42
    assert set(tables["authorships"]["person_id"]) == {PERSON_ID}
    for ordinal in (5, 11, 37):
        item = (
            tables["profile_publications"]
            .loc[
                tables["profile_publications"]["author_pub_id"]
                == rows[ordinal - 1]["author_pub_id"]
            ]
            .iloc[0]
        )
        assert not item["collaboration_eligible"]
        assert item["work_match_method"] == "source_native_id"


def test_anderson_offline_snapshot_is_exact_and_publisher_still_holds(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "anderson.parquet"
    manifest_path = tmp_path / "anderson.manifest.json"
    manifest = build_snapshot(
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        cache_dir=tmp_path / "empty-cache",
        author_site_path=CAPTURE,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
    assert manifest["author_site_rows"] == 42
    assert manifest["publication_rows"] == 70
    frame, _ = validate_snapshot(
        snapshot_path,
        manifest_path,
        people_path=ROOT / "registry/people.csv",
        memberships_path=ROOT / "registry/memberships.csv",
        departments_path=ROOT / "registry/departments.csv",
        author_site_path=CAPTURE,
    )
    assert len(frame.loc[frame["source_type"] == "author_site"]) == 42
    with pytest.raises(ValueError, match="Author-site capture does not match"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            people_path=ROOT / "registry/people.csv",
            memberships_path=ROOT / "registry/memberships.csv",
            departments_path=ROOT / "registry/departments.csv",
        )


@pytest.mark.parametrize(
    "key", ["html", "source_csv", "cv_pdf", "cv_crosswalk_csv", "review_csv"]
)
def test_anderson_rejects_changed_source_files(tmp_path: Path, key: str) -> None:
    manifest_path = _copy_capture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest_path.parent / manifest[f"{key}_file"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=f"{key} checksum changed"):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_changed_cv_pair_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_capture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crosswalk_path = manifest_path.parent / manifest["cv_crosswalk_csv_file"]
    with crosswalk_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["cv_number"] = "41"
    with crosswalk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rebind_hash(manifest_path, "cv_crosswalk_csv")
    with pytest.raises(ValueError, match="one-to-one crosswalk"):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_false_appendix_authorship_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_capture(tmp_path)
    _change_csv_row(
        manifest_path,
        "review_csv",
        4,
        "authors_as_listed",
        "Anderson, Theresa C. and Lemke Oliver, Robert J. Appendix to: Kim, Elena.",
    )
    with pytest.raises(ValueError, match="Appendix host author"):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_non_web_source_url() -> None:
    with pytest.raises(ValueError, match="Invalid Anderson source URL"):
        _url("ftp://example.org/entry")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_type", "scholar", "identity or scope changed"),
        ("captured_at_utc", "2026-10-25T11:42:24", "lacks timezone"),
        ("html_file", "wrong.html", "html filename changed"),
    ],
)
def test_anderson_rejects_changed_capture_manifest(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    manifest_path = _copy_capture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_unverified_registry_identity() -> None:
    registry = _registry()
    people = tuple(
        dataclasses.replace(person, scholar_id="unverifiedcandidate")
        if person.person_id == PERSON_ID
        else person
        for person in registry.people
    )
    with pytest.raises(ValueError, match="verified registry identity"):
        load_author_site_rows(CAPTURE, dataclasses.replace(registry, people=people))


def test_anderson_rejects_wrong_author_section_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_capture(tmp_path)
    html_path = manifest_path.parent / "theresa-anderson-2026-10-25.html"
    original = html_path.read_text(encoding="utf-8")
    html_path.write_text(
        original.replace("Publications and preprints", "Publications and preprintz", 1),
        encoding="utf-8",
    )
    _rebind_hash(manifest_path, "html")
    with pytest.raises(ValueError, match="no longer has 42"):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_partial_bibliography_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_capture(tmp_path)
    source_path = manifest_path.parent / "theresa-anderson-2026-10-25.source.csv"
    source_path.write_text(
        "\n".join(source_path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )
    _rebind_hash(manifest_path, "source_csv")
    with pytest.raises(ValueError, match="bibliography or CV crosswalk is incomplete"):
        load_author_site_rows(manifest_path, _registry())


def test_anderson_rejects_duplicate_citation_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_capture(tmp_path)
    with (manifest_path.parent / "theresa-anderson-2026-10-25.source.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        first = next(csv.DictReader(handle))["raw_citation"]
    _change_csv_row(manifest_path, "source_csv", 1, "raw_citation", first)
    with pytest.raises(ValueError, match="repeats a source citation"):
        load_author_site_rows(manifest_path, _registry())


@pytest.mark.parametrize(
    ("key", "index", "field", "value", "message"),
    [
        (
            "source_csv",
            0,
            "raw_citation",
            "Anderson, Theresa C. Changed",
            "source or CV row changed",
        ),
        (
            "review_csv",
            10,
            "source_output_type",
            "publication",
            "metadata or genre review changed",
        ),
        (
            "review_csv",
            0,
            "title_as_listed",
            "Invented title",
            "title or author parsing changed",
        ),
        ("review_csv", 29, "year_for_staging", "2021", "year lacks source support"),
        (
            "review_csv",
            0,
            "venue_and_status_as_listed",
            "Invented venue",
            "venue or status changed",
        ),
    ],
)
def test_anderson_rejects_review_drift_even_with_new_checksum(
    tmp_path: Path, key: str, index: int, field: str, value: str, message: str
) -> None:
    manifest_path = _copy_capture(tmp_path)
    _change_csv_row(manifest_path, key, index, field, value)
    with pytest.raises(ValueError, match=message):
        load_author_site_rows(manifest_path, _registry())
