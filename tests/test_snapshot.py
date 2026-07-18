import datetime as dt
from pathlib import Path

import pytest

from map_of_research.io import atomic_write_json
from map_of_research.snapshot import build_snapshot, validate_snapshot
from tests.registry_helpers import write_registry

NOW = dt.datetime(2026, 7, 17, 12, 0, tzinfo=dt.UTC)


def make_registry(root: Path) -> tuple[Path, Path, Path]:
    return write_registry(
        root,
        people=[
            {
                "person_id": "person-alpha",
                "display_name": "Alpha Person",
                "scholar_id": "alphaAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-alpha",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Alpha",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-alpha",
                "department_id": "cmu-silicon-valley",
                "role": "teaching",
                "included": "true",
                "legacy_label": "A. Alpha",
                "source_url": "https://www.sv.cmu.edu/directory/index.html",
                "verified_at": "2026-07-17",
            },
        ],
        departments=[
            {
                "department_id": "ece",
                "title": "ECE",
                "directory_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "reviewed_at": "2026-07-17",
                "review_notes": "",
            },
            {
                "department_id": "cmu-silicon-valley",
                "title": "CMU Silicon Valley",
                "directory_url": "https://www.sv.cmu.edu/directory/index.html",
                "reviewed_at": "2026-07-17",
                "review_notes": "",
            },
        ],
    )


def write_cache(cache_dir: Path) -> None:
    atomic_write_json(
        cache_dir / "alphaAAAAJ.json",
        {
            "schema_version": 1,
            "scholar_id": "alphaAAAAJ",
            "display_name": "Alpha",
            "fetched_at_utc": NOW.isoformat(),
            "publication_count": 1,
            "publications": [
                {
                    "title": "A Paper",
                    "author": "A. Alpha",
                    "pub_year": "2025",
                    "citation": "Journal, 2025",
                    "author_pub_id": "alphaAAAAJ:paper",
                    "num_citations": 4,
                    "pages": "1-10",
                }
            ],
        },
    )


def test_snapshot_preserves_source_record_and_normalized_memberships(
    tmp_path: Path,
) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    cache_dir = tmp_path / "authors"
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    cache_dir.mkdir()
    write_cache(cache_dir)

    manifest = build_snapshot(
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    frame, validated_manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        now=NOW,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
    )

    assert manifest == validated_manifest
    assert len(frame) == 1
    assert frame.loc[0, "person_id"] == "person-alpha"
    assert set(frame.loc[0, "department_ids"]) == {
        "ece",
        "cmu-silicon-valley",
    }
    assert len(frame.loc[0, "memberships"]) == 2
    assert frame.loc[0, "year"] == 2025
    assert frame.loc[0, "citation_count"] == 4
    assert '"pages":"1-10"' in frame.loc[0, "source_record_json"]


def test_snapshot_validation_rejects_checksum_mismatch(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    cache_dir = tmp_path / "authors"
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    cache_dir.mkdir()
    write_cache(cache_dir)
    build_snapshot(
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    with snapshot_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            now=NOW,
            people_path=people_path,
            memberships_path=memberships_path,
            departments_path=departments_path,
        )
