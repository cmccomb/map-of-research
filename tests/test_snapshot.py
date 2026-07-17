import csv
import datetime as dt
from pathlib import Path

import pytest

from map_of_research.io import atomic_write_json
from map_of_research.snapshot import build_snapshot, validate_snapshot

NOW = dt.datetime(2026, 7, 17, 12, 0, tzinfo=dt.UTC)


def write_registry(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("map_slug", "department", "faculty", "scholar_id"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "map_slug": "map-of-ece",
                    "department": "ECE",
                    "faculty": "Alpha",
                    "scholar_id": "alphaAAAAJ",
                },
                {
                    "map_slug": "map-of-cmu-silicon-valley",
                    "department": "CMU Silicon Valley",
                    "faculty": "A. Alpha",
                    "scholar_id": "alphaAAAAJ",
                },
            ]
        )


def test_snapshot_normalizes_cross_appointment_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "faculty.csv"
    cache_dir = tmp_path / "authors"
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    cache_dir.mkdir()
    write_registry(registry_path)
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
                }
            ],
        },
    )

    manifest = build_snapshot(
        registry_path=registry_path,
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    frame, validated_manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        now=NOW,
    )

    assert manifest == validated_manifest
    assert len(frame) == 1
    assert set(frame.loc[0, "map_slugs"]) == {
        "map-of-ece",
        "map-of-cmu-silicon-valley",
    }
    assert len(frame.loc[0, "memberships"]) == 2
    assert frame.loc[0, "year"] == 2025
    assert frame.loc[0, "citation_count"] == 4


def test_snapshot_validation_rejects_checksum_mismatch(tmp_path: Path) -> None:
    registry_path = tmp_path / "faculty.csv"
    cache_dir = tmp_path / "authors"
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    cache_dir.mkdir()
    write_registry(registry_path)
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
                    "author_pub_id": "alphaAAAAJ:paper",
                    "num_citations": 0,
                }
            ],
        },
    )
    build_snapshot(
        registry_path=registry_path,
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    with snapshot_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        validate_snapshot(snapshot_path, manifest_path, now=NOW)
