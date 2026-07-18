import datetime as dt
import json
from pathlib import Path

import pandas
import pytest

import map_of_research.snapshot as snapshot
from map_of_research.io import atomic_write_json
from map_of_research.registry import load_registry, unique_profiles
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
            },
            {
                "person_id": "person-beta",
                "display_name": "Beta Person",
                "scholar_id": "betaAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "Uncached test profile",
            },
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
            {
                "person_id": "person-beta",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Beta",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
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


def build_fixture(tmp_path: Path):
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
    return (
        snapshot_path,
        manifest_path,
        people_path,
        memberships_path,
        departments_path,
        manifest,
    )


def test_snapshot_scalar_normalizers_and_venue_fallbacks(tmp_path: Path) -> None:
    assert snapshot._parse_year(None) is None
    assert snapshot._parse_year("") is None
    assert snapshot._parse_year("bad") is None
    assert snapshot._parse_year(1499) is None
    assert snapshot._parse_year(2025.0) == 2025
    assert snapshot._citation_count("4") == 4
    assert snapshot._citation_count(-2) == 0
    assert snapshot._citation_count("bad") == 0

    paths = make_registry(tmp_path / "registry")
    profile = unique_profiles(load_registry(*paths))[0]
    base = {"author_pub_id": "id:one", "title": "One"}
    for field in ("venue", "journal", "conference", "citation"):
        publication = {**base, field: f"From {field}"}
        row = snapshot._publication_row(profile, publication, fetched_at_utc="now")
        assert row["venue"] == f"From {field}"
        assert row["citation_count"] == 0
    assert len(snapshot._memberships(profile)) == 2


def test_registry_digest_is_independent_of_argument_order(tmp_path: Path) -> None:
    paths = make_registry(tmp_path / "registry")
    assert snapshot._registry_sha256(paths) == snapshot._registry_sha256(paths[::-1])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "not an object"),
        ({}, "schema"),
        (
            {
                "schema_version": 1,
                "scholar_id": "other",
                "publications": [],
                "fetched_at_utc": "now",
            },
            "identity mismatch",
        ),
        (
            {
                "schema_version": 1,
                "scholar_id": "alphaAAAAJ",
                "publications": {},
                "fetched_at_utc": "now",
            },
            "publications are invalid",
        ),
        (
            {
                "schema_version": 1,
                "scholar_id": "alphaAAAAJ",
                "publications": [],
                "fetched_at_utc": None,
            },
            "timestamp is invalid",
        ),
    ],
)
def test_profile_cache_validation_rejects_malformed_documents(
    tmp_path: Path,
    payload,
    message,
) -> None:
    paths = make_registry(tmp_path / "registry")
    profile = unique_profiles(load_registry(*paths))[0]
    cache_path = tmp_path / "cache.json"
    atomic_write_json(cache_path, payload)
    with pytest.raises(ValueError, match=message):
        snapshot._load_profile_cache(cache_path, profile)


def valid_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                **{column: "value" for column in snapshot.REQUIRED_COLUMNS},
                "scholar_id": "one",
                "author_pub_id": "one:paper",
                "title": "A paper",
                "citation_count": 1,
            }
        ]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="title"), "missing columns"),
        (lambda frame: frame.iloc[0:0], "no publication rows"),
        (
            lambda frame: frame.assign(scholar_id=None),
            "identities cannot be null",
        ),
        (
            lambda frame: pandas.concat([frame, frame], ignore_index=True),
            "duplicate profile-publication",
        ),
        (lambda frame: frame.assign(title=" "), "blank publication"),
        (lambda frame: frame.assign(citation_count=-1), "negative citation"),
    ],
)
def test_frame_validation_rejects_each_integrity_failure(mutate, message) -> None:
    frame = mutate(valid_frame())
    with pytest.raises(ValueError, match=message):
        snapshot._validate_frame(frame)


def test_snapshot_skips_uncached_profiles_and_rejects_non_object_publications(
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    write_cache(cache_dir)
    cache = json.loads((cache_dir / "alphaAAAAJ.json").read_text())
    cache["publications"] = ["bad"]
    atomic_write_json(cache_dir / "alphaAAAAJ.json", cache)
    with pytest.raises(ValueError, match="Invalid publication"):
        build_snapshot(
            people_path=paths[0],
            memberships_path=paths[1],
            departments_path=paths[2],
            cache_dir=cache_dir,
            snapshot_path=tmp_path / "snapshot.parquet",
            manifest_path=tmp_path / "manifest.json",
            now=NOW,
        )


def test_snapshot_removes_temporary_parquet_after_write_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    write_cache(cache_dir)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pandas.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk full"):
        build_snapshot(
            people_path=paths[0],
            memberships_path=paths[1],
            departments_path=paths[2],
            cache_dir=cache_dir,
            snapshot_path=tmp_path / "snapshot.parquet",
            manifest_path=tmp_path / "manifest.json",
            now=NOW,
        )
    assert not list(tmp_path.glob(".snapshot.parquet.*.tmp"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: [], "must be an object"),
        (lambda manifest: {"schema_version": 1}, "missing fields"),
        (
            lambda manifest: {**manifest, "schema_version": 9},
            "manifest schema is unsupported",
        ),
        (
            lambda manifest: {**manifest, "snapshot_schema_version": 9},
            "data schema is unsupported",
        ),
        (lambda manifest: {**manifest, "status": "failure"}, "did not report success"),
        (
            lambda manifest: {**manifest, "snapshot_file": "other.parquet"},
            "filename does not match",
        ),
        (
            lambda manifest: {**manifest, "registry_sha256": "bad"},
            "Registry files do not match",
        ),
        (
            lambda manifest: {**manifest, "created_at_utc": "2026-07-17T12:00:00"},
            "timestamp must include a timezone",
        ),
        (
            lambda manifest: {
                **manifest,
                "created_at_utc": "2026-07-17T12:06:00+00:00",
            },
            "unexpectedly in the future",
        ),
        (
            lambda manifest: {
                **manifest,
                "created_at_utc": "2026-06-01T12:00:00+00:00",
            },
            "too old to publish",
        ),
        (
            lambda manifest: {**manifest, "publication_rows": 2},
            "row count does not match",
        ),
        (
            lambda manifest: {**manifest, "cached_profiles": 0},
            "profile count does not match",
        ),
        (
            lambda manifest: {
                **manifest,
                "cached_profiles": 1,
                "registry_profiles": 0,
            },
            "more profiles than the registry",
        ),
    ],
)
def test_manifest_validation_rejects_every_provenance_failure(
    tmp_path: Path,
    mutate,
    message,
) -> None:
    paths = build_fixture(tmp_path)
    manifest = mutate(paths[5])
    atomic_write_json(paths[1], manifest)
    with pytest.raises(ValueError, match=message):
        validate_snapshot(
            paths[0],
            paths[1],
            now=NOW,
            people_path=paths[2],
            memberships_path=paths[3],
            departments_path=paths[4],
        )


def test_snapshot_command_entrypoints_emit_manifest(tmp_path: Path, capsys) -> None:
    people, memberships, departments = make_registry(tmp_path / "registry")
    cache = tmp_path / "cache"
    cache.mkdir()
    write_cache(cache)
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "manifest.json"
    common = [
        "--snapshot",
        str(snapshot_path),
        "--manifest",
        str(manifest_path),
        "--people",
        str(people),
        "--memberships",
        str(memberships),
        "--departments",
        str(departments),
    ]
    assert snapshot.build_main([*common, "--cache-dir", str(cache)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert snapshot.validate_main(common) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
