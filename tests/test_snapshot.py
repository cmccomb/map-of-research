import datetime as dt
import json
from dataclasses import replace
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


def make_curated_only_registry(root: Path) -> tuple[Path, Path, Path]:
    return write_registry(
        root,
        people=[
            {
                "person_id": "person-math",
                "display_name": "Math Person",
                "scholar_id": "",
            },
            {
                "person_id": "person-affiliate",
                "display_name": "Affiliate Person",
                "scholar_id": "",
            },
        ],
        memberships=[
            {
                "person_id": "person-math",
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://www.cmu.edu/math/people/faculty.html",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-math",
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-affiliate",
                "department_id": "math",
                "role": "affiliate",
                "included": "false",
                "source_url": "https://www.cmu.edu/math/people/faculty.html",
                "verified_at": "2026-10-24",
            },
        ],
        departments=[
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematical Sciences",
                "directory_url": "https://www.cmu.edu/math/people/faculty.html",
            },
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "ECE",
                "directory_url": "https://www.ece.cmu.edu/directory/faculty.html",
            },
        ],
        schools=[
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://www.cmu.edu/mcs/",
            },
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://www.cmu.edu/engineering/",
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
    assert "schools.csv" in manifest["registry_files"]
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
    assert len(snapshot._memberships(profile.memberships)) == 2


def test_registry_digest_is_independent_of_argument_order(tmp_path: Path) -> None:
    paths = make_registry(tmp_path / "registry")
    assert snapshot._registry_sha256(paths) == snapshot._registry_sha256(paths[::-1])
    registry_paths = snapshot._registry_paths(*paths)
    assert registry_paths[-1] == paths[2].with_name("schools.csv")


def test_snapshot_validation_rejects_changed_school_hierarchy(tmp_path: Path) -> None:
    (
        snapshot_path,
        manifest_path,
        people_path,
        memberships_path,
        departments_path,
        _,
    ) = build_fixture(tmp_path)
    schools_path = departments_path.with_name("schools.csv")
    schools_path.write_text(
        schools_path.read_text(encoding="utf-8").replace(
            "College of Engineering", "Engineering College"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Registry files do not match"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            now=NOW,
            people_path=people_path,
            memberships_path=memberships_path,
            departments_path=departments_path,
        )


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
                "source_type": "scholar",
                "source_native_id": "",
            }
        ]
    )


def curated_payload() -> dict:
    return {
        "schema_version": 1,
        "external_coauthors": [
            {
                "person_id": "external-example",
                "display_name": "External Example",
                "source_url": "https://example.test/author",
            }
        ],
        "works": [
            {
                "person_id": "person-alpha",
                "doi": "10.1000/example",
                "title": "A Verified Paper",
                "authors": "Alpha Person and External Example",
                "year": 1995,
                "venue": "Journal",
                "citation": "Journal 1 (1995)",
                "source_url": "https://example.test/paper",
                "source_verification_urls": ["https://example.test/paper"],
                "source_verified_at": "2026-10-24",
                "external_coauthor_ids": ["external-example"],
            }
        ],
    }


def test_curated_rows_are_provenance_bound_and_not_scholar_fetches(
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    cache_dir = tmp_path / "authors"
    cache_dir.mkdir()
    write_cache(cache_dir)
    curated_path = tmp_path / "data/curated/works.json"
    curated_path.parent.mkdir(parents=True)
    atomic_write_json(curated_path, curated_payload())
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    manifest = build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    frame, _ = validate_snapshot(
        snapshot_path,
        manifest_path,
        now=NOW,
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
    )
    assert manifest["cached_profiles"] == 1
    assert manifest["curated_work_rows"] == 1
    assert manifest["curated_works_sha256"] == snapshot.sha256_file(curated_path)
    scholar = frame.loc[frame["source_type"] == "scholar"].iloc[0]
    curated = frame.loc[frame["source_type"] == "curated_verified"].iloc[0]
    assert scholar["fetched_at_utc"] == NOW.isoformat()
    assert curated["scholar_id"] == ""
    assert curated["fetched_at_utc"] == ""
    assert curated["source_doi"] == "10.1000/example"
    assert curated["source_verified_at"] == "2026-10-24"
    assert curated["external_coauthors"][0]["person_id"] == "external-example"

    edited = curated_payload()
    edited["works"][0]["title"] = "Tampered title"
    atomic_write_json(curated_path, edited)
    with pytest.raises(ValueError, match="Curated work source"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            now=NOW,
            people_path=paths[0],
            memberships_path=paths[1],
            departments_path=paths[2],
        )
    atomic_write_json(curated_path, curated_payload())
    manifest["curated_work_rows"] = 0
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="curated row count"):
        validate_snapshot(
            snapshot_path,
            manifest_path,
            now=NOW,
            people_path=paths[0],
            memberships_path=paths[1],
            departments_path=paths[2],
        )


def test_curated_native_id_survives_snapshot_without_scholar_identity(
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    curated_path = tmp_path / "data/curated/works.json"
    curated_path.parent.mkdir(parents=True)
    payload = curated_payload()
    work = payload["works"][0]
    work.pop("doi")
    work["source_native_id"] = "EGU2012:EGU2012-2516-1"
    atomic_write_json(curated_path, payload)
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"

    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=tmp_path / "no-scholar-cache",
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    frame, manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        now=NOW,
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
    )
    assert manifest["cached_profiles"] == 0
    assert manifest["curated_work_rows"] == len(frame) == 1
    row = frame.iloc[0]
    assert row["author_pub_id"] == (
        "curated-native:person-alpha:egu2012:egu2012-2516-1"
    )
    assert row["source_native_id"] == "egu2012:egu2012-2516-1"
    assert row["source_doi"] == row["scholar_id"] == row["fetched_at_utc"] == ""
    assert row["source_type"] == "curated_verified"


def test_curated_work_for_included_faculty_without_scholar_profile(
    tmp_path: Path,
) -> None:
    paths = make_curated_only_registry(tmp_path / "registry")
    curated_path = tmp_path / "data/curated/works.json"
    curated_path.parent.mkdir(parents=True)
    payload = curated_payload()
    payload["works"][0].update(
        person_id="person-math",
        title="A Verified Mathematics Paper",
        authors="Math Person and External Example",
    )
    atomic_write_json(curated_path, payload)
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"

    manifest = build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=tmp_path / "no-scholar-cache",
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=NOW,
    )
    frame, validated_manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        now=NOW,
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
    )

    assert manifest == validated_manifest
    assert manifest["registry_people"] == 2
    assert manifest["registry_profiles"] == manifest["cached_profiles"] == 0
    assert manifest["curated_work_rows"] == len(frame) == 1
    row = frame.iloc[0]
    assert row["person_id"] == "person-math"
    assert row["scholar_id"] == row["fetched_at_utc"] == ""
    assert row["department_ids"].tolist() == ["math"]
    assert {membership["department_id"] for membership in row["memberships"]} == {
        "math",
        "ece",
    }
    assert row["source_type"] == "curated_verified"
    assert row["source_doi"] == "10.1000/example"
    assert row["source_verified_at"] == "2026-10-24"
    assert row["source_verification_urls"].tolist() == ["https://example.test/paper"]

    payload["works"][0]["person_id"] = "person-affiliate"
    atomic_write_json(curated_path, payload)
    with pytest.raises(ValueError, match="registered faculty"):
        snapshot._load_curated_rows(curated_path, load_registry(*paths))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("__root__",), [], "schema"),
        (("schema_version",), 2, "schema"),
        (("works",), {}, "must be lists"),
        (("external_coauthors",), {}, "must be lists"),
        (("external_coauthors", 0), "bad", "must be an object"),
        (("external_coauthors", 0, "person_id"), "person-example", "identity"),
        (("external_coauthors", 0, "display_name"), "", "identity"),
        (("external_coauthors", 0, "source_url"), "http://example.test", "HTTPS"),
        (("works", 0), "bad", "must be an object"),
        (("works", 0, "person_id"), "person-missing", "registered faculty"),
        (("works", 0, "doi"), "bad-doi", "invalid DOI"),
        (("works", 0, "source_native_id"), "bad id", "source_native_id"),
        (("works", 0, "title"), "", "title, authors, and venue"),
        (("works", 0, "authors"), "", "title, authors, and venue"),
        (("works", 0, "venue"), "", "title, authors, and venue"),
        (("works", 0, "publication_type"), "invented_type", "publication type"),
        (("works", 0, "year"), "unknown", "year is invalid"),
        (("works", 0, "source_verified_at"), "yesterday", "YYYY-MM-DD"),
        (("works", 0, "source_verification_urls"), "bad", "verification URLs"),
        (("works", 0, "source_verification_urls"), [], "verification URLs"),
        (("works", 0, "source_verification_urls"), ["http://bad"], "HTTPS"),
        (("works", 0, "external_coauthor_ids"), "bad", "unknown external"),
        (
            ("works", 0, "external_coauthor_ids"),
            ["external-unknown"],
            "unknown external",
        ),
        (("works", 0, "source_url"), "http://bad", "HTTPS"),
    ],
)
def test_curated_source_rejects_unverified_or_invalid_entries(
    tmp_path: Path, path: tuple, replacement, message: str
) -> None:
    registry = load_registry(*make_registry(tmp_path / "registry"))
    payload = curated_payload()
    if path == ("__root__",):
        payload = replacement
    else:
        target = payload
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = replacement
    source_path = tmp_path / "curated.json"
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match=message):
        snapshot._load_curated_rows(source_path, registry)


def test_curated_source_rejects_duplicate_ids_and_works(tmp_path: Path) -> None:
    registry = load_registry(*make_registry(tmp_path / "registry"))
    source_path = tmp_path / "curated.json"
    payload = curated_payload()
    payload["external_coauthors"].append(payload["external_coauthors"][0])
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match="identity"):
        snapshot._load_curated_rows(source_path, registry)

    payload = curated_payload()
    payload["works"].append(payload["works"][0])
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match="Duplicate curated faculty DOI"):
        snapshot._load_curated_rows(source_path, registry)

    payload = curated_payload()
    payload["works"][0].pop("doi")
    payload["works"][0]["source_native_id"] = "egu2012:egu2012-2516-1"
    payload["works"].append(payload["works"][0])
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match="Duplicate curated faculty source_native_id"):
        snapshot._load_curated_rows(source_path, registry)


@pytest.mark.parametrize(
    ("doi", "native_id", "source_url", "message"),
    [
        ("", "", "https://example.test/paper", "exactly one"),
        ("10.1000/example", "egu:example", "https://example.test/paper", "exactly one"),
        ("", "bad id", "https://example.test/paper", "invalid source_native_id"),
        ("", "egu:example", "https://example.test/other", "URL needs verification"),
    ],
)
def test_curated_native_identity_requires_exact_source_evidence(
    tmp_path: Path,
    doi: str,
    native_id: str,
    source_url: str,
    message: str,
) -> None:
    registry = load_registry(*make_registry(tmp_path / "registry"))
    payload = curated_payload()
    payload["works"][0].update(
        doi=doi,
        source_native_id=native_id,
        source_url=source_url,
    )
    source_path = tmp_path / "curated.json"
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match=message):
        snapshot._load_curated_rows(source_path, registry)


def test_curated_source_requires_active_faculty_and_can_omit_external_ids(
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    registry = load_registry(*paths)
    excluded = replace(
        registry,
        memberships=tuple(
            replace(item, role="affiliate", included=False)
            for item in registry.memberships
        ),
    )
    source_path = tmp_path / "curated.json"
    payload = curated_payload()
    atomic_write_json(source_path, payload)
    with pytest.raises(ValueError, match="registered faculty"):
        snapshot._load_curated_rows(source_path, excluded)

    payload["works"][0].pop("external_coauthor_ids")
    atomic_write_json(source_path, payload)
    rows = snapshot._load_curated_rows(source_path, registry)
    assert rows[0]["external_coauthors"] == []
    assert snapshot._curated_path(paths[0], source_path) == source_path
    assert (
        snapshot._build_parser()
        .parse_args(["--curated-works", str(source_path)])
        .curated_works
        == source_path
    )


def test_snapshot_rejects_invalid_source_identity_fields() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        snapshot._validate_frame(valid_frame().assign(source_type="unknown"))
    with pytest.raises(ValueError, match="Scholar identity"):
        snapshot._validate_frame(valid_frame().assign(scholar_id=""))
    with pytest.raises(ValueError, match="curated source ID"):
        snapshot._validate_frame(valid_frame().assign(source_native_id="egu:example"))
    with pytest.raises(ValueError, match="exactly one DOI or source_native_id"):
        snapshot._validate_frame(
            valid_frame().assign(
                source_type="curated_verified", source_doi="", source_native_id=""
            )
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
