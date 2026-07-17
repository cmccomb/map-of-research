"""Build and validate the raw central publication snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .collector import CACHE_SCHEMA_VERSION
from .io import atomic_write_json, load_json, sha256_file
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    AuthorProfile,
    load_registry,
    unique_profiles,
)

SNAPSHOT_SCHEMA_VERSION = 4
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_SNAPSHOT_PATH = Path("snapshots/cmu-engineering-publications.parquet")
DEFAULT_MANIFEST_PATH = Path(
    "snapshots/cmu-engineering-publications.parquet.manifest.json"
)
DEFAULT_MAX_AGE_DAYS = 14
REQUIRED_COLUMNS = {
    "scholar_id",
    "person_id",
    "display_name",
    "faculty",
    "department_ids",
    "memberships",
    "author_pub_id",
    "title",
    "authors",
    "year",
    "venue",
    "citation",
    "citation_count",
    "source_url",
    "source_record_json",
    "fetched_at_utc",
}


def _parse_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    current_year = dt.datetime.now(dt.UTC).year + 1
    return year if 1500 <= year <= current_year else None


def _citation_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _memberships(profile: AuthorProfile) -> list[dict[str, Any]]:
    return [
        {
            "department_id": membership.department_id,
            "role": membership.role,
            "included": membership.included,
            "legacy_label": membership.legacy_label,
            "source_url": membership.source_url,
            "verified_at": membership.verified_at,
        }
        for membership in profile.memberships
    ]


def _publication_row(
    profile: AuthorProfile,
    publication: Mapping[str, Any],
    *,
    fetched_at_utc: str,
) -> dict[str, Any]:
    citation = str(publication.get("citation") or "").strip()
    venue = str(
        publication.get("venue")
        or publication.get("journal")
        or publication.get("conference")
        or citation
        or ""
    ).strip()
    return {
        "scholar_id": profile.scholar_id,
        "person_id": profile.person_id,
        "display_name": profile.display_name,
        "faculty": profile.display_name,
        "department_ids": list(profile.department_ids),
        "memberships": _memberships(profile),
        "author_pub_id": str(publication["author_pub_id"]),
        "title": str(publication["title"]),
        "authors": str(publication.get("author") or "").strip(),
        "year": _parse_year(publication.get("pub_year", publication.get("year"))),
        "venue": venue,
        "citation": citation,
        "citation_count": _citation_count(publication.get("num_citations")),
        "source_url": str(publication.get("pub_url") or "").strip(),
        "source_record_json": json.dumps(
            dict(publication),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "fetched_at_utc": fetched_at_utc,
    }


def _registry_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_profile_cache(path: Path, profile: AuthorProfile) -> dict[str, Any]:
    cache = load_json(path)
    if not isinstance(cache, dict):
        raise ValueError(f"Profile cache is not an object: {path}")
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported profile cache schema: {path}")
    if cache.get("scholar_id") != profile.scholar_id:
        raise ValueError(f"Profile cache identity mismatch: {path}")
    if not isinstance(cache.get("publications"), list):
        raise ValueError(f"Profile cache publications are invalid: {path}")
    if not isinstance(cache.get("fetched_at_utc"), str):
        raise ValueError(f"Profile cache timestamp is invalid: {path}")
    return cache


def _validate_frame(frame: Any) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Snapshot has no publication rows")
    if frame[["scholar_id", "author_pub_id"]].isnull().any().any():
        raise ValueError("Snapshot publication identities cannot be null")
    duplicate_mask = frame.duplicated(
        subset=["scholar_id", "author_pub_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError("Snapshot contains duplicate profile-publication rows")
    if frame["title"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Snapshot contains blank publication titles")
    if (frame["citation_count"].fillna(0) < 0).any():
        raise ValueError("Snapshot contains a negative citation count")


def build_snapshot(
    *,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build one normalized row per unique profile publication."""

    import pandas

    created_at = now or dt.datetime.now(dt.UTC)
    registry = load_registry(people_path, memberships_path, departments_path)
    profiles = unique_profiles(registry)
    rows: list[dict[str, Any]] = []
    cached_profile_ids: list[str] = []
    for profile in profiles:
        cache_path = cache_dir / f"{profile.scholar_id}.json"
        if not cache_path.is_file():
            continue
        cache = _load_profile_cache(cache_path, profile)
        fetched_at_utc = cache["fetched_at_utc"]
        for publication in cache["publications"]:
            if not isinstance(publication, Mapping):
                raise ValueError(f"Invalid publication in {cache_path}")
            rows.append(
                _publication_row(
                    profile,
                    publication,
                    fetched_at_utc=fetched_at_utc,
                )
            )
        cached_profile_ids.append(profile.scholar_id)

    frame = pandas.DataFrame(rows)
    _validate_frame(frame)
    frame["year"] = frame["year"].astype("Int64")
    frame.sort_values(
        ["scholar_id", "title", "author_pub_id"],
        key=lambda column: column.astype(str).str.casefold(),
        inplace=True,
        ignore_index=True,
    )

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=snapshot_path.parent,
        prefix=f".{snapshot_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        os.replace(temporary_path, snapshot_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "success",
        "created_at_utc": created_at.astimezone(dt.UTC).isoformat(),
        "snapshot_file": snapshot_path.name,
        "snapshot_sha256": sha256_file(snapshot_path),
        "registry_sha256": _registry_sha256(
            (people_path, memberships_path, departments_path)
        ),
        "registry_files": [
            people_path.name,
            memberships_path.name,
            departments_path.name,
        ],
        "publication_rows": len(frame),
        "registry_people": len(registry.people),
        "registry_memberships": len(registry.memberships),
        "registry_profiles": len(profiles),
        "cached_profiles": len(cached_profile_ids),
        "cached_profile_ids_sha256": hashlib.sha256(
            "\n".join(sorted(cached_profile_ids)).encode("utf-8")
        ).hexdigest(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: dt.datetime | None = None,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Validate snapshot provenance and contents before any upload."""

    import pandas

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Snapshot manifest must be an object")
    required_manifest_fields = {
        "schema_version",
        "snapshot_schema_version",
        "status",
        "created_at_utc",
        "snapshot_file",
        "snapshot_sha256",
        "publication_rows",
        "registry_profiles",
        "cached_profiles",
        "registry_sha256",
    }
    missing = required_manifest_fields.difference(manifest)
    if missing:
        raise ValueError(f"Snapshot manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Snapshot manifest schema is unsupported")
    if manifest["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Snapshot data schema is unsupported")
    if manifest["status"] != "success":
        raise ValueError("Snapshot builder did not report success")
    if manifest["snapshot_file"] != snapshot_path.name:
        raise ValueError("Snapshot filename does not match the manifest")
    if sha256_file(snapshot_path) != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot checksum does not match the manifest")
    if (
        _registry_sha256((people_path, memberships_path, departments_path))
        != manifest["registry_sha256"]
    ):
        raise ValueError("Registry files do not match the snapshot manifest")

    created_at = dt.datetime.fromisoformat(str(manifest["created_at_utc"]))
    if created_at.tzinfo is None:
        raise ValueError("Snapshot timestamp must include a timezone")
    current_time = now or dt.datetime.now(dt.UTC)
    age = current_time - created_at.astimezone(dt.UTC)
    if age < dt.timedelta(minutes=-5):
        raise ValueError("Snapshot timestamp is unexpectedly in the future")
    if age > dt.timedelta(days=max_age_days):
        raise ValueError("Snapshot is too old to publish")

    frame = pandas.read_parquet(snapshot_path, engine="pyarrow")
    _validate_frame(frame)
    if len(frame) != int(manifest["publication_rows"]):
        raise ValueError("Snapshot row count does not match the manifest")
    cached_profiles = frame["scholar_id"].nunique()
    if cached_profiles != int(manifest["cached_profiles"]):
        raise ValueError("Snapshot profile count does not match the manifest")
    if int(manifest["cached_profiles"]) > int(manifest["registry_profiles"]):
        raise ValueError("Snapshot contains more profiles than the registry")
    return frame, manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a normalized raw snapshot")
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument(
        "--memberships",
        type=Path,
        default=DEFAULT_MEMBERSHIPS_PATH,
    )
    parser.add_argument(
        "--departments",
        type=Path,
        default=DEFAULT_DEPARTMENTS_PATH,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser


def build_main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_snapshot(
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
        cache_dir=args.cache_dir,
        snapshot_path=args.snapshot,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a raw snapshot")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument(
        "--memberships",
        type=Path,
        default=DEFAULT_MEMBERSHIPS_PATH,
    )
    parser.add_argument(
        "--departments",
        type=Path,
        default=DEFAULT_DEPARTMENTS_PATH,
    )
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = parser.parse_args(argv)
    _, manifest = validate_snapshot(
        args.snapshot,
        args.manifest,
        max_age_days=args.max_age_days,
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0
