"""Build and validate the raw central publication snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .author_site import load_author_site_rows
from .collector import CACHE_SCHEMA_VERSION
from .io import atomic_write_json, load_json, sha256_file
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    DEFAULT_SCHOOLS_PATH,
    AuthorProfile,
    Membership,
    Registry,
    load_registry,
    unique_profiles,
)

SNAPSHOT_SCHEMA_VERSION = 7
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_CURATED_WORKS_PATH = Path("data/curated/works.json")
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
    "source_type",
    "source_doi",
    "source_native_id",
    "source_verified_at",
    "source_verification_urls",
    "external_coauthors",
}
CURATED_SCHEMA_VERSION = 1
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+\Z", re.IGNORECASE)
NATIVE_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*:[a-z0-9][a-z0-9._:/-]*\Z")


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


def _memberships(memberships: Sequence[Membership]) -> list[dict[str, Any]]:
    return [
        {
            "department_id": membership.department_id,
            "role": membership.role,
            "included": membership.included,
            "legacy_label": membership.legacy_label,
            "source_url": membership.source_url,
            "verified_at": membership.verified_at,
        }
        for membership in memberships
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
        "memberships": _memberships(profile.memberships),
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
        "source_type": "scholar",
        "source_doi": str(publication.get("doi") or "").strip(),
        "source_native_id": "",
        "source_verified_at": "",
        "source_verification_urls": [],
        "external_coauthors": [],
        "source_output_type": "publication",
        "source_citation_count_known": publication.get("num_citations") is not None,
        "source_work_url_as_linked": "",
    }


def _curated_path(people_path: Path, curated_path: Path | None) -> Path:
    if curated_path is not None:
        return curated_path
    return people_path.parent.parent / DEFAULT_CURATED_WORKS_PATH


def _https_url(value: Any, *, field: str) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Curated {field} must be an HTTPS URL")
    return url


def _load_curated_rows(path: Path, registry: Registry) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    source = load_json(path)
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != CURATED_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported curated-work schema: {path}")
    works = source.get("works")
    coauthors = source.get("external_coauthors")
    if not isinstance(works, list) or not isinstance(coauthors, list):
        raise ValueError("Curated works and external coauthors must be lists")
    external_by_id: dict[str, dict[str, str]] = {}
    for item in coauthors:
        if not isinstance(item, dict):
            raise ValueError("Curated external coauthor must be an object")
        person_id = str(item.get("person_id") or "").strip()
        display_name = str(item.get("display_name") or "").strip()
        if (
            not person_id.startswith("external-")
            or not display_name
            or person_id in external_by_id
        ):
            raise ValueError(
                "Curated external coauthor identity is invalid or duplicated"
            )
        external_by_id[person_id] = {
            "person_id": person_id,
            "display_name": display_name,
            "source_url": _https_url(
                item.get("source_url"), field="coauthor source_url"
            ),
        }
    people_by_id = registry.people_by_id
    memberships_by_person: dict[str, list[Membership]] = {}
    for membership in registry.memberships:
        memberships_by_person.setdefault(membership.person_id, []).append(membership)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for work in works:
        if not isinstance(work, dict):
            raise ValueError("Curated work must be an object")
        person_id = str(work.get("person_id") or "").strip()
        person = people_by_id.get(person_id)
        memberships = memberships_by_person.get(person_id, [])
        department_ids = [
            membership.department_id
            for membership in memberships
            if membership.included
        ]
        if person is None or not department_ids:
            raise ValueError(
                f"Curated work has no active registered faculty: {person_id}"
            )
        doi = str(work.get("doi") or "").strip().casefold()
        native_id = str(work.get("source_native_id") or "").strip().casefold()
        if doi and not DOI_PATTERN.fullmatch(doi):
            raise ValueError(f"Curated work has an invalid DOI: {doi}")
        if native_id and not NATIVE_ID_PATTERN.fullmatch(native_id):
            raise ValueError(
                f"Curated work has an invalid source_native_id: {native_id}"
            )
        if bool(doi) == bool(native_id):
            raise ValueError("Curated work needs exactly one DOI or source_native_id")
        identifier = f"doi:{doi}" if doi else f"native:{native_id}"
        key = (person_id, identifier)
        if key in seen:
            kind = "DOI" if doi else "source_native_id"
            raise ValueError(f"Duplicate curated faculty {kind}: {key}")
        seen.add(key)
        title = str(work.get("title") or "").strip()
        authors = str(work.get("authors") or "").strip()
        venue = str(work.get("venue") or "").strip()
        if not title or not authors or not venue:
            raise ValueError("Curated work needs a title, authors, and venue")
        year = _parse_year(work.get("year"))
        if year is None:
            raise ValueError("Curated work year is invalid")
        verified_at = str(work.get("source_verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Curated source_verified_at must use YYYY-MM-DD"
            ) from error
        verification_urls = work.get("source_verification_urls")
        if not isinstance(verification_urls, list) or not verification_urls:
            raise ValueError("Curated work needs verification URLs")
        verified_urls = [
            _https_url(url, field="source_verification_urls")
            for url in verification_urls
        ]
        source_url = _https_url(work.get("source_url"), field="source_url")
        if native_id and source_url not in verified_urls:
            raise ValueError("Curated source_native_id URL needs verification")
        external_ids = work.get("external_coauthor_ids", [])
        if not isinstance(external_ids, list) or any(
            str(item) not in external_by_id for item in external_ids
        ):
            raise ValueError("Curated work references an unknown external coauthor")
        rows.append(
            {
                "scholar_id": "",
                "person_id": person_id,
                "display_name": person.display_name,
                "faculty": person.display_name,
                "department_ids": department_ids,
                "memberships": _memberships(memberships),
                "author_pub_id": (
                    f"curated-doi:{person_id}:{doi}"
                    if doi
                    else f"curated-native:{person_id}:{native_id}"
                ),
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "citation": str(work.get("citation") or "").strip(),
                "citation_count": 0,
                "source_url": source_url,
                "source_record_json": json.dumps(
                    work, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "fetched_at_utc": "",
                "source_type": "curated_verified",
                "source_doi": doi,
                "source_native_id": native_id,
                "source_verified_at": verified_at,
                "source_verification_urls": verified_urls,
                "external_coauthors": [
                    external_by_id[str(item)] for item in external_ids
                ],
                "source_output_type": "publication",
                "source_citation_count_known": False,
                "source_work_url_as_linked": "",
            }
        )
    return rows


def _registry_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _registry_paths(
    people_path: Path,
    memberships_path: Path,
    departments_path: Path,
) -> tuple[Path, ...]:
    """Include the school hierarchy in snapshot provenance."""

    return (
        people_path,
        memberships_path,
        departments_path,
        departments_path.with_name(DEFAULT_SCHOOLS_PATH.name),
    )


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
    if (
        not frame["source_type"]
        .isin({"scholar", "curated_verified", "author_site"})
        .all()
    ):
        raise ValueError("Snapshot contains an unknown source type")
    if frame.loc[frame["source_type"] == "scholar", "scholar_id"].eq("").any():
        raise ValueError("Scholar observations need a Scholar identity")
    if frame.loc[frame["source_type"] == "scholar", "source_native_id"].ne("").any():
        raise ValueError("Scholar observations cannot claim a curated source ID")
    curated = frame.loc[frame["source_type"] == "curated_verified"]
    if curated["source_doi"].eq("").eq(curated["source_native_id"].eq("")).any():
        raise ValueError(
            "Curated observations need exactly one DOI or source_native_id"
        )
    author_site = frame.loc[frame["source_type"] == "author_site"]
    if not author_site.empty and "source_citation_count_known" not in frame:
        raise ValueError("Author-site citation provenance is missing")
    if not author_site.empty and (
        author_site["scholar_id"].ne("").any()
        or author_site["fetched_at_utc"].ne("").any()
        or author_site["source_native_id"].eq("").any()
        or author_site["source_citation_count_known"].any()
    ):
        raise ValueError(
            "Author-site observations cannot claim Scholar or citation counts"
        )
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


def _author_site_paths(
    value: Path | Sequence[Path] | None,
) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(path) for path in value)


def _author_site_specs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Identify each capture without depending on CLI argument order."""

    specs: list[dict[str, Any]] = []
    person_ids: set[str] = set()
    filenames: set[str] = set()
    for path in paths:
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise ValueError("Author-site capture manifest must be an object")
        person_id = manifest.get("person_id")
        row_count = manifest.get("expected_entry_count")
        if (
            not isinstance(person_id, str)
            or not person_id
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise ValueError("Author-site capture identity or row count is invalid")
        if person_id in person_ids:
            raise ValueError(f"Duplicate author-site person: {person_id}")
        if path.name in filenames:
            raise ValueError(f"Duplicate author-site manifest filename: {path.name}")
        person_ids.add(person_id)
        filenames.add(path.name)
        specs.append(
            {
                "person_id": person_id,
                "file": path.name,
                "sha256": sha256_file(path),
                "rows": row_count,
                "path": path,
            }
        )
    return sorted(specs, key=lambda item: item["person_id"])


def _capture_metadata(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: spec[key] for key in ("person_id", "file", "sha256", "rows")}
        for spec in specs
    ]


def _load_author_site_captures(
    specs: Sequence[Mapping[str, Any]], registry: Registry
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    native_ids: set[str] = set()
    for spec in specs:
        captured = load_author_site_rows(spec["path"], registry)
        if len(captured) != spec["rows"] or any(
            row["person_id"] != spec["person_id"] for row in captured
        ):
            raise ValueError(f"Incomplete author-site capture: {spec['person_id']}")
        for row in captured:
            native_id = str(row["source_native_id"])
            if native_id in native_ids:
                raise ValueError(f"Duplicate author-site native ID: {native_id}")
            native_ids.add(native_id)
        rows.extend(captured)
    return rows


def _source_value(value: Any) -> Any:
    """Normalize parquet's array and nullable scalars for exact row comparison."""

    import pandas

    if isinstance(value, Mapping):
        return {key: _source_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_source_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _source_value(value.tolist())
    if pandas.isna(value):
        return None
    return value


def build_snapshot(
    *,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    curated_path: Path | None = None,
    author_site_path: Path | Sequence[Path] | None = None,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build one normalized row per unique profile publication."""

    import pandas

    created_at = now or dt.datetime.now(dt.UTC)
    registry = load_registry(people_path, memberships_path, departments_path)
    profiles = unique_profiles(registry)
    source_path = _curated_path(people_path, curated_path)
    capture_specs = _author_site_specs(_author_site_paths(author_site_path))
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

    rows.extend(_load_curated_rows(source_path, registry))
    rows.extend(_load_author_site_captures(capture_specs, registry))

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
            _registry_paths(people_path, memberships_path, departments_path)
        ),
        "registry_files": [
            people_path.name,
            memberships_path.name,
            departments_path.name,
            DEFAULT_SCHOOLS_PATH.name,
        ],
        "publication_rows": len(frame),
        "registry_people": len(registry.people),
        "registry_memberships": len(registry.memberships),
        "registry_profiles": len(profiles),
        "cached_profiles": len(cached_profile_ids),
        "curated_works_file": source_path.name if source_path.is_file() else "",
        "curated_works_sha256": sha256_file(source_path)
        if source_path.is_file()
        else "",
        "curated_work_rows": int((frame["source_type"] == "curated_verified").sum()),
        "author_site_capture_file": capture_specs[0]["file"]
        if len(capture_specs) == 1
        else "",
        "author_site_capture_sha256": capture_specs[0]["sha256"]
        if len(capture_specs) == 1
        else "",
        "author_site_captures": _capture_metadata(capture_specs),
        "author_site_rows": int((frame["source_type"] == "author_site").sum()),
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
    curated_path: Path | None = None,
    author_site_path: Path | Sequence[Path] | None = None,
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
        "curated_works_file",
        "curated_works_sha256",
        "curated_work_rows",
        "author_site_capture_file",
        "author_site_capture_sha256",
        "author_site_rows",
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
        _registry_sha256(
            _registry_paths(people_path, memberships_path, departments_path)
        )
        != manifest["registry_sha256"]
    ):
        raise ValueError("Registry files do not match the snapshot manifest")
    source_path = _curated_path(people_path, curated_path)
    current_curated_name = source_path.name if source_path.is_file() else ""
    current_curated_sha = sha256_file(source_path) if source_path.is_file() else ""
    if (
        manifest["curated_works_file"] != current_curated_name
        or manifest["curated_works_sha256"] != current_curated_sha
    ):
        raise ValueError("Curated work source does not match the snapshot manifest")
    capture_specs = _author_site_specs(_author_site_paths(author_site_path))
    legacy_file = capture_specs[0]["file"] if len(capture_specs) == 1 else ""
    legacy_sha = capture_specs[0]["sha256"] if len(capture_specs) == 1 else ""
    if (
        manifest["author_site_capture_file"] != legacy_file
        or manifest["author_site_capture_sha256"] != legacy_sha
    ):
        raise ValueError("Author-site capture does not match the snapshot manifest")
    if "author_site_captures" in manifest:
        if manifest["author_site_captures"] != _capture_metadata(capture_specs):
            raise ValueError(
                "Author-site capture list does not match the snapshot manifest"
            )
    elif len(capture_specs) > 1:
        raise ValueError("Legacy snapshot manifest cannot bind multiple author sites")

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
    cached_profiles = frame.loc[
        frame["source_type"] == "scholar", "scholar_id"
    ].nunique()
    if cached_profiles != int(manifest["cached_profiles"]):
        raise ValueError("Snapshot profile count does not match the manifest")
    curated_rows = int((frame["source_type"] == "curated_verified").sum())
    if curated_rows != int(manifest["curated_work_rows"]):
        raise ValueError("Snapshot curated row count does not match the manifest")
    author_site_rows = int((frame["source_type"] == "author_site").sum())
    if author_site_rows != int(manifest["author_site_rows"]):
        raise ValueError("Snapshot author-site row count does not match the manifest")
    if capture_specs:
        registry = load_registry(people_path, memberships_path, departments_path)
        expected_rows = _load_author_site_captures(capture_specs, registry)
        observed = frame.loc[frame["source_type"] == "author_site"]
        if observed["source_native_id"].duplicated().any():
            raise ValueError("Snapshot has duplicate author-site native IDs")
        for spec in capture_specs:
            person_rows = observed.loc[observed["person_id"] == spec["person_id"]]
            if len(person_rows) != spec["rows"]:
                person = registry.people_by_id[spec["person_id"]].display_name.split()[
                    -1
                ]
                raise ValueError(f"Incomplete {person} author-site profile")
        actual = {row.author_pub_id: row for row in observed.itertuples(index=False)}
        if {row["author_pub_id"] for row in expected_rows} != set(actual):
            raise ValueError("Snapshot author-site observation identities changed")
        for source_row in expected_rows:
            observed_row = actual[source_row["author_pub_id"]]
            if any(
                _source_value(source_value)
                != _source_value(getattr(observed_row, field))
                for field, source_value in source_row.items()
            ):
                raise ValueError("Snapshot author-site observation changed")
    elif author_site_rows:
        raise ValueError("Snapshot author-site captures are incomplete")
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
    parser.add_argument("--curated-works", type=Path)
    parser.add_argument("--author-site-capture", type=Path, action="append")
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
        curated_path=args.curated_works,
        author_site_path=args.author_site_capture,
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
    parser.add_argument("--curated-works", type=Path)
    parser.add_argument("--author-site-capture", type=Path, action="append")
    args = parser.parse_args(argv)
    _, manifest = validate_snapshot(
        args.snapshot,
        args.manifest,
        max_age_days=args.max_age_days,
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
        curated_path=args.curated_works,
        author_site_path=args.author_site_capture,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0
