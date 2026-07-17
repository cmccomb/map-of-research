#!/usr/bin/env python3
"""Seed the central cache from the last good department repository data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map_of_research.collector import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    normalize_publications,
)
from map_of_research.io import atomic_write_json  # noqa: E402
from map_of_research.registry import load_registry, unique_profiles  # noqa: E402


def _git_timestamp(repository: Path, relative_path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "log",
            "-1",
            "--format=%cI",
            "--",
            str(relative_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    timestamp = result.stdout.strip()
    if timestamp:
        return dt.datetime.fromisoformat(timestamp).astimezone(dt.UTC).isoformat()
    return dt.datetime.now(dt.UTC).isoformat()


def _candidate(
    repos_root: Path,
    map_slug: str,
    faculty: str,
) -> tuple[Path, str, list[dict[str, Any]]] | None:
    repository = repos_root / map_slug
    for data_directory in ("data", "data_old"):
        relative_path = Path(data_directory) / f"{faculty}.json"
        path = repository / relative_path
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            legacy = json.load(handle)
        if not isinstance(legacy, list):
            continue
        publications = normalize_publications(legacy)
        if publications:
            return path, _git_timestamp(repository, relative_path), publications
    return None


def migrate(
    *,
    repos_root: Path,
    people_path: Path,
    memberships_path: Path,
    maps_path: Path,
    cache_dir: Path,
    state_path: Path,
) -> dict[str, Any]:
    profiles = unique_profiles(load_registry(people_path, memberships_path, maps_path))
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "profiles": {},
    }
    migrated = 0
    missing: list[str] = []
    for profile in profiles:
        candidates = []
        for membership in profile.memberships:
            candidate = _candidate(
                repos_root,
                membership.map_slug,
                membership.legacy_label,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            missing.append(profile.scholar_id)
            continue
        source_path, fetched_at_utc, publications = max(
            candidates,
            key=lambda item: (len(item[2]), item[1], str(item[0])),
        )
        migration_source = str(source_path.relative_to(repos_root))
        atomic_write_json(
            cache_dir / f"{profile.scholar_id}.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "scholar_id": profile.scholar_id,
                "display_name": profile.display_name,
                "fetched_at_utc": fetched_at_utc,
                "publication_count": len(publications),
                "publications": publications,
                "migration_source": migration_source,
            },
        )
        state["profiles"][profile.scholar_id] = {
            "display_name": profile.display_name,
            "last_attempt_at_utc": fetched_at_utc,
            "last_success_at_utc": fetched_at_utc,
            "last_error_type": None,
            "publication_count": len(publications),
            "migration_source": migration_source,
        }
        migrated += 1
    atomic_write_json(state_path, state)
    return {
        "registry_profiles": len(profiles),
        "migrated_profiles": migrated,
        "missing_profile_ids": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, required=True)
    parser.add_argument(
        "--people",
        type=Path,
        default=Path("registry/people.csv"),
    )
    parser.add_argument(
        "--memberships",
        type=Path,
        default=Path("registry/memberships.csv"),
    )
    parser.add_argument("--maps", type=Path, default=Path("registry/maps.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/authors"))
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("status/authors.json"),
    )
    args = parser.parse_args()
    result = migrate(
        repos_root=args.repos_root,
        people_path=args.people,
        memberships_path=args.memberships,
        maps_path=args.maps,
        cache_dir=args.cache_dir,
        state_path=args.state_file,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["migrated_profiles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
