#!/usr/bin/env python3
"""Migrate the retained flat registry into normalized schema-v2 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

MAPS = {
    "map-of-eng": (
        "CMU Engineering Research",
        "",
        "Aggregate of included memberships in all maps.",
    ),
    "map-of-bme": (
        "Biomedical Engineering Research",
        "https://www.cmu.edu/bme/People/Faculty/",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-cheme": (
        "Chemical Engineering Research",
        "https://www.cheme.engineering.cmu.edu/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-civil": (
        "Civil & Environmental Engineering Research",
        "https://cee.engineering.cmu.edu/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-ece": (
        "Electrical & Computer Engineering Research",
        "https://www.ece.cmu.edu/directory/faculty.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-epp": (
        "Engineering & Public Policy Research",
        "https://epp.engineering.cmu.edu/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-iii": (
        "Integrated Innovation Institute Research",
        "https://www.cmu.edu/iii/people/faculty-staff/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-ini": (
        "Information Networking Institute Research",
        "https://www.cmu.edu/ini/about/team/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-cmu-silicon-valley": (
        "CMU Silicon Valley Research",
        "https://www.sv.cmu.edu/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-cmu-africa": (
        "CMU-Africa Research",
        "https://www.africa.engineering.cmu.edu/about/contact/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-mech": (
        "Mechanical Engineering Research",
        "https://www.meche.engineering.cmu.edu/directory/index.html",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
    "map-of-mse": (
        "Materials Science & Engineering Research",
        "https://www.mse.engineering.cmu.edu/directory/",
        "Include faculty, teaching faculty, and emeriti; exclude affiliates.",
    ),
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "unknown"


def _person_id(identity: str, display_name: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    return f"person-{_slug(display_name)}-{digest}"


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_registry(legacy_path: Path, output_dir: Path) -> tuple[int, int]:
    with legacy_path.open(newline="", encoding="utf-8-sig") as handle:
        legacy_rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in legacy_rows:
        scholar_id = (row.get("scholar_id") or "").strip()
        faculty = (row.get("faculty") or "").strip()
        identity = (
            f"scholar:{scholar_id}" if scholar_id else f"name:{faculty.casefold()}"
        )
        grouped.setdefault(identity, []).append(row)

    people_rows: list[dict[str, str]] = []
    person_ids: dict[str, str] = {}
    for identity, rows in sorted(grouped.items()):
        names = Counter((row.get("faculty") or "").strip() for row in rows)
        display_name = min(
            names,
            key=lambda name: (
                -names[name],
                -name.count(" "),
                len(name),
                name.casefold(),
            ),
        )
        person_id = _person_id(identity, display_name)
        person_ids[identity] = person_id
        scholar_ids = {
            (row.get("scholar_id") or "").strip()
            for row in rows
            if (row.get("scholar_id") or "").strip()
        }
        if len(scholar_ids) > 1:
            raise ValueError(f"Legacy identity has conflicting Scholar IDs: {identity}")
        people_rows.append(
            {
                "person_id": person_id,
                "display_name": display_name,
                "scholar_id": next(iter(scholar_ids), ""),
                "scholar_id_source_url": "",
                "scholar_id_verified_at": "",
                "orcid": "",
                "homepage_url": "",
                "notes": (
                    "Migrated from the legacy map registries; full-name review pending."
                ),
            }
        )

    membership_rows: list[dict[str, str]] = []
    seen_memberships: set[tuple[str, str]] = set()
    for row in legacy_rows:
        scholar_id = (row.get("scholar_id") or "").strip()
        faculty = (row.get("faculty") or "").strip()
        map_slug = (row.get("map_slug") or "").strip()
        identity = (
            f"scholar:{scholar_id}" if scholar_id else f"name:{faculty.casefold()}"
        )
        key = (person_ids[identity], map_slug)
        if key in seen_memberships:
            continue
        seen_memberships.add(key)
        membership_rows.append(
            {
                "person_id": person_ids[identity],
                "map_slug": map_slug,
                "role": "faculty",
                "included": "true",
                "legacy_label": faculty,
                "source_url": MAPS[map_slug][1],
                "verified_at": "",
            }
        )

    map_rows = [
        {
            "map_slug": slug,
            "title": title,
            "directory_url": directory_url,
            "reviewed_at": "",
            "review_notes": notes,
        }
        for slug, (title, directory_url, notes) in sorted(MAPS.items())
    ]
    people_rows.sort(key=lambda row: row["person_id"])
    membership_rows.sort(key=lambda row: (row["map_slug"], row["person_id"]))
    _write_csv(
        output_dir / "people.csv",
        (
            "person_id",
            "display_name",
            "scholar_id",
            "scholar_id_source_url",
            "scholar_id_verified_at",
            "orcid",
            "homepage_url",
            "notes",
        ),
        people_rows,
    )
    _write_csv(
        output_dir / "memberships.csv",
        (
            "person_id",
            "map_slug",
            "role",
            "included",
            "legacy_label",
            "source_url",
            "verified_at",
        ),
        membership_rows,
    )
    _write_csv(
        output_dir / "maps.csv",
        ("map_slug", "title", "directory_url", "reviewed_at", "review_notes"),
        map_rows,
    )
    return len(people_rows), len(membership_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-registry",
        type=Path,
        default=Path("registry/faculty.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("registry"))
    args = parser.parse_args()
    people, memberships = build_registry(args.legacy_registry, args.output_dir)
    print(f"Wrote {people} people and {memberships} memberships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
