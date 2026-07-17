#!/usr/bin/env python3
"""One-time deterministic migration of the department faculty registries."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

MAPS = (
    ("map-of-bme", "Biomedical Engineering"),
    ("map-of-cheme", "Chemical Engineering"),
    ("map-of-civil", "Civil & Environmental Engineering"),
    ("map-of-ece", "Electrical & Computer Engineering"),
    ("map-of-epp", "Engineering & Public Policy"),
    ("map-of-iii", "Integrated Innovation Institute"),
    ("map-of-ini", "Information Networking Institute"),
    ("map-of-cmu-silicon-valley", "CMU Silicon Valley"),
    ("map-of-cmu-africa", "CMU-Africa"),
    ("map-of-mech", "Mechanical Engineering"),
    ("map-of-mse", "Materials Science & Engineering"),
)


def build_registry(repos_root: Path, output_path: Path) -> int:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for map_slug, department in MAPS:
        source_path = repos_root / map_slug / "faculty.csv"
        with source_path.open(newline="", encoding="utf-8-sig") as handle:
            for source_row in csv.DictReader(handle):
                row = {
                    "map_slug": map_slug,
                    "department": department,
                    "faculty": (source_row.get("name") or "").strip(),
                    "scholar_id": (source_row.get("id") or "").strip(),
                }
                identity = (
                    row["map_slug"],
                    row["scholar_id"],
                    row["faculty"].casefold(),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("map_slug", "department", "faculty", "scholar_id"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("registry/faculty.csv"),
    )
    args = parser.parse_args()
    row_count = build_registry(args.repos_root, args.output)
    print(f"Wrote {row_count} canonical memberships to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
