import csv
from pathlib import Path


def _write(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_registry(
    root: Path,
    *,
    people: list[dict[str, object]],
    memberships: list[dict[str, object]],
    maps: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    map_rows = maps or [
        {
            "map_slug": "map-of-eng",
            "title": "Engineering",
            "directory_url": "",
            "reviewed_at": "2026-07-17",
            "review_notes": "Aggregate",
        },
        {
            "map_slug": "map-of-ece",
            "title": "Electrical & Computer Engineering",
            "directory_url": "https://www.ece.cmu.edu/directory/faculty.html",
            "reviewed_at": "2026-07-17",
            "review_notes": "Test",
        },
        {
            "map_slug": "map-of-bme",
            "title": "Biomedical Engineering",
            "directory_url": "https://www.cmu.edu/bme/People/Faculty/",
            "reviewed_at": "2026-07-17",
            "review_notes": "Test",
        },
    ]
    people_path = root / "people.csv"
    memberships_path = root / "memberships.csv"
    maps_path = root / "maps.csv"
    _write(
        people_path,
        ("person_id", "display_name", "scholar_id", "orcid", "homepage_url", "notes"),
        people,
    )
    _write(
        memberships_path,
        (
            "person_id",
            "map_slug",
            "role",
            "included",
            "legacy_label",
            "source_url",
            "verified_at",
        ),
        memberships,
    )
    _write(
        maps_path,
        ("map_slug", "title", "directory_url", "reviewed_at", "review_notes"),
        map_rows,
    )
    return people_path, memberships_path, maps_path
