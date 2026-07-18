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
    departments: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    department_rows = departments or [
        {
            "department_id": "ece",
            "title": "Electrical & Computer Engineering",
            "directory_url": "https://www.ece.cmu.edu/directory/faculty.html",
            "reviewed_at": "2026-07-17",
            "review_notes": "Test",
        },
        {
            "department_id": "bme",
            "title": "Biomedical Engineering",
            "directory_url": "https://www.cmu.edu/bme/People/Faculty/",
            "reviewed_at": "2026-07-17",
            "review_notes": "Test",
        },
    ]
    people_path = root / "people.csv"
    memberships_path = root / "memberships.csv"
    departments_path = root / "departments.csv"
    _write(
        people_path,
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
        people,
    )
    _write(
        memberships_path,
        (
            "person_id",
            "department_id",
            "role",
            "included",
            "legacy_label",
            "source_url",
            "verified_at",
        ),
        memberships,
    )
    _write(
        departments_path,
        ("department_id", "title", "directory_url", "reviewed_at", "review_notes"),
        department_rows,
    )
    return people_path, memberships_path, departments_path
