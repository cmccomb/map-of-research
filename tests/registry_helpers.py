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
    schools: list[dict[str, object]] | None = None,
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
    schools_path = root / "schools.csv"
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
            "primary_department_id",
            "primary_department_source_url",
            "primary_department_verified_at",
            "primary_department_basis",
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
        (
            "department_id",
            "school_id",
            "title",
            "directory_url",
            "reviewed_at",
            "review_notes",
        ),
        [{"school_id": "engineering", **row} for row in department_rows],
    )
    _write(
        schools_path,
        ("school_id", "title", "directory_url"),
        schools
        or [
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://example.test/engineering",
            },
            {
                "school_id": "scs",
                "title": "School of Computer Science",
                "directory_url": "https://example.test/scs",
            },
        ],
    )
    return people_path, memberships_path, departments_path
