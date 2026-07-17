import csv
from pathlib import Path

import pytest

from map_of_research.registry import load_registry, map_catalog, unique_profiles


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("map_slug", "department", "faculty", "scholar_id"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_registry_collapses_cross_appointments_and_preserves_aliases(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "faculty.csv"
    write_registry(
        registry_path,
        [
            {
                "map_slug": "map-of-bme",
                "department": "Biomedical Engineering",
                "faculty": "BYu",
                "scholar_id": "Fz3_tukAAAAJ",
            },
            {
                "map_slug": "map-of-ece",
                "department": "Electrical & Computer Engineering",
                "faculty": "Yu",
                "scholar_id": "Fz3_tukAAAAJ",
            },
            {
                "map_slug": "map-of-ece",
                "department": "Electrical & Computer Engineering",
                "faculty": "Missing ID",
                "scholar_id": "",
            },
        ],
    )

    memberships = load_registry(registry_path)
    profiles = unique_profiles(memberships)

    assert len(memberships) == 3
    assert len(profiles) == 1
    assert profiles[0].display_name == "Yu"
    assert profiles[0].map_slugs == ("map-of-bme", "map-of-ece")
    assert map_catalog(memberships) == {
        "map-of-bme": "Biomedical Engineering",
        "map-of-ece": "Electrical & Computer Engineering",
    }


def test_registry_rejects_conflicting_map_titles(tmp_path: Path) -> None:
    registry_path = tmp_path / "faculty.csv"
    write_registry(
        registry_path,
        [
            {
                "map_slug": "map-of-ece",
                "department": "ECE",
                "faculty": "One",
                "scholar_id": "oneAAAAJ",
            },
            {
                "map_slug": "map-of-ece",
                "department": "Different",
                "faculty": "Two",
                "scholar_id": "twoAAAAJ",
            },
        ],
    )

    with pytest.raises(ValueError, match="conflicting department"):
        load_registry(registry_path)
