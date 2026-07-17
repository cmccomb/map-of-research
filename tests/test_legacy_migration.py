import json
from pathlib import Path

from map_of_research.collector import normalize_publications
from scripts import migrate_legacy_cache
from tests.registry_helpers import write_registry


def test_normalizer_accepts_flattened_legacy_publications() -> None:
    publications = normalize_publications(
        [
            {
                "title": "Legacy Paper",
                "author": "A. Author",
                "pub_year": "2024",
                "author_pub_id": "authorAAAAJ:paper",
                "num_citations": 5,
            }
        ]
    )

    assert publications == [
        {
            "title": "Legacy Paper",
            "author": "A. Author",
            "pub_year": "2024",
            "author_pub_id": "authorAAAAJ:paper",
            "num_citations": 5,
            "pub_url": "",
        }
    ]


def test_migration_provenance_is_relative(tmp_path: Path, monkeypatch) -> None:
    repos_root = tmp_path / "repos"
    source_path = repos_root / "map-of-ece/data/Alpha.json"
    source_path.parent.mkdir(parents=True)
    people_path, memberships_path, maps_path = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-alpha",
                "display_name": "Alpha Person",
                "scholar_id": "alphaAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-alpha",
                "map_slug": "map-of-ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Alpha",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )
    monkeypatch.setattr(
        migrate_legacy_cache,
        "_candidate",
        lambda *_: (
            source_path,
            "2026-07-17T12:00:00+00:00",
            [
                {
                    "title": "A Paper",
                    "author_pub_id": "alphaAAAAJ:paper",
                    "num_citations": 1,
                }
            ],
        ),
    )
    cache_dir = tmp_path / "cache"

    migrate_legacy_cache.migrate(
        repos_root=repos_root,
        people_path=people_path,
        memberships_path=memberships_path,
        maps_path=maps_path,
        cache_dir=cache_dir,
        state_path=tmp_path / "state.json",
    )
    cache = json.loads((cache_dir / "alphaAAAAJ.json").read_text())

    assert cache["migration_source"] == "map-of-ece/data/Alpha.json"
