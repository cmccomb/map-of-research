import datetime as dt
from pathlib import Path

from map_of_research.collector import (
    collect_profiles,
    load_collection_state,
    select_profiles,
)
from map_of_research.io import atomic_write_json, load_json
from map_of_research.registry import load_registry, unique_profiles
from tests.registry_helpers import write_registry

NOW = dt.datetime(2026, 7, 17, 12, 0, tzinfo=dt.UTC)


def make_registry(root: Path) -> tuple[Path, Path, Path]:
    people = []
    memberships = []
    for name in ("Alpha", "Beta"):
        person_id = f"person-{name.casefold()}"
        scholar_id = f"{name.casefold()}AAAAJ"
        people.append(
            {
                "person_id": person_id,
                "display_name": name,
                "scholar_id": scholar_id,
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        )
        memberships.append(
            {
                "person_id": person_id,
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": name,
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        )
    return write_registry(root, people=people, memberships=memberships)


def test_selection_prioritizes_missing_cache_then_oldest_attempt(
    tmp_path: Path,
) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    cache_dir = tmp_path / "authors"
    cache_dir.mkdir()
    profiles = unique_profiles(
        load_registry(people_path, memberships_path, departments_path)
    )
    atomic_write_json(cache_dir / "betaAAAAJ.json", {})
    state = {
        "schema_version": 1,
        "profiles": {
            "alphaAAAAJ": {"last_attempt_at_utc": "2026-07-16T00:00:00+00:00"},
            "betaAAAAJ": {
                "last_attempt_at_utc": "2020-01-01T00:00:00+00:00",
                "last_success_at_utc": "2020-01-01T00:00:00+00:00",
            },
        },
    }

    selected = select_profiles(
        profiles,
        state,
        cache_dir,
        now=NOW,
        min_age_days=365,
        max_profiles=2,
    )

    assert [profile.scholar_id for profile in selected] == [
        "alphaAAAAJ",
        "betaAAAAJ",
    ]


def test_failed_fetch_preserves_cache_and_stops_run(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    cache_dir = tmp_path / "authors"
    state_path = tmp_path / "authors.json"
    status_path = tmp_path / "last-collection.json"
    cache_dir.mkdir()
    sentinel_path = cache_dir / "alphaAAAAJ.json"
    sentinel_path.write_text('{"last_good": true}\n', encoding="utf-8")
    (cache_dir / "betaAAAAJ.json").write_text(
        '{"last_good": true}\n',
        encoding="utf-8",
    )

    class MaxTriesExceededException(RuntimeError):
        pass

    calls: list[str] = []

    def blocked_fetcher(profile, *, publication_limit):
        calls.append(profile.scholar_id)
        raise MaxTriesExceededException("blocked")

    result = collect_profiles(
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        cache_dir=cache_dir,
        state_path=state_path,
        status_path=status_path,
        min_age_days=0,
        max_profiles=2,
        request_delay_seconds=0,
        now=NOW,
        fetcher=blocked_fetcher,
    )

    assert result.status == "failure"
    assert result.blocked is True
    assert calls == ["alphaAAAAJ"]
    assert sentinel_path.read_text(encoding="utf-8") == '{"last_good": true}\n'
    assert load_json(status_path)["failed_profile_ids"] == ["alphaAAAAJ"]
    assert (
        load_collection_state(state_path)["profiles"]["alphaAAAAJ"]["last_error_type"]
        == "MaxTriesExceededException"
    )


def test_successful_collection_writes_atomic_cache_and_state(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    cache_dir = tmp_path / "authors"
    state_path = tmp_path / "authors.json"
    status_path = tmp_path / "last-collection.json"

    def fetcher(profile, *, publication_limit):
        return [
            {
                "title": f"Paper by {profile.display_name}",
                "author_pub_id": f"{profile.scholar_id}:pub",
                "num_citations": 3,
            }
        ]

    result = collect_profiles(
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        cache_dir=cache_dir,
        state_path=state_path,
        status_path=status_path,
        max_profiles=1,
        request_delay_seconds=0,
        now=NOW,
        fetcher=fetcher,
    )

    assert result.status == "success"
    assert result.refreshed_profile_ids == ("alphaAAAAJ",)
    cache = load_json(cache_dir / "alphaAAAAJ.json")
    assert cache["publication_count"] == 1
    assert cache["publications"][0]["num_citations"] == 3
