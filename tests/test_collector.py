import datetime as dt
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import map_of_research.collector as collector
from map_of_research.collector import (
    ScholarAccessRefused,
    _fail_fast_scholarly,
    _parse_timestamp,
    collect_profiles,
    fetch_author_publications,
    load_collection_state,
    normalize_publications,
    select_profiles,
)
from map_of_research.io import atomic_write_json, load_json
from map_of_research.registry import load_registry, unique_profiles
from tests.registry_helpers import write_registry

NOW = dt.datetime(2026, 7, 17, 12, 0, tzinfo=dt.UTC)


class ProxyManager:
    def __init__(self, has_proxy: bool = False) -> None:
        self._has_proxy = has_proxy

    def has_proxy(self) -> bool:
        return self._has_proxy


class Session:
    def __init__(self) -> None:
        self.event_hooks = {"request": [], "response": []}


class Navigator:
    def __init__(self) -> None:
        self.pm1 = ProxyManager()
        self.pm2 = ProxyManager()
        self._session1 = Session()
        self._session2 = Session()

    @staticmethod
    def _requests_has_captcha(text: str) -> bool:
        return "captcha" in text.casefold()


class ScholarlyClient:
    def __init__(self) -> None:
        self._Scholarly__nav = Navigator()
        self.retries = None
        self.timeout = None

    def set_retries(self, value: int) -> None:
        self.retries = value

    def set_timeout(self, value: int) -> None:
        self.timeout = value


def test_scholarly_guard_stops_first_refusal_without_proxy() -> None:
    class Response:
        status_code = 403
        text = "forbidden"

        @staticmethod
        def read() -> None:
            return None

    client = ScholarlyClient()
    with (
        pytest.raises(ScholarAccessRefused, match="HTTP 403"),
        _fail_fast_scholarly(client),
    ):
        hook = client._Scholarly__nav._session1.event_hooks["response"][0]
        hook(Response())

    assert client.retries == 1
    assert client.timeout == 10
    assert client._Scholarly__nav._session1.event_hooks["response"] == []
    assert client._Scholarly__nav._session2.event_hooks["response"] == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda client: setattr(client, "_Scholarly__nav", None), "fail-fast guard"),
        (lambda client: setattr(client._Scholarly__nav, "pm1", None), "no-proxy"),
        (
            lambda client: setattr(client._Scholarly__nav, "pm1", ProxyManager(True)),
            "does not permit proxies",
        ),
        (
            lambda client: setattr(client._Scholarly__nav, "_session1", None),
            "fail-fast",
        ),
        (
            lambda client: setattr(
                client._Scholarly__nav,
                "_requests_has_captcha",
                None,
            ),
            "CAPTCHA guard",
        ),
    ],
)
def test_scholarly_guard_refuses_unverifiable_clients(mutate, message) -> None:
    client = ScholarlyClient()
    mutate(client)
    with pytest.raises(RuntimeError, match=message), _fail_fast_scholarly(client):
        pass


@pytest.mark.parametrize(
    ("status_code", "text", "message"),
    [(429, "slow down", "HTTP 429"), (200, "Captcha", "CAPTCHA")],
)
def test_scholarly_guard_stops_rate_limits_and_captchas(
    status_code,
    text,
    message,
) -> None:
    reads = []
    response = SimpleNamespace(
        status_code=status_code,
        text=text,
        read=lambda: reads.append(True),
    )
    client = ScholarlyClient()
    with (
        pytest.raises(ScholarAccessRefused, match=message),
        _fail_fast_scholarly(client),
    ):
        client._Scholarly__nav._session1.event_hooks["response"][0](response)
    assert reads == ([True] if status_code == 200 else [])


def test_scholarly_guard_ignores_normal_responses_and_preserves_other_hooks() -> None:
    client = ScholarlyClient()

    def sentinel(response) -> None:
        return None

    client._Scholarly__nav._session1.event_hooks["response"].append(sentinel)
    reads = []
    response = SimpleNamespace(
        status_code=200,
        text="normal profile",
        read=lambda: reads.append(True),
    )
    with _fail_fast_scholarly(client):
        hook = client._Scholarly__nav._session1.event_hooks["response"][-1]
        hook(response)
        hook(SimpleNamespace(status_code=204))
        client._Scholarly__nav._session2.event_hooks["response"].remove(hook)
    assert client._Scholarly__nav._session1.event_hooks["response"] == [sentinel]
    assert reads == [True]


def test_timestamp_parsing_is_utc_and_fail_closed() -> None:
    minimum = dt.datetime.min.replace(tzinfo=dt.UTC)
    assert _parse_timestamp(None) == minimum
    assert _parse_timestamp("not-a-time") == minimum
    assert _parse_timestamp("2026-01-02T03:04:05") == dt.datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC
    )
    assert _parse_timestamp("2026-01-02T03:04:05-05:00").hour == 8


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


def test_collection_state_initializes_and_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    assert load_collection_state(path) == {"schema_version": 1, "profiles": {}}
    invalid_documents = [
        [],
        {},
        {"schema_version": 99, "profiles": {}},
        {"schema_version": 1},
    ]
    for index, document in enumerate(invalid_documents):
        candidate = tmp_path / f"invalid-{index}.json"
        atomic_write_json(candidate, document)
        with pytest.raises(ValueError):
            load_collection_state(candidate)


def test_selection_validates_limits_and_skips_fresh_cache(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = make_registry(
        tmp_path / "registry"
    )
    profiles = unique_profiles(
        load_registry(people_path, memberships_path, departments_path)
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for profile in profiles:
        atomic_write_json(cache_dir / f"{profile.scholar_id}.json", {})
    fresh = NOW.isoformat()
    state = {
        "profiles": {
            profiles[0].scholar_id: {"last_success_at_utc": fresh},
            profiles[1].scholar_id: "malformed",
        }
    }
    selected = select_profiles(
        profiles,
        state,
        cache_dir,
        now=NOW,
        min_age_days=1,
        max_profiles=2,
    )
    assert [item.scholar_id for item in selected] == [profiles[1].scholar_id]
    assert (
        len(
            select_profiles(
                profiles,
                state,
                cache_dir,
                now=NOW,
                min_age_days=1,
                max_profiles=2,
                force=True,
            )
        )
        == 2
    )
    for kwargs in (
        {"min_age_days": -1, "max_profiles": 1},
        {"min_age_days": 1, "max_profiles": 0},
        {"min_age_days": 1, "max_profiles": collector.HARD_MAX_PROFILES + 1},
    ):
        with pytest.raises(ValueError):
            select_profiles(profiles, state, cache_dir, now=NOW, **kwargs)
    with pytest.raises(ValueError, match="must be a mapping"):
        select_profiles(
            profiles,
            {"profiles": []},
            cache_dir,
            now=NOW,
            min_age_days=1,
            max_profiles=1,
        )


def test_publication_normalization_is_stable_json_and_deduplicated() -> None:
    publications = [
        {
            "author_pub_id": "id:2",
            "bib": {
                "title": " Beta ",
                "authors": ("A", "B"),
                "score": math.nan,
                "nested": {1: object()},
            },
            "num_citations": math.inf,
        },
        {
            "author_pub_id": "id:1",
            "title": "Alpha",
            "pub_url": "https://example.test/alpha",
        },
        {"author_pub_id": "id:1", "bib": {"title": "Duplicate"}},
        {"author_pub_id": "", "bib": {"title": "Missing ID"}},
        {"author_pub_id": "id:3", "bib": {"title": ""}},
    ]
    normalized = normalize_publications(publications)
    assert [record["author_pub_id"] for record in normalized] == ["id:1", "id:2"]
    assert normalized[1]["authors"] == ["A", "B"]
    assert normalized[1]["score"] is None
    assert normalized[1]["num_citations"] is None
    assert isinstance(normalized[1]["nested"]["1"], str)


def test_fetch_author_publications_validates_each_scholarly_outcome(
    monkeypatch,
) -> None:
    profile = SimpleNamespace(scholar_id="alphaAAAAJ")

    class Client(ScholarlyClient):
        def __init__(self, author, filled):
            super().__init__()
            self.author = author
            self.filled = filled

        def search_author_id(self, scholar_id):
            return self.author

        def fill(self, author, **kwargs):
            assert kwargs == {"sections": ["publications"], "publication_limit": 4}
            return self.filled

    outcomes = [
        (None, None, "was not found"),
        ({"id": 1}, None, "could not be filled"),
        ({"id": 1}, {"publications": {}}, "invalid publication list"),
        ({"id": 1}, {"publications": []}, "no usable publications"),
    ]
    for author, filled, message in outcomes:
        client = Client(author, filled)
        monkeypatch.setitem(sys.modules, "scholarly", SimpleNamespace(scholarly=client))
        with pytest.raises(RuntimeError, match=message):
            fetch_author_publications(profile, publication_limit=4)

    client = Client(
        {"id": 1},
        {"publications": [{"author_pub_id": "id:1", "bib": {"title": "One"}}]},
    )
    monkeypatch.setitem(sys.modules, "scholarly", SimpleNamespace(scholarly=client))
    assert fetch_author_publications(profile, publication_limit=4)[0]["title"] == "One"


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


def test_collection_supports_dry_run_and_noop(tmp_path: Path) -> None:
    paths = make_registry(tmp_path / "registry")
    common = {
        "people_path": paths[0],
        "memberships_path": paths[1],
        "departments_path": paths[2],
        "cache_dir": tmp_path / "cache",
        "state_path": tmp_path / "state.json",
        "status_path": tmp_path / "status.json",
        "now": NOW,
    }
    planned = collect_profiles(**common, dry_run=True)
    assert planned.status == "planned"
    assert planned.dry_run is True
    assert planned.exit_code == 0

    common["cache_dir"].mkdir()
    for scholar_id in ("alphaAAAAJ", "betaAAAAJ"):
        atomic_write_json(common["cache_dir"] / f"{scholar_id}.json", {})
    atomic_write_json(
        common["state_path"],
        {
            "schema_version": 1,
            "profiles": {
                scholar_id: {"last_success_at_utc": NOW.isoformat()}
                for scholar_id in ("alphaAAAAJ", "betaAAAAJ")
            },
        },
    )
    noop = collect_profiles(**common)
    assert noop.status == "noop"
    assert noop.selected_profile_ids == ()


def test_collection_delays_between_profiles_and_can_finish_partial(
    tmp_path: Path,
) -> None:
    paths = make_registry(tmp_path / "registry")
    sleeps = []
    calls = []

    def fetcher(profile, *, publication_limit):
        calls.append((profile.scholar_id, publication_limit))
        if len(calls) == 2:
            raise RuntimeError("temporary failure")
        return [{"title": "One", "author_pub_id": f"{profile.scholar_id}:one"}]

    result = collect_profiles(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=tmp_path / "cache",
        state_path=tmp_path / "state.json",
        status_path=tmp_path / "status.json",
        max_profiles=2,
        publication_limit=3,
        request_delay_seconds=2.5,
        now=NOW,
        fetcher=fetcher,
        sleeper=sleeps.append,
    )
    assert result.status == "partial"
    assert result.exit_code == 1
    assert result.blocked is False
    assert sleeps == [2.5]
    assert [item[0] for item in calls] == ["alphaAAAAJ", "betaAAAAJ"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"publication_limit": 0},
        {"publication_limit": collector.HARD_PUBLICATION_LIMIT + 1},
        {"request_delay_seconds": -0.1},
    ],
)
def test_collection_validates_request_budget(tmp_path: Path, kwargs) -> None:
    with pytest.raises(ValueError):
        collect_profiles(status_path=tmp_path / "status.json", **kwargs)


def test_workflow_url_compatibility_entrypoint_and_parser(monkeypatch) -> None:
    for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    assert collector._workflow_url() is None
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    assert (
        collector._workflow_url() == "https://github.test/owner/repo/actions/runs/123"
    )
    expected = collector.CollectionResult("noop", (), (), (), False, False)
    monkeypatch.setattr(collector, "collect_profiles", lambda **kwargs: expected)
    assert collector.scrape_faculty_data() is expected
    assert collector._parse_args(["--force", "--dry-run"]).force is True


@pytest.mark.parametrize(("raises", "exit_code"), [(False, 0), (True, 1)])
def test_main_reports_startup_failures(
    monkeypatch,
    tmp_path: Path,
    raises,
    exit_code,
) -> None:
    status_path = tmp_path / "status.json"

    def run(**kwargs):
        if raises:
            raise RuntimeError("cannot start")
        return collector.CollectionResult("success", (), (), (), False, False)

    monkeypatch.setattr(collector, "collect_profiles", run)
    result = collector.main(["--status-file", str(status_path)])
    assert result == exit_code
    if raises:
        status = json.loads(status_path.read_text())
        assert status["status"] == "failure"
        assert status["error_type"] == "RuntimeError"
