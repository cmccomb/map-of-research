"""Cache-first, bounded collection of public Scholar profile metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    AuthorProfile,
    load_registry,
    unique_profiles,
)

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_STATE_PATH = Path("status/authors.json")
DEFAULT_STATUS_PATH = Path("status/last-collection.json")
DEFAULT_MIN_AGE_DAYS = 365
DEFAULT_MAX_PROFILES = 1
HARD_MAX_PROFILES = 5
DEFAULT_PUBLICATION_LIMIT = 200
HARD_PUBLICATION_LIMIT = 500
DEFAULT_REQUEST_DELAY_SECONDS = 90.0
SCHOLARLY_REQUEST_TIMEOUT_SECONDS = 10


class ScholarAccessRefused(RuntimeError):
    """Scholar explicitly refused an automated request."""


class _StopScholarAccess(BaseException):
    """Escape scholarly's internal retry loop after the first refusal."""


@contextmanager
def _fail_fast_scholarly(client: Any) -> Iterator[None]:
    """Make pinned scholarly fail closed on a block without using proxies.

    Scholarly 1.7.11 retries HTTP 403 responses internally, including long
    sleeps that do not advance its retry counter. Response hooks let us stop
    before that retry path while continuing to use scholarly for profile
    lookup and parsing. The dependency is pinned, and this guard deliberately
    refuses to run if the expected no-proxy sessions are unavailable.
    """

    client.set_retries(1)
    client.set_timeout(SCHOLARLY_REQUEST_TIMEOUT_SECONDS)
    navigator = getattr(client, "_Scholarly__nav", None)
    if navigator is None:
        raise RuntimeError("Cannot install the scholarly fail-fast guard")

    proxy_managers = [getattr(navigator, name, None) for name in ("pm1", "pm2")]
    if any(manager is None for manager in proxy_managers):
        raise RuntimeError("Cannot verify scholarly's no-proxy configuration")
    if any(manager.has_proxy() for manager in proxy_managers):
        raise RuntimeError("Scholar collection does not permit proxies")

    sessions = [getattr(navigator, name, None) for name in ("_session1", "_session2")]
    if any(
        session is None
        or not isinstance(getattr(session, "event_hooks", None), dict)
        or "response" not in session.event_hooks
        for session in sessions
    ):
        raise RuntimeError("Cannot install the scholarly fail-fast guard")

    captcha_detector = getattr(navigator, "_requests_has_captcha", None)
    if not callable(captcha_detector):
        raise RuntimeError("Cannot install the scholarly CAPTCHA guard")

    def stop_on_refusal(response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code in {403, 429}:
            raise _StopScholarAccess(f"Scholar returned HTTP {status_code}")
        if status_code == 200:
            response.read()
            if captcha_detector(response.text):
                raise _StopScholarAccess("Scholar returned a CAPTCHA")

    for session in sessions:
        session.event_hooks["response"].append(stop_on_refusal)
    try:
        yield
    except _StopScholarAccess as error:
        raise ScholarAccessRefused(str(error)) from None
    finally:
        for session in sessions:
            hooks = session.event_hooks["response"]
            if stop_on_refusal in hooks:
                hooks.remove(stop_on_refusal)


@dataclass(frozen=True)
class CollectionResult:
    """Observable outcome of one bounded collection pass."""

    status: str
    selected_profile_ids: tuple[str, ...]
    refreshed_profile_ids: tuple[str, ...]
    failed_profile_ids: tuple[str, ...]
    blocked: bool
    dry_run: bool

    @property
    def exit_code(self) -> int:
        return 0 if self.status in {"success", "noop", "planned"} else 1


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _isoformat(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat()


def _parse_timestamp(value: Any) -> dt.datetime:
    if not isinstance(value, str) or not value:
        return dt.datetime.min.replace(tzinfo=dt.UTC)
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return dt.datetime.min.replace(tzinfo=dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def load_collection_state(path: Path) -> dict[str, Any]:
    """Load strict resumable state, or initialize a new state document."""

    if not path.exists():
        return {"schema_version": STATE_SCHEMA_VERSION, "profiles": {}}
    state = load_json(path)
    if not isinstance(state, dict):
        raise ValueError("Collection state must be a JSON object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Collection state schema is unsupported")
    profiles = state.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Collection state profiles must be a JSON object")
    return state


def select_profiles(
    profiles: Sequence[AuthorProfile],
    state: Mapping[str, Any],
    cache_dir: Path,
    *,
    now: dt.datetime,
    min_age_days: int,
    max_profiles: int,
    force: bool = False,
) -> list[AuthorProfile]:
    """Select missing or stale profiles, oldest attempted first."""

    if min_age_days < 0:
        raise ValueError("min_age_days cannot be negative")
    if not 1 <= max_profiles <= HARD_MAX_PROFILES:
        raise ValueError(f"max_profiles must be between 1 and {HARD_MAX_PROFILES}")

    profile_state = state.get("profiles", {})
    if not isinstance(profile_state, Mapping):
        raise ValueError("Collection state profiles must be a mapping")
    cutoff = now - dt.timedelta(days=min_age_days)
    candidates: list[tuple[bool, dt.datetime, str, AuthorProfile]] = []
    for profile in profiles:
        cached = (cache_dir / f"{profile.scholar_id}.json").is_file()
        record = profile_state.get(profile.scholar_id, {})
        if not isinstance(record, Mapping):
            record = {}
        last_success = _parse_timestamp(record.get("last_success_at_utc"))
        if cached and not force and last_success > cutoff:
            continue
        last_attempt = _parse_timestamp(record.get("last_attempt_at_utc"))
        candidates.append((cached, last_attempt, profile.scholar_id, profile))

    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates[:max_profiles]]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def normalize_publications(
    publications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce Scholar publication stubs to stable JSON-compatible records."""

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for publication in publications:
        bibliography = publication.get("bib")
        if not isinstance(bibliography, Mapping):
            # Legacy map repositories already flattened ``bib`` into each row.
            bibliography = publication
        title = str(bibliography.get("title") or "").strip()
        author_pub_id = str(publication.get("author_pub_id") or "").strip()
        if not title or not author_pub_id or author_pub_id in seen_ids:
            continue
        seen_ids.add(author_pub_id)
        record = {str(key): _json_safe(value) for key, value in bibliography.items()}
        record["title"] = title
        record["author_pub_id"] = author_pub_id
        record["num_citations"] = _json_safe(publication.get("num_citations", 0))
        record["pub_url"] = _json_safe(
            publication.get("pub_url", bibliography.get("pub_url", ""))
        )
        normalized.append(record)
    return sorted(
        normalized,
        key=lambda record: (
            str(record["title"]).casefold(),
            str(record["author_pub_id"]),
        ),
    )


def fetch_author_publications(
    profile: AuthorProfile,
    *,
    publication_limit: int,
) -> list[dict[str, Any]]:
    """Fetch one public profile without proxying, searching, or retries."""

    from scholarly import scholarly

    LOGGER.info("Fetching Scholar profile %s", profile.scholar_id)
    with _fail_fast_scholarly(scholarly):
        author = scholarly.search_author_id(profile.scholar_id)
        if not author:
            raise RuntimeError(f"Scholar profile {profile.scholar_id} was not found")
        filled_author = scholarly.fill(
            author,
            sections=["publications"],
            publication_limit=publication_limit,
        )
    if not filled_author:
        raise RuntimeError(f"Scholar profile {profile.scholar_id} could not be filled")
    publications = filled_author.get("publications", [])
    if not isinstance(publications, list):
        raise RuntimeError("Scholar returned an invalid publication list")
    normalized = normalize_publications(publications)
    if not normalized:
        raise RuntimeError("Scholar returned no usable publications")
    return normalized


def _looks_blocked(error: BaseException) -> bool:
    """Recognize scholarly's stop signals without depending on private imports."""

    return type(error).__name__ in {
        "DOSException",
        "MaxTriesExceededException",
        "CaptchaException",
        "ScholarAccessRefused",
    }


def _workflow_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def collect_profiles(
    *,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
    min_age_days: int = DEFAULT_MIN_AGE_DAYS,
    max_profiles: int = DEFAULT_MAX_PROFILES,
    publication_limit: int = DEFAULT_PUBLICATION_LIMIT,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    force: bool = False,
    dry_run: bool = False,
    now: dt.datetime | None = None,
    fetcher: Callable[..., list[dict[str, Any]]] = fetch_author_publications,
    sleeper: Callable[[float], None] = time.sleep,
) -> CollectionResult:
    """Refresh a bounded set of profiles and preserve the last good cache."""

    if not 1 <= publication_limit <= HARD_PUBLICATION_LIMIT:
        raise ValueError(
            f"publication_limit must be between 1 and {HARD_PUBLICATION_LIMIT}"
        )
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    started_at = now or utc_now()
    registry = load_registry(people_path, memberships_path, departments_path)
    # Retain excluded and historical people in snapshots, but do not spend
    # Scholar requests refreshing profiles outside the included faculty catalog.
    profiles = unique_profiles(registry, included_only=True)
    state = load_collection_state(state_path)
    selected = select_profiles(
        profiles,
        state,
        cache_dir,
        now=started_at,
        min_age_days=min_age_days,
        max_profiles=max_profiles,
        force=force,
    )
    selected_ids = tuple(profile.scholar_id for profile in selected)

    base_status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": _isoformat(started_at),
        "finished_at_utc": None,
        "workflow_url": _workflow_url(),
        "registry_people": len(registry.people),
        "registry_memberships": len(registry.memberships),
        "unique_profiles": len(profiles),
        "selected_profile_ids": list(selected_ids),
        "refreshed_profile_ids": [],
        "failed_profile_ids": [],
        "min_age_days": min_age_days,
        "max_profiles": max_profiles,
        "publication_limit": publication_limit,
        "request_delay_seconds": request_delay_seconds,
        "force": force,
        "dry_run": dry_run,
        "blocked": False,
    }

    if dry_run:
        base_status.update(
            {
                "status": "planned",
                "finished_at_utc": _isoformat(started_at),
            }
        )
        atomic_write_json(status_path, base_status)
        return CollectionResult("planned", selected_ids, (), (), False, True)

    if not selected:
        base_status.update(
            {
                "status": "noop",
                "finished_at_utc": _isoformat(started_at),
            }
        )
        atomic_write_json(status_path, base_status)
        return CollectionResult("noop", (), (), (), False, False)

    state_profiles = state["profiles"]
    refreshed: list[str] = []
    failed: list[str] = []
    blocked = False
    for index, profile in enumerate(selected):
        if index and request_delay_seconds:
            sleeper(request_delay_seconds)
        attempted_at = utc_now() if now is None else started_at
        record = state_profiles.setdefault(profile.scholar_id, {})
        record.update(
            {
                "display_name": profile.display_name,
                "last_attempt_at_utc": _isoformat(attempted_at),
            }
        )
        try:
            publications = fetcher(
                profile,
                publication_limit=publication_limit,
            )
            cache_payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "scholar_id": profile.scholar_id,
                "display_name": profile.display_name,
                "fetched_at_utc": _isoformat(attempted_at),
                "publication_count": len(publications),
                "publications": publications,
            }
            atomic_write_json(
                cache_dir / f"{profile.scholar_id}.json",
                cache_payload,
            )
            record.update(
                {
                    "last_success_at_utc": _isoformat(attempted_at),
                    "last_error_type": None,
                    "publication_count": len(publications),
                }
            )
            refreshed.append(profile.scholar_id)
        except Exception as error:  # stop after one failed profile by policy
            blocked = _looks_blocked(error)
            record.update(
                {
                    "last_error_type": type(error).__name__,
                    "last_error_message": str(error)[:500],
                }
            )
            failed.append(profile.scholar_id)
            LOGGER.exception("Scholar collection stopped after a profile failure")
        finally:
            atomic_write_json(state_path, state)
        if failed:
            break

    if failed and refreshed:
        outcome = "partial"
    elif failed:
        outcome = "failure"
    else:
        outcome = "success"
    finished_at = utc_now() if now is None else started_at
    base_status.update(
        {
            "status": outcome,
            "finished_at_utc": _isoformat(finished_at),
            "refreshed_profile_ids": refreshed,
            "failed_profile_ids": failed,
            "blocked": blocked,
        }
    )
    atomic_write_json(status_path, base_status)
    return CollectionResult(
        outcome,
        selected_ids,
        tuple(refreshed),
        tuple(failed),
        blocked,
        False,
    )


def scrape_faculty_data() -> CollectionResult:
    """Backward-compatible entry point for the original console command."""

    return collect_profiles()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument(
        "--memberships",
        type=Path,
        default=DEFAULT_MEMBERSHIPS_PATH,
    )
    parser.add_argument(
        "--departments",
        type=Path,
        default=DEFAULT_DEPARTMENTS_PATH,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS)
    parser.add_argument("--max-profiles", type=int, default=DEFAULT_MAX_PROFILES)
    parser.add_argument(
        "--publication-limit",
        type=int,
        default=DEFAULT_PUBLICATION_LIMIT,
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = _parse_args(argv)
    try:
        result = collect_profiles(
            people_path=args.people,
            memberships_path=args.memberships,
            departments_path=args.departments,
            cache_dir=args.cache_dir,
            state_path=args.state_file,
            status_path=args.status_file,
            min_age_days=args.min_age_days,
            max_profiles=args.max_profiles,
            publication_limit=args.publication_limit,
            request_delay_seconds=args.request_delay_seconds,
            force=args.force,
            dry_run=args.dry_run,
        )
        return result.exit_code
    except Exception as error:
        LOGGER.exception("Collection could not start")
        atomic_write_json(
            args.status_file,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "status": "failure",
                "started_at_utc": _isoformat(utc_now()),
                "finished_at_utc": _isoformat(utc_now()),
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "workflow_url": _workflow_url(),
            },
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
