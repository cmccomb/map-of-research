"""Fail closed before publishing an incomplete Mathematical Sciences profile."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .collector import COMPLETE_VISIBLE_PROFILE
from .io import load_json, sha256_file
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    Registry,
    load_registry,
)
from .snapshot import DEFAULT_MANIFEST_PATH, DEFAULT_SNAPSHOT_PATH, validate_snapshot

MATH_DEPARTMENT_ID = "math"
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_CAPTURE_DIR = Path("data/captures")


class _VisibleRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.more_disabled = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr" and "gsc_a_tr" in (attributes.get("class") or "").split():
            self.count += 1
        if attributes.get("id") == "gsc_bpf_more":
            self.more_disabled = "disabled" in attributes


def _require(condition: bool, person: str, message: str) -> None:
    if not condition:
        raise ValueError(f"Math release gate: {person}: {message}")


def _capture_file(
    capture_dir: Path,
    capture: Mapping[str, Any],
    kind: str,
    person: str,
) -> Path:
    filename = capture.get(f"source_{kind}_filename")
    digest = capture.get(f"source_{kind}_sha256")
    _require(
        isinstance(filename, str)
        and filename == Path(filename).name
        and filename.endswith(f".{kind}"),
        person,
        f"invalid saved {kind} filename",
    )
    path = capture_dir / filename
    _require(path.is_file(), person, f"saved {kind} capture is missing: {filename}")
    _require(
        isinstance(digest, str) and sha256_file(path) == digest,
        person,
        f"saved {kind} capture checksum differs: {filename}",
    )
    return path


def _citation_id(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    values = parse_qs(urlparse(url).query).get("citation_for_view", [])
    return values[0] if len(values) == 1 else ""


def validate_math_coverage(
    frame: Any,
    registry: Registry,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    capture_dir: Path = DEFAULT_CAPTURE_DIR,
) -> dict[str, int]:
    """Check every registered Math ID against its full saved Scholar capture.

    A person without a verified Scholar ID stays in the roster and is counted
    as an identity gap. Other departments' cache coverage is not gated here.
    """

    if MATH_DEPARTMENT_ID not in registry.departments_by_id:
        raise ValueError("Math release gate: Mathematical Sciences is not registered")
    people = registry.people_by_id
    math_ids = sorted(
        {
            membership.person_id
            for membership in registry.memberships
            if membership.department_id == MATH_DEPARTMENT_ID and membership.included
        }
    )
    if not math_ids:
        raise ValueError("Math release gate: no included Mathematical Sciences people")

    complete = 0
    identity_gaps = 0
    publication_rows = 0
    for person_id in math_ids:
        person = people[person_id]
        home = registry.primary_affiliation(person_id)
        _require(
            bool(home.home_id and home.source_url and home.verified_at),
            person.display_name,
            "primary academic home is unresolved",
        )
        if not person.scholar_id:
            identity_gaps += 1
            continue

        scholar_id = person.scholar_id
        cache_path = cache_dir / f"{scholar_id}.json"
        _require(cache_path.is_file(), person.display_name, "Scholar cache is missing")
        cache = load_json(cache_path)
        _require(
            isinstance(cache, dict)
            and cache.get("schema_version") == 1
            and cache.get("scholar_id") == scholar_id,
            person.display_name,
            "Scholar cache identity or schema differs",
        )
        publications = cache.get("publications")
        capture = cache.get("browser_capture")
        _require(
            isinstance(publications, list)
            and bool(publications)
            and isinstance(capture, dict)
            and capture.get("coverage") == COMPLETE_VISIBLE_PROFILE
            and capture.get("more_remain") is False,
            person.display_name,
            "a complete visible-profile capture is required",
        )
        count = len(publications)
        _require(
            cache.get("publication_count") == count
            and capture.get("displayed_count") == count,
            person.display_name,
            "cache row counts differ from the completed capture",
        )
        json_path = _capture_file(capture_dir, capture, "json", person.display_name)
        html_path = _capture_file(capture_dir, capture, "html", person.display_name)
        raw = load_json(json_path)
        _require(
            isinstance(raw, dict)
            and raw.get("scholar_user_id") == scholar_id
            and raw.get("more_remain") is False
            and raw.get("challenge") is False
            and not raw.get("page_error")
            and raw.get("displayed_count") == count
            and isinstance(raw.get("works"), list)
            and len(raw["works"]) == count,
            person.display_name,
            "saved Scholar JSON is incomplete or belongs to another profile",
        )
        html = _VisibleRows()
        html.feed(html_path.read_text(encoding="utf-8"))
        _require(
            html.count == count and html.more_disabled,
            person.display_name,
            "saved Scholar HTML rows or Show More state differ",
        )

        cache_ids: set[str] = set()
        cache_indices: set[int] = set()
        for publication in publications:
            _require(
                isinstance(publication, dict),
                person.display_name,
                "invalid cached publication",
            )
            citation_id = publication.get("author_pub_id")
            index = publication.get("capture_row_index")
            _require(
                isinstance(citation_id, str)
                and citation_id.startswith(f"{scholar_id}:")
                and _citation_id(publication.get("pub_url")) == citation_id
                and isinstance(index, int)
                and not isinstance(index, bool),
                person.display_name,
                "cached publication lacks a valid Scholar row identity",
            )
            cache_ids.add(citation_id)
            cache_indices.add(index)
        _require(
            len(cache_ids) == count and cache_indices == set(range(1, count + 1)),
            person.display_name,
            "cached citations or capture row indices are missing or duplicated",
        )

        raw_ids: set[str] = set()
        raw_indices: set[int] = set()
        for work in raw["works"]:
            _require(isinstance(work, dict), person.display_name, "invalid raw row")
            citation_id = _citation_id(work.get("citation_url"))
            index = work.get("index")
            _require(
                citation_id.startswith(f"{scholar_id}:")
                and isinstance(index, int)
                and not isinstance(index, bool),
                person.display_name,
                "raw Scholar row identity or index is invalid",
            )
            raw_ids.add(citation_id)
            raw_indices.add(index)
        _require(
            raw_ids == cache_ids
            and len(raw_ids) == count
            and raw_indices == set(range(1, count + 1)),
            person.display_name,
            "raw Scholar citation IDs or row indices differ from cache",
        )

        rows = frame.loc[
            (frame["source_type"] == "scholar") & (frame["scholar_id"] == scholar_id)
        ]
        snapshot_by_id = rows.set_index("author_pub_id").to_dict("index")
        _require(
            len(rows) == count
            and set(snapshot_by_id) == cache_ids
            and rows["person_id"].eq(person_id).all(),
            person.display_name,
            "snapshot omits or misattributes Scholar rows",
        )
        for publication in publications:
            citation_id = publication["author_pub_id"]
            _require(
                json.loads(snapshot_by_id[citation_id]["source_record_json"])
                == publication,
                person.display_name,
                f"snapshot Scholar record differs from cache: {citation_id}",
            )
        complete += 1
        publication_rows += count

    return {
        "math_people": len(math_ids),
        "complete_scholar_profiles": complete,
        "unresolved_scholar_id_people": identity_gaps,
        "complete_scholar_rows": publication_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate Math Scholar completeness")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument("--memberships", type=Path, default=DEFAULT_MEMBERSHIPS_PATH)
    parser.add_argument("--departments", type=Path, default=DEFAULT_DEPARTMENTS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    args = parser.parse_args(argv)
    registry = load_registry(args.people, args.memberships, args.departments)
    frame, _ = validate_snapshot(
        args.snapshot,
        args.manifest,
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
    )
    result = validate_math_coverage(
        frame,
        registry,
        cache_dir=args.cache_dir,
        capture_dir=args.capture_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0
