"""Require complete Scholar source receipts for new or changed profile caches."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .io import load_json, sha256_file
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    load_registry,
    unique_profiles,
)

DEFAULT_BASELINE_PATH = Path("registry/legacy-scholar-cache-baseline.json")
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_CAPTURE_DIR = Path("data/captures")


def _normalized(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _citation_id(url: str) -> str:
    values = parse_qs(urlparse(url).query).get("citation_for_view", [])
    return values[0] if len(values) == 1 else ""


class _ProfileHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.row_count = 0
        self.citation_ids: list[str] = []
        self.more_buttons: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr" and "gsc_a_tr" in (attributes.get("class") or "").split():
            self.in_row = True
            self.row_count += 1
        elif tag == "a" and self.in_row:
            citation_id = _citation_id(attributes.get("href") or "")
            if citation_id:
                self.citation_ids.append(citation_id)
        elif tag == "button" and attributes.get("id") == "gsc_bpf_more":
            self.more_buttons.append("disabled" in attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr":
            self.in_row = False


def _source_file(captures_dir: Path, filename: Any, digest: Any) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Capture source filename must be a basename")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Capture source checksum is invalid")
    path = captures_dir / filename
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Capture source checksum mismatch: {path}")
    return path


def validate_complete_receipt(
    scholar_id: str, cache: dict[str, Any], captures_dir: Path
) -> None:
    """Reconcile a complete browser claim to saved rows and HTML evidence."""

    rows = cache.get("publications")
    receipt = cache.get("browser_capture")
    if not isinstance(rows, list) or not isinstance(receipt, dict):
        raise ValueError(f"Complete source receipt is missing: {scholar_id}")
    count = len(rows)
    if (
        receipt.get("coverage") != "complete_visible_profile"
        or receipt.get("more_remain") is not False
        or receipt.get("displayed_count") != count
        or cache.get("publication_count") != count
        or not isinstance(receipt.get("limit"), int)
        or receipt["limit"] < count
        or parse_qs(urlparse(str(receipt.get("profile_url") or "")).query).get("user")
        != [scholar_id]
    ):
        raise ValueError(f"Complete visible-profile claim is invalid: {scholar_id}")

    source_json = _source_file(
        captures_dir,
        receipt.get("source_json_filename"),
        receipt.get("source_json_sha256"),
    )
    source_html = _source_file(
        captures_dir,
        receipt.get("source_html_filename"),
        receipt.get("source_html_sha256"),
    )
    capture = load_json(source_json)
    if (
        not isinstance(capture, dict)
        or capture.get("scholar_user_id") != scholar_id
        or capture.get("displayed_count") != count
        or capture.get("more_remain") is not False
        or capture.get("challenge") not in (None, False, "")
        or capture.get("page_error") not in (None, False, "")
        or not isinstance(capture.get("works"), list)
        or len(capture["works"]) != count
    ):
        raise ValueError(f"Saved capture is incomplete or mismatched: {scholar_id}")

    html = _ProfileHTML()
    html.feed(source_html.read_text(encoding="utf-8"))
    if (
        html.row_count != count
        or len(html.citation_ids) != count
        or html.more_buttons != [True]
    ):
        raise ValueError(f"Saved HTML does not show a complete list: {scholar_id}")

    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid cached publication row: {scholar_id}")
        index = row.get("capture_row_index")
        if not isinstance(index, int) or index in by_index or not 1 <= index <= count:
            raise ValueError(f"Capture row indexes are invalid: {scholar_id}")
        by_index[index] = row

    for index, source_row in enumerate(capture["works"], start=1):
        if not isinstance(source_row, dict):
            raise ValueError(f"Invalid source publication row: {scholar_id}")
        cached_row = by_index[index]
        citation_url = str(source_row.get("citation_url") or "")
        citation_id = _citation_id(citation_url)
        citation_count = _normalized(source_row.get("citation_count")).replace(",", "")
        if (
            source_row.get("index") != index
            or not citation_id.startswith(f"{scholar_id}:")
            or citation_id != html.citation_ids[index - 1]
            or citation_id != cached_row.get("author_pub_id")
            or citation_url != cached_row.get("pub_url")
            or any(
                _normalized(cached_row.get(cache_key))
                != _normalized(source_row.get(source_key))
                for cache_key, source_key in (
                    ("title", "title"),
                    ("author", "authors"),
                    ("venue", "venue"),
                    ("pub_year", "year"),
                )
            )
            or (citation_count != "" and not citation_count.isdecimal())
            or cached_row.get("num_citations") != int(citation_count or "0")
        ):
            raise ValueError(
                f"Cached row differs from saved source: {scholar_id}:{index}"
            )


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    baseline = load_json(path)
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema_version") != 1
        or not isinstance(baseline.get("profiles"), dict)
        or not isinstance(baseline.get("frozen_snapshot_commit"), str)
        or not isinstance(baseline.get("frozen_snapshot_sha256"), str)
    ):
        raise ValueError("Legacy Scholar cache baseline is invalid")
    profiles = baseline["profiles"]
    for scholar_id, record in profiles.items():
        if (
            not isinstance(scholar_id, str)
            or not isinstance(record, dict)
            or not isinstance(record.get("person_id"), str)
            or not isinstance(record.get("included"), bool)
            or not isinstance(record.get("cache_sha256"), str)
            or not isinstance(record.get("author_pub_ids"), list)
            or record["author_pub_ids"] != sorted(set(record["author_pub_ids"]))
        ):
            raise ValueError(f"Legacy Scholar cache entry is invalid: {scholar_id}")
    return profiles


def validate_coverage(
    *,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    captures_dir: Path = DEFAULT_CAPTURE_DIR,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    snapshot_path: Path | None = None,
) -> dict[str, int]:
    """Permit frozen legacy caches and verify every newer Scholar source."""

    registry = load_registry(people_path, memberships_path, departments_path)
    baseline = _load_baseline(baseline_path)
    frame = None
    if snapshot_path is not None:
        import pandas

        frame = pandas.read_parquet(snapshot_path, engine="pyarrow")
        frame = frame.loc[frame["source_type"] == "scholar"]
    expected_profile_ids: set[str] = set()
    result = {"legacy_profiles": 0, "complete_profiles": 0}

    for profile in unique_profiles(registry):
        path = cache_dir / f"{profile.scholar_id}.json"
        if not path.is_file():
            continue
        cache = load_json(path)
        if (
            not isinstance(cache, dict)
            or cache.get("scholar_id") != profile.scholar_id
            or not isinstance(cache.get("publications"), list)
            or not isinstance(cache.get("fetched_at_utc"), str)
        ):
            raise ValueError(f"Invalid Scholar cache: {path}")
        expected_profile_ids.add(profile.scholar_id)
        prior = baseline.get(profile.scholar_id)
        if prior is not None:
            previous_ids = set(prior["author_pub_ids"])
            current_ids = {
                row.get("author_pub_id")
                for row in cache["publications"]
                if isinstance(row, dict)
            }
            if not previous_ids.issubset(current_ids):
                raise ValueError(
                    f"Previously published Scholar IDs disappeared: {path}"
                )
        included_now = any(membership.included for membership in profile.memberships)
        unchanged_legacy = (
            isinstance(prior, dict)
            and prior.get("person_id") == profile.person_id
            and prior.get("cache_sha256") == sha256_file(path)
            and (prior.get("included") is True or not included_now)
        )
        if unchanged_legacy:
            result["legacy_profiles"] += 1
        else:
            validate_complete_receipt(profile.scholar_id, cache, captures_dir)
            result["complete_profiles"] += 1

        if frame is not None:
            profile_rows = frame.loc[frame["scholar_id"] == profile.scholar_id]
            cached_records = {
                row["author_pub_id"]: json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                for row in cache["publications"]
            }
            snapshot_records = {
                str(row.author_pub_id): str(row.source_record_json)
                for row in profile_rows.itertuples(index=False)
            }
            if (
                len(profile_rows) != len(cache["publications"])
                or cached_records != snapshot_records
                or profile_rows["person_id"].ne(profile.person_id).any()
                or profile_rows["fetched_at_utc"].ne(cache["fetched_at_utc"]).any()
            ):
                raise ValueError(f"Snapshot rows differ from Scholar cache: {path}")

    if not set(baseline).issubset(expected_profile_ids):
        raise ValueError("A previously published Scholar cache is missing")
    if frame is not None and set(frame["scholar_id"]) != expected_profile_ids:
        raise ValueError("Snapshot Scholar profiles differ from source caches")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument("--memberships", type=Path, default=DEFAULT_MEMBERSHIPS_PATH)
    parser.add_argument("--departments", type=Path, default=DEFAULT_DEPARTMENTS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    result = validate_coverage(
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
        cache_dir=args.cache_dir,
        captures_dir=args.captures_dir,
        baseline_path=args.baseline,
        snapshot_path=args.snapshot,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
