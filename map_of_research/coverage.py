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
_TINA_REPAIR_LEDGER = Path("status/physics-kahniashvili-cache-repair-2026-10-24.json")
_TINA_RAW_TITLE = (
    "ity (see Kofman & Starobinsky, Lahav et al. 5). All spectra were normalized"
)
_TINA_REPAIRED_TITLE = (
    "Large Scale Structure Formation in Mixed Dark Matter Models "
    "with a Cosmological Constant"
)


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
        self.rows: list[dict[str, str]] = []
        self.more_buttons: list[bool] = []
        self._cell = ""
        self._gray_count = 0
        self._field = ""
        self._field_depth = 0
        self._field_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if self._field:
            if self._field == "title" and tag == "svg" and "gs_fsvg" in classes:
                # Scholar renders some title glyphs as labeled SVG icons.
                self._field_text.append(attributes.get("aria-label") or "")
            self._field_depth += 1
            return
        if tag == "tr" and "gsc_a_tr" in (attributes.get("class") or "").split():
            self.in_row = True
            self.row_count += 1
            self._cell = ""
            self._gray_count = 0
            self.rows.append({})
        elif self.in_row and tag == "td":
            self._cell = next(
                (name for name in ("gsc_a_t", "gsc_a_c", "gsc_a_y") if name in classes),
                "",
            )
        elif (
            self.in_row
            and self._cell == "gsc_a_t"
            and tag == "a"
            and "gsc_a_at" in classes
        ):
            self.citation_ids.append(_citation_id(attributes.get("href") or ""))
            self._begin_field("title")
        elif (
            self.in_row
            and self._cell == "gsc_a_t"
            and tag == "div"
            and "gs_gray" in classes
        ):
            self._gray_count += 1
            if self._gray_count <= 2:
                self._begin_field("authors" if self._gray_count == 1 else "venue")
        elif (
            self.in_row
            and self._cell == "gsc_a_c"
            and tag == "a"
            and "gsc_a_ac" in classes
        ):
            self._begin_field("citation_count")
        elif (
            self.in_row
            and self._cell == "gsc_a_y"
            and tag == "span"
            and "gsc_a_h" in classes
        ):
            self._begin_field("year")
        elif tag == "button" and attributes.get("id") == "gsc_bpf_more":
            self.more_buttons.append("disabled" in attributes)

    def _begin_field(self, field: str) -> None:
        self._field = field
        self._field_depth = 0
        self._field_text = []

    def handle_data(self, data: str) -> None:
        if self._field:
            self._field_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._field:
            if self._field_depth:
                self._field_depth -= 1
            else:
                self.rows[-1][self._field] = "".join(self._field_text).strip()
                self._field = ""
            return
        if tag == "td":
            self._cell = ""
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


def _is_reviewed_tina_title_repair(
    scholar_id: str,
    index: int,
    citation_id: str,
    source_title: Any,
    cache_title: Any,
    captures_dir: Path,
) -> bool:
    """Accept one evidence-reviewed cache title while retaining its raw capture."""

    if (
        scholar_id != "au_cLfoAAAAJ"
        or index != 116
        or citation_id != "au_cLfoAAAAJ:VL0QpB8kHFEC"
        or source_title != _TINA_RAW_TITLE
        or cache_title != _TINA_REPAIRED_TITLE
    ):
        return False

    ledger_path = captures_dir.parent.parent / _TINA_REPAIR_LEDGER
    if not ledger_path.is_file():
        return False
    try:
        ledger = load_json(ledger_path)
    except (OSError, ValueError):
        return False
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != 1
        or ledger.get("review_state") != "source_reviewed_cache_repaired"
        or ledger.get("capture_receipt")
        != "status/browser-capture-physics-tina-kahniashvili-2026-10-24.json"
        or not isinstance(ledger.get("repairs"), list)
        or len(ledger["repairs"]) != 1
        or not isinstance(ledger["repairs"][0], dict)
    ):
        return False
    repair = ledger["repairs"][0]
    expected = {
        "scholar_id": scholar_id,
        "capture_row": index,
        "author_pub_id": citation_id,
        "cache_field": "title",
        "captured_value": _TINA_RAW_TITLE,
        "repaired_value": _TINA_REPAIRED_TITLE,
        "evidence_url": "https://books.google.com/books?id=9sFUDwAAQBAJ&pg=PA269",
        "corroborating_url": (
            "https://www.mao.kiev.ua/biblio/docs/special/"
            "2007-04-19_Novosyadlyj_aref.pdf"
        ),
    }
    return all(repair.get(key) == value for key, value in expected.items())


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
        or len(html.rows) != count
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
        html_row = html.rows[index - 1]
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
                _normalized(html_row.get(key)) != _normalized(source_row.get(key))
                for key in ("title", "authors", "venue", "year", "citation_count")
            )
            or (
                _normalized(cached_row.get("title"))
                != _normalized(source_row.get("title"))
                and not _is_reviewed_tina_title_repair(
                    scholar_id,
                    index,
                    citation_id,
                    source_row.get("title"),
                    cached_row.get("title"),
                    captures_dir,
                )
            )
            or any(
                _normalized(cached_row.get(cache_key))
                != _normalized(source_row.get(source_key))
                for cache_key, source_key in (
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
