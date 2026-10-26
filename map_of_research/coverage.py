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
DEFAULT_PRIOR_ID_CROSSWALK_PATH = Path(
    "registry/reviews/2026-10-25-chem-donahue-prior-id-crosswalk.json"
)
DEFAULT_CACHE_DIR = Path("data/authors")
DEFAULT_CAPTURE_DIR = Path("data/captures")
_TINA_REPAIR_LEDGER = Path("status/physics-kahniashvili-cache-repair-2026-10-24.json")
_BREIVIK_ROW115_REPAIR_LEDGER = Path(
    "status/physics-breivik-row115-cache-repair-2026-10-24.json"
)
_BREIVIK_ROW116_REPAIR_LEDGER = Path(
    "status/physics-breivik-row116-cache-repair-2026-10-25.json"
)
_PHYSICS_SVG_TITLE_REPAIR_LEDGER = Path(
    "status/physics-eight-svg-title-repairs-2026-10-25.json"
)
_PHYSICS_LEGACY_SVG_TITLE_REPAIR_LEDGER = Path(
    "status/physics-seven-legacy-svg-title-repairs-2026-10-25.json"
)
_PHYSICS_LEGACY_SVG_SCHOLARS = frozenset(
    {
        "8AqWr_4AAAAJ",
        "JbSQsXAAAAAJ",
        "jrRg4PAAAAAJ",
        "jcb81yAAAAAJ",
        "xKrmD5IAAAAJ",
        "O0NnwB0AAAAJ",
        "79uTAy8AAAAJ",
    }
)
_PHYSICS_NINE_SVG_TITLE_REPAIR_LEDGER = Path(
    "status/physics-nine-svg-title-repairs-2026-10-25.json"
)
_PHYSICS_NINE_SVG_SCHOLARS = frozenset(
    {
        "6pPZKd0AAAAJ",
        "NhBa8hAAAAAJ",
        "jkKrk0sAAAAJ",
        "wex9ft0AAAAJ",
        "wJ_gQLoAAAAJ",
    }
)
_KATOCH_CACHE_REPAIR_LEDGER = Path(
    "status/physics-katoch-penco-cache-repairs-2026-10-24.json"
)
_KATOCH_SOURCE_HASHES = (
    "3e28b68e0d98fbf0a6e14654745d9140765369aa69ba7fc5b5c96b5712bdf485",
    "0155fe015c516f3c271e6d7021025758dee1319b05856d42f04b0cda4777157a",
)
_KATOCH_LEGACY_REPAIRS = {
    (22, "title"): {
        "author_pub_id": "xKrmD5IAAAAJ:2osOgNQ5qMEC",
        "captured_value": (
            "Low Bias Electron Scattering in Structure-Identified Single Wall "
            "Carbon Nanotubes:<? format?> Role of Substrate Polar Phonons"
        ),
        "repaired_value": (
            "Low Bias Electron Scattering in Structure-Identified Single Wall "
            "Carbon Nanotubes: Role of Substrate Polar Phonons"
        ),
        "evidence_url": (
            "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.146601"
        ),
    },
    (40, "author"): {
        "author_pub_id": "xKrmD5IAAAAJ:WF5omc3nYNoC",
        "captured_value": "J Katoch, D Le, S Singh, R Rao, TSRM Ishigami",
        "repaired_value": "J Katoch, D Le, S Singh, R Rao, TS Rahman, M Ishigami",
        "evidence_url": (
            "https://api.crossref.org/works/10.1088%2F0953-8984%2F28%2F11%2F115301"
        ),
    },
}
_PHYSICS_REVIEW_REPAIR_LEDGERS = {
    "wJ_gQLoAAAAJ": Path(
        "status/physics-cremonesi-cache-title-repairs-2026-10-24.json"
    ),
    "LibSWRMAAAAJ": Path(
        "status/physics-schumacher-cache-title-repairs-2026-10-24.json"
    ),
}
_PHYSICS_REVIEW_RECEIPTS = {
    "wJ_gQLoAAAAJ": "status/browser-capture-physics-matteo-cremonesi-2026-10-24.json",
    "LibSWRMAAAAJ": (
        "status/browser-capture-physics-reinhard-schumacher-2026-10-24.json"
    ),
}
_PHYSICS_REVIEW_SOURCE_HASHES = {
    "wJ_gQLoAAAAJ": (
        "a9090001adfae87a95ea4925beb9c0335db0622e0250ce6a7d199b5e92ed49ba",
        "59c1fbb55bfe30f22796d518cf5c81ded79f7f11667ea1ee95e9ee4dcb7d0017",
    ),
    "LibSWRMAAAAJ": (
        "050aa1cffa53fec64523f27b8d7093a6973c4cb7b267697c80f31c17ec20eb20",
        "5ce76d360feff9abf784e45ed2afa5c2c2e3df0f5fbe18b6b005d1017d2ee93d",
    ),
}
_TINA_RAW_TITLE = (
    "ity (see Kofman & Starobinsky, Lahav et al. 5). All spectra were normalized"
)
_TINA_REPAIRED_TITLE = (
    "Large Scale Structure Formation in Mixed Dark Matter Models "
    "with a Cosmological Constant"
)
DEFAULT_CURATED_WORKS_PATH = Path("data/curated/works.json")
DEFAULT_EXCLUSIONS_PATH = Path("data/curated/scholar-exclusions.json")


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
        self.svg_title_rows: set[int] = set()
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
                # MathJax pads accessible labels for visual positioning. The
                # padding is not part of the citation title, including when a
                # hyphen follows the SVG with no intervening space.
                self._field_text.append((attributes.get("aria-label") or "").strip())
                self.svg_title_rows.add(self.row_count)
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


def _is_reviewed_breivik_author_repair(
    scholar_id: str,
    index: int,
    citation_id: str,
    source_author: Any,
    cache_author: Any,
    captures_dir: Path,
) -> bool:
    """Accept only the two AAS-backed Breivik cache byline corrections."""

    if index == 115:
        expected_id = "AcW2EXwAAAAJ:roLk4NBRz8UC"
        captured = "K Breivik, S Chatterjee, J Andrews"
        repaired = "J Andrews, S Chatterjee, K Breivik"
        ledger_name = _BREIVIK_ROW115_REPAIR_LEDGER
    elif index == 116:
        expected_id = "AcW2EXwAAAAJ:qjMakFHDy7sC"
        captured = "A Polzin, A Geller, A Miller, K Breivik"
        repaired = "A Geller, A Polzin, K Breivik, A Miller"
        ledger_name = _BREIVIK_ROW116_REPAIR_LEDGER
    else:
        return False
    if (
        scholar_id != "AcW2EXwAAAAJ"
        or citation_id != expected_id
        or source_author != captured
        or cache_author != repaired
    ):
        return False

    ledger_path = captures_dir.parent.parent / ledger_name
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
        != "status/browser-capture-physics-katelyn-breivik-2026-10-24.json"
        or not isinstance(ledger.get("repairs"), list)
        or len(ledger["repairs"]) != 1
        or not isinstance(ledger["repairs"][0], dict)
    ):
        return False
    expected = {
        "scholar_id": scholar_id,
        "capture_row": index,
        "author_pub_id": citation_id,
        "cache_field": "author",
        "captured_value": captured,
        "repaired_value": repaired,
        "evidence_url": "https://aas.org/files/aas233_abstracts.pdf",
    }
    return all(
        ledger["repairs"][0].get(key) == value for key, value in expected.items()
    )


def _is_reviewed_svg_title_repair(
    scholar_id: str,
    index: int,
    citation_id: str,
    source_title: Any,
    html_title: Any,
    cache_title: Any,
    receipt: dict[str, Any],
    captures_dir: Path,
    has_svg_title: bool,
) -> bool:
    """Accept only a hash-bound, ledger-listed saved-HTML SVG title repair."""

    if (
        not has_svg_title
        or not isinstance(source_title, str)
        or not isinstance(html_title, str)
        or cache_title != html_title
    ):
        return False
    legacy = scholar_id in _PHYSICS_LEGACY_SVG_SCHOLARS
    if legacy:
        ledger_name = _PHYSICS_LEGACY_SVG_TITLE_REPAIR_LEDGER
    elif scholar_id in _PHYSICS_NINE_SVG_SCHOLARS:
        ledger_name = _PHYSICS_NINE_SVG_TITLE_REPAIR_LEDGER
    else:
        ledger_name = _PHYSICS_SVG_TITLE_REPAIR_LEDGER
    ledger_path = captures_dir.parent.parent / ledger_name
    if not ledger_path.is_file():
        return False
    try:
        ledger = load_json(ledger_path)
    except (OSError, ValueError):
        return False
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != 1
        or ledger.get("review_state") != "saved_html_svg_title_repair"
        or (
            legacy
            and (
                ledger.get("scope") != "seven_published_legacy_physics_profiles"
                or ledger.get("raw_capture_and_receipt_unchanged") is not True
                or ledger.get("profile_count") != 7
                or ledger.get("svg_mismatch_count") != 61
                or ledger.get("cache_title_change_count") != 59
            )
        )
        or not isinstance(ledger.get("entries"), list)
        or not ledger["entries"]
    ):
        return False
    required = {
        "scholar_id",
        "capture_row",
        "author_pub_id",
        "captured_value",
        "repaired_value",
        "source_json_sha256",
        "source_html_sha256",
    }
    entries = ledger["entries"]
    if any(
        not isinstance(item, dict) or not required <= item.keys() for item in entries
    ):
        return False
    matches = [
        item
        for item in entries
        if item["scholar_id"] == scholar_id
        and item["capture_row"] == index
        and item["author_pub_id"] == citation_id
    ]
    if len(matches) != 1:
        return False
    repair = matches[0]
    return (
        repair["captured_value"] == source_title
        and repair["repaired_value"] == html_title
        and repair["source_json_sha256"] == receipt.get("source_json_sha256")
        and repair["source_html_sha256"] == receipt.get("source_html_sha256")
    )


def _is_reviewed_katoch_cache_repair(
    scholar_id: str,
    index: int,
    citation_id: str,
    field: str,
    source_value: Any,
    cache_value: Any,
    receipt: dict[str, Any],
    captures_dir: Path,
) -> bool:
    """Accept only two prior Katoch repairs from frozen sources and ledger."""

    if (
        scholar_id != "xKrmD5IAAAAJ"
        or (index, field) not in _KATOCH_LEGACY_REPAIRS
        or (
            receipt.get("source_json_sha256"),
            receipt.get("source_html_sha256"),
        )
        != _KATOCH_SOURCE_HASHES
    ):
        return False
    ledger_path = captures_dir.parent.parent / _KATOCH_CACHE_REPAIR_LEDGER
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
        or ledger.get("raw_capture_and_receipt_unchanged") is not True
        or not isinstance(ledger.get("capture_receipts"), dict)
        or ledger["capture_receipts"].get(scholar_id)
        != "status/browser-capture-physics-jyoti-katoch-2026-10-24.json"
        or not isinstance(ledger.get("repairs"), list)
    ):
        return False
    matches = [
        item
        for item in ledger["repairs"]
        if isinstance(item, dict)
        and item.get("scholar_id") == scholar_id
        and item.get("capture_row") == index
        and item.get("author_pub_id") == citation_id
        and item.get("cache_field") == field
    ]
    if len(matches) != 1:
        return False
    item = matches[0]
    expected = _KATOCH_LEGACY_REPAIRS[index, field]
    return (
        item.get("captured_value") == source_value
        and item.get("repaired_value") == cache_value
        and all(item.get(key) == value for key, value in expected.items())
    )


def _reviewed_physics_capture(scholar_id: str, receipt: dict[str, Any]) -> bool:
    """Bind the two older held Physics captures to their saved source bytes."""

    hashes = _PHYSICS_REVIEW_SOURCE_HASHES.get(scholar_id)
    return hashes is not None and hashes == (
        receipt.get("source_json_sha256"),
        receipt.get("source_html_sha256"),
    )


def _is_reviewed_physics_cache_repair(
    scholar_id: str,
    index: int,
    citation_id: str,
    field: str,
    source_value: Any,
    cache_value: Any,
    receipt: dict[str, Any],
    captures_dir: Path,
) -> bool:
    """Accept an exact, previously reviewed title or venue cache correction."""

    ledger_name = _PHYSICS_REVIEW_REPAIR_LEDGERS.get(scholar_id)
    if (
        ledger_name is None
        or field not in {"title", "venue"}
        or not _reviewed_physics_capture(scholar_id, receipt)
    ):
        return False
    ledger_path = captures_dir.parent.parent / ledger_name
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
        or ledger.get("raw_capture_and_receipt_unchanged") is not True
        or not isinstance(ledger.get("capture_receipts"), dict)
        or ledger["capture_receipts"].get(scholar_id)
        != _PHYSICS_REVIEW_RECEIPTS[scholar_id]
        or not isinstance(ledger.get("repairs"), list)
    ):
        return False
    matches = [
        item
        for item in ledger["repairs"]
        if isinstance(item, dict)
        and item.get("scholar_id") == scholar_id
        and item.get("capture_row") == index
        and item.get("author_pub_id") == citation_id
        and item.get("cache_field") == field
    ]
    if len(matches) != 1:
        return False
    item = matches[0]
    return (
        item.get("captured_value") == source_value
        and item.get("repaired_value") == cache_value
        and urlparse(str(item.get("evidence_url") or "")).scheme == "https"
    )


def _is_reviewed_physics_split_venue(
    scholar_id: str,
    source_venue: Any,
    source_year: Any,
    html_venue: Any,
    receipt: dict[str, Any],
) -> bool:
    """The two saved extractors split Scholar's trailing year from venue."""

    return (
        _reviewed_physics_capture(scholar_id, receipt)
        and _normalized(html_venue)
        == f"{_normalized(source_venue)}, {_normalized(source_year) or '0'}"
    )


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
        source_title = source_row.get("title")
        html_title = html_row.get("title")
        cached_title = cached_row.get("title")
        svg_title_repaired = _normalized(html_title) != _normalized(
            source_title
        ) and _is_reviewed_svg_title_repair(
            scholar_id,
            index,
            citation_id,
            source_title,
            html_title,
            cached_title,
            receipt,
            captures_dir,
            index in html.svg_title_rows,
        )
        if (
            source_row.get("index") != index
            or not citation_id.startswith(f"{scholar_id}:")
            or citation_id != html.citation_ids[index - 1]
            or citation_id != cached_row.get("author_pub_id")
            or citation_url != cached_row.get("pub_url")
            or (
                _normalized(html_title) != _normalized(source_title)
                and not svg_title_repaired
            )
            or any(
                _normalized(html_row.get(key)) != _normalized(source_row.get(key))
                for key in ("authors", "year", "citation_count")
            )
            or (
                _normalized(html_row.get("venue"))
                != _normalized(source_row.get("venue"))
                and not _is_reviewed_physics_split_venue(
                    scholar_id,
                    source_row.get("venue"),
                    source_row.get("year"),
                    html_row.get("venue"),
                    receipt,
                )
            )
            or (
                _normalized(cached_title) != _normalized(source_title)
                and not svg_title_repaired
                and not _is_reviewed_tina_title_repair(
                    scholar_id,
                    index,
                    citation_id,
                    source_title,
                    cached_title,
                    captures_dir,
                )
                and not _is_reviewed_physics_cache_repair(
                    scholar_id,
                    index,
                    citation_id,
                    "title",
                    source_title,
                    cached_title,
                    receipt,
                    captures_dir,
                )
                and not _is_reviewed_katoch_cache_repair(
                    scholar_id,
                    index,
                    citation_id,
                    "title",
                    source_title,
                    cached_title,
                    receipt,
                    captures_dir,
                )
            )
            or (
                _normalized(cached_row.get("author"))
                != _normalized(source_row.get("authors"))
                and not _is_reviewed_breivik_author_repair(
                    scholar_id,
                    index,
                    citation_id,
                    source_row.get("authors"),
                    cached_row.get("author"),
                    captures_dir,
                )
                and not _is_reviewed_katoch_cache_repair(
                    scholar_id,
                    index,
                    citation_id,
                    "author",
                    source_row.get("authors"),
                    cached_row.get("author"),
                    receipt,
                    captures_dir,
                )
            )
            or (
                _normalized(cached_row.get("venue"))
                != _normalized(source_row.get("venue"))
                and not _is_reviewed_physics_cache_repair(
                    scholar_id,
                    index,
                    citation_id,
                    "venue",
                    source_row.get("venue"),
                    cached_row.get("venue"),
                    receipt,
                    captures_dir,
                )
            )
            or _normalized(cached_row.get("pub_year"))
            != _normalized(source_row.get("year"))
            or (citation_count != "" and not citation_count.isdecimal())
            or cached_row.get("num_citations") != int(citation_count or "0")
        ):
            raise ValueError(
                f"Cached row differs from saved source: {scholar_id}:{index}"
            )


def _load_baseline(path: Path) -> dict[str, Any]:
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
    return baseline


def _reconcile_prior_ids(
    *,
    scholar_id: str,
    person_id: str,
    missing_ids: set[str],
    current_rows: list[dict[str, Any]],
    cache_path: Path,
    baseline: dict[str, Any],
    prior: dict[str, Any],
    crosswalk_paths: tuple[Path, ...],
    curated_works_path: Path,
    exclusions_path: Path,
) -> int:
    """Allow only a reviewed, exact historical-ID replacement for this cache."""

    crosswalks = []
    for path in crosswalk_paths:
        if path.is_file():
            document = load_json(path)
            if not isinstance(document, dict):
                raise ValueError("Historical Scholar ID crosswalk is invalid")
            crosswalks.append(document)
    if not crosswalks:
        raise ValueError(f"Previously published Scholar IDs disappeared: {cache_path}")
    if len(crosswalks) == 1:
        crosswalk = crosswalks[0]
    else:
        matching = [item for item in crosswalks if item.get("scholar_id") == scholar_id]
        if len(matching) != 1:
            raise ValueError(
                f"Previously published Scholar IDs disappeared: {cache_path}"
            )
        crosswalk = matching[0]
    if not isinstance(crosswalk, dict) or set(crosswalk) != {
        "schema_version",
        "scholar_id",
        "person_id",
        "frozen_snapshot_commit",
        "frozen_snapshot_sha256",
        "prior_cache_sha256",
        "replacement_cache_sha256",
        "reviewed_at",
        "entries",
    }:
        raise ValueError("Historical Scholar ID crosswalk is invalid")
    if (
        crosswalk["schema_version"] != 1
        or crosswalk["scholar_id"] != scholar_id
        or crosswalk["person_id"] != person_id
        or crosswalk["frozen_snapshot_commit"] != baseline["frozen_snapshot_commit"]
        or crosswalk["frozen_snapshot_sha256"] != baseline["frozen_snapshot_sha256"]
        or crosswalk["prior_cache_sha256"] != prior["cache_sha256"]
        or crosswalk["replacement_cache_sha256"] != sha256_file(cache_path)
        or not isinstance(crosswalk["reviewed_at"], str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", crosswalk["reviewed_at"])
        or not isinstance(crosswalk["entries"], list)
    ):
        raise ValueError("Historical Scholar ID crosswalk does not match this release")

    entries = crosswalk["entries"]
    expected_keys = {
        "prior_author_pub_id",
        "legacy_title",
        "legacy_year",
        "legacy_row_sha256",
        "classification",
        "target_kind",
        "target_id",
        "target_title",
        "target_year",
        "evidence_url",
        "review_note",
    }
    if any(
        not isinstance(entry, dict) or set(entry) != expected_keys for entry in entries
    ):
        raise ValueError("Historical Scholar ID crosswalk has an invalid entry")
    entry_ids = [entry["prior_author_pub_id"] for entry in entries]
    if len(entry_ids) != len(set(entry_ids)) or set(entry_ids) != missing_ids:
        raise ValueError("Historical Scholar ID crosswalk omits or adds prior IDs")

    current_by_id: dict[str, dict[str, Any]] = {}
    for row in current_rows:
        if not isinstance(row, dict) or not isinstance(row.get("author_pub_id"), str):
            raise ValueError("Historical Scholar ID crosswalk needs valid current rows")
        row_id = row["author_pub_id"]
        if row_id in current_by_id:
            raise ValueError(
                "Historical Scholar ID crosswalk found duplicate current IDs"
            )
        current_by_id[row_id] = row

    curated = load_json(curated_works_path)
    exclusions = load_json(exclusions_path)
    if (
        not isinstance(curated, dict)
        or not isinstance(curated.get("works"), list)
        or not isinstance(exclusions, dict)
        or not isinstance(exclusions.get("exclusions"), list)
    ):
        raise ValueError("Historical Scholar ID crosswalk targets are invalid")

    allowed = {
        "duplicate_rekeyed": "current_scholar",
        "malformed_duplicate": "current_scholar",
        "supporting_material_duplicate": "current_scholar",
        "nonpublication_rekeyed": "current_scholar",
        "false_profile_attribution_rekeyed": "current_scholar",
        "supporting_material": "curated_doi",
        "historical_conference_abstract": "curated_native",
        "historical_conference_proceeding": "curated_native",
    }
    for entry in entries:
        kind = entry["target_kind"]
        target_id = entry["target_id"]
        evidence_url = entry["evidence_url"]
        if (
            entry["classification"] not in allowed
            or allowed[entry["classification"]] != kind
            or not isinstance(entry["prior_author_pub_id"], str)
            or not entry["prior_author_pub_id"].startswith(f"{scholar_id}:")
            or not isinstance(entry["legacy_title"], str)
            or not entry["legacy_title"].strip()
            or entry["legacy_year"] is not None
            and not isinstance(entry["legacy_year"], str)
            or not isinstance(entry["legacy_row_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["legacy_row_sha256"])
            or not isinstance(target_id, str)
            or not target_id
            or not isinstance(entry["target_title"], str)
            or not entry["target_title"].strip()
            or not isinstance(entry["target_year"], str)
            or not isinstance(evidence_url, str)
            or urlparse(evidence_url).scheme != "https"
            or not urlparse(evidence_url).netloc
            or not isinstance(entry["review_note"], str)
            or not entry["review_note"].strip()
        ):
            raise ValueError("Historical Scholar ID crosswalk entry is invalid")

        if kind == "current_scholar":
            target = current_by_id.get(target_id)
            if (
                target is None
                or not target_id.startswith(f"{scholar_id}:")
                or _normalized(target.get("title"))
                != _normalized(entry["target_title"])
                or str(target.get("pub_year") or "") != entry["target_year"]
            ):
                raise ValueError("Historical Scholar ID crosswalk target is absent")
            if entry["classification"] in {
                "nonpublication_rekeyed",
                "false_profile_attribution_rekeyed",
            }:
                reason_code = (
                    "nonpublication"
                    if entry["classification"] == "nonpublication_rekeyed"
                    else "false_profile_attribution"
                )
                matches = [
                    item
                    for item in exclusions["exclusions"]
                    if isinstance(item, dict)
                    and item.get("scholar_id") == scholar_id
                    and item.get("author_pub_id") == target_id
                    and item.get("reason_code") == reason_code
                    and item.get("evidence_url") == evidence_url
                ]
                if len(matches) != 1:
                    raise ValueError("Historical reviewed exclusion is absent")
            elif any(
                isinstance(item, dict)
                and item.get("scholar_id") == scholar_id
                and item.get("author_pub_id") == target_id
                for item in exclusions["exclusions"]
            ):
                raise ValueError("Historical Scholar replacement is excluded")
        else:
            key = "doi" if kind == "curated_doi" else "source_native_id"
            matches = [
                item
                for item in curated["works"]
                if isinstance(item, dict)
                and item.get("person_id") == person_id
                and item.get(key) == target_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Historical curated work target is absent or duplicated"
                )
            target = matches[0]
            if (
                _normalized(target.get("title")) != _normalized(entry["target_title"])
                or str(target.get("year") or "") != entry["target_year"]
                or evidence_url
                not in [
                    target.get("source_url"),
                    *(target.get("source_verification_urls") or []),
                ]
            ):
                raise ValueError("Historical curated work evidence does not match")
    return len(entries)


def validate_coverage(
    *,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    captures_dir: Path = DEFAULT_CAPTURE_DIR,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    prior_id_crosswalk_path: Path = DEFAULT_PRIOR_ID_CROSSWALK_PATH,
    additional_prior_id_crosswalk_paths: tuple[Path, ...] = (),
    curated_works_path: Path = DEFAULT_CURATED_WORKS_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
    snapshot_path: Path | None = None,
) -> dict[str, int]:
    """Permit frozen legacy caches and verify every newer Scholar source."""

    registry = load_registry(people_path, memberships_path, departments_path)
    baseline_document = _load_baseline(baseline_path)
    baseline = baseline_document["profiles"]
    frame = None
    if snapshot_path is not None:
        import pandas

        frame = pandas.read_parquet(snapshot_path, engine="pyarrow")
        frame = frame.loc[frame["source_type"] == "scholar"]
    expected_profile_ids: set[str] = set()
    result = {"legacy_profiles": 0, "complete_profiles": 0, "reconciled_prior_ids": 0}

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
        missing_ids: set[str] = set()
        if prior is not None:
            previous_ids = set(prior["author_pub_ids"])
            current_ids = {
                row.get("author_pub_id")
                for row in cache["publications"]
                if isinstance(row, dict)
            }
            missing_ids = previous_ids - current_ids
            if missing_ids:
                result["reconciled_prior_ids"] += _reconcile_prior_ids(
                    scholar_id=profile.scholar_id,
                    person_id=profile.person_id,
                    missing_ids=missing_ids,
                    current_rows=cache["publications"],
                    cache_path=path,
                    baseline=baseline_document,
                    prior=prior,
                    crosswalk_paths=(
                        prior_id_crosswalk_path,
                        *additional_prior_id_crosswalk_paths,
                    ),
                    curated_works_path=curated_works_path,
                    exclusions_path=exclusions_path,
                )
        included_now = any(membership.included for membership in profile.memberships)
        unchanged_legacy = (
            isinstance(prior, dict)
            and not missing_ids
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
    parser.add_argument(
        "--prior-id-crosswalk", type=Path, default=DEFAULT_PRIOR_ID_CROSSWALK_PATH
    )
    parser.add_argument(
        "--additional-prior-id-crosswalk", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--curated-works", type=Path, default=DEFAULT_CURATED_WORKS_PATH
    )
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    result = validate_coverage(
        people_path=args.people,
        memberships_path=args.memberships,
        departments_path=args.departments,
        cache_dir=args.cache_dir,
        captures_dir=args.captures_dir,
        baseline_path=args.baseline,
        prior_id_crosswalk_path=args.prior_id_crosswalk,
        additional_prior_id_crosswalk_paths=tuple(args.additional_prior_id_crosswalk),
        curated_works_path=args.curated_works,
        exclusions_path=args.exclusions,
        snapshot_path=args.snapshot,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
