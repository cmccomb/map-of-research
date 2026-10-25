"""Protect reviewed Scholar cache corrections and their captured evidence."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "status/physics-katoch-penco-cache-repairs-2026-10-24.json"


def test_reviewed_physics_cache_repairs_are_preserved() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    repairs = ledger["repairs"]
    assert len(repairs) == 4
    assert {(repair["author_pub_id"], repair["cache_field"]) for repair in repairs} == {
        ("xKrmD5IAAAAJ:u-x6o8ySG0sC", "title"),
        ("xKrmD5IAAAAJ:2osOgNQ5qMEC", "title"),
        ("xKrmD5IAAAAJ:WF5omc3nYNoC", "author"),
        ("O0NnwB0AAAAJ:_Qo2XoVZTnwC", "title"),
    }

    caches = {}
    captures = {}
    for scholar_id, receipt_path in ledger["capture_receipts"].items():
        cache_path = ROOT / "data/authors" / f"{scholar_id}.json"
        caches[scholar_id] = json.loads(cache_path.read_text(encoding="utf-8"))
        receipt = json.loads((ROOT / receipt_path).read_text(encoding="utf-8"))
        assert receipt["scholar_id"] == scholar_id
        for kind in ("json", "html"):
            source_path = ROOT / receipt[f"source_{kind}_path"]
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            assert digest == receipt[f"source_{kind}_sha256"]
        captures[scholar_id] = json.loads(
            (ROOT / receipt["source_json_path"]).read_text(encoding="utf-8")
        )

    for repair in repairs:
        scholar_id = repair["scholar_id"]
        author_pub_id = repair["author_pub_id"]
        publication = next(
            row
            for row in caches[scholar_id]["publications"]
            if row["author_pub_id"] == author_pub_id
        )
        assert publication[repair["cache_field"]] == repair["repaired_value"]

        work = next(
            row
            for row in captures[scholar_id]["works"]
            if row["index"] == repair["capture_row"]
        )
        citation_id = parse_qs(urlsplit(work["citation_url"]).query)[
            "citation_for_view"
        ]
        assert citation_id == [author_pub_id]
        raw_field = {"author": "authors", "title": "title"}[repair["cache_field"]]
        assert work[raw_field] == repair.get(
            "raw_capture_value", repair["captured_value"]
        )
