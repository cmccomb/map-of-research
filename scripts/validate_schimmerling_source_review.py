"""Validate Schimmerling's pinned, release-held source crosswalk.

This does not create snapshot rows or infer a Google Scholar account.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/ernest-schimmerling-2026-10-25"
)


class _AuthorPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_heading = False
        self.in_item = False
        self.section = ""
        self.parts: list[str] = []
        self.rows: list[tuple[str, int, str]] = []
        self.counts: Counter[str] = Counter()

    def _finish_item(self) -> None:
        if self.in_item and self.section:
            self.counts[self.section] += 1
            self.rows.append(
                (
                    self.section,
                    self.counts[self.section],
                    " ".join(" ".join(self.parts).split()),
                )
            )
        self.in_item = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1":
            self._finish_item()
            self.in_heading = True
            self.parts = []
        elif tag == "li":
            self._finish_item()
            self.in_item = True
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.in_heading or self.in_item:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.in_heading:
            value = "".join(" ".join(self.parts).split()).casefold()
            self.section = {
                "articles": "Articles",
                "books": "Books",
                "otherpublications": "Other publications",
            }.get(value, "")
            self.in_heading = False
        elif tag == "li" and self.in_item:
            self._finish_item()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _elements_payload(path: Path, offset: int) -> list[dict]:
    document = path.read_text(encoding="utf-8")
    for match in re.finditer(r"<script(?:\s[^>]*)?>(.*?)</script>", document, re.S):
        script = match.group(1).strip()
        if not script.startswith("self.__next_f.push("):
            continue
        try:
            frame = json.loads(
                script.removeprefix("self.__next_f.push(").removesuffix(")")
            )[1]
            data = json.loads(frame.split(":", 1)[1])[3]["data"]
            records = data["resource"]
            if not records or not all("objectId" in record for record in records):
                continue
        except (ValueError, IndexError, KeyError, TypeError):
            continue
        assert data["pagination"] == {
            "startFrom": offset,
            "perPage": 25,
            "total": 45,
            "totalIsLowerBound": False,
        }
        return records
    raise ValueError(f"No CMU Elements publications payload in {path.name}")


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["person_id"] == "person-ernestschimmerling-2b6af882"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["cmu_elements_user_id"] == 1550
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest

    page_records = _elements_payload(source_dir / "elements-page-1.html", 0)
    page_records += _elements_payload(source_dir / "elements-page-2.html", 25)
    elements = _csv(source_dir / "elements-45.csv")
    assert len(page_records) == len(elements) == 45
    assert [int(row["cmu_elements_object_id"]) for row in elements] == [
        record["objectId"] for record in page_records
    ]
    for row, record in zip(elements, page_records, strict=True):
        assert row["title"] == record["title"]
        assert row["doi"] == record.get("doi", "")
        assert row["type"] == record["objectTypeDisplayName"]
        assert 1550 in record["linkedObjectIds"]["users"]
        assert any(
            author.get("resolvedUser", {}).get("objectId") == 1550
            for author in record["authors"]
        )

    author_page = _AuthorPage()
    author_page.feed(
        (source_dir / "author-publications.html").read_text(encoding="utf-8")
    )
    author_page._finish_item()
    author = _csv(source_dir / "author-34.csv")
    assert len(author) == 34
    assert author_page.counts == {"Articles": 27, "Books": 2, "Other publications": 5}
    assert [
        (row["section"], int(row["source_ordinal"]), row["raw_citation"])
        for row in author
    ] == author_page.rows

    crosswalk = _csv(source_dir / "author-elements-crosswalk-34.csv")
    assert len(crosswalk) == len(author)
    assert [(row["section"], row["source_ordinal"]) for row in crosswalk] == [
        (row["section"], row["source_ordinal"]) for row in author
    ]
    linked = [
        int(object_id)
        for row in crosswalk
        for object_id in row["elements_object_ids"].split("|")
        if object_id
    ]
    assert len(linked) == len(set(linked)) == 41
    element_ids = {int(row["cmu_elements_object_id"]) for row in elements}
    assert set(linked) <= element_ids
    assert element_ids - set(linked) == {767570, 327734, 420492, 58367}
    assert [
        (row["section"], row["source_ordinal"])
        for row in crosswalk
        if row["relation"] == "author_only"
    ] == [("Other publications", "4")]
    assert (
        next(
            row
            for row in crosswalk
            if row["section"] == "Books" and row["source_ordinal"] == "2"
        )["genre"]
        == "edited_volume"
    )
    assert all(
        row["relation"] in {"author_only", "author_and_elements"} for row in crosswalk
    )

    return {
        "elements_observations": len(elements),
        "author_observations": len(author),
        "author_matched_elements": len(linked),
        "elements_only_observations": len(element_ids - set(linked)),
        "author_only_observations": sum(
            row["relation"] == "author_only" for row in crosswalk
        ),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
