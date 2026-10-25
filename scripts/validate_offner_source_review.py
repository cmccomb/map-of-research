"""Check the pinned, review-only David Offner source crosswalk.

This deliberately does not create snapshot rows or infer a Scholar account.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[1] / "data/source_reviews/david-offner-2026-10-25"
)
AUTHOR_PAGE_CANONICAL_IDS = [
    532091,
    428523,
    428533,
    428525,
    428532,
    458958,
    428472,
    428527,
    428476,
    428529,
    428534,
    428528,
    428531,
    428530,
    739552,
    798473,
    791730,
    791732,
]


class AuthorPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.entry_depth: int | None = None
        self.section = "journal"
        self.heading = False
        self.paragraph = False
        self.parts: list[str] = []
        self.links: list[str] = []
        self.rows: list[tuple[str, int, str, list[str]]] = []
        self.counts: Counter[str] = Counter()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "div":
            self.depth += 1
            if (
                self.entry_depth is None
                and "entry-content" in (attributes.get("class") or "").split()
            ):
                self.entry_depth = self.depth
        if self.entry_depth is None:
            return
        if tag == "h1":
            self.heading = True
            self.parts = []
        elif tag == "p":
            self.paragraph = True
            self.parts = []
            self.links = []
        elif tag == "a" and self.paragraph and attributes.get("href"):
            self.links.append(str(attributes["href"]))

    def handle_data(self, data: str) -> None:
        if self.heading or self.paragraph:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.heading:
            heading = " ".join("".join(self.parts).split())
            if heading == "Preprints":
                self.section = "preprint"
            elif heading == "Other Publications":
                self.section = "other_publication"
            self.heading = False
        elif tag == "p" and self.paragraph:
            self.paragraph = False
            self.counts[self.section] += 1
            self.rows.append(
                (
                    self.section,
                    self.counts[self.section],
                    " ".join("".join(self.parts).split()),
                    self.links.copy(),
                )
            )
        elif tag == "div":
            if self.entry_depth == self.depth:
                self.entry_depth = None
            self.depth -= 1


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _elements(path: Path, offset: int) -> list[dict]:
    html = path.read_text(encoding="utf-8")
    for match in re.finditer(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.S):
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
            "total": 29,
            "totalIsLowerBound": False,
        }
        return records
    raise ValueError(f"No CMU Elements publication payload in {path.name}")


def _surname(name: str) -> str:
    value = unicodedata.normalize("NFKD", name.split()[-1].casefold())
    return "".join(letter for letter in value if letter.isalpha())


def _title_key(title: str) -> str:
    plain = re.sub(r"<[^>]*>", "", html.unescape(title))
    return "".join(letter for letter in plain.casefold() if letter.isalnum())


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["person_id"] == "person-davidoffner-91600258"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["cmu_elements_user_id"] == 5568
    assert len(manifest["files"]) == 7
    for filename, digest in manifest["files"].items():
        actual_digest = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        assert actual_digest == digest

    records = _elements(source_dir / "elements-page-1.html", 0) + _elements(
        source_dir / "elements-page-2.html", 25
    )
    elements = _csv(source_dir / "elements-29.csv")
    crosswalk = _csv(source_dir / "elements-crosswalk-29.csv")
    assert len(records) == len(elements) == len(crosswalk) == 29
    ids = {record["objectId"] for record in records}
    assert len(ids) == 29
    kinds = Counter(record["objectTypeDisplayName"] for record in records)
    assert kinds == {"Journal article": 15, "Preprint": 14}
    by_id = {record["objectId"]: record for record in records}
    for ordinal, (record, element, review) in enumerate(
        zip(records, elements, crosswalk, strict=True), start=1
    ):
        assert 5568 in record["linkedObjectIds"]["users"]
        assert any(
            author.get("resolvedUser", {}).get("objectId") == 5568
            for author in record["authors"]
        )
        assert int(element["ordinal"]) == int(review["elements_ordinal"]) == ordinal
        assert (
            int(element["cmu_elements_object_id"])
            == int(review["cmu_elements_object_id"])
            == record["objectId"]
        )
        assert element["title"] == review["elements_title_as_listed"] == record["title"]
        assert (
            element["type"]
            == review["elements_type_as_listed"]
            == record["objectTypeDisplayName"]
        )
        assert (
            element["year"]
            == review["elements_year_as_listed"]
            == str(record["publicationDate"]["year"])
        )
        assert (
            element["date"]
            == review["elements_date_as_listed"]
            == record["publicationDate"]["dateTime"]
        )
        assert element["doi"] == (record.get("doi") or "")
        names = [
            author.get("fullName")
            or author.get("firstNameLastName")
            or author.get("nameShortFormat")
            for author in record["authors"]
        ]
        assert element["authors"] == " | ".join(names)
        assert int(element["author_count"]) == len(names)

    pairs = {
        item["preprint_object_id"]: item["journal_object_id"]
        for item in manifest["version_pairs"]
    }
    assert len(pairs) == 10 and len(set(pairs.values())) == 10
    assert not set(pairs).intersection(pairs.values())
    for preprint_id, journal_id in pairs.items():
        preprint, journal = by_id[preprint_id], by_id[journal_id]
        assert preprint["objectTypeDisplayName"] == "Preprint"
        assert journal["objectTypeDisplayName"] == "Journal article"
        assert [_surname(a["fullName"]) for a in preprint["authors"]] == [
            _surname(a["fullName"]) for a in journal["authors"]
        ]
        if preprint_id not in {791731, 791729}:
            assert _title_key(preprint["title"]) == _title_key(journal["title"])
        assert any(
            int(row["cmu_elements_object_id"]) == preprint_id
            and int(row["canonical_cmu_object_id"]) == journal_id
            and row["relation"] == "preprint_version"
            for row in crosswalk
        )
    canonical = {int(row["canonical_cmu_object_id"]) for row in crosswalk}
    assert len(canonical) == 19
    assert Counter(row["relation"] for row in crosswalk) == {
        "journal_canonical": 15,
        "preprint_version": 10,
        "standalone_preprint": 4,
    }
    for row in crosswalk:
        oid = int(row["cmu_elements_object_id"])
        cid = int(row["canonical_cmu_object_id"])
        assert cid == pairs.get(oid, oid)
        expected_year = (
            "2016" if cid == 791732 else str(by_id[cid]["publicationDate"]["year"])
        )
        assert row["canonical_display_year_for_review"] == expected_year
    assert (
        next(row for row in crosswalk if int(row["cmu_elements_object_id"]) == 791732)[
            "external_evidence_url"
        ]
        == "https://arxiv.org/abs/1607.03471"
    )

    page = AuthorPage()
    page.feed((source_dir / "author-research.html").read_text(encoding="utf-8"))
    author_rows = _csv(source_dir / "author-20.csv")
    assert len(page.rows) == len(author_rows) == 20
    assert page.counts == {"journal": 14, "preprint": 4, "other_publication": 2}
    for ordinal, ((section, section_ordinal, citation, links), row) in enumerate(
        zip(page.rows, author_rows, strict=True), start=1
    ):
        assert int(row["author_page_ordinal"]) == ordinal
        assert row["section"] == section
        assert int(row["section_ordinal"]) == section_ordinal
        assert row["raw_citation"] == citation
        assert json.loads(row["linked_urls_json"]) == links
        if section == "other_publication":
            assert not row["canonical_cmu_object_id"]
            assert row["collaboration_hold"] == "not_research_article"
        else:
            assert (
                int(row["canonical_cmu_object_id"])
                == AUTHOR_PAGE_CANONICAL_IDS[ordinal - 1]
            )
    assert [row["genre"] for row in author_rows[-2:]] == [
        "book_review",
        "problem_solution",
    ]
    assert [row["byline_review"] for row in author_rows[-2:]] == [
        "David Offner",
        "John Bonomo | David Offner",
    ]
    assert "Donald Knuth" in author_rows[-1]["raw_citation"]
    assert "Donald Knuth" not in author_rows[-1]["byline_review"]
    assert canonical - {
        int(row["canonical_cmu_object_id"])
        for row in author_rows
        if row["canonical_cmu_object_id"]
    } == {791733}
    actual = {
        "elements_observations": len(records),
        "elements_journal_observations": kinds["Journal article"],
        "elements_preprint_observations": kinds["Preprint"],
        "preprint_journal_pairs": len(pairs),
        "distinct_research_works": len(canonical),
        "author_page_observations": len(author_rows),
        "author_page_journal": page.counts["journal"],
        "author_page_preprint": page.counts["preprint"],
        "author_page_other": page.counts["other_publication"],
        "source_union_distinct_outputs_excluding_thesis": len(canonical) + 2,
    }
    assert manifest["counts"] == actual
    return actual


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
