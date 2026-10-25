"""Validate the held, exact Iyer/Slepčev arXiv version crosswalk."""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class CitationMeta(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        attributes = dict(attrs)
        name = attributes.get("name")
        if name and name.startswith("citation_"):
            self.values.setdefault(name, []).append(attributes.get("content") or "")


def check() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["release_state"] != "source_only_release_held":
        raise ValueError("Source review release hold changed")
    for name, digest in manifest["files_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Pinned source changed: {name}")
    expected_authors = ["Han, Ruiyu", "Iyer, Gautam", "Slepčev, Dejan"]
    titles = [
        "Polynomial complexity sampling from multimodal distributions using Sequential Monte Carlo",
        "Time-complexity of sampling from a multimodal distribution using sequential Monte Carlo",
    ]
    for version, title in ((1, titles[0]), (2, titles[1])):
        page = (ROOT / f"arxiv-v{version}.html").read_text()
        metadata = CitationMeta()
        metadata.feed(page)
        if (
            metadata.values.get("citation_arxiv_id") != ["2508.02763"]
            or metadata.values.get("citation_title") != [title]
            or metadata.values.get("citation_author") != expected_authors
            or f"arXiv:2508.02763v{version}" not in page
        ):
            raise ValueError(f"arXiv v{version} metadata changed")
    scholar = json.loads((ROOT / "scholar-row.json").read_text())
    publication = scholar["publication"]
    if (
        scholar["scholar_id"] != "tWQsy4UAAAAJ"
        or publication["author_pub_id"] != "tWQsy4UAAAAJ:Tiz5es2fbqcC"
        or publication["title"] != titles[0]
        or publication["author"] != "R Han, G Iyer, D Slepcev"
        or publication["pub_year"] != "2025"
        or "2508.02763" not in publication["venue"]
        or "user=tWQsy4UAAAAJ" not in publication["pub_url"]
    ):
        raise ValueError("Frozen Slepčev Scholar observation changed")
    author = json.loads((ROOT / "author-site-row.json").read_text())["row"]
    if (
        author["person_id"] != "person-gautamiyer-3894621b"
        or author["row_index"] != "4"
        or author["title_as_listed"].rstrip(".") != titles[0]
        or author["authors_as_listed"] != "R. Han, G. Iyer, D. Slepčev"
        or author["year_as_listed"] != "2025"
        or author["doi_as_linked"] != ""
    ):
        raise ValueError("Frozen Iyer author-site row changed")
    if manifest["proposed_native_work_key"] != "native:arxiv:2508.02763":
        raise ValueError("Reviewed native work identity changed")


if __name__ == "__main__":
    check()
    print("Iyer row-4 source review verified; graph release remains held")
