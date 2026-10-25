# Author-owned bibliography capture

This lane stores a public page capture and exact entry extraction when no Google
Scholar account is verified. It is separate from `curated_verified` DOI records
and from Scholar caches. The `author_site` source label preserves that boundary.

`gautam-iyer-2026-10-25.html` is the exact 163,654-byte author page. Its companion
JSON pins the page SHA-256, extraction CSV and DOI audit SHA-256 values, person
identity, page selector, 47 expected entries, 20 reviewed linked DOIs, and one
thesis classification. The raw author title, year, venue, links, and byline stay
in `source_record_json`. Normalized year uses a linked DOI's publisher-supplied
publication year only after its exact review; the 27 rows without linked DOI
keep the author-listed year. Blank venues remain blank.

Snapshot building requires `--author-site-capture` explicitly. The resulting
snapshot marks all 47 rows `author_site`, leaves Scholar ID and fetch time empty,
and pins the capture manifest checksum. This branch does not build or publish
a central source snapshot. The publisher currently rejects this source type
until an exact release gate and a public citation/provenance display are added.
This draft must remain unmerged until the in-progress Sullivan/Math composite
release finishes, because schema v7 cannot validate its schema v6 source
snapshot. A later source rebuild may pair this capture with the 49 reviewed
Math primary homes in PR #317; that registry change is not part of this draft.
