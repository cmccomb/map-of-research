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

`theresa-anderson-2026-10-25.json` separately pins all 42 entries in Theresa
Anderson's CMU hosted "Publications and preprints" section, the exact source
HTML and extraction CSV, her linked August 2026 CV PDF, a one-to-one 42-entry
CV crosswalk, and a reviewed title/byline/genre index. The six source/CV title
differences remain visible in the original wording; no canonical publisher
title or DOI is inferred from an unlinked citation. Five newest entries have
no source-supported year. Entry 30 uses the linked CV's explicit 2020 year.
Entry 5 is an appendix by Anderson and Lemke Oliver to a Kim article; Kim is
not asserted as their coauthor. Entry 11 is a personal essay and entry 37 a
thesis. These three are labeled with non-publication output types and cannot
form collaboration links. No external coauthors are inferred from the
bibliography's free-text bylines.

The loader exposes all 42 only when the manifest is explicitly passed to
`--author-site-capture`. It preserves the exact source citation, CV wording,
raw links, and review record in `source_record_json`. Its citation counts are
unknown, Scholar IDs and fetch times are empty, and the publisher still rejects
all `author_site` snapshots. The capture establishes completeness of the
displayed author page and current CV, not independently exhaustive career
coverage or a Google Scholar profile.

The current snapshot CLI selects one author-site manifest at a time. Combining
the Iyer and Anderson captures in the same source snapshot requires a separate
multi-manifest pipeline change before publication.
