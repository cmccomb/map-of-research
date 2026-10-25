# Gautam Iyer author-site publication review (2026-10-25)

This is a **review extraction**, not a Google Scholar capture or a released source-graph update. The [CMU Mathematical Sciences faculty page](https://www.cmu.edu/math/people/faculty/iyer.html) identifies Gautam Iyer and links to his [CMU-hosted personal website](https://www.math.cmu.edu/~gautam/sj/index.html). Its **All Publications** tab currently displays 47 entries. This means 47 displayed entries were captured, not that every work in his academic history has been independently established.

## Capture and integrity

- Source: `https://www.math.cmu.edu/~gautam/sj/index.html`
- Captured at: `2026-10-25T09:48:40+00:00` from an HTTP 200 response; 163,654 HTML bytes.
- Source HTML SHA-256: `264cca147620a53e60d51b83b97339b3556646cb906cc46fdae53df6e500107b`.
- Extraction boundary: the 47 direct `<li>` children of `div#pubs-all > ul`. The Recent Publications and topic tabs repeat many of these works and were excluded.
- [Author-site CSV](2026-10-25-math-gautam-iyer-author-site.csv): 47 consecutive row indices, 47 unique normalized title/year pairs, 47 titles, 47 author strings containing Iyer, 47 years, and 47 linked work URLs. Its SHA-256 is `e3f481cb22ede0500ccb5bc1f950a012e0c442058865876644b5bf69a01dde79`.
- The CSV keeps the author's title, initials, year, and venue **as listed**. `work_url_as_linked` is the venue/publisher link when shown, otherwise an author-hosted manuscript PDF. These links were recorded, not all independently opened. Row 44 is a Ph.D. thesis.

## Metadata gaps

- 20 of 47 entries have a DOI link on the author page. The other 27 have **no DOI link there**; this does not establish that they lack a DOI.
- 40 of 47 entries show a venue. The seven newest entries have no venue shown, though each has an author-hosted manuscript link.
- The [DOI audit CSV](2026-10-25-math-gautam-iyer-doi-audit.csv) checks all 20 linked DOIs against the Crossref [Works API](https://api.crossref.org/works/). Each DOI's registry author list includes an Iyer. Ten registry publication years differ from the year displayed on the author page (rows 8, 11, 14, 16, 17, 18, 20, 22, 23, 26). Row 22 also has a title variant in the DOI record. These may reflect manuscript and final-publication versions; the records need reconciliation before ingestion. The audit CSV SHA-256 is `188388e7b8bcbdcbeeea40946cd4baa27b91456d5075685a1215f6311e4017ee`.
- No exact Google Scholar account is registered for Iyer in `registry/people.csv` at this review. If one is verified later, Scholar should remain the project's primary source for his displayed profile, with this author-site list used to inspect omissions and conflicts.

## Ingestion decision

**Hold the whole-Iyer batch outside the live graph.** The present `curated_verified` loader in `map_of_research/snapshot.py` requires a valid DOI and a nonempty venue for every row, so it cannot represent all 47 as listed. Adding only the 20 DOI-linked works would violate the whole-researcher capture priority. No source snapshot, curated work, Scholar cache, or publisher artifact changes in this review.

An all-at-once author-site route would need a distinct provenance label with optional DOI and venue, a source-page URL/date/hash and per-work URL, an identity rule for non-DOI works, and a 47-row count/uniqueness gate. DOI matches could merge directly; title/year or manuscript-to-journal matches need an ambiguity hold rather than an automatic coauthor edge. The publisher would need to show the source label and unknown citation counts without changing the Scholar refresh date. A verified Scholar account should supersede this route as the primary profile record while retaining the author-site evidence for reconciliation.
