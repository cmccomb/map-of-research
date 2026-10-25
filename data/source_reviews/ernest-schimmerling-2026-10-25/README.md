# Ernest Schimmerling: complete visible-source review

**Status:** Review only. These files are not loaded by the source-snapshot
builder and create no map edges. They establish complete capture of two
specific displayed bibliographies on 25 October 2026, not an exhaustive
career bibliography or a verified Google Scholar account.

[CMU's Math roster](https://www.cmu.edu/math/people/faculty/schimmerling.html)
identifies Ernest Schimmerling as the included person
`person-ernestschimmerling-2b6af882`. The two retained
[CMU Scholars publication pages](https://scholars.cmu.edu/1550-ernest-schimmerling/publications)
show 25 and 20 records, with a reported total of 45 and
`totalIsLowerBound=false`. `elements-45.csv` preserves every displayed CMU
object ID, title, date, type, DOI, venue, and complete ordered author list.
Each record links CMU person 1550 and resolves Schimmerling as an author.

His [CMU-hosted bibliography](https://www.math.cmu.edu/~eschimme/publications.html)
shows 27 entries under Articles, two Books, and five Other publications.
`author-34.csv` preserves all 34 source citations by section and ordinal.
`author-elements-crosswalk-34.csv` assigns 33 author entries to 41 CMU
observations, with one author-only appendix. The remaining four CMU
observations cover a 2026 preprint, a 2016 article review, and two records
with the same 1997 preprint title. The crosswalk groups evidence for review;
it is not a release-ready canonical works or authorships table.

The crosswalk keeps four identical journal-DOI duplicate pairs, the
three-observation Jensen covering group, and preprint/journal pairs visible
as separate source records. An Elements row labeled “Journal article” can
still be a book or article review, so the author bibliography's sections
inform the proposed genre. In particular, Schimmerling and Cummings edit
*Appalachian Set Theory 2006–2012*; an editor must not become a coauthor of
the book's chapters. The author-listed *Wellorderings, ordinals and
cardinals* appendix is absent from Elements; its container's editors are
not appendix coauthors. The author page calls *Covering at limit cardinals
of K* submitted in 2019, while Elements has its 2024 journal DOI; both dates
are retained.

There is no verified Schimmerling Scholar profile. The author's page ends
with a 2019 submission; Elements has a 2026 preprint. More works may exist
outside these sources. All 34 author rows and 45 CMU observations remain
held from the snapshot and public maps pending identity, version, genre,
byline, and career-completeness review. Citation counts are unknown here.

Run `python3 scripts/validate_schimmerling_source_review.py` from the repo
root to verify the pinned files, embedded CMU payloads, crosswalk coverage,
and person identity.
