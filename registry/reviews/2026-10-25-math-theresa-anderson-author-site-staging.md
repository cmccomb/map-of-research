# Theresa Anderson: staged complete displayed bibliography

**State:** source-only draft. No verified Google Scholar account. No source
snapshot, Hub dataset, or public graph has been changed.

[CMU's faculty profile](https://www.cmu.edu/math/people/faculty/anderson.html)
identifies Theresa Anderson in Mathematical Sciences. Her [CMU hosted research
page](https://faculty.math.cmu.edu/anderson/research/) lists 42 entries under
"Publications and preprints". Its linked [August 2026
CV](https://faculty.math.cmu.edu/anderson/wp-content/uploads/sites/2/2026/08/CV2026.pdf)
numbers 42 entries. The exact HTML, PDF, 42 source citations, one-to-one CV
crosswalk, and source-exact parsing review are pinned by
[`data/author_sites/theresa-anderson-2026-10-25.json`](../../data/author_sites/theresa-anderson-2026-10-25.json).
The loader rejects a missing row, altered capture byte, changed CV pairing, or
false Scholar/citation provenance.

The staged source rows preserve the original page and CV wording in
`source_record_json`. They do not assert any DOI. Six source/CV title differences
remain under explicit review at source ordinals **21, 22, 23, 24, 25, and 33**.
Five newest entries have no stated publication year; entry 30 takes its 2020
year from the linked CV. Ordinal **5** is an appendix by Anderson and Lemke
Oliver to an Elena Kim article; Kim is recorded in the title of the source
observation, not as an appendix coauthor. Ordinal **11** is a personal essay and
**37** is a thesis. All three have non-publication output types and cannot
create collaboration links. The other 39 have no asserted external coauthors:
free-text bylines alone do not establish registry identity or edges.

All 42 observations receive `source_type=author_site`, empty Scholar ID and
fetch time, and unknown citation counts. The author-site loader is opt-in for
offline source snapshots, while the publisher rejects any snapshot containing
this source type. Google Scholar remains the preferred field source if an exact
Anderson profile is later verified and captured in full. The current source
capture proves coverage of the displayed bibliography and CV, not all lifetime
academic publications. Resolve DOI/work identity, title variants, and author
identity before any graph release; run an exact no-upload publisher preview.

The current CLI accepts one `--author-site-capture` manifest per snapshot.
Therefore the 47-row Iyer capture and this 42-row Anderson capture are separately
testable but cannot yet coexist in one source snapshot. A separate composition
change is required before any combined author-site release.
