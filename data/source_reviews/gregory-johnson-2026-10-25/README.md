# Gregory Johnson: held whole-researcher source review

Captured 2026-10-25 for `person-gregoryjohnson-26c9645f`. This is a source
review, not a complete publication snapshot or release input. No exact Google
Scholar account or current author-maintained complete bibliography was verified.
The registry's Scholar ID remains blank.

## Identity chain

The [current CMU profile](https://www.cmu.edu/math/people/faculty/johnson.html)
identifies Gregory Johnson, Math Associate Teaching Professor, with email
`greggo@andrew.cmu.edu`, a University of Maryland PhD, and research in model
theory and abstract elementary classes. The [older CMU profile](https://www.math.cmu.edu/math/visiting_faculty/johnson.html)
uses **Greggo Johnson** with the same email and research area. The [University
of Maryland dissertation record](https://drum.lib.umd.edu/items/95910634-bbe1-4228-8bf7-b2049a005c6e)
identifies **Gregory Mitchell Johnson**, a 2008 PhD dissertation in that area,
advised by David W. Kueker. The [publisher-registered DOI metadata](https://api.crossref.org/works/10.1215%2F00294527-2010-022)
identifies a 2010 journal article with the same title and byline **Gregory M.
Johnson**. Thesis and journal article are different genres and manifestations.

The later AI education bylines say only **Gregory Johnson**. The [CMU HCII
conference listing](https://hcii.cmu.edu/news/cmu-festival-learning-2026)
places Gregory Johnson and John Mackey on the AIED chapter. The three arXiv
records share these coauthors and describe a linked classroom study. This is
strong provisional attribution, but no author-controlled Scholar account
confirms the entire publication history. Keep the lineage held.

## Captured source observations

The eight attributed source observations in `source-observations-8.csv` refer
to **six provisional manifestations** across two research lineages. Neither
count represents lifetime output:

| Manifestation | Primary source | Treatment |
| --- | --- | --- |
| 2008 UMD doctoral dissertation | [UMD repository](https://drum.lib.umd.edu/items/95910634-bbe1-4228-8bf7-b2049a005c6e) | Separate thesis genre. |
| 2010 *Notre Dame Journal of Formal Logic* article | [DOI metadata](https://api.crossref.org/works/10.1215%2F00294527-2010-022) | Same title as thesis, separate journal manifestation, DOI `10.1215/00294527-2010-022`. |
| 2025 AI proof-learning preprint | [arXiv:2509.16778](https://arxiv.org/abs/2509.16778) | Earlier version of later classroom-study line. |
| February 2026 AI proof-learning preprint | [arXiv:2602.18807](https://arxiv.org/abs/2602.18807) | AIED camera-ready manuscript line. |
| AIED conference chapter | [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-29763-1_41), [Crossref](https://api.crossref.org/works/10.1007%2F978-3-032-29763-1_41), [CMU listing](https://hcii.cmu.edu/news/cmu-festival-learning-2026) | DOI `10.1007/978-3-032-29763-1_41`; do not count the three observations as three works. |
| October 2026 extended manuscript | [arXiv:2610.31481](https://arxiv.org/abs/2610.31481) | Explicitly says it revises and extends the February 2026 and 2025 preprints. |

The Springer HTML `citation_publication_date` says **2027**, while Crossref
`published` says **2026-06-27** and CMU lists the chapter under AIED 2026.
The conflicting dates are retained as observed; no publication year is chosen
for the graph.

## Genre and attribution holds

- The [official AMS meeting abstract](https://jointmathematicsmeetings.org/meetings/national/jmm/1067-z1-1898.pdf)
  for a 2011 statistics teaching talk has the exact CMU `greggo` email, but
  direct PDF retrieval returned HTTP 403. It remains an uncaptured meeting
  lead, not an ingested journal paper.
- The [2017 Giuga paper](https://scholar.rose-hulman.edu/rhumj/vol18/iss1/5/)
  lists Gregory Johnson as **faculty sponsor** and names four students as
  authors. It is excluded from his authored-publication set.
- An unrelated CMU computational biology Gregory R. Johnson must not be used
  to assign this Math faculty member an ORCID or Scholar ID.

Raw source responses and SHA-256 hashes are pinned in `manifest.json`. All
observations and leads are `graph_eligible=false`. Do not make publication or
coauthor edges, upload rows to the Hub, or call the six manifestations a
complete career without an exact verified Scholar account and fully paginated
profile reconciliation. A current complete author bibliography plus an
explicit exception decision would be needed if no Scholar account exists.

Run `PYTHONPATH=. uv run --extra dev python scripts/validate_gregory_johnson_source_review.py`
to check source preservation, version lineage, genre, sponsor exclusion, and
release holds.
