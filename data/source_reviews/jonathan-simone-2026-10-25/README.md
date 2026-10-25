# Jonathan Simone: whole visible author bibliography

**Status:** One complete author-site candidate batch, source only. The 11 public
work families in `candidate-11.json` are outside the snapshot builder and add
no rows or edges to the public maps. No Google Scholar account is verified.

[CMU's Math profile](https://www.cmu.edu/math/people/faculty/simone.html)
identifies Jonathan Simone as an Assistant Teaching Professor and links his
researcher-managed site. Its explicitly titled [Publications and
Preprints](https://www.math.cmu.edu/~jsimone/research.html) section has **10
public entries**, with his **PhD dissertation in a separate section**. His
[current CV](https://www.math.cmu.edu/~jsimone/JonathanSimoneCV.pdf) has nine
publication entries, the dissertation in Education, and one manuscript
explicitly marked **in preparation**. The page retains the 2017 arXiv-only
*Tight contact structures on some plumbed 3-manifolds*, omitted from the
current CV. The CV's in-preparation *X- and H-plumbings bounding rational
homology 4-balls* has no public paper in these sources and is held outside the
11 public families.

An [exact arXiv author query](https://export.arxiv.org/api/query?search_query=au:%22Simone%2C_Jonathan%22&start=0&max_results=100)
reports **10 of 10** entries, each naming Jonathan Simone; their 10 stable
arXiv IDs match the 10 author-page papers one-to-one. Eight have individual
[Crossref DOI registrations](https://api.crossref.org/works/10.1016/j.topol.2025.109578)
whose titles and full bylines identify the mathematical researcher. Each
preprint and its journal article is one family, never two publications. The
remaining two arXiv families are the author-asserted accepted article without
a confirmed journal DOI and the older arXiv-only preprint. The
[University of Virginia's thesis registration](https://api.datacite.org/dois/10.18130/V39Z90B6J)
and [defense announcement](https://math.virginia.edu/2018/03/Simone-defense/)
independently identify the separate 2018 dissertation, DOI
`10.18130/v39z90b6j`.

| Genre | Families | Treatment |
| --- | ---: | --- |
| Journal article with DOI and linked arXiv version | 8 | One family per article |
| Accepted preprint, no verified journal DOI | 1 | arXiv:2405.00500; author says “to appear” |
| arXiv-only preprint | 1 | arXiv:1710.06702; distinct from the later toroidal article |
| Doctoral dissertation | 1 | Distinct university thesis DOI |
| In-preparation CV manuscript | 0 public | Held separately, no map edge |

The author page and CV call *On χ-slice pretzel links* **Topology Appl.
377 (2026), article 109578**. The [DOI registration](https://api.crossref.org/works/10.1016/j.topol.2025.109578)
and [publisher landing page](https://www.sciencedirect.com/science/article/pii/S0166864125003761)
give **volume 375, December 2025, article 109578**. The candidate uses the
publisher's year and volume, while preserving the conflicting author pages
unchanged. Volume 377 (2026), article 109602 belongs to Simone's distinct
*Tight contact structures on toroidal plumbed 3-manifolds*.

The [Math Scholar account review](../../../registry/reviews/2026-10-25-math-scholar-account-coverage.json)
found no exact verified Scholar profile in a bounded author search. The CMU
profile, researcher site, and CV do not link one. Crossref's fuzzy Jonathan
Simone results also include an unrelated neuroscience/cannabis author, so this
batch admits only DOI records matching the Math publication page, arXiv ID,
title, and byline. If a Scholar profile is later verified, its whole visible
profile must be captured and reconciled before treating it as ground truth.

**Completeness claim:** This is the complete *visible, researcher-declared*
public bibliography on the current “Publications and Preprints” page, checked
against his current CV and all 10 exact-author arXiv records. Those independent
checks found no additional public Simone math work. They cannot prove the
absence of works he has never listed or deposited. The candidate is staged as
one batch because these are all currently visible author-declared public works,
not a selected list or a per-paper DOI sample. Release integration must keep
this scope and provenance visible; `manifest.json` pins the captures and the
candidate, and the validator rejects lost families or source drift.

Run `python scripts/validate_simone_source_review.py` from the repository
root. The bundle saves source captures from CMU, the researcher site, arXiv,
Crossref, DataCite, and Virginia, plus text extracted from the current CV.
Trailing horizontal whitespace was removed from the two HTML copies and
indentation tabs were expanded in the CMU copy; the manifest retains the
original response SHA-256 values. It also records the original CV PDF's
SHA-256 without committing the PDF.
