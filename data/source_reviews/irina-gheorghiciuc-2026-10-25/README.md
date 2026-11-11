# Irina Gheorghiciuc: held whole-researcher source review

Captured 2026-10-25 for `person-irinagheorghiciuc-c53bff46`. This directory is
evidence for a bounded review, not a publication snapshot or a release input.
No Google Scholar ID or self-asserted ORCID was verified. The existing registry
identity and blank Scholar ID are unchanged.

## What the captured sources actually cover

| Source | Attributed observations | Scope |
| --- | ---: | --- |
| [CMU faculty profile](https://www.cmu.edu/math/people/faculty/gheorghiciuc.html) | 3 | Explicitly **Select Publications**, not a full bibliography. The three citations are transcribed in `cmu-selected-3.csv`. |
| [arXiv exact-author query](https://export.arxiv.org/api/query?search_query=au%3A%22Irina+Gheorghiciuc%22&start=0&max_results=100) | 4 | All four results in the captured query; a second page is empty. This establishes only the query result, not lifetime publication completeness. |
| [Penn Math doctoral history](https://www.math.upenn.edu/about/department-history/our-phds) | 1 | A 2004 doctoral thesis, a separate genre from her later related journal article. |
| [DMTCS publisher record](https://dmtcs.episciences.org/articles/3553) | 1 | The correlation-polynomials work is in **DMTCS Proceedings vol. AH**, AofA 2007, DOI `10.46298/dmtcs.3553`. |
| [Journal of Integer Sequences article](https://www.maths.tcd.ie/EMIS/journals/JIS/VOL17/Allen/gheo.html) | 1 | The weighted-interpretation work is in **Vol. 17, Article 14.10.7**, published 2014-11-05. CMU's selected citation says “10,” so that field needs correction at reconciliation. |
| [KiltHub search](https://api.figshare.com/v2/articles/search) | 0 | The one name-search hit is Emily Allen's thesis; the repository byline has no Irina Gheorghiciuc. It is excluded, not inferred as a coauthored work. |

The 10 attributed observations form **six provisional work groups** after
linking matching title/byline preprint and publisher versions. Neither number
is a lifetime publication count. The source rows, version IDs, bylines, dates,
and provisional links are in `source-observations-10.csv`. The four arXiv rows
also remain in `arxiv-exact-author-4.csv`; all raw source responses are pinned
with SHA-256 hashes in `manifest.json`.

## Holds and source limitations

- The CMU profile points to an [old personal home page](https://www.math.cmu.edu/~gheorghi/), but its `research.html` link returned HTTP 404. The response and HTTP receipt are captured.
- A single direct [Scholar author search](https://scholar.google.com/citations?view_op=search_authors&mauthors=Irina+Gheorghiciuc&hl=en) redirected to Google sign-in. The denied response was not retained; it provided no author result. No Scholar ID was inferred from a name match.
- The [ScienceDirect publisher page](https://www.sciencedirect.com/science/article/pii/S0196885806000984) returned HTTP 403 to the capture client; its denied HTML was not retained. The CMU citation and arXiv record still anchor that title.
- The 2014 arXiv record “On super Catalan polynomials” and the 2020 “Super-Catalan Numbers of the Third and Fourth Kind” remain preprint observations. The captured sources do not establish later journal manifestations. The Journal of Integer Sequences PDF cites the former as a preprint.
- The Penn thesis is not collapsed into the 2007 article merely because their topics overlap. A complete Scholar profile could present either or both, and that decision belongs in the later Scholar reconciliation.

Every observation is marked `graph_eligible=false` with the same completeness
hold. Do not create publication or coauthor edges, upload these rows to the Hub,
or present the six groups as a completed researcher until an exact Scholar
account is verified and paginated fully against these sources. If no Scholar
account exists, a researcher-controlled complete bibliography and an explicit
exception decision would be required instead.

Run `PYTHONPATH=. uv run --extra dev python scripts/validate_gheorghiciuc_source_review.py`
to check the pinned responses, row preservation, attribution, and holds.
