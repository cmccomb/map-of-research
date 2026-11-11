# Po-Shen Loh bibliography source audit (review only)

**Decision:** Do not publish or ingest these rows as a whole-researcher capture. There is no verified Google Scholar profile ID in the registry. The 46-row author bibliography is incomplete against current first-party CMU records. The 84-row CMU list is also unverified for whole-career completeness and contains duplicate versions and truncated coauthor displays.

Po-Shen Loh is `person-poshenloh-b545d989`, included in Mathematical Sciences. His [official department profile](https://www.cmu.edu/math/people/faculty/loh.html) links to the [author-owned research/publications page](https://www.math.cmu.edu/~ploh/docs.shtml). The [CMU Scholars profile](https://scholars.cmu.edu/3056-poshen-loh) identifies the same person and an ORCID, but does not establish an exact Google Scholar account.

## Observed sources

| Source | Scope observed on 2026-10-25 | Capture |
| --- | --- | --- |
| [Author bibliography](https://www.math.cmu.edu/~ploh/docs.shtml) | 41 rows under “Papers” and 5 under “Olympiad related”; page states “Last updated 28 February 2021” but includes a 2022 citation | [46-row review CSV](2026-10-25-math-poshen-loh-author-site-observations.csv); [exact source HTML](2026-10-25-math-poshen-loh-author-page.html) |
| [CMU Scholars works](https://scholars.cmu.edu/3056-poshen-loh/publications) | 84 displayed records across four pages (25, 25, 25, 9); 55 distinct titles under strict alphanumeric normalization; 29 additional records repeat a normalized title, often a preprint and journal version | [84-row review CSV](2026-10-25-math-poshen-loh-elements-observations.csv); page URLs and SHA-256 checksums in the [manifest](2026-10-25-math-poshen-loh-bibliography-manifest.json) |
| [Spring 2026 CV](https://www.math.cmu.edu/~lohp/cv.pdf) | Selected publications are described as drawn from 40; this is a CV count with its own scope, not a reconciliation of all author-page categories | Manifest records the PDF checksum; PDF is not copied here |
| [Earlier CMU-hosted CV](https://www.math.cmu.edu/~lohp/cv-general.pdf) | Names A. Frieze and P. Loh’s 2013 *The Probabilistic Method* chapter in *Handbook of Graph Theory*, absent from the author bibliography | Manifest records the PDF checksum; PDF is not copied here |

The author bibliography lacks CMU Scholars records dated [2023](https://scholars.cmu.edu/3056-poshen-loh/publications), [2024](https://scholars.cmu.edu/3056-poshen-loh/publications), [2025](https://scholars.cmu.edu/3056-poshen-loh/publications), and [2026](https://scholars.cmu.edu/3056-poshen-loh/publications). The five newer title observations are rows 1–5 of the CMU Scholars review CSV; the older author page also omits the 2019 preprint *A Simple Proof of the Quadratic Formula* (CMU Scholars row 12). These observations establish missing entries, without establishing that all CMU Scholars records belong in the collaboration graph.

Of the 84 CMU Scholars cards, 45 abbreviate their coauthor display. The CSV preserves only what each card actually displays and labels that limitation. Exact title overlap with the author page is 42 of 46 rows; the other four include wording/version differences and the edited Olympiad book. Neither title overlap nor the 84-card count is a count of distinct verified works. CMU Scholars also displays some arXiv dates that may be later revisions, so the CSV retains them as display dates.

## Collection gate

1. Verify an exact Google Scholar account if one exists. Google Scholar remains the preferred source for this project, and no ID is assigned from a namesake or title search.
2. If no Scholar account exists, obtain or reconcile a current complete author-approved bibliography. Resolve the 2013 chapter, 2019 preprint, and 2023–2026 CMU records against it.
3. Before graph ingestion, reconcile preprint/journal versions, publication dates, full coauthor lists, publication types, and title aliases. The two CSVs are observations, not `author_site` or Scholar records in a release snapshot.

This change contains only review data. It changes no registry identity, source snapshot, pipeline, or published map.
