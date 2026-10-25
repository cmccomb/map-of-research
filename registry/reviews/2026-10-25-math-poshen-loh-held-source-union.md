# Po-Shen Loh held source union

**Status:** Review only. `person-poshenloh-b545d989` has no verified exact Google
Scholar account or proven complete career bibliography. None of these records
can enter the publication graph. This draft depends on the pinned bibliography
in PR #336 and the crosswalk in PR #353.

## Sources and accounting

| Collection | Observations | Evidence |
| --- | ---: | --- |
| [Loh's author bibliography](https://www.math.cmu.edu/~ploh/docs.shtml) | 46 | Pinned HTML and row CSV from PR #336; 41 Papers and 5 Olympiad related entries. The page states an update date of 28 February 2021. |
| [CMU Scholars publications](https://scholars.cmu.edu/3056-poshen-loh/publications) | 84 | Four captured pages and row CSV from PR #336; 45 cards abbreviate coauthors. |
| [arXiv exact-author result](2026-10-25-math-poshen-loh-arxiv-author-crosswalk-manifest.json) | 38 | Complete result-row CSV and Atom hash from PR #353; the raw Atom body was not committed there. |
| [2013 chapter DOI](https://api.crossref.org/works/10.1201%2Fb16132-60) | 1 | Full Crossref response pinned in `data/source_reviews/po-shen-loh-2026-10-25/`; also listed in Loh's [CMU-hosted CV](https://www.math.cmu.edu/~lohp/cv-general.pdf). |
| **Total source observations, not unique works** | **169** | Every existing CSV row is copied into `raw_fields` in the held JSONL. |

The deterministic builder `scripts/build_poshen_loh_held_union.py` validates
the prior source counts and hashes, preserves every original row, and writes
[the JSONL](2026-10-25-math-poshen-loh-held-source-union.jsonl),
[eight review cases](2026-10-25-math-poshen-loh-held-reconciliation-cases.json),
and a [manifest](2026-10-25-math-poshen-loh-held-source-union-manifest.json)
with hashes of all inputs and outputs. Run `PYTHONPATH=. uv run --extra dev
python scripts/build_poshen_loh_held_union.py --check` to reproduce and compare
the committed bytes. Each observation has `graph_eligible: false` and a hold
reason. Candidate correspondence IDs are associations for review, not merged
publications or asserted coauthor edges.

The prior crosswalk links 45 author rows to 77 CMU cards; the remaining author
row is an edited Olympiad volume. CMU cards 1–5, 12, and 20 have no author-page
row. All 38 arXiv results have CMU card candidates, and 35 have author-page
candidates. These counts verify this bounded union; they do not establish
whole-career completeness.

## Resolved evidence and open relationships

- [Frieze and Loh's 2013 *The Probabilistic Method* chapter](https://api.crossref.org/works/10.1201%2Fb16132-60)
  is missing from both lists. Its publisher-registered DOI and the older CV
  support one additional held source observation.
- CMU cards 82 and 84 share the title *Finding Shortest Paths With Computational
  Geometry*. The [2006 chapter DOI](https://api.crossref.org/works/10.1142%2F9789812773296_0013)
  and [2003 journal DOI](https://api.crossref.org/works/10.7155%2Fjgaa.00071)
  confirm distinct genres. Whether the chapter reprints or derives from the
  article remains open, so both cards remain separate observations.
- CMU card 20's [Lavrov–Loh DOI](https://api.crossref.org/works/10.1002%2Frsa.20592)
  resolves the older CV's reversed title opening. Crossref records online
  publication in 2015; CMU and the CV use its 2016 volume. This is one DOI
  identity with two publication date contexts.
- The author page and its linked arXiv/citation evidence connect a preprint,
  conference version, and journal version of *Anarchy is free in network
  creation*. The same evidence connects the title variant of the F-saturated
  preprint and journal article. Their manifestations remain separate.
- [arXiv:2011.04650](https://arxiv.org/abs/2011.04650) was first submitted
  in 2020; CMU's 2026 date reflects a later revision. The two seven-author
  COVID survey records remain a possible version pair pending content review.
- The author-page Olympiad volume is an edited book, not a research article;
  the author page supplies its broad section label and the older CV supports
  the editor interpretation.

No exact Google Scholar profile was corroborated by a CMU or researcher-owned
link. The author page has documented gaps; CMU Elements has shortened bylines
and no evidence that it captures every career output. The seven CMU-only cards,
the 2013 chapter, and genre/version differences make promotion of a partial
person especially unsafe. Obtain a corroborated Scholar ID and complete Scholar
capture, or a current author-approved whole-career bibliography, before a
separate ingestion review.
