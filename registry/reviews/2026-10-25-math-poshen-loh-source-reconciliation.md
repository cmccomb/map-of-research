# Po-Shen Loh source reconciliation (review only)

**Decision:** Keep `person-poshenloh-b545d989` out of whole-researcher ingestion. No exact Google Scholar account is attributable to him from his [CMU profile](https://www.cmu.edu/math/people/faculty/loh.html), [author site](https://www.math.cmu.edu/~ploh/cmu.shtml), [CMU Scholars profile](https://scholars.cmu.edu/3056-poshen-loh), linked CVs, or indexed direct-profile searches. This is an unresolved identity, not an assertion that he has no account. The author bibliography has documented omissions, and the CMU Scholars cards are not a verified complete career list.

This review extends the [pinned 46-row author bibliography and 84-row CMU Scholars capture](2026-10-25-math-poshen-loh-bibliography-audit.md) without changing either source, the person registry, the source snapshot, or the graph. The original [manifest](2026-10-25-math-poshen-loh-bibliography-manifest.json) records their source URL and SHA-256 evidence. The [46-row crosswalk](2026-10-25-math-poshen-loh-author-elements-crosswalk.csv) and [seven-card unmatched list](2026-10-25-math-poshen-loh-elements-only-crosswalk.csv) account for every displayed row.

| Reconciliation | Author rows | CMU cards | Basis |
| --- | ---: | ---: | --- |
| Matching displayed titles | 43 | 74 | Case, accent, punctuation, and HTML-tag-normalized title; multiple CMU cards can share one author row. |
| Source-backed title variants | 2 | 3 | The author-linked arXiv IDs identify CMU cards 9 and 10; the author citation also identifies the journal version at card 8. |
| Only in the author list | 1 | 0 | The edited *USA and International Mathematical Olympiads 2004* volume is absent from the displayed CMU cards. |
| Only in the CMU cards | 0 | 7 | Cards 1–5, 12, and 20 have no author-page row; five are labeled preprints and two journals. |

For this crosswalk, a title key removes HTML tags, folds Unicode accents and case, and keeps letters and digits. It is only a candidate association for review. The earlier manifest's stricter title metric (55 title strings and 29 repeated cards) is unchanged. Neither title-key count equals a count of distinct verified publications, and this crosswalk does not merge any preprint with a journal article. CMU cards abbreviate coauthor display on 45 of 84 rows.

A bounded [arXiv exact-author search](2026-10-25-math-poshen-loh-arxiv-author-crosswalk-manifest.json) returned 38 entries; the [row-level arXiv crosswalk](2026-10-25-math-poshen-loh-arxiv-author-crosswalk.csv) records its complete displayed author lists and first submission dates. All 38 map to a CMU card by arXiv DOI or normalized title, and 35 map to an author-page row. The three author-page gaps are arXiv:1910.06709, 2011.04650, and 2504.10664. This checks the observed arXiv result set only; it does not prove either list covers every publication genre.

The seven unmatched CMU cards have independent authorship evidence in the CSV. Three deserve particular care:

- Card 1 is dated 2026 by CMU because [arXiv version 4](https://arxiv.org/abs/2011.04650) was revised on 27 May 2026; the work was first submitted on 8 November 2020. It is absent from the author page, but should not be counted as a newly first-published 2026 work.
- Cards 4 and 5 name the same seven-person team in [journal DOI metadata](https://api.crossref.org/works/10.4018/IJHISI.353901) and the [medRxiv title page](https://www.medrxiv.org/content/10.1101/2023.07.20.23292950v1.full.pdf). They may be versions of one study; their distinct titles and outputs stay separate observations pending content review.
- Card 20 is a real missing research paper, [*Increasing Hamiltonian paths in random edge orderings*](https://onlinelibrary.wiley.com/doi/abs/10.1002/rsa.20592), by Mikhail Lavrov and Loh. Wiley and the [earlier CMU-hosted CV](https://www.math.cmu.edu/~lohp/cv-general.pdf) both identify it. The CV reverses the first two words of the title; the DOI and journal citation identify the same paper.

Matching titles can also hide missing version metadata. Author-page paper 22 cites the 2013 conference version of *Anarchy is free in network creation*; CMU cards 22, 41, and 45 display its 2016 journal, 2013 preprint, and 2013 conference records. The earlier CMU-hosted CV names the journal version. The crosswalk connects those cards for review but preserves their separate genres and dates.

The earlier CV also lists Frieze and Loh's 2013 chapter [*The Probabilistic Method*](https://www.routledge.com/Handbook-of-Graph-Theory/Gross-Yellen-Zhang/p/book/9781439880180) in *Handbook of Graph Theory*. It is absent from both the 46-row author page and the 84 displayed CMU cards. This is a separate chapter observation, not a ninth source card or an inferred graph edge.

**Release hold:** First identify an exact Scholar account, if one exists, or assemble an author-approved current whole-career bibliography. Reconcile the seven unmatched CMU cards, the 2013 chapter, the edited book, publication genres, version pairs, dates, and complete coauthor lists. The crosswalk files are review artifacts with no release eligibility.
