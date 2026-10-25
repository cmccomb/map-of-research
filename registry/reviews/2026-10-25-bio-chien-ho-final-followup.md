# Chien Ho: final exact-ID attribution follow-up — 2026-10-25 UTC

This follow-up applies to the saved **582-row visible Scholar capture** for `8iLLOQgAAAAJ`. The [initial disposition](2026-10-25-bio-chien-ho-582-disposition.csv) is frozen. The [reconciled disposition](2026-10-25-bio-chien-ho-582-followup-reconciled.csv) records each later source check, exact citation ID, evidence URL, and final review state. The saved capture itself remains unchanged.

| Decision | Exact Scholar rows | Change from initial review |
| --- | ---: | ---: |
| Independently confirmed Chien Ho attribution | 424 | +155 |
| Source-backed exclusion from the map | 88 | +53 |
| Held pending attribution or publication decision | 70 | −208 |
| **Total raw observations retained** | **582** | **0** |

The [positive attribution manifest](../../data/curated/scholar-attribution-reviews.json) allows only those 424 exact IDs through the mixed-identity profile gate. Its original 269 entries cite numbered works in [Ho's CMU bibliography](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf). Later confirmations cite original research, publisher-deposited DOI records, official conference abstracts, or institutional author sources; entries without a CMU bibliography number include the source-specific review basis. The ordinary publication-quality rules still apply after attribution confirmation.

The [reviewed exclusion manifest](../../data/curated/scholar-exclusions.json) now contains 88 exact Chien Ho-profile decisions. They cover another person's work imported into this profile, entries with no Ho author, malformed nonworks, and duplicate Scholar records whose canonical version is retained. The source URL and reason are stored on each exact ID. The [reconciled ledger](2026-10-25-bio-chien-ho-582-followup-reconciled.csv) records the retained counterpart for duplicate decisions when identified.

The **70 held rows** include 48 still lacking adequate exact-work evidence, 9 with unresolved metadata, 12 confirmed Ho author/context cases whose map publication status remains undecided (including patents, a withdrawn paper, and a panel or edited-work entry), and one potential canonical-row swap. They are absent from mapped works and coauthor links while remaining in the raw observation table. In particular, the 1974 Pratt–Ho *Federation Proceedings* abstract and the 2007 Song et al. *Biophysical Journal* abstract are still held because the original abstract pages or an author-owned exact citation were not found. Later related papers were not used to infer those abstracts' bylines.

The last six dated holds received a separate primary-source pass. Original *Biophysical Journal* pages [88a (1985)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1435322/), [A298 (1998)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1913948/), and [441a (2002)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721761/) confirm three exact IDs. The [publisher-deposited registration for a 2006 *Critical Care Medicine* abstract](https://api.crossref.org/works/10.1097%2F00003246-200612002-00017) confirms another; its original page scan was not inspected. All four remain distinguishable from related full articles.

A final dated source pass resolved 14 additional positive IDs and three exclusions. It left a [withdrawn article](https://pubmed.ncbi.nlm.nih.gov/15194702/) on hold pending publication treatment. A coauthor's CV identifies another exact work but conflicts with Scholar's page number, so that row remains held for metadata repair.

A subsequent exact-work check excludes Scholar row 495 (`A8cqit5AE6sC`) as a corrupted duplicate of retained row 170 (`JoZmwDi-zQgC`). [Publisher-deposited DOI metadata](https://api.crossref.org/works/10.1021%2Fbi048289a) and the [NLM record](https://pubmed.ncbi.nlm.nih.gov/15882059/) identify one *Biochemistry* 44:7207–7217 article with Chien Ho; row 495's title is a broken author list. Both original Scholar observations remain in the raw capture.

This is a review of the **visible profile captured on 2026-10-24**, with 70 decisions still held. It does not establish that Scholar contains every publication in Ho's career, and it does not release a partial Chien Ho graph as complete. Source capture PR #162, this review PR, identity assignment, snapshot build, and public deployment remain separate steps.
