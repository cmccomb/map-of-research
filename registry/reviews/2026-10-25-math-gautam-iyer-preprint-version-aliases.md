# Gautam Iyer row 22: two exact Pego preprint versions

Reviewed 2026-10-25. This draft stacks on the six Iyer/Pego DOI aliases in PR #341. It adds only Pego Scholar observations `neJ3NIUAAAAJ:VL0QpB8kHFEC` and `neJ3NIUAAAAJ:NJ774b8OgUMC` to the existing Iyer author-site DOI work `10.4310/cms.2020.v18.n7.a2`. Iyer's source remains `author_site`, with no inferred Scholar account. The trusted publisher still blocks author-site observations.

## Exact version evidence

| Observation | Observed record | Independent version evidence |
| --- | --- | --- |
| Pego `VL0QpB8kHFEC` | “Anomalous diffusion in one and two dimensional combs,” 2018, “To Be Submitted” | [arXiv:1809.01601 v1](https://arxiv.org/abs/1809.01601v1) has this title, the four listed authors, and report `18-CNA-018`; [CMU's report](https://www.math.cmu.edu/CNA/Publications/publication_auto.php?cnacode=18-CNA-018) has the same title, authors, and abstract. |
| Pego `NJ774b8OgUMC` | “Anomalous diffusion in comb-shaped domains and graphs,” 2018, arXiv:1809.01601 | [arXiv:1809.01601 v2](https://arxiv.org/abs/1809.01601v2) retains the identifier, report number, and four authors after the retitle; [Iyer's author manuscript](https://www.math.cmu.edu/~gautam/sj/research/papers/201809-fcomb.pdf) has the revised title and byline. |
| Iyer author-site row 22 | Older 2018 title and venue as listed; reviewed DOI and journal year 2020 | The author's linked DOI resolves to the 2020 *Communications in Mathematical Sciences* article in the [Crossref DOI record](https://api.crossref.org/works/10.4310%2FCMS.2020.v18.n7.a2). The author-site capture includes a reviewed `publisher_title` and `publisher_year` pinned by its raw source-record SHA-256. The prior [DOI audit](2026-10-25-math-gautam-iyer-doi-audit.csv) records the stale author-page title/year separately. |

The two Scholar observations are explicitly `preprint_of` the DOI work. Each ledger entry pins its Scholar raw-record hash, the exact Iyer person/publication/observation identity, DOI, target raw-record hash, and the 47-row capture SHA-256 `421e40cf104781b1500644f9f74ad81ce6768881edcf55daf9463626d69ff6eb`. The published display is an exact override for this target only: title “Anomalous diffusion in comb-shaped domains and graphs,” year 2020, venue “Communications in Mathematical Sciences.” Validation requires the title, year, DOI, and registry URL to match the pinned target's `doi_review`. All three raw titles, years, venues, bylines, citations, URLs, and source records remain as observed.

## Frozen full-snapshot diff

The offline comparison used the unchanged 51,926-observation Parquet snapshot with SHA-256 `9bf9570577618313c333155836187e800c55e50526950d98650decc08321e364`. Previous 58-alias and candidate 60-alias builds both retain 51,926 profile observations, 50,899 authorships, and 2,001 people. Canonical works change from 47,783 to 47,781: only `work-0b3cb4a05af32592b65c12b4128d61b03be90fab4c6e22d99cb333c7bc2b23b0` and `work-cc2354a1925de7667dfff90271d86698f3e80abc91fbf9a5412cd25152908c0b` disappear, joining existing DOI work `work-776e058801538dbc2f51a1a60a756ea17573cf20df038fe5d9f309240a738236`.

The complete diff proves equality of 51,924 unaffected profile rows, 50,897 unaffected authorship rows, 47,780 unaffected work rows, and 2,000 unaffected people rows. Robert Pego alone changes derived `unique_work_count` and `mapped_work_count` from 191 to 190; his 196 Scholar observations remain. The DOI work holds three observations, both faculty, both 2018 and 2020 year variants, all three title variants, and citation count 1 rather than a sum. The full suite passed 1,149 tests with 100% coverage; Ruff check and format passed.

Iyer row 4 and the Slepčev arXiv retitle remain held. They have an exact arXiv version identity but no reviewed DOI target; a separately validated native-ID alias lane would be needed. This draft contains no general title merge. The source snapshot, Hub dataset, and public maps were not changed.
