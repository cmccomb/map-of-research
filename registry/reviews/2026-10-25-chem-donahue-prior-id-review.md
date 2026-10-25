# Neil M. Donahue historical Scholar ID reconciliation

This review reconciles the **12 citation IDs** in Donahue's 551-row frozen
Scholar cache that do not appear in the saved 623-row complete visible-profile
capture. The [machine-readable crosswalk](2026-10-25-chem-donahue-prior-id-crosswalk.json)
is an exact exception to the prior-ID publication hold; it does not put older
rows back into the current Scholar cache. The original observations remain in
the immutable snapshot at commit `4dbafc2a627930a0ef80aa2c645089546c0d874a`
and its cache (`data/authors/ILambBMAAAAJ.json` at that commit). Each crosswalk
entry records the prior row's canonical JSON SHA-256, its observed title and
year, a resolution, and review evidence.

| Prior citation ID suffix | Resolution in the complete capture or curated catalog | Evidence and limit |
| --- | --- | --- |
| `CMvovTBb2okC` | Supporting information for the 2006 food-cooking paper. The main article is curated as DOI `10.1021/es060781p`. | [ACS article](https://pubs.acs.org/doi/10.1021/es060781p); [PubMed authors and metadata](https://pubmed.ncbi.nlm.nih.gov/17256533/). The supplement is retained as a historical observation, not counted as another article. |
| `E8ajGqO0XoUC` | Site-maintenance text now appears as `ufKn5pxu7C0C`; the reviewed exclusion classifies it as a nonpublication. | [Saved current Scholar row](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ILambBMAAAAJ&cstart=500&pagesize=100&citation_for_view=ILambBMAAAAJ:ufKn5pxu7C0C). |
| `GHsHDPAyICYC` | Malformed author-list tail for the current ACP article `LkGwnXOMwfcC`. | [ACP article and byline](https://acp.copernicus.org/articles/10/4625/2010/). |
| `I858iXPj1OkC` | Older, incomplete title of the current HOM review `QsKbpXNoaWkC`. | [ACS Chemical Reviews article](https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00395). |
| `Nnq8S6OXqDYC` | Malformed title of the current dimethyl-sulfide paper `eFf2swCANGcC`. | [ACS Environmental Science & Technology article](https://pubs.acs.org/doi/10.1021/acs.est.2c05154). |
| `U0iAMwwPxtsC` | Supporting information for the Part 1 source-apportionment paper, current Scholar ID `MXK_kJrjxJIC`. | [ACS article and supporting material](https://pubs.acs.org/doi/10.1021/es0510414). The supplement is retained as a historical observation, not counted as another article. |
| `Ul_CLA4dPeMC` | Same 2025 EGU abstract as current ID `hB2aVRuWZNwC`. | [Original EGU abstract](https://meetingorganizer.copernicus.org/EGU25/EGU25-15281.html). |
| `b1wdh0AR-JQC` | Historical EGU2012 abstract absent from the current profile; one curated record `egu2012:egu2012-2516-1`. | [Original EGU abstract](https://meetingorganizer.copernicus.org/EGU2012/EGU2012-2516-1.pdf) names N. M. Donahue. |
| `c1AJUTjuCtUC` | Historical A&WMA proceeding absent from the current profile; one curated record `awma2007:ace-07-cd:3094`. | [Published proceedings contents](https://www.proceedings.com/content/002/002176webtoc.pdf), printed p. 3094, supplies the complete title and N. Donahue in the author list. The individual paper was not recovered. |
| `feST4K8J0scC` | Same 2013 Faraday discussion item as current ID `qSd0DAb9jMoC`. | [RSC item](https://pubs.rsc.org/en/content/articlehtml/2013/fd/c3fd90032k) verifies title, issue, pages, and DOI. The [CU Boulder institutional record](https://vivo-cub.colorado.edu/display/pubid_386319) lists `Donahue N` among the authors but does not expand the initial; the current Scholar profile associates this item with Neil M. Donahue. |
| `t6usbXjVLHcC` | Historical AGU2009 presentation absent from the current profile; one curated record `agu2009:a21h-08`. | [AGU scientific program copy held by UNOLS](https://www.unols.org/sites/default/files/AGU_FM09_Scientific_Program.pdf), p. 103, lists N. M. Donahue. |
| `tOudhMTPpwUC` | Historical EGU2009 abstract absent from the current profile; one curated record `egu2009:egu2009-9565`. | [Original EGU abstract](https://meetingorganizer.copernicus.org/EGU2009/EGU2009-9565.pdf) names N. M. Donahue. |

The complete-capture gate requires the exact frozen snapshot commit and digest,
the frozen cache SHA-256 `783d1ad3c2be87fd30f55cd46b3f8fac3927cece9be00134a92b1fb85afa6678`,
and the replacement 623-row cache SHA-256
`706baa33811d7970d623176601f0a293807c74ca489e7c93d63db2ffbb4ef4ec`.
It accepts only the exact set of 12 missing IDs, verifies every current Scholar
target's ID/title/year or every curated target's unique person/DOI or native
ID/title/year/evidence, and requires the maintenance target's exact
nonpublication exclusion. An academic replacement with an exclusion fails.
The source JSON, HTML, final disabled Show More, and raw snapshot comparison
still run. Future cache changes require a new reviewed crosswalk version.

The staged 623-row HTML has SHA-256
`4c2cb0ee0144c9850133931feb1f7e327da30b788d2b46cc13e8846f7eb45494`.
The initial draft Donahue cache declared the SHA-256 of an **unstaged**
original HTML file, `80a3f8345bc2d6e2536b1059aff8a9ef127b2277c6a4a5aed6482791489a20f6`.
The staged HTML passed all 623 row IDs, visible title text (including SVG
labels), JSON details, and the disabled final Show More check. Draft source
PR #139 commit `f6266aa` changed only
`data/authors/ILambBMAAAAJ.json`'s `browser_capture.source_html_sha256` to
the staged digest above. That is the only difference between the original
cache SHA-256
`424269fcfed5f480ffda279f6013a3a6841f49f5849e5e559a949871106fdba3`
and the replacement SHA-256 pinned above.

This exception preserves the full **current visible Scholar profile** and the
verified historical academic works found in the prior cache. It does not claim
that Scholar lists every lifetime publication. The 12 obsolete IDs are not
retained as current raw observations; they remain inspectable in the frozen
snapshot and this crosswalk. Publishing the 623-row cache still depends on
merging the curated historical works, the maintenance exclusion, and the
repaired draft source cache.
