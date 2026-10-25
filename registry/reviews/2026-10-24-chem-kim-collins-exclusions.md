# Chemistry source quality: Hyung J. Kim and Terrence J. Collins

**Current integration (2026-10-25):** This branch carries only the 31 exact
Collins decisions into producer `main` at `22480ba`. All 1,157 exclusions
already on that main are retained. The eight Kim decisions described below
remain in draft PR #119; they are not part of this integration and Kim has no
canonical author cache. The seven unresolved Collins rows in the
[provisional-hold decision](2026-10-25-chem-collins-provisional-holds.md) are
disjoint from the 31 exclusions and remain raw observations.

Reviewed 2026-10-24 from the complete, paired offline Google Scholar captures in draft source PR #118. Kim's visible profile has 918 rows; Collins's has 361. Both saved HTML pages have a disabled Show More button. All Collins JSON fields match the HTML rows after ordinary whitespace normalization. Kim's JSON citation counts omit the visible `*` annotation on rows 81 and 600; those annotations remain in the saved HTML, and the other compared fields match. The raw captures remain intact. This PR adds only exact Scholar observation IDs to the producer exclusion manifest; it does not change either capture or a map snapshot.

| Profile | JSON SHA-256 | HTML SHA-256 | Source status |
| --- | --- | --- | --- |
| Hyung J. Kim, `bC4aq-4AAAAJ` | `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867` | `a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94` | Raw capture only; canonical cache withheld while the other 910 rows undergo attribution review. |
| Terrence J. Collins, `0e1jwkkAAAAJ` | `d21519b8e5126f785181a84d74d0d75c30341b27db12538a937d144c02200231` | `ce518668aeefa35d81895ed9d47c6b3e35cce46178da38f9d53cf6eccf35d4a9` | Complete 361-row cache staged in source PR; source must remain draft until these exclusions land. |

## Exact exclusions

The eight Kim exclusions are false profile attributions. Seven rows name `H. J. Kim` but [Kyungpook National University](https://research.knu.ac.kr/en/persons/hong-joo-kim/) links that author to physicist **Hong Joo Kim**, distinct from [CMU Chemistry's Hyung J. Kim](https://www.cmu.edu/chemistry/people/faculty/kim.html). For the NEOS journal article, [APS gives H. J. Kim's Kyungpook Physics affiliation](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.121802), and [Kyungpook expands the name on the related NEOS conference paper](https://research.knu.ac.kr/en/publications/sterile-neutrino-search-at-the-neos-experiment-2/). Each other glass, scintillator, COSINE, or KIMS manifest entry links its own exact Kyungpook output page with the author linked to Hong Joo Kim's profile. The remaining [COCOLAB report's society program](https://www.ipsj.or.jp/09sig/kaikoku/2005/UBI8.html) names **Hyeonsook Kim** and no Hyung J. Kim. The exact Kim capture indices are 3, 14, 22, 34, 50, 226, 378, and 695.

The initial nine Collins exclusions are three false attributions, one corrupted duplicate, and five nonpublication/source artifacts. The [publisher's issue PDF](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/s-0037-1610870.pdf) names five authors other than Collins for capture row 180. Capture row 235 is a sole-authored `T Wu` dissertation; the [Garcia-Bosch group](https://garcia-boschgroup.com/lab-news) identifies Tong Wu as its 2023 doctoral graduate. [ACS identifies only Schnoor and Gentleman](https://pubs.acs.org/doi/10.1021/es1018527) as authors of the `Changes to Viewpoints` editorial at row 220. Collins row 349 prepends `Research Article` to the [2007 Wiley paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/clen.200720011) already retained as `0e1jwkkAAAAJ:_xSYboBqXhAC`. Rows 351, 357, and 359 are supporting information for separately indexed ACS, Nature, and RSC articles; their retained row IDs are named in the manifest. Row 352 is a person-profile title, `Prof Terry Collins`, without a publication venue or year. Row 253 is the [publisher-labeled editorial board masthead](https://www.tandfonline.com/doi/pdf/10.1080/17518253.2010.541062), on which Collins is a board member. The exact Collins capture indices are 180, 220, 235, 253, 349, 351, 352, 357, and 359. A [systematic retained-row audit](2026-10-24-chem-collins-retained-row-audit.md) adds 14 more exact Collins decisions and documents records still needing resolution.

The 31 Collins IDs are disjoint from the 1,157 producer exclusions on `main` at
`22480ba`, giving 1,188 unique exact decisions in this branch: 15
`corrupted_duplicate`, 12 `nonpublication`, and four
`false_profile_attribution` for Collins. Every new decision names an exact
saved Scholar ID and supporting source page. The exclusions prevent only
those observations from making eligible map works and coauthor edges. They do
not claim that all other captured rows are correct.

## Retained and deferred

**Do not use topic mismatch as a deletion rule for Kim.** His [Pittsburgh Quantum Institute record](https://www.pqi.org/hyung-kim) explicitly lists the 1985 supergravity paper (capture row 1) and a 2019 microfluidic neural-circuit paper (row 165). These remain despite their subject matter. The institute page also lists his ionic-liquid supercapacitor paper (row 9). Capture row 907 resembles an alternate rendering of the 1985 title, but its bibliographic relationship needs separate verification before exclusion.

Kim's 910 other raw observations remain candidate-only, **not verified publications**. Many `H. J. Kim` records still need author identity checks against full names and affiliations. Collins's undetermined cover and fragment-like rows remain eligible pending exact source comparison; patents, conference outputs, authored reports, and legitimate early organometallic work are retained. [CMU News lists `Call to restrict neonicotinoids`](https://www.cmu.edu/news/experts/terrence.collins) among Collins's work, so that signatory row stays eligible. [UCL](https://discovery.ucl.ac.uk/id/eprint/10178597/) names Terrence J. Collins on `A vision for safer food contact materials`, and [Ecologic](https://www.ecologic.eu/19670) lists him on `Primary plastic polymers`; these cross-field works also stay eligible. No broad author-name, subject, year, citation-count, or output-type filter was applied. Seven unresolved Collins rows will instead be held from map contribution by the source-bound provisional gate in draft PR #243. The cache and producer PRs remain draft and unmerged.
