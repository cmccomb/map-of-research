# Published mathematics Scholar captures: offline source triage

Reviewed 2026-10-24. This is an **offline screen for false attribution, nonpublications, and corrupted duplicates**, not independent confirmation of every retained work. No new Scholar request was made. The exact-ID decisions are in the [checked-in exclusion manifest](../../data/curated/scholar-exclusions.json). This review alone does not upload or deploy a new dataset.

## Capture integrity and scope

| Scholar account | Visible rows | Capture | Evidence files | Exclusions proposed |
| --- | ---: | --- | --- | ---: |
| Giovanni Leoni `0Nj-v7sAAAAJ` | 134 | Complete visible profile; limit 200; `more_remain=false` | `work/scholar-bootstrap/leoni-visible-134-2026-10-24.json` SHA256 `78e1bb2e564fb313d1b4c4b6bb11b9543fd32b30e2ad2ff3100a63597a20aeb7`; paired HTML SHA256 `3047d9e35beee4a82f0995da0b41239f1872cabe26f356871c445bdab3e12b3c` | 20 |
| Tomasz Tkocz `4JAHlmcAAAAJ` | 69 | Complete visible profile; limit 1000; `more_remain=false` | Paired source JSON SHA256 `49510fbacf65f6828344579b9753ca02fb8fdbdf6c3f1f58c449d8ecc570cd56`; HTML SHA256 `dc4d7714ac97a5b3dd89fd4174445c07b2dfa03405401f3270e5eb1b144cf7ff` | 0 |
| John Mackey `uOCVq1cAAAAJ` | 19 | Complete visible profile; limit 5000; `more_remain=false` | Paired source JSON SHA256 `ff01bd32a2c0f6a96a3ccd73716c10d3eb8f576fdd10bc77eef0ea0e20144a40`; HTML SHA256 `21ccb3110a685f37f932c2bf9e5c7fac3325ec23d0fa4e67e7bedd201a808f8b` | 1 |

The canonical author caches and paired source files are under `data/authors/` and `data/captures/` on the snapshot branch. SHA256 of all six local source files matched the canonical cache metadata. All 21 proposed citation IDs occur exactly once in their corresponding caches. Complete capture means all *visible* profile rows were saved; it does not prove Scholar itself has every work.

## Exact-ID exclusion decisions

The manifest fragment records the precise citation ID, reason code, one-line rationale, primary evidence URL, and verification date for every decision. The evidence groups are:

| Group | IDs (citation suffix after account ID) | Primary check |
| --- | --- | --- |
| Seven Leoni grant records | `_Re3VWB3Y0AC`, `UHK10RUVsp4C`, `tzM49s52ZIMC`, `uLbwQdceFCQC`, `W5xh706n7nkC`, `JQOojiI6XY0C`, `_Ybze24A_UAC` | [NSF award search](https://www.nsf.gov/awardsearch/) IDs 0635983, 1412095, 2108784, 0708039, 0405423, 1714098, 1007989. The official NSF API titles matched the rows; they are awards, not publications. |
| Leoni nonpublication headings and documents | `5ugPr518TE4C`, `cFHS6HbyZ2cC`, `M3NEmzRMIkIC`, `UxriW0iASnsC`, `5awf1xo2G04C`, `EUQCXRtRnyEC` | [MSP article PDF](https://msp.org/apde/2015/8-2/apde-v8-n2-p02-p.pdf), [SISSA slides](https://iris.sissa.it/retrieve/dd8a4bf8-b908-20a0-e053-d805fe0a8cb0/Leoni_SISSA.pdf), [CVGMT](https://cvgmt.sns.it/person/220/), [Piovano's CMU CV](https://www.math.cmu.edu/~ppiovano/PaoloPiovano_files/Curriculum%20Vitae%20CMU%20Website.pdf), [CMU Libraries](https://www.library.cmu.edu/about/news/2023-10/kilthub), [SIAM meeting minutes](https://www.siam.org/media/igpnlvld/siag_pde_11.pdf). These rows represent a journal header, talk slides, repository header, another person's CV, repository name, and meeting minutes respectively. |
| Leoni false attribution | `fPk4N6BV_jEC` | The [JAMS article DOI](https://doi.org/10.1090/S0894-0347-98-00277-X) corresponds to Daskalopoulos and Hamilton, pp. 899–965; Leoni is absent. The imported title itself is an author-list artifact. |
| Leoni corrupted duplicates | `NMxIlDl6LWMC`, `LPZeul_q3PIC`, `blknAaTinKkC`, `vV6vV6tmYwMC`, `fQNAKQ3IYiAC`, `J_g5lzvAfSwC` | [CVGMT preprint](https://cvgmt.sns.it/paper/1602/), [JFA corrigendum](https://www.sciencedirect.com/science/article/pii/S0022123613004242), [Leoni CVGMT bibliography](https://cvgmt.sns.it/person/220/), [CMU 2001 report](https://www.math.cmu.edu/CNA/Publications/publications2001/009abs/009abs.htm), [thin-film erratum](https://www.gillesfrancfort.com/assets/files/bending-errata-2007-october-16%20copy.pdf), and [Francfort publication list](https://gillesfrancfort.com/page3.html). Valid historical reports, published versions, and substantive corrections remain in the cache. |
| Mackey corrupted duplicate | `Se3iqnhoufwC` | [Springer CICM proceedings](https://link.springer.com/book/10.1007/978-3-031-66997-2) and the [author PDF](https://www.cs.cmu.edu/~mheule/publications/CICM24.pdf) give the four-author published chapter, pp. 21–41, retained as `ufrVoPGSRksC`. The 2023 arXiv version remains too. |

## Explicit retentions

- Leoni's substantive [2014 JFA corrigendum](https://www.sciencedirect.com/science/article/pii/S0022123613004242) `QIV2ME_5wuYC` and [2008 thin-film erratum](https://www.gillesfrancfort.com/assets/files/bending-errata-2007-october-16%20copy.pdf) `qUcmZB5y_30C` are retained, along with the original articles. The proposed exclusions affect only corrupted copies.
- Leoni's historical CMU reports `BqipwSGYUEgC` and `dhFuZR0502QC`, the 2001/2002 two-gradient preprint/publication family, [2017 separate Cahn–Hilliard note](https://arxiv.org/abs/1705.00606) `dshw04ExmUIC`, and [authored BIRS workshop report](https://www.birs.ca/workshops/2018/18w5094/report18w5094.pdf) `tS2w5q8j5-wC` are retained.
- Tkocz has no proposed exclusions. His two captured lecture-note sets `TQgYirikUcIC` and `blknAaTinKkC` are genuine authored academic documents ([2020 notes](https://www.math.cmu.edu/users/ttkocz/teaching/1920/conv-discr-geom-notes.pdf), [2019 notes](https://www.math.cmu.edu/~ttkocz/teaching/1819/asympt-conv-geom-notes.pdf)). His [personal paper list](https://www.math.cmu.edu/~ttkocz/mathematics.php) supports the broader research line; this review did not independently check all 69 works.
- Tkocz's 2026 `GnPB-g6toBAC` is a genuine three-author [Mathematische Zeitschrift article](https://link.springer.com/article/10.1007/s00209-026-04016-5); Scholar appends a stray “: J. Jakimiuk et al.” citation fragment to the title. Keep the row and repair its display metadata separately.
- Mackey's 1994 thesis `IjCSPb-OGe4C`, [2009 RIT technical report](https://repository.rit.edu/article/983/) `qjMakFHDy7sC`, short [SAT 2020](https://tuhat.helsinki.fi/ws/files/142452772/sc2020_proceedings.pdf) `Y0pCki6q_DkC` and [MaxSAT 2024](https://helda.helsinki.fi/server/api/core/bitstreams/5066b1c4-72e6-465e-b5c4-c52a5d7d6837/content) `_FxGoFyzp5QC` contributions, and [2023 arXiv preprint](https://arxiv.org/abs/2311.03645) `WF5omc3nYNoC` are retained. CMU's [departmental publication list](https://www.cmu.edu/math/research/docs/sample-faculty-publications-2018-2023.pdf) and [SCS faculty page](https://www.csd.cmu.edu/people/faculty/john-mackey) establish his mathematical research identity.

## Unresolved; no exclusion proposed

| Account / citation ID | Issue and next check |
| --- | --- |
| Leoni `0Nj-v7sAAAAJ:JV2RwH3_ST0C` | Generic 2012 “Center for Nonlinear Analysis” could be an institutional page, but the exact underlying document was not identified. |
| Leoni `0Nj-v7sAAAAJ:ZfRJV9d4-WMC` | “Density estimates for phase transitions with spatially dependent wells” is tied to a 2026 meeting in Scholar. Need the meeting abstract or an independent preprint/paper to classify it. |
| Leoni `0Nj-v7sAAAAJ:zA6iFVUQeVQC` and `0Nj-v7sAAAAJ:tKAzc9rXhukC` | Two title variants may represent the legitimate 2001 preprint of the 2002 two-gradient paper; retain until the versions can be aligned. |
| Leoni `0Nj-v7sAAAAJ:NhqRSupF_l8C` and `0Nj-v7sAAAAJ:uWQEDVKXjbEC` | [arXiv 1605.08315](https://arxiv.org/abs/1605.08315) has an earlier water-waves title and later free-boundary title. They refer to the same evolving preprint; retain version history pending a work-level dedup policy. |
| Leoni `0Nj-v7sAAAAJ:SdhP9T11ey4C`, `0Nj-v7sAAAAJ:GnPB-g6toBAC`, `0Nj-v7sAAAAJ:2P1L_qKh6hAC` | Garbled title, “x1. Introduction.” fragment, and mojibake respectively. They are probably extraction artifacts, but no primary source linked to each exact underlying document was found. Avoid discarding a historical work without that match. |
| Tkocz `4JAHlmcAAAAJ:LkGwnXOMwfcC` and `4JAHlmcAAAAJ:RHpTSmoSYBkC` | Genuine articles whose titles lose the final math notation. The [ECP publisher](https://www.maths.tcd.ie/EMIS/journals/EJP-ECP/article/view/1599.html) gives the first title ending “in C^n”; the [author manuscript](https://arxiv.org/abs/2211.01986) and [Technion record](https://cris.technion.ac.il/en/publications/resilience-of-cube-slicing-in-lp/) give the second ending “in ℓ_p”. Keep both and repair display metadata separately. |
| Mackey `uOCVq1cAAAAJ:zYLM7Y9cAGgC` | Genuine distinct [2022 Journal of Automated Reasoning article](https://doi.org/10.1007/s10817-022-09623-5), following the 2020 IJCAR proceedings version. Scholar appends a stray “: J. Brakensiek et al.” to the title. Keep; display metadata may be repaired. |

## Reproduction check

The fragment uses the repository's `schema_version: 1` exclusion shape with reason codes `false_profile_attribution`, `nonpublication`, and `corrupted_duplicate`. A local validation confirmed that all 21 IDs are unique and present exactly once in the three author caches, all six paired-capture SHA256 hashes match the cache metadata, and all three captures say `complete_visible_profile` and `more_remain=false`. No producer data or site artifact was modified by this triage.
