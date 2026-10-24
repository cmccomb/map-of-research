# Neil Donahue: exact Scholar graph decisions (2026-10-24)

The [CMU faculty page](https://www.cmu.edu/chemistry/people/faculty/donahue.html) links Neil M. Donahue to Google Scholar profile `ILambBMAAAAJ`. [Draft source PR #56](https://github.com/cmccomb/map-of-research/pull/56) stages the complete visible 623-row profile, its capture receipt, a 43-row suspect ledger, and the 12-ID comparison with the 2025 cache. This producer review was checked against that saved capture. It adds 43 **exact citation-ID** decisions to `data/curated/scholar-exclusions.json`; it does not edit or shorten the 623 raw rows. The excluded observations remain auditable in `profile_publications` but do not create graph authorships. Three of these IDs already occur in the current 551-row cache; the intended release merges these decisions with the full-profile source before rebuilding and publishing the snapshot.

## Decisions

| Captured rows | Graph treatment | Evidence |
| --- | --- | --- |
| 24 NSF award records | Exclude each exact ID as `nonpublication`. Every award number and title was checked against its corresponding [NSF award API](https://resources.research.gov/common/webapi/awardapisearch-v1.htm) record; the manifest links the individual records. | Scholar row plus 24 matching NSF records, all naming Donahue. |
| One NASA proposal record | Exclude exact ID `ILambBMAAAAJ:lRPiJ3GhvscC` as `nonpublication`. | Captured venue says `NASA Proposal ID. 18-ACMAP18-0008`; no paper is inferred. |
| Three imported website or table fragments | Exclude exact IDs `ZYsTHYU9jrMC`, `dhpJJ7xvgBgC`, and `ufKn5pxu7C0C` as `nonpublication`. | Saved Scholar rows contain a library access notice, table text, and a scheduled-maintenance message, respectively. |
| Seven malformed article citations | Exclude each malformed ID as `corrupted_duplicate`; keep the seven complete Scholar citations. | Each manifest entry names the complete candidate ID and links the matching publisher article, including [ACP 2016](https://acp.copernicus.org/articles/16/1693/2016/), [ACP 2020](https://acp.copernicus.org/articles/20/11809/2020/), and [ACP 2018](https://acp.copernicus.org/articles/18/845/2018/). |
| Two conference session titles | Exclude exact IDs `BPS1z4jHU5cC` and `j2GSQqY3pL0C` as `nonpublication`. | [NOAA's AGU23 planner](https://csl.noaa.gov/news/2023/AGU-FM23planner.pdf) distinguishes a session from an individual presentation; the [AGU25 listing](https://agu.confex.com/agu/f/EarthCoveringAGU25) identifies the latter session. |

The six initially unresolved suspect rows were handled individually:

| Exact Scholar suffix | Decision | Basis and limit |
| --- | --- | --- |
| `s2G-WRnXBicC` | Exclude the captured row as `nonpublication`. | The saved title is only “General Discussion in,” with author `C Wagner`, no year, and venue `Thermodynamics, 0`. It cannot identify a Donahue publication. The underlying document, if any, remains unidentified; this decision applies only to the broken row. |
| `3x-KLxxGyuUC` | Exclude as `corrupted_duplicate`. | The title begins `submitter:` and has no venue. The candidate also has a complete 2025 EGUsphere preprint `1wZ_wKGpLuwC` and a complete 2026 article `IExZWSxeYXUC`; the [publisher article](https://amt.copernicus.org/articles/19/3363/2026/) confirms the work. Those complete observations remain. |
| `RVqaWcrwK10C` | Exclude the captured row as `nonpublication`. | “Laboratory Studies in Atmospheric Sciences II Online Poster Discussion” is a session or discussion title, not an individually titled presentation. Its Scholar year, 2022, was not independently corroborated. This decision does not remove any separately documented talk. |
| `JdL-Xu2nR38C` | Exclude as `nonpublication`. | The [ACS issue contents](https://pubs.acs.org/toc/jpcafh/120/9) place this one-page names list in the festschrift preface, adjacent to the separately captured “James G. Anderson Tribute” article `6_hjMsCP8ZoC`, which remains eligible. |
| `TeJ9juy8vcMC` | **Retain.** | The [AMS 2016 program](https://ams.confex.com/ams/96Annual/webprogram/18ATCHEM.html) names Donahue, Hakala, and Hofbauer for the individually titled J10.5 invited presentation. It is an actual profile-listed academic presentation. |
| `F9HO9s0W2bwC` | Exclude as `nonpublication`. | “Processes & Impacts” is a journal-title fragment with no year or venue. [Aarhus University's record](https://pure.au.dk/portal/en/publications/atmospheric-surfaces/) identifies the actual 2018 editorial by Freedman, Bilde, Donahue, and Glasius as “Atmospheric surfaces,” captured completely as `sfnaS5RM6jYC`. |

The same capture also contains `ILambBMAAAAJ:lK9BDNCuzFgC`, “Environmental Science Processes & Impacts,” with no year or venue and a longer contaminated author list. This seventh exact decision was found outside the initial suspect ledger. It is another journal-title fragment, excluded as `nonpublication` while `sfnaS5RM6jYC` remains eligible.

The four independently documented historical conference items among the 12 IDs absent from the current visible Scholar profile remain in the source PR's provenance ledger. This review neither silently carries them forward as 2026 Scholar observations nor claims they were withdrawn. A separate historical-source decision can add them with explicit provenance. The 2025 cache and present graph are unchanged until the full-profile source, this producer manifest, and a rebuilt snapshot pass the normal release sequence.
