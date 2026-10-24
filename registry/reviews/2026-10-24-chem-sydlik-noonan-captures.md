# Chemistry Scholar evidence: Sydlik and Noonan — 2026-10-24

This branch preserves two **complete visible-profile** Google Scholar captures as raw, paired HTML and JSON. It does not add canonical author caches or change the map snapshot. The saved HTML has both hidden `xsrf` values redacted as `[REDACTED]`; the visible publication rows remain unmodified.

| Person and primary department | Scholar identity | Visible rows | Capture checks |
| --- | --- | ---: | --- |
| Stefanie A. Sydlik, Chemistry | The [CMU Chemistry faculty page](https://www.cmu.edu/chemistry/people/faculty/sydlik.html) and [CMU Scholars record](https://scholars.cmu.edu/4721-stefanie-sydlik/about) corroborate the mapped faculty identity and linked Scholar ID [`rQ79IvMAAAAJ`](https://scholar.google.com/citations?user=rQ79IvMAAAAJ&hl=en). | 114 | Show More reached disabled state after three page loads; no challenge or visible error. |
| Kevin Noonan, Chemistry | The [CMU Chemistry faculty page](https://www.cmu.edu/chemistry/people/faculty/noonan.html) and [CMU Scholars record](https://scholars.cmu.edu/3742-kevin-noonan/about) corroborate the mapped faculty identity and linked Scholar ID [`wmP8_2AAAAAJ`](https://scholar.google.com/citations?user=wmP8_2AAAAAJ&hl=en). | 112 | Show More reached disabled state after three page loads; no challenge or visible error. |

The batch used direct profile URLs in a bounded headed Chrome session and closed the session after capture. Complete visible-profile coverage means every row then shown by the profile through the disabled Show More button; it does not prove that Scholar itself is exhaustive or that every row is a research publication.

| Source file stem under `data/captures/` | JSON SHA-256 | Redacted HTML SHA-256 | Offline paired-capture checks |
| --- | --- | --- | --- |
| `sydlik-visible-114-2026-10-24` | `5ba5732342a521b4c0ac633d1d9741337c6926d21ea712aa70aeac445a85e5cc` | `75b4f7923b51e4fb82a6ac8071a0085b59314b338dc970e6b97fb9e43b4b19cb` | 114 unique publication IDs; 5 blank years, 5 blank venues; 53 zero/blank citations; no missing author. |
| `noonan-visible-112-2026-10-24` | `92da37f9f476c9744844dae25736022a4eccf6320e2f66ceb04d162d004c0b5f` | `52cf9bfaf6ade45fab57c1ae6884a010bb3777b5956e1728c8a083e24bc40c9c` | 112 unique publication IDs; 5 blank years, 8 blank venues; 49 zero/blank citations; two rows with missing author. |

The offline converter matched every row's title, authors, venue, year, citation display, and citation URL between HTML and JSON. No missing fields were guessed. Candidate caches were made outside this branch for the later coordinated release.

## Exact producer exclusions proposed separately

The ten rows below remain in the raw capture and are staged for producer review in [PR #62](https://github.com/cmccomb/map-of-research/pull/62). They are not research publications or are a duplicate. Their exact `author_pub_id` values permit surgical exclusion from map authorship edges.

| Scholar row ID | Decision and primary source |
| --- | --- |
| `rQ79IvMAAAAJ:4OULZ7Gr8RgC` | [NSF award 1905665](https://api.nsf.gov/services/v1/awards/1905665.json) |
| `rQ79IvMAAAAJ:cFHS6HbyZ2cC` | [NSF award 2318652](https://api.nsf.gov/services/v1/awards/2318652.json) |
| `wmP8_2AAAAAJ:xtRiw3GOFMkC` | [NSF award 1455136](https://api.nsf.gov/services/v1/awards/1455136.json) |
| `wmP8_2AAAAAJ:NhqRSupF_l8C` | [NSF award 1809658](https://api.nsf.gov/services/v1/awards/1809658.json) |
| `wmP8_2AAAAAJ:CHSYGLWDkRkC` | [NSF award 2109065](https://api.nsf.gov/services/v1/awards/2109065.json) |
| `wmP8_2AAAAAJ:1sJd4Hv_s6UC` | [NSF award 2404367](https://api.nsf.gov/services/v1/awards/2404367.json) |
| `wmP8_2AAAAAJ:tOudhMTPpwUC` | [NSF award 2318652](https://api.nsf.gov/services/v1/awards/2318652.json) |
| `wmP8_2AAAAAJ:zA6iFVUQeVQC` | [Cornell eCommons dataset](https://ecommons.cornell.edu/entities/publication/bda05268-8a12-463d-85af-63df2038a7e7), separate from its article. |
| `wmP8_2AAAAAJ:XiSMed-E-HIC` | [Cornell eCommons dataset](https://ecommons.cornell.edu/entities/publication/e7e1913d-3ee9-4342-80cb-95263484044e), separate from its article. |
| `wmP8_2AAAAAJ:l7t_Zn2s7bgC` | Damaged duplicate of the complete `wmP8_2AAAAAJ:WbkHhVStYXYC` KillSwitch record; the [ACS article](https://pubs.acs.org/doi/10.1021/jacs.6c11419) confirms one 2026 publication. Retain the complete row. |

Noonan's `wmP8_2AAAAAJ:K3LRdlH-MEoC` is an ACS URL string without an author or year and is already rejected by the generic quality rule. Other ambiguous metadata remain eligible pending exact source review: Sydlik's editorial masthead `rQ79IvMAAAAJ:OU6Ihb5iCvQC` and garbled thesis duplicate `rQ79IvMAAAAJ:UxriW0iASnsC`; Noonan's reviewer list `wmP8_2AAAAAJ:Tiz5es2fbqcC`, journal front matter `wmP8_2AAAAAJ:KlAtU1dfN6UC`, and truncated title `wmP8_2AAAAAJ:u_35RYKgDlwC`. These are unresolved output/provenance decisions, not a basis to discard the complete captures.
