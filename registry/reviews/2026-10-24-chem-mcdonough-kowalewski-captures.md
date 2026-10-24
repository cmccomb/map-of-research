# Chemistry Scholar evidence: McDonough and Kowalewski — 2026-10-24

This review accompanies **raw, complete visible-profile captures only**. It does not add author caches, alter the canonical map snapshot, or approve every Scholar row as a publication. The saved HTML contains `[REDACTED]` in both hidden `xsrf` values per profile; the visible publication rows are unmodified. Candidate publication decisions below are separate from those source files.

## Identity and capture completeness

| Person and primary department | Scholar ID and identity evidence | Visible rows | Capture checks |
| --- | --- | ---: | --- |
| Carrie McDonough, Chemistry | [`Xq5HrPYAAAAJ`](https://scholar.google.com/citations?user=Xq5HrPYAAAAJ&hl=en) is directly linked from [CMU's McDonough profile](https://energy.cmu.edu/directory/bios/mcdonough-carrie.html); her [CMU lab page](https://groups.chem.cmu.edu/mcdonough/) states Associate Professor of Chemistry. | 55 | One Show More; disabled at articles 1–55; no challenge or page error. |
| Tomasz Kowalewski, Chemistry | The user supplied [`6VKJW4kAAAAJ`](https://scholar.google.com/citations?user=6VKJW4kAAAAJ&hl=en). Its displayed profile showed Tomasz Kowalewski, Professor of Chemistry at Carnegie Mellon University, with verified `andrew.cmu.edu` email; his [CMU Chemistry faculty page](https://www.cmu.edu/chemistry/people/faculty/kowalewski.html) independently confirms the identity and department. The CMU page did not directly link this exact ID. | 351 | Four Show More actions; disabled at articles 1–351; no challenge or page error. |

The first profile was loaded at 21:47:39 UTC and the second at 21:49:17 UTC in one headed Chrome session. The shared Scholar gate was reserved before each profile open; the session was then closed and the gate released. No search results, citation traversal, proxies, or access challenges were used. These counts mean **all rows visible on each profile at capture time**; they do not prove that Scholar itself is exhaustive or that every row is a journal article.

| Source file stem under `data/captures/` | JSON SHA-256 | Redacted HTML SHA-256 | Offline paired-capture checks |
| --- | --- | --- | --- |
| `mcdonough-visible-55-2026-10-24` | `ecc9e4144ac237a8678f56482e1f8ab283b1cf8b5a014889c3a4172fe3226cde` | `a03324e76b799f91e6c9740317e69a45bd8c9109a435c75e280b61e5b54e6917` | 55 unique publication IDs; no missing author; 3 blank years, 1 blank venue, 10 zero/blank citation displays. |
| `kowalewski-visible-351-2026-10-24` | `b8f5879531a46ce29020b434713ec13be35243c76e869c34ec497ca3e99d5561` | `0e7b5e6d4a2f1cb62b82e46109106c03e83259f35442c1b9b3882766e53ae039` | 351 unique publication IDs; no missing author; 4 blank years, 9 blank venues, 127 zero/blank citation displays. Row 153 retains Scholar's `19*` citation annotation. |

The offline converter checked the paired HTML/JSON before making local candidate caches outside this branch. It preserved the displayed fields and did not fabricate missing years or venues. Those candidate caches are withheld pending the producer review and coordinated snapshot release.

## Ten exact Kowalewski exclusions proposed for producer review

The following nine rows are NSF **award records** whose numbers, titles, and investigator lists match the official NSF Award Search API. The tenth is a standalone supplemental file for a retained ACS article. All ten rows remain intact in the raw captures. The `author_pub_id` values allow a later producer change to exclude only those rows from map authorship edges.

| `author_pub_id` | Reason and source |
| --- | --- |
| `6VKJW4kAAAAJ:hsZV8lGYWTMC` | [NSF award 1809658](https://api.nsf.gov/services/v1/awards/1809658.json) |
| `6VKJW4kAAAAJ:-mN3Mh-tlDkC` | [NSF award 2404367](https://api.nsf.gov/services/v1/awards/2404367.json) |
| `6VKJW4kAAAAJ:L1USKYWJimsC` | [NSF award 2109065](https://api.nsf.gov/services/v1/awards/2109065.json) |
| `6VKJW4kAAAAJ:XUvXOeBm_78C` | [NSF award 2117523](https://api.nsf.gov/services/v1/awards/2117523.json) |
| `6VKJW4kAAAAJ:1yWc8FF-_SYC` | [NSF award 0521079](https://api.nsf.gov/services/v1/awards/0521079.json) |
| `6VKJW4kAAAAJ:prdVHNxh-e8C` | [NSF award 9871874](https://api.nsf.gov/services/v1/awards/9871874.json) |
| `6VKJW4kAAAAJ:kw52XkFRtyQC` | [NSF award 0304508](https://api.nsf.gov/services/v1/awards/0304508.json) |
| `6VKJW4kAAAAJ:yxmsSjX2EkcC` | [NSF award 0210247](https://api.nsf.gov/services/v1/awards/0210247.json) |
| `6VKJW4kAAAAJ:Br1UauaknNIC` | [NSF award 0824188](https://api.nsf.gov/services/v1/awards/0824188.json) |
| `6VKJW4kAAAAJ:1taIhTC69MYC` | Standalone Supporting Information file for the [ACS 2024 article](https://pubs.acs.org/doi/10.1021/acsaem.3c02822); retain the main article `6VKJW4kAAAAJ:v1_lew4L6wgC`. |

Two high-citation rows that initially resembled namesake mistakes are **verified coauthorships**. The [2004 JBC paper](https://www.sciencedirect.com/science/article/pii/S0021925820771966) lists Tomasz Kowalewski after the Scholar display's shortened author list, so retain `6VKJW4kAAAAJ:Y0pCki6q_DkC`. The [2020 ACS Catalysis accepted manuscript](https://par.nsf.gov/servlets/purl/10159101) lists both Jakub and Tomasz Kowalewski, so retain `6VKJW4kAAAAJ:sJsF-0ZLhtgC`. A truncated Scholar author display alone cannot support exclusion. The damaged-looking `6VKJW4kAAAAJ:43bX7VzcjpAC` ("Chemistry—A European") lacks year and venue, but its exact provenance is unresolved; leave it eligible pending an exact-source check. Conference abstracts, patents, and theses also remain pending a project-wide output-type policy.

## Carrie's three undated records

| `author_pub_id` | Source finding and current disposition |
| --- | --- |
| `Xq5HrPYAAAAJ:rO6llkc54NcC` | The [SETAC 2025 abstract book](https://www.setac.org/asset/B97B2E92-74A5-49F5-90D36FAB051BBC2A/) contains the same title and McDonough authorship as abstract `4.01.P-Tu-035` in the 46th North America meeting. It is a genuine conference abstract, but SETAC states its abstracts were not peer reviewed and should not be cited. **Pending output-type policy**; do not invent a Scholar year or exclude this exact row yet. |
| `Xq5HrPYAAAAJ:3s1wT3WcHBgC` | The Scholar row identifies the SETAC North America 42nd Annual Meeting. [SETAC's 2021 abstract-book listing](https://www.setac.org/resource/setac-na-virtual-2021-abstract-book.html) confirms the meeting and year, but the exact abstract is behind SETAC login and was not independently matched. **Pending exact match and output-type policy**. |
| `Xq5HrPYAAAAJ:O3NaXMp0MMsC` | This is a real McDonough–Scott article, although Scholar omits its year and venue. [McDonough's CMU lab publication list](https://groups.chem.cmu.edu/mcdonough/) gives 2023, *LCGC Supplements: Hot Topics in PFAS*, 41(s9), 12–16, DOI `10.56530/lcgc.na.nr4267c9`; the [publisher page](https://www.chromatographyonline.com/view/tackling-pfas-complexity-with-hrms-and-bioanalytical-techniques) confirms it. **Retain**; any year/venue repair belongs in a separate source-backed curation step, not a change to the Scholar capture. |

No clear Carrie namesake or award-record row was identified in this pass. Preserve the three uncertain Scholar rows and their raw metadata until the project decides whether meeting abstracts contribute collaboration edges. The two SETAC entries should be assessed as abstracts rather than silently treated as peer-reviewed articles.
