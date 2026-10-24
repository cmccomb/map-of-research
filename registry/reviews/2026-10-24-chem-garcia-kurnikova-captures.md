# Chemistry Scholar evidence: Garcia-Bosch and Kurnikova — 2026-10-24

The user supplied two Scholar IDs without a name mapping. A gated, direct-profile Chrome batch resolved both to included CMU Chemistry faculty and captured their **complete visible profiles**. The saved HTML/JSON preserve every displayed publication row, including dubious rows; hidden Scholar `xsrf` values were redacted as `[REDACTED]`. This capture-only branch does not assign registry IDs, add canonical author caches, or alter the map snapshot.

| CMU person and primary department | Identity evidence | Visible rows | Capture state |
| --- | --- | ---: | --- |
| Isaac Garcia Bosch, Chemistry | [`-HzcsYQAAAAJ`](https://scholar.google.com/citations?user=-HzcsYQAAAAJ&hl=en) displayed **Isaac Garcia-Bosch**, Associate Professor at Department of Chemistry, verified `andrew.cmu.edu` email, and linked the [CMU-hosted Garcia-Bosch group](https://groups.chem.cmu.edu/garcia-bosch/). His [CMU Chemistry faculty page](https://www.cmu.edu/chemistry/people/faculty/garcia_bosch.html) corroborates the person. Registry ID: `person-isaacgarciabosch-e93d4d92`. | 91 | One Show More, then disabled at articles 1–91; no challenge or page error. |
| Maria Kurnikova, Chemistry | [`IggMbacAAAAJ`](https://scholar.google.com/citations?user=IggMbacAAAAJ&hl=en) displayed **Maria Kurnikova**, Professor of Chemistry at Carnegie Mellon University, verified `andrew.cmu.edu` email, and linked her [CMU Chemistry faculty page](https://www.cmu.edu/chemistry/people/faculty/kurnikova.html). Registry ID: `person-mariakurnikova-215a60c6`. | 170 | Two Show More actions, then disabled at articles 1–170; no challenge or page error. |

The guard reserved `-HzcsYQAAAAJ` at 21:58:35 UTC and `IggMbacAAAAJ` at 22:00:12 UTC, more than 90 seconds apart. The session was closed and the guard released after capture. There was no search-results traversal or challenge handling. Complete visible-profile coverage is bounded by what Scholar displayed at that time, and does not classify every row as a research publication.

| Source file stem under `data/captures/` | JSON SHA-256 | Redacted HTML SHA-256 | Offline paired-capture checks |
| --- | --- | --- | --- |
| `garcia-bosch-visible-91-2026-10-24` | `b512987955af886407c216d3a45ac4aabf4b11a7f7f7236175db6906451e00ea` | `2118769917810444f5d0308ecb9f15b957a0c8b0f1a8c93fb603d42f79fd6675` | 91 unique publication IDs; 0 blank years, 2 blank venues, 0 missing authors; 2 hidden `xsrf` values redacted. |
| `kurnikova-visible-170-2026-10-24` | `4c9a5d6b654df3398b917fe3b48f9fe462cacd0885d568aee38a1d96b8e9e937` | `552cedb95a0e83e7480d749055007817f5f5af9f0a5f7fdcaf27a34b1959ab96` | 170 unique publication IDs; 6 blank years, 9 blank venues, 0 missing authors; 2 hidden `xsrf` values redacted. |

The offline converter matched every saved HTML row to its JSON row, including title, authors, venue, year, citation display, and citation URL. Candidate caches and a validation report are staged outside Git at `work/chem-polymer-capture-staging/` pending source and producer sequencing.

## Producer review before cache release

Two exact Scholar rows are **NSF award records**, not publications. Official NSF API records match the award numbers, titles, investigators, and CMU:

| `author_pub_id` | Primary evidence |
| --- | --- |
| `-HzcsYQAAAAJ:SeFeTyx0c_EC` | [NSF award 2153109](https://api.nsf.gov/services/v1/awards/2153109.json), Isaac Garcia-Bosch PI. |
| `IggMbacAAAAJ:hkOj_22Ku90C` | [NSF award 1818213](https://api.nsf.gov/services/v1/awards/1818213.json), Maria Kurnikova PI. |

Do not delete those source rows. Stage exact producer exclusions before the caches contribute collaboration edges.

Other rows warrant focused review but remain **unexcluded**: Garcia-Bosch `-HzcsYQAAAAJ:7PzlFSSx8tAC` is a garbled one-word “Company” title; Kurnikova `IggMbacAAAAJ:ZuybSZzF8UAC` is a one-word “III” title with no year; and `IggMbacAAAAJ:cFHS6HbyZ2cC` is labeled “SPECIAL ISSUE ON ION CHANNELS—BIONANOTUBES” with no year. The current generic quality rule already rejects Garcia-Bosch's three author-list titles `:RGFaLdJalmkC`, `:ZeXyd9-uunAC`, and `:9yKSN-GCB0IC`. An exact source or output-type decision is needed for the remaining suspicious entries.

Retain Kurnikova `IggMbacAAAAJ:4TOpqqG69KYC`, “A community effort to discover small molecule SARS-CoV-2 inhibitors.” Although Scholar truncates its long author list and leaves the venue blank, the [author manuscript on ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/642d8e79a41dec1a5686286f/original/a-community-effort-to-discover-small-molecule-sars-co-v-2-inhibitors.pdf) explicitly names Maria Kurnikova and Olexandr Isayev. It is genuine CMU coauthorship, not a namesake artifact.
