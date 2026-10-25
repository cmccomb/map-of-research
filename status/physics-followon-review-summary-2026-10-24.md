# Physics source attribution review, updated 2026-10-25

The original source review covers ten complete visible Google Scholar profiles: **2,332 raw observations**. Morningstar and Parno (428 rows) are already on the live source branch. This restack carries the other eight complete captures (1,904 rows) from the current source parent. Saved HTML, extracted JSON, converted author caches, and receipt hashes agree for every row. The review records exact observation decisions without altering source rows, producer exclusions, the snapshot, or publication state.

| Profile | Raw rows | Exact proposed exclusions | Remaining individual candidates |
| --- | ---: | ---: | ---: |
| Simranjeet Singh | 53 | 1 | 0 |
| Benjamin Hunt | 95 | 7 | 1 |
| Colin Morningstar | 272 | 20 | 0 |
| Diana Parno | 156 | 4 | 0 |
| Markus Deserno | 233 | 12 | 1 |
| Matthew Walker | 413 | 19 | 11 |
| Randall Feenstra | 422 | 11 | 0 |
| Robert H. Swendsen | 202 | 6 | 0 |
| Michael Widom | 215 | 3 | 0 |
| Sara Majetich | 271 | 27 | 2 |
| **Total** | **2,332** | **110** | **15** |

The 15 individual candidates do not count same-title, missing-venue, or meeting-abstract groups. Those grouped flags are review leads, not blanket exclusion rules. The [exact proposed ledger](physics-followon-reviewed-observation-exclusions-2026-10-24.json) contains the full citation IDs, reasons, and primary-source URLs. Its 22 new decisions and the separately reviewed retains are explained in the [2026-10-25 addendum](physics-followon-exact-resolution-2026-10-25.md). The 24 Morningstar and Parno exclusions already occur on current `main`; the other 86 ledger IDs remain source proposals, not producer changes.

## Decisions and limits

- **Morningstar and Parno** have no unresolved individual candidate rows or profile-wide identity concern in the saved evidence. Source/producer pairs [#133](https://github.com/cmccomb/map-of-research/pull/133)/[#134](https://github.com/cmccomb/map-of-research/pull/134) and [#131](https://github.com/cmccomb/map-of-research/pull/131)/[#132](https://github.com/cmccomb/map-of-research/pull/132) merged on 2026-10-25, carving 428 raw rows and 24 exact exclusions. Do not apply those 24 producer decisions twice.
- **Walker remains held as a whole profile.** Same-name records occur throughout all 413 rows. The source ledger now has 19 exact Walker proposals, including [RMIT's author-entered report record](https://pub.orcid.org/v3.0/0000-0003-1086-6222/work/121411569), the [original Nottingham IJIC abstract](https://ijic.org/articles/785/files/submission/proof/785-1-2513-1-10-20110721.pdf), and an [original University of Washington course report](https://depts.washington.edu/soslab/mw/images/7/7d/Group2MS2Report.pdf). Rows 404, 410, and 411 remain preliminary: the Natural Hazards PDF is a different report that only references those RMIT titles in its bibliography. Eleven individual Walker candidates and grouped flags still prevent publication of the whole profile.
- **Thirty additional NSF award observations** are exact nonpublications: Feenstra 9, Swendsen 3, Majetich 18. Each saved row has a linked official NSF award record. Feenstra row 381's truncated saved number maps to [award 0079485](https://api.nsf.gov/services/v1/awards/0079485.json), corroborated by [Katayun Barmak's Columbia CV](https://www.apam.columbia.edu/files/seas/content/apam_image/barmak-katayun_cv-102019.pdf). The new exact committee, index, and malformed-row decisions still leave grouped flags and report-type support to review before the held profiles are carved.
- Singh row 38 and Hunt row 31 are [UCF](https://stars.library.ucf.edu/etd/4606/) and [Cornell](https://www.cmu.edu/physics/people/faculty/documents/hunt_cv.pdf) dissertations by the intended physicists; both are explicitly retained despite blank Scholar venue fields. Genuine Singh, Feenstra, and Majetich patents are retained under the existing producer patent policy. Hunt row 95 remains unresolved. Hunt row 91's dead Adelaide link is replaced with the [university thesis PDF](https://digital.library.adelaide.edu.au/server/api/core/bitstreams/6953eea0-c673-45bc-9ade-b86527ca533c/content). The Parno DOI and Widom chapter links were repaired to accessible metadata or author-bibliography sources.

Every scholar in this review has one included Physics academic membership, so the current registry and live map derive Physics as the primary home. Hunt, Widom, and Majetich retain separate MSE courtesy memberships. The eight new raw caches must remain a draft until a producer release manifest binds each one to its complete cache and holds all unresolved profiles. Walker's entire profile remains held pending its separate identity review.
