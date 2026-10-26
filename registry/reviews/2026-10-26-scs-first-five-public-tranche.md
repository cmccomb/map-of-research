# First five SCS Scholar sources: bounded public tranche

Five exact Scholar IDs were linked on named CMU SCS faculty pages before the browser opened. One isolated Playwright session loaded every visible row for each exact profile; the final **Show more** control was disabled in every case. Saved HTML, row captures, cache hashes, and browser receipts are on the paired source branch. The browser is closed. These captures establish the visible Scholar profiles, not an independently complete lifetime bibliography.

| Faculty | Stable ID | Scholar ID | CMU identity page | Primary academic home | Source rows | Exact provisional holds | Ordinary gate candidates |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Carolyn Rosé | `person-carolynrose-0ff383b5` | `BMydCgcAAAAJ` | [HCII page](https://hcii.cmu.edu/index%2ephp/people/carolyn-rose) | LTI, institute; retains HCII and LTI memberships | 677 | 139 | 538 |
| Mona Diab | `person-monadiab-7b90c255` | `-y6SIhQAAAAJ` | [LTI page](https://www.lti.cs.cmu.edu/people/faculty/diab-mona.html) | LTI, institute | 413 | 107 | 306 |
| Tim Dettmers | `person-timdettmers-6d9c0030` | `lHI3w5kAAAAJ` | [CSD page](https://www.csd.cs.cmu.edu/people/faculty/tim-dettmers) | MLD, department; retains CSD and MLD memberships | 31 | 5 | 26 |
| Aayush Jain | `person-aayushjain-aac61f01` | `VZ6S70MAAAAJ` | [CSD page](https://www.csd.cs.cmu.edu/people/faculty/aayush-jain) | CSD, department | 78 | 7 | 71 |
| Aviral Kumar | `person-aviralkumar-aa7154c0` | `zBUwaGkAAAAJ` | [CSD page](https://www.csd.cs.cmu.edu/people/faculty/aviral-kumar) | CSD, department; retains CSD and MLD memberships | 110 | 19 | 91 |

Mona's LTI page directly links her exact Scholar ID, which was missing from `registry/people.csv`; the same page labels her LTI faculty. Aayush's CSD faculty page identifies his sole included academic membership, resolving his previously blank primary department. Carolyn, Tim, and Aviral retain their previously reviewed primary homes. The source and producer registry rows have matching updates for Mona and Aayush.

All 1,309 saved rows enter the raw source snapshot. The paired producer decisions release the five complete sources with **277 exact provisional holds** and no new exclusion claims. A row is held if its saved byline does not visibly name the faculty member, its year is missing or malformed, the same normalized title and year occur more than once in that profile, or its title appears to be a whole conference volume, program, committee, or website artifact. Each hold is bound to the exact capture row, Scholar citation ID, cache hash, and saved citation URL. The other **1,032 rows are candidates for the existing publication quality, attribution, membership, and map gates**; this review does not assert that all 1,032 become map points.

The source branch's schema 6 Parquet was built with the current producer snapshot builder and the 43 existing curated work rows. It has 60,743 observations: all 59,434 prior observations, field for field, plus exactly 1,309 new rows owned by these five stable IDs. The source manifest validates. No upload or deployment belongs to this draft pair; the release coordinator must run an exact-parent no-upload preview against the then-current immutable Hub revision and inspect the resulting preservation and hold-leakage receipts before any merge.
