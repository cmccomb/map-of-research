# Combined SCS first 15 Scholar sources

This paired draft integrates the three separately reviewed, CI-passing SCS batches: first five source/producer PRs #442/#441, second CSD #443/#444, and HCII #445/#446. No original batch PR is merged. The source branch is based on receipt tip `ee538308c6950c2520c2cd05ee27a6ee855c1b70`; the producer is based on main `03eb745bc4475fb883f66fb2486914ef96d8ecfb`. The immutable preview parent is Hugging Face revision `ceee8b0d534a1ded300f27d7408804fc1a3bbf93`.

| Batch | Complete verified profiles | Captured raw rows | Exact provisional holds | Ordinary gate candidates |
| --- | ---: | ---: | ---: | ---: |
| First SCS five | 5 | 1,309 | 277 | 1,032 |
| Second CSD five | 5 | 828 | 321 | 507 |
| HCII five | 5 | 1,590 | 509 | 1,081 |
| **Combined** | **15** | **3,727** | **1,107** | **2,620** |

The source snapshot has 63,161 rows: all 59,434 prior observations remain field for field equal, including 43 curated rows, plus exactly 3,727 captured Scholar rows owned by 15 distinct stable person IDs. Every complete capture reconciles to saved rendered HTML, row JSON, and its hash-bound cache. The source and producer registries use identical bytes. Producer release and hold manifests bind the exact 15 cache hashes, source row counts, and 1,107 held citation IDs. No new exclusion claim is made by this integration.

All 15 included scholars have one evidenced primary academic home. The first five retain or receive the documented LTI, MLD, and CSD choices; the second batch's five sole CSD memberships resolve to CSD; and the third batch's five sole HCII memberships resolve to the HCII institute. Existing joint memberships remain present. The three individual review documents and source receipts record the identity and attribution decisions for each profile.

This is a draft pair for central review. Merging the source branch would trigger the upload workflow, so the release coordinator must first review an exact-parent seven-file no-upload preview, its staged hashes, prior point/authorship/membership/pair preservation, and held-row leakage checks. Neither branch should merge independently.
