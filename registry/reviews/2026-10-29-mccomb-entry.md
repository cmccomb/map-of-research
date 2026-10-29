# Christopher McComb registry review

Reviewed 2026-10-29 against the [current CMU Mechanical Engineering faculty bio](https://meche.engineering.cmu.edu/directory/bios/mccomb-christopher.html).

The bio names Christopher McComb a Professor of Mechanical Engineering, labels Civil and Environmental Engineering a courtesy appointment, directly links the registered [Google Scholar profile](https://scholar.google.com/citations?user=0P9w_S0AAAAJ), and links [his personal site](https://cmccomb.com/). The existing `person_id` and Scholar ID remain unchanged. The MechE membership is included; the CEE courtesy membership is retained but excluded from map inclusion.

The registry derives MechE as the display primary from the sole included academic department with `single_included_membership` basis. Explicit primary fields stay empty because this review does not assert an HR primary or replace the registry's derived-home contract. Scholar publication capture and dataset release have separate provenance and validation receipts on `automation/map-snapshot`.

## Publication source review

The signed-out [public Scholar profile](https://scholar.google.com/citations?user=0P9w_S0AAAAJ&hl=en) displayed 265 rows on 2026-10-29 after two `Show more` clicks. The control was disabled at the end. The saved JSON rows and HTML extracted directly from the rendered article table agree on all 265 citation IDs, titles, authors, venues, citation counts, and years. The complete source and its checksums are pinned in `data/authors/0P9w_S0AAAAJ.json` on `automation/map-snapshot`.

The frozen 2025 cache had 222 rows. The refreshed profile retains 219 IDs, adds 46, and lacks three historical IDs. The [exact crosswalk](2026-10-29-mccomb-prior-id-crosswalk.json) retains the old row hashes and their reviewed replacements:

- Two rows with arXiv identifier `2405.01319` map to one current 2025 Journal of Computational Physics row. [arXiv v1](https://arxiv.org/abs/2405.01319v1) has the old *Data Scoping* title; [arXiv v2](https://arxiv.org/abs/2405.01319v2) has the *Enforcing the Principle of Locality* title. The [journal publication](https://www.sciencedirect.com/science/article/pii/S0021999125004140) uses the latter title and lists Christopher McComb.
- The old shortened CAADRIA 2025 title maps to the fuller current title. The [official proceedings](https://caadria.org/new/wp-content/Downloads/CAADRIA2025_Volume-1.pdf) list Yuval Kahlon, Christopher McComb, and Haruyuki Fujii at pages 501–510.

The raw snapshot comparison has 69,012 prior rows and 69,055 new rows: 222 to 265 for Chris, while all 68,790 other raw rows are byte-for-byte equivalent at the table-value level. The separate 267-row scrape-my-publications release intentionally retains two older off-profile Scholar IDs; this map capture represents the current complete visible profile and does not import that release as source evidence.
