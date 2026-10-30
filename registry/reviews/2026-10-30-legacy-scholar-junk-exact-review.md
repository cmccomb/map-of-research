# Legacy Scholar cache quality review — 2026-10-30

## Scope and frozen inputs

This review checks older migrated Google Scholar rows that reached the public
map. The source was read from `automation/map-snapshot` at `0ede6a185`; the
producer baseline is `cdbdf9c29`. The public comparison is the map and graph
artifact generated at `2026-10-30T13:30:17Z` from current Hub revision
`42031e014886e67c4b606261c2e0dee217ee762c` (53,642 map points), plus
the saved collaboration artifact sourced from the preceding Hub revision
`5073c24450e2f239ca49f674cbc81287b915045b` (850 edges). All exact
observations below still map to one point in the newer Hub artifact. No
Scholar profile was re-collected for this review.

The verified rows below are appended to `data/curated/scholar-exclusions.json`
by exact `(scholar_id, author_pub_id)`. All 17 migrated raw observations remain
in the cache and lossless snapshot. They currently produce 16 distinct public
map points. Each row was matched to its saved cache observation and to exactly
one current point by the observed title. The current v3 title-quality rule
marks all 17 eligible, so these are source-backed corrections to known false
positives, not a general missing-metadata exclusion.

## Exact decisions

### DOI-registered author indexes and boilerplate

The first eight saved titles contain IEEE copyright price/ISBN boilerplate.
Their embedded DOIs all resolve in the DOI registry to **Author Index**, with
no listed author. A ninth row contains a garbled URL/DOI footer; its DOI
resolves to **Author Index for Volume 3**. These are index/front-matter records,
not faculty research works.

| Saved Scholar observation | Owner label | Primary DOI record |
| --- | --- | --- |
| `ygiY7C8AAAAJ:BqipwSGYUEgC` | Amvrosiadis | [10.1109/IPDPSW63119.2024.00212](https://api.crossref.org/works/10.1109/IPDPSW63119.2024.00212) |
| `4PU5ZFYV-iEC:BwyfMAYsbu0C` | Erdogmus | [10.1109/CAIN58948.2023.00037](https://api.crossref.org/works/10.1109/CAIN58948.2023.00037) |
| `yjJPtE4AAAAJ:SeFeTyx0c_EC` | Jia | [10.1109/EUROSP53844.2022.00051](https://api.crossref.org/works/10.1109/EUROSP53844.2022.00051) |
| `yqIoH34AAAAJ:sNmaIFBj_lkC` | Joshi | [10.1109/CIC56439.2022.00029](https://api.crossref.org/works/10.1109/CIC56439.2022.00029) |
| `SHcIvuoAAAAJ:HbR8gkJAVGIC` | Franchetti | [10.1109/ISPASS57527.2023.00052](https://api.crossref.org/works/10.1109/ISPASS57527.2023.00052) |
| `F9kqUXkAAAAJ:WC9gN4BGCRcC` | Gibbons | [10.1109/ISPASS55109.2022.00057](https://api.crossref.org/works/10.1109/ISPASS55109.2022.00057) |
| `yjJPtE4AAAAJ:eMMeJKvmdy0C` | Jia | [10.1109/EUROSP60621.2024.00054](https://api.crossref.org/works/10.1109/EUROSP60621.2024.00054) |
| `WfaqxlgAAAAJ:HFi42EZPDb4C` | Kumar | [10.1109/IPSN54338.2022.00073](https://api.crossref.org/works/10.1109/IPSN54338.2022.00073) |
| `fiZryxEAAAAJ:Nw_I7GeUguwC` | Grossman | [10.1006/mben.2001.0202](https://api.crossref.org/works/10.1006/mben.2001.0202) |

### Other source artifacts

| Saved Scholar observation | Public title or defect | Primary check |
| --- | --- | --- |
| `l6oz33MAAAAJ:VLnqNzywnoUC` and `F9kqUXkAAAAJ:anf4URPfarAC` | Identical reference-list fragment beginning “in SIGIR, pp. 50–57…”; captured venue `Computer 42(8):30–37` | [Actual article](https://api.crossref.org/works/10.1109/MC.2009.263) is by Koren, Bell, and Volinsky, with a different title. |
| `-syEyyAAAAAJ:O3NaXMp0MMsC` | “2015 IEEE International Symposium on Performance Analysis of Systems and Software ISPASS 2015” is an event/container label. | [Organizer's conference site](https://ispass.org/ispass2015/home_home.php); [Ken Mai's actual paper](https://ieeexplore.ieee.org/abstract/document/7095780) is separately identified. |
| `pARKzQsAAAAJ:LO7wyVUgiFcC` and `pARKzQsAAAAJ:8d8msizDQcsC` | “IISWC-2006 People” and “IISWC-2007 People” are site navigation labels. | [2006](https://iiswc.org/iiswc2006/home.html) and [2007](https://iiswc.org/iiswc2007/home.html) conference sites. |
| `Rn_BmTYAAAAJ:kVjdVfd2voEC` | “SIGMETRICS People” is a society contact section. | [SIGMETRICS newsletter](https://www.math.vu.nl/~mei/articles/2026/PER.pdf) prints this heading above officer and board contacts, including Fanti. |
| `DK7TEvkAAAAJ:pS0ncopqnHgC` | `" School of Computer Science` is an affiliation fragment with no work metadata. | [Exact saved Scholar observation](https://scholar.google.com/citations?view_op=view_citation&user=DK7TEvkAAAAJ&citation_for_view=DK7TEvkAAAAJ:pS0ncopqnHgC) and frozen cache row. |
| `DFtapYEAAAAJ:U_HPUtbDl20C` | DSN-S 2024 proceedings title plus IEEE copyright/ISBN footer and a truncated DOI; no paper title or author. | [University library catalog](https://dspace.apps.iitd.ac.in/node/156280) identifies the exact ISBN as the proceedings volume. |

## Current-map impact and limits

Excluding these exact observations is expected to remove 16 current points.
Only the Ganger/Gibbons reference fragment has two listed faculty; it adds one
spurious shared work to their existing edge. The edge has 18 other works in the
current graph, so its presence should remain. No coauthor relation should be
inferred from this corrupted fragment. The exact public delta must be checked
in a full no-write release preview against the then-current Hub parent before
publication; the saved artifacts above are a fixed comparison, not a release
receipt.

## Broader older-source audit queue

The frozen 70,443-row public snapshot contains 36,542 rows from 208 caches
fetched in 2024–25. Those legacy rows all lack saved bylines and source URLs;
the source branch also contains one old alternate cache outside that frozen
snapshot. There were no exact reviewed exclusions or profile-release records
for the legacy profiles before this change. In the preceding Hub map revision,
30,459 of 53,642 points overlap a work ID from a legacy observation. This is provenance
exposure, not a count of false publications. An exact title match between two
profiles still needs byline verification before treating the pair as confirmed
coauthors. A focused queue of 1,146 live points has no author, year, venue, or
DOI, zero citations, and one observation; legitimate reports and preprints
also meet that description, so the queue is for manual review only.

The legacy source has two separate cache-identity defects, neither currently
creating a visible person or edge because its registry membership is excluded:

- `wP966k4AAAAJ` (“KYu”) has 139/139 `author_pub_id` values beginning
  `Fz3_tukAAAAJ:` and duplicated in the Byron Yu cache. The
  [CMU Engineering bio](https://engineering.cmu.edu/directory/bios/yu-byron.html)
  links Byron Yu to that latter Scholar ID.
- `_7HjzLYAAAAJ` (“Wayne”) has 33/33 IDs beginning `e0v5iEgAAAAJ:`.
  [CMU Engineering](https://engineering.cmu.edu/directory/bios/wayne-elizabeth.html)
  identifies Elizabeth Wayne; an owner-linked source for the latter Scholar ID
  is still needed before any reassignment.

Keep the bad cache bytes for provenance and resolve the two identities through
source/registry review. Do not remove all older rows, infer authors from title
alone, or promote one-sided legacy listings to verified coauthorship.

A separate 2026-captured Schwartz row, `6ArMETUAAAAJ:UeHWp8X0CEIC`, carries
the title “BMC genomics Volume: 9 ISSN…” rather than a work title. The
[actual Sridhar–Schwartz paper](https://bmcgenomics.biomedcentral.com/counter/pdf/10.1186/1471-2164-9-389)
is already mapped correctly. That row needs a separate exact Bio release review;
it is outside this legacy-source batch.
