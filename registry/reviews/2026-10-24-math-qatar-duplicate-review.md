# Mathematical Sciences: two reviewed Scholar duplicates

The complete visible Google Scholar captures for Hasan Demirkoparan and Niraj Khare each contain one malformed second observation of a publication already listed on the same profile. This review excludes those two observations from derived map data by exact Scholar row identity. The source captures and profile caches remain intact.

| Profile | Exclude | Retain | Evidence |
| --- | --- | --- | --- |
| Hasan Demirkoparan, `yssfFIsAAAAJ` | Row 49, `yssfFIsAAAAJ:W7OEmFMy1HYC`: “On dissolution and reassembly of filamentary”; three authors, no year or venue | Row 9, `yssfFIsAAAAJ:LkGwnXOMwfcC`: full title, same three authors, 2009 journal venue, 33 citations in capture | [Crossref DOI 10.1098/rspa.2008.0360](https://api.crossref.org/works/10.1098/rspa.2008.0360) and [Demirkoparan's CMU-hosted publication list](https://www.contrib.andrew.cmu.edu/~hasand/index_files/Page412.htm) identify the full title and authors. |
| Niraj Khare, `8Vj5Ps4AAAAJ` | Row 8, `8Vj5Ps4AAAAJ:W7OEmFMy1HYC`: “Research Article Graphs Whose Edge Set Can Be Partitioned into Maximum Matchings”; no venue | Row 9, `8Vj5Ps4AAAAJ:zYLM7Y9cAGgC`: title without the prefix, 2013 journal venue | The [publisher article page](https://onlinelibrary.wiley.com/doi/10.1155/2013/358527) uses “Research Article” as a type label and gives the title and author separately. |

The paired capture JSON and HTML hashes match their cache metadata (`hasan-visible-50-2026-10-24`, 50/50 rows; `niraj-visible-10-2026-10-24`, 10/10 rows), and both captures report `more_remain: false`. The exact retained and excluded IDs above appear in those saved rows. The exclusions remove duplicate observations in the derived data; they do not reduce either profile's source-history coverage.

**Retained despite the short Scholar author field:** Hasan capture row 46, `yssfFIsAAAAJ:mB3voiENLucC`, lists only A. Hadoush for “Homogeneous Deformation Of Internally Balanced Compressible Hyperelastic Materials.” [CMU Qatar's research forum report](https://www.qatar.cmu.edu/news/cmu-faculty-and-students-win-top-awards-at-annual-research-forum/) credits Demirkoparan, Hadoush, and Pence for this work. The row remains eligible; a shortened Scholar author field alone is insufficient evidence for exclusion.
