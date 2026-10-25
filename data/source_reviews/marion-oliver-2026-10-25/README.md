# Marion Oliver: held whole-researcher source review

This review covers `person-marionoliver-1975d99a`, an included Mathematical
Sciences emeritus member. [CMU's faculty profile](https://www.cmu.edu/math/people/faculty/oliver.html)
calls him **Teaching Professor Emeritus** at CMU Qatar. [CMU's retirement
profile](https://www.cmu.edu/math/news-events/articles/20231218_cmuq-marion-retirement.html)
describes his teaching and advising career. Neither page gives a publication
list. No exact Google Scholar account or complete researcher-owned bibliography
was verified. **Lifetime publication completeness is unestablished.**

## Identity and bounded research evidence

A [1970 CMU faculty archive](https://iiif.library.cmu.edu/file/1967-71_Ser5_1970_Sep_9_1970/1967-71_Ser5_1970_Sep_9_1970.pdf)
names **Marion Leon Oliver** as an instructor in Mathematics. The
[1972 Springer article](https://link.springer.com/article/10.1007/BF00255666)
names **Marion L. Oliver** and gives the Department of Mathematics at
Carnegie Mellon as his affiliation. The exact OpenAlex author cluster
`A5014311491` has three 1972–1975 mixtures articles with the matching
Carnegie Mellon affiliation. These are strong identity links for the three
publisher-backed records, but the OpenAlex cluster is a discovery aid and its
three-row count cannot establish a whole career.

| Year | Publisher-backed research article | Deposited authors |
| --- | --- | --- |
| 1972 | [On balanced interactions in mixtures](https://link.springer.com/article/10.1007/BF00255666) | Marion L. Oliver |
| 1973 | [On balance of forces for mixtures](https://doi.org/10.1090/qam/99715) | Morton E. Gurtin; Marion L. Oliver; William O. Williams |
| 1975 | [Formulation of balance of forces in mixture theories](https://doi.org/10.1090/qam/449210) | M. L. Oliver; W. O. Williams |

`candidate-crosswalk-3.csv` preserves each publisher title, date, venue,
pagination, deposited byline, DOI, and identity evidence. All three rows are
source-only: Scholar, release, map, and collaboration flags are false. The
loader supplies no canonical work IDs or coauthor IDs. In particular, the two
Williams relationships do **not** enter the public graph from this review.

## Boundaries and provenance

The [CMU Qatar task-force report](https://www.cmu.edu/cmuexp-taskforce/task-force-final-report-november-2019.pdf)
names Oliver as a co-chair. It is institutional service evidence, not a
research article. The CMU teaching and retirement pages are identity and role
evidence, not publication rows. CMU confirms a Mathematics doctorate, but no
named dissertation record was located, so this review invents no thesis row.

A broad OpenAlex name search returned six author clusters. The two medical
clusters (Toledo/Spectrum, 41 indexed works; Advocate Health, eight) are
excluded from the CMU mathematician. Two other similarly named clusters have
no corroborating CMU affiliation and remain unresolved. `name-collision-review.json`
records every returned cluster and the reason for its decision. The prior Math
account audit found no matching Scholar account; a direct exact-name Scholar
search redirected to Google sign-in rather than exposing a verifiable profile.
The KiltHub exact-name and exact-title searches returned zero records.

`manifest.json` pins every committed source and derivative by hash, records
query endpoints and capture time, and states the hold. Raw publisher, OpenAlex,
and KiltHub responses and gzipped CMU/Springer pages are preserved. The 1970
CMU PDF itself is not committed; `cmu-1970-archive-index.json` records its
source URL, hash, size, and the inspected page.

Run `python scripts/validate_oliver_source_review.py` to check hashes, identity
links, all six name-search decisions, exact DOI/byline coverage, and the release
hold. This directory has no import path into collection, snapshot building,
or upload. A future release needs a verified Scholar account or a separately
accepted whole-researcher bibliography, plus a reviewed atomic source graph.
