# Chemistry decisions in the frozen Bernhard region release

The [exact 19-row manifest](../../data/curated/scholar-chem-region-review.json)
binds each decision to its saved Scholar citation, complete profile cache,
frozen source row, `bb88ddf0` work fingerprint, and independent evidence.
The frozen `f8d7ffec` raw snapshot and every full Scholar cache remain intact.
The post-#337 Chemistry release applies these decisions to the immutable
published #337 dataset after validating the original curated manifests. The
Sullivan/Math release gate remains pinned to its own `bb88ddf0` parent.

| Historical change | Include | Hold | Exclude |
| --- | ---: | ---: | ---: |
| Restore | 3 | 3 | 1 |
| Reexclude | 7 | 2 | 3 |

The two Donahue AGU meeting presentations are explicitly typed as
`conference_presentation` and included as scholarly meeting outputs. They are
not journal papers; one is solo and adds no coauthor edge.
Matyjaszewski's 1979 *Polimery* article is the third restored output. Seven
previously visible works retain their points: two Kietrys posters, Collins's
ACS response and viewpoint, Armitage's News & Views article, Matyjaszewski's
book section, and his *Macromolecules* article.

Against the merged-main #337 stage, the scoped correction adds no authorship
rows and removes nine held/excluded observations: three Collins, three
Matyjaszewski, and one each for Gil, Kurnikova, and Washburn. The two Donahue
presentations each retain their existing Donahue authorship. Their candidate
points have only Donahue as verified faculty and no external coauthor IDs, so
neither adds a verified collaboration edge. The offline stage preserved all
49,620 shared authorship payloads and all 42,904 shared map point payloads.

The two already-visible Kietrys poster rows retain their generic Scholar
`publication` type in this visibility review. Organizer entries confirm the
poster format, but their full abstract/byline has not been recovered; a
separate context and collaboration review should resolve that type and any
coauthor links.

Five rows stay raw under holds. They are the two Collins translated features,
Kurnikova's HARLEM software row with an unsupported 1999 journal venue,
Gil's multi-group signed support feature, and a Collins ACS meeting abstract
whose exact byline still needs the original abstract. Four exact rows are
excluded from mapping and coauthor edges: Washburn's spliced citation,
Matyjaszewski's falsely attributed Cesa column, the unattributed publisher
notice in *Progress in Polymer Science*, and the truncated duplicate
"Living" Scholar row. The publisher-deposited Crossref record for
[the notice](https://api.crossref.org/works/10.1016/S0079-6700(02)00114-4)
has no authors; the adjacent [Berry–Matyjaszewski editorial](https://api.crossref.org/works/10.1016/S0079-6700(02)00115-6)
has a separate DOI and page.

The #337 parent is expected to contain 42,909 map points. The scoped Chemistry
decisions add three and remove five, for an expected 42,907 points. The parent
must first be published at a new immutable Hub revision whose seven release
files match the independently validated #337 staged receipt byte for byte.
Then run a fresh exact-parent, first-write-intercepted preview from the exact
reviewed code head and frozen source commit `8be6a240`, checking all four
Parquet configs, both map files, and README. Every non-Chemistry row and shared
map point must match that Hub parent. The old `bb88ddf0` preview for draft
PR #338 is source-review evidence only and cannot authorize this restack.

Two clean #337 producer previews serialized one otherwise identical work with
the same `venue_variants` in opposite order. The cause is `_variants()` in
`dataset.py`: it sorts a `set` by `str.casefold`, so strings differing only in
capitalization tie and retain the set's process-dependent order. A later
publisher run could therefore produce a different `works` file hash. This
restack pins the merged-main #337 receipt and its exact live file bytes. A
separate producer fix should sort by `(value.casefold(), value)` before a
future full rebuild; it is outside this Chemistry delta release.
