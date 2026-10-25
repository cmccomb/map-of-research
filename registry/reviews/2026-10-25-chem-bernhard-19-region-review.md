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

The merged-main #337 parent `people.unique_work_count` equals the number of
distinct authored work IDs for all eight reviewed Chemistry people. Donahue's
two hidden AGU works already have authorship rows and are included in his 577
unique works, so revealing their map points does not change that count. The
nine removed observations each represent a distinct authored work for their
person. `mapped_work_count` is already offset from actual map-point membership
for several people in #337 because the frozen region decisions were applied
after the initial people count. This release preserves that inherited offset
and applies only the exact map-visibility delta:

| Person | Unique-work delta | Mapped-work delta |
| --- | ---: | ---: |
| Donahue | 0 | +2 |
| Matyjaszewski | −3 | −2 |
| Collins | −3 | −1 |
| Gil | −1 | −1 |
| Kurnikova | −1 | 0 |
| Washburn | −1 | 0 |

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

The #337 parent is now published at immutable Hub revision
`38f119d71692cdb9112210217b532a7dd7c1633a`, with successful
`automation/map-snapshot/status/last-upload.json` and producer `a68fbf0c`.
Its seven files were downloaded and independently checked against that receipt.
It has 42,909 map points. The scoped Chemistry decisions add three and remove
five, for an expected 42,907 points. Run a fresh exact-parent,
first-write-intercepted preview from the exact reviewed code head and frozen
source commit `8be6a240`, checking all four Parquet configs, both map files,
and README. Every non-Chemistry row and shared map point must match the live
Hub parent. The old `bb88ddf0` preview for draft PR #338 is source-review
evidence only and cannot authorize this restack.

The release path requires a new local intent and a clean snapshot-branch
checkout whose last successful receipt names the exact live #337 parent. Before
any Hub write it commits the nonce, seven stage hashes, and parent to
`automation/map-snapshot/status/last-upload.json`, pushes without force, fetches
that remote commit, and compares the retained status bytes. A failed push,
readback, or branch check stops before the Hub write. The prior successful
receipt remains intact until all seven immutable Hub files have been verified;
only then does the path advance both status files. The first-write-intercepted
preview simulates this Git checkpoint in a local output directory and makes no
canonical Git remote mutation, so its receipt must label that simulation.

The separate manual `chem-region-release.yml` workflow is a future production
entrypoint. It requires a reviewed main producer SHA, a requested immutable Hub
parent, an explicit Hub token, and three distinct checkouts: clean reviewed
producer code, frozen raw snapshot and complete Scholar caches at `8be6a240`,
and current `automation/map-snapshot` status. The production CLI checks those
commits and source bytes before building the seven files. The existing #337
upload workflow is unchanged. The Git checkpoint is retained on the status
branch before `HfApi.create_commit`; missing credentials, prior status, remote
push, or fetched-byte readback abort before that write. This draft workflow is
not an upload authorization and has not been dispatched.

The standing 100% generic coverage metric excludes both new Chemistry release
modules. A separate CI gate measures their branch coverage: the synthetic
19-row/eight-cache source loader and exact-delta tests currently cover 96% of
`chem_region_review.py` and 76% of `chem_delta_release.py` (78% combined,
minimum 75%). The full seven-file serialization in `stage_chem_delta` was
exercised against the merged-main #337 staged parent in the offline simulation;
the live-parent, first-write-intercepted preview remains required before any
Chemistry write.

The merged-main local preview, hosted Linux no-write preview
[40196085001](https://github.com/cmccomb/map-of-research/actions/runs/40196085001),
and actual successful upload
[40197924026](https://github.com/cmccomb/map-of-research/actions/runs/40197924026)
did not produce identical seven-file bytes. The hosted preview and live Hub
contain exactly the same 42,909 ordered publication IDs and final eligibility,
attribution, and metadata. Eleven *new Sullivan* points, outside all nineteen
Chemistry decisions, have different layout coordinates; two derived
unclassified keyword centroids also move. The manifest's intermediate
`region_continuity.prior_eligible_overrides` and `prior_excluded_overrides`
counts are `100/62` in the hosted preview and `79/134` in the live Hub, while
the final decisions match. The local preview additionally differed in the
order of two case-varied `title_variants` for one unrelated work; deterministic
variant ordering was subsequently fixed in producer PR #343. The **actual live
Hub bytes** and its successful release receipt, rather than either preview,
are the exact Chemistry parent. The nineteen decisions and frozen source hashes
have not changed.
