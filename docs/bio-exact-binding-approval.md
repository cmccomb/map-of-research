# Bio work-binding review on the dcb Hub parent

The read-only W3 diagnostic against Hub revision
`38f119d71692cdb9112210217b532a7dd7c1633a` found 11 still-visible
Google Scholar observations whose work IDs changed. All 11 belong to included
Biological Sciences people; each retained collaboration eligibility. The
preview found no unexplained visible gain or loss and stopped before a Hub
write because these bindings lacked exact review. The subsequent Chemistry
release at `dcbdfacd7e8cfeaefd4c91081c4b347b7c690061` retains the same
prior work IDs and eligibility for all 11 observations.

The [binding manifest](../data/curated/bio-exact-binding-approval-dcb.json)
records each observation, previous and proposed work ID, evidence identity,
and source URL. Seven observations acquire reviewed DOI identities: one each
for Alison Barth, C. Joel McManus, Gordon Rule, and Russell Schwartz, and
three for Robert F. Murphy. Four observations join independently verified
native records: one for Amanda Willard and three for Murphy. The separate
McManus poster approval covers one map-point loss; this manifest does not
approve that loss or any new collaboration eligibility.

The validator pins the manifest bytes, immutable Hub parent, exact W3 source
commit and Parquet digest, DOI aliases, reviewed exclusions, the exact Math
person hold, curated works, profile releases, provisional holds, and poster
approval. The continuity gate requires all 11 old/new work and collaboration
tuples and rejects an extra, missing, or
changed tuple. The source review documents are
[four publication identities](bio-four-reviewed-region-gains.md),
[Murphy's caption and figure publications](murphy-caption-figure-source-review.md),
and [Willard's education article](amanda-willard-2014-native-source-review.md).

The release gate also requires three exact Murphy Scholar observations to
leave the map as reviewed corrupted duplicates. Their retained PSB, BioLINK,
and KDD publications must remain mapped, and the three losses must appear in
the continuity receipt. The exclusion file hash and per-record evidence are
pinned; any additional mapped Bio29 loss fails the gate.

This approval binds the immutable post-Chemistry Hub parent. The Bio producer
and source PRs remain in draft until the full read-only preview and hosted
checks pass on their exact final commits.
