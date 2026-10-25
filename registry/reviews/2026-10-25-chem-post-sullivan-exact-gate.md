# Chemistry metadata release after Sullivan: exact continuity candidate

This draft combines the source reviews for [Silva and Whitehead's strawberry
versions](2026-10-25-chem-silva-strawberry-version.md), [Gil's unresolved
Cor-Ciencia observation](2026-10-25-chem-gil-archive-provisional-hold.md), and
[Donahue's discussion/final article](2026-10-25-chem-donahue-water-version.md).
Its [manifest](../../data/curated/chemistry-post-sullivan-exact-continuity.json)
names exactly seven Scholar observations: one Gil visible-to-held loss, one
Silva held-to-visible gain, and five still-visible work-ID changes (two
Donahue, one Silva, two Whitehead). The complete four profile captures retain
967 raw rows, with unchanged `source_record_json` values.

The proposed source snapshot is immutable commit
`8be6a2400cf72dd959c874bfcada8624540820ef`, SHA-256
`f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2`.
The proposed parent is the seven-file **no-write Sullivan stage** recorded at
`/tmp/sullivan-addendum-no-write-20261025-run2/preview-receipt.json`.
The manifest pins all seven staged file hashes. A local comparison found the
same 51,879 observation identities and raw JSON values in that stage and the
frozen source snapshot. This is evidence for a candidate parent, not proof
that those bytes are hosted.

The manifest intentionally has `prior_revision: null` and a pending status.
The publisher rejects it before a Hub write. To activate this specific
candidate after Sullivan is published, record the **actual immutable Hub
revision**, verify the hosted parent contains exactly the seven pinned file
hashes, set `status: reviewed_exact_parent`, update the manifest SHA-256 pinned
in `chemistry_post_sullivan_approval.py`, run the full production no-write
preview against that revision, and review its complete region-continuity
receipt. The release must show only the seven named transitions, no unrelated
reviewed loss, no unused approval, 967 preserved raw profile rows, and one
atomic write intent with no unexplained graph change. A changed parent needs
a new comparison and manifest, rather than substituting its revision here.

In particular, a Sullivan plus Bernhard composite parent may differ from the
seven-file Sullivan-only stage. The exact file-hash check will reject it until
the Chemistry deltas are replayed and independently reviewed against that
actual hosted parent. No Hub upload or live map update is authorized by this
draft.
