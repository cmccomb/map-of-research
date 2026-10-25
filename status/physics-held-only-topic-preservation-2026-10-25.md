# Physics held-only topic preservation, local candidate

The read-only atomic preview against immutable Hub parent
`38f119d71692cdb9112210217b532a7dd7c1633a` finished with
`status: passed_no_upload`, `hub_writes_completed: 0`, and the first
`HfApi.create_commit` intercepted before transmission. It staged 1,904 new
Physics raw rows under eight held profile decisions and no new public map
points. The preview files are outside this branch.

The staged map still reclustered every point during ordinary keyword
generation. Among 42,909 mapped works, 41,436 `keyword_ids` arrays changed;
the keyword catalog fell from 152 to 150 entries, and 15 pending-review
labels disappeared. The parent topics include reviewed labels from the
Sullivan/Math composite release, so a held-source-only upload must preserve
those parent decisions.

This candidate copies all eight topic fields from works at the pinned Hub
parent only when the region-continuity receipt reports no visible observation
gains, losses, or binding changes. It then checks exact released observation
and work rows, mapped work IDs, the non-topic payload of every point, and the
parent work/map topic relationship. The staged map must match the parent's
`points` and `keywords` arrays and every other field except
`generated_at_utc` before the atomic Hub write. A mixed release with new
visible observations follows the existing topic path. Bio and composite
release gates remain separate.

This branch is local and has not dispatched an upload. The Chemistry release
will change the Hub parent; the Physics no-upload preview and all parent-bound
checks must run again against that new immutable revision before any merge or
upload decision.
