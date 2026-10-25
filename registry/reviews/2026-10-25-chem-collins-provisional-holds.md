# Collins complete Scholar capture: provisional map holds

This draft decision applies to Terrence J. Collins (`0e1jwkkAAAAJ`), whose CMU
Experts page [links the exact Scholar profile](https://experts.cmu.edu/expert/terrence.collins/terry-collins).
The captured source has 361 unique visible Scholar rows after Show More was
exhausted. The canonical cache SHA-256 is
`71ad4333c8f43adb4b2438be05ee3d8365c46a6da3f67b71b61ed740300bc83c`.
The paired source capture and receipt are staged in
[draft source PR #118](https://github.com/cmccomb/map-of-research/pull/118).
This is a complete *visible profile* capture,
not a claim that Scholar lists every publication in Collins's career.

The 31 exact source-backed exclusions already on producer `main` remain
exclusions. The following four rows lack enough evidence for a positive
attribution or output-type decision. They are **provisional holds, not false
attributions**:

| Capture row | Exact `author_pub_id` | Reason for hold |
| --- | --- | --- |
| 214 | `0e1jwkkAAAAJ:vRqMK49ujn8C` | The lead-poisoning pilot study shows a truncated `T Collins` byline; full author identity and document type remain unclear. |
| 274 | `0e1jwkkAAAAJ:cK4Rrx0J3m0C` | The UK histopathology meeting abstract has a truncated byline with no visible Collins; the complete primary byline was not found. |
| 299 | `0e1jwkkAAAAJ:BJbdYPG6LGMC` | The *Sunday Times* item is credited to `B Collins`; a primary source has not linked it to Terrence J. Collins. |
| 355 | `0e1jwkkAAAAJ:yqoGN6RLRZoC` | The ACS abstract names `TJ Collins`, but its publication type and relationship to separately indexed work remain unclear. |

The [retained-row audit in PR #119](https://github.com/cmccomb/map-of-research/pull/119)
and [second-pass review in PR #204](https://github.com/cmccomb/map-of-research/pull/204)
explain why these rows remain unresolved. The exact Scholar citation URLs are recorded in
`data/curated/scholar-provisional-holds.json`, each matched to the row index and
ID in the pinned cache.

The producer now requires a released profile decision to pin the 361-row cache,
the four held IDs, their count, and their SHA-256. Before any dataset upload it
checks that the source snapshot has the complete cache, each held ID still
matches its original capture row, and no held observation contributes an
authorship or mapped-work source. All 361 rows remain in the public
`profile_publications` table if released; held rows carry
`attribution_provisional_hold` and `review_attribution_status=held`. The
remaining rows pass the existing quality and exact-exclusion gates. This
decision does not individually certify all 326 other raw candidates.

**Release dependency:** Keep this PR draft and unmerged until the central
Donahue/Chien upload and live-map verification finish, draft source PR #118 is
rebased and merged, and Collins's 31 exact exclusions from stacked producer PRs
#119/#204 are confirmed on the resulting main branch. Rebuild a combined
snapshot, run the complete-source coverage and no-upload table preview, and
inspect the four held IDs before the next trusted upload.
