# Ryan C. Sullivan canonical source staging (2026-10-25)

This draft copies the exact 250-row candidate cache from source review #244 to
`data/authors/wXSgFr4AAAAJ.json`. The canonical file has SHA-256
`960b296af5cf0c9e3aab6be32844ec2fe2ad7f2a2b3e331cfc4c857f4df57fd7`,
matching the candidate byte for byte. The saved Scholar JSON and HTML contain
all 250 visible rows, and the final Show more control was disabled. The earlier
review notes in #244 describe that branch's 196-row canonical state; this child
branch is a later staging state.

Producer draft `codex/chem-sullivan-retention-holds-20261025` binds this exact
cache hash to 56 exact exclusions, 35 provisional holds, and four historical-ID
dispositions. The two old ORNL variants must map to the current row's exact
`false_profile_attribution` exclusion. The AGU 2017 abstract remains a separate
verified curated work; the EGU25 abstract maps to its current Scholar ID. No
old-only row is appended to the 250-row current source.

The source snapshot has **not** been rebuilt on this draft branch. Rebase on the
latest completed Math-inclusive source snapshot, merge the producer gate first,
rebuild and validate one combined snapshot, inspect the 250 raw Sullivan rows
and the exclusion/hold contributions, then upload and pin the immutable revision
in both public maps. Until those steps pass, this branch is not merge-ready and
no public publication count or graph link changes.
