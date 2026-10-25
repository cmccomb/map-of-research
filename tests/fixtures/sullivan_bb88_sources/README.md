# Historical Sullivan approval sources

These five compressed JSON files expand to byte-for-byte copies of their paths at producer commit
`b289b1a2ecdfb74e398d5a4a5b2c87d78ee5f335` (the merged Sullivan gate).
Their SHA-256 values are pinned by the unchanged
`data/curated/sullivan-exact-losses-bb88.json` approval. The Sullivan regression
test must use these historical inputs because later Chemistry releases extend
the live curation files. The production publisher applies that approval only
when the Hub parent is `bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd` and
the source snapshot is `f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2`.

The approval manifest itself stays in its production path; this fixture does
not grant any approval for a later release.

The files were compressed with `gzip.compress(..., compresslevel=9, mtime=0)`.
The validator checks each expanded file against the manifest's pinned SHA-256.
