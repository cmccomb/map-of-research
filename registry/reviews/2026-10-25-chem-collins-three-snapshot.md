# Collins and three Chemistry faculty: no-upload release preview

Built 2026-10-25 from the Math-live source snapshot at `31b1492`, plus the
complete source captures later merged through #118 and #250. The original
candidate used trusted producer #253 (`15ecc203`). An independent frozen-pair
rebuild from producer `9347ff51f9ea5c345bfd6f69d6933fbdc7fc8e98` and source
`134d1f20f6c8716ebdfb07a0c8f4763885f29d8b` produced byte-identical
Parquet and the same manifest fields except the build timestamp. The producer's
paired registry and exclusion/hold manifests supply the release decisions.
This review branch does not upload or change either public map.

| Scholar profile | Complete raw rows | Exact exclusions | Provisional holds | Eligible observations before map region filtering |
| --- | ---: | ---: | ---: | ---: |
| Terrence J. Collins `0e1jwkkAAAAJ` | 361 | 31 | 7 | 323 |
| Nimer Murshid `MPxFaHYAAAAJ` | 49 | 3 | 2 | 44 |
| Gloria L. Silva `vF51LxIAAAAJ` | 41 | 0 | 2 | 39 |
| Simon Faulkner `tPp2aaoAAAAJ` | 5 | 1 | 0 | 4 |

All 51,139 existing `(scholar_id, author_pub_id)` keys and their raw
`source_record_json` values remain unchanged. The 456 new keys exactly match
the four complete canonical caches. The combined snapshot has 51,595 raw rows,
309 cached Scholar profiles, and 26 preserved source-verified curated rows.
Its SHA-256 is
`b72df04cf28e5af1450b4010ac05a91c5f68b1b671fa19d2a2e7a93a22d46e58`.

The local no-upload preview used the producer's `validate_snapshot`, primary
home check, Math coverage check, full Scholar coverage check, exact cache-bound
profile release gate, exclusion and provisional-hold checks, and relationship
table builder. Math remained at 36 complete profiles/4,080 rows; full Scholar
coverage recognized 23 complete and 286 frozen legacy profiles. The transformed
tables contained 47,495 works and 49,478 authorships before embedding, layout,
topic, and low-information-region processing. The release gate confirmed that
held observations contributed no public works or authorships. The machine-readable
receipt is `work/chem-collins-three-preview-results.json` in the task workspace.

Source PRs #118 and #250 and producer PRs #243 and #253 have merged with green
exact-head CI. The final frozen producer and source heads above match this
preview. The upload workflow adds `--provisional-holds` to the already present
primary-home gate. After this snapshot PR passes final CI, run its protected
trusted upload, then verify the immutable Hub receipt and both map deployments.
