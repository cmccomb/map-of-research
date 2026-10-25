# Hyung J. Kim: complete Scholar capture and held release proposal

This is a **nonoperational review proposal**. Hyung J. Kim remains off the
public map. The complete browser capture from source PR #118 has 918 distinct
Scholar citation IDs and no remaining profile rows. Its unchanged JSON SHA-256
is `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`;
the paired redacted HTML SHA-256 is
`a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94`.
The capture is a complete **visible Scholar profile**, not proof that every
publication in Kim's academic history appears there.

## Release blocker measured against the current publisher

At producer head `fa8b087`, I converted the saved 918 rows to the ordinary
snapshot shape in memory, added inert embedding vectors, and called
`build_dataset_tables` with the existing `scholar-exclusions.json`. This made
**409 of 410 unresolved observations map eligible**. The one exception,
`bC4aq-4AAAAJ:KaMxkj08jr0C` (“I-SEEC 2012”), was rejected by the generic
`event_or_session_label` quality rule. All 394 exact exclusions were
suppressed, and all 114 positive person signals were eligible. The publisher
currently has no hold set, so installing the complete cache before a verified
hold gate would put unresolved attributions into graph works and edges.

## Conservative exact-ID proposal

The [prepared release fragment](proposals/scholar-profile-releases.kim-proposal.json)
and [114 source confirmations](proposals/scholar-attribution-reviews.kim-proposal.json)
partition every captured ID as follows:

| Disposition | IDs | Sorted-ID SHA-256, joined by newline without a final newline |
| --- | ---: | --- |
| Confirmed source-backed works for a possible reviewed subset | 114 | `a7fa4ee3c74d12e25cd0e17e011710efbc62e61ac3bfb50f13d6f2f22f599a3a` |
| Exact false-attribution, duplicate, or nonpublication exclusions | 394 | `e074bac97acf1dd2c4aa762e71f1551b148172125d13561330b9bf6b13e14a8e` |
| Held from any map release | 410 | `544356f5eacd3d84060d4ba213b0b9669ba527e0acc6e827319bf20d88337b66` |

The 410 holds comprise observations that still lack enough output-level review
for this conservative subset. The 114 confirmed entries have exact Scholar
IDs and work-specific publisher or CMU evidence in the linked fragment and in
the [historical](2026-10-25-chem-kim-historical-primary-retentions.md),
[group bibliography](2026-10-25-chem-kim-lab-bibliography-follow-up.md), and
[five latest](2026-10-25-chem-kim-five-primary-retentions.md) review notes,
plus the [first eight CMU group matches](2026-10-25-chem-kim-group-eight-primary-retentions.md)
and [ten further group/publisher matches](2026-10-25-chem-kim-group-ten-primary-retentions.md),
followed by [eleven AIP/PLOS/BKCS matches](2026-10-25-chem-kim-group-eleven-primary-retentions.md)
and [ten recent CMU faculty/publisher matches](2026-10-25-chem-kim-faculty-ten-primary-retentions.md),
followed by [twelve systematic official-bibliography/publisher matches](2026-10-25-chem-kim-official-bibliography-twelve-primary-retentions.md)
and [28 full-name publisher matches with CMU or matching ORCID identity](2026-10-25-chem-kim-publisher-fullname-twentyeight-primary-retentions.md),
then [the final 20 exact CMU ORCID signals with original publisher and institutional evidence](2026-10-25-chem-kim-remaining-twenty-primary-retentions.md).
Published ACS/RSC correction notices at rows 643, 709, and 796 have their
own DOI/page evidence and remain distinct works. The 114 confirmed rows do
**not** represent Kim's whole publication history.
The [additional high-citation check](2026-10-25-chem-kim-three-high-impact-holds.md)
keeps three more initials-only records held despite suggestive related sources.

The offline [preparer](prepare_chem_kim_reviewed_subset.py) recreates a
candidate `data/authors/bC4aq-4AAAAJ.json`-shaped cache and both proposal
fragments from the saved capture and review ledger. The candidate cache is
kept outside canonical `data/authors` and has byte SHA-256
`f4b7b37efcd3310a05ff82057488ecb286450e3db0dc390b4e26b125fc378b2e`.
A no-upload snapshot build/validation from that candidate retained 918 Kim
Scholar rows plus 18 existing curated observations, with 918 unique Kim IDs.
Its `browser_capture.coverage` marker protects the complete capture from a
later bounded collector refresh.

To reproduce locally with the verified source file:

```sh
python3 -m registry.reviews.prepare_chem_kim_reviewed_subset \
  --capture /path/to/kim-visible-918-2026-10-24.json \
  --output-dir /tmp/kim-review-proposal
```

Before any public use, a trusted producer gate must compare the cache SHA-256,
all 918 exact IDs, the 114 confirmed IDs, and the 394 exact exclusion IDs to
these lists; it must prevent all 410 held rows from creating works,
authorships, or coauthor edges. Even after that guard exists, Kim should stay
off public maps until the subset is clearly represented as incomplete or the
remaining outputs receive work-level review. This proposal does not edit the
canonical cache, publisher manifests, snapshot, or public map.
