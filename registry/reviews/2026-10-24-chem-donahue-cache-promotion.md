# Neil Donahue: complete visible cache promotion

The reviewed `data/captures/donahue-visible-623-2026-10-24.candidate-cache.json`
has SHA-256 `424269fcfed5f480ffda279f6013a3a6841f49f5849e5e559a949871106fdba3`.
The canonical `data/authors/ILambBMAAAAJ.json` has the same 623 publication
rows, but its `browser_capture.source_html_sha256` was corrected from the
unstaged original HTML digest to the staged HTML digest
`4c2cb0ee0144c9850133931feb1f7e327da30b788d2b46cc13e8846f7eb45494`.
That single field change gives the canonical cache SHA-256
`706baa33811d7970d623176601f0a293807c74ca489e7c93d63db2ffbb4ef4ec`.
The saved HTML, corrected row extraction, receipt, and identity review are in
[the capture review](2026-10-24-chem-donahue-capture.md). The final Scholar
**Show more** control was disabled after 623 rows. This is the full profile
visible at capture time, not a claim that Scholar lists every work Donahue
ever authored.

The old 551-row cache and the current 623-row profile share 539 citation IDs.
The new profile adds 84 IDs and omits 12. Every omitted ID is classified in
`status/chem-donahue-disappeared-2026-10-24.csv`. Four are independently
documented historical conference records absent from the current profile;
producer [PR #123](https://github.com/cmccomb/map-of-research/pull/123)
proposes them as separate curated works without invented current Scholar IDs.
The other eight are rekeys, malformed or supporting-material duplicates, and
site text. The old cache remains in Git history and the exact-ID ledger.

## Reproducible source preview

Use the **trusted producer `main` snapshot builder**, the current source-branch
registry and caches, and the producer curated catalog. The source branch holds
historical package code that does not yet read curated works, so importing its
`map_of_research.snapshot` directly will drop them. The collection workflow
already installs the trusted producer package and copies the producer registry
and curated catalog into the source checkout before rebuilding.

In two sibling checkouts named `producer` (`main`) and `snapshot`
(`automation/map-snapshot`), the workflow's local equivalent is:

```bash
python -m pip install './producer[collector]'
cp producer/registry/{departments,schools,memberships,people}.csv snapshot/registry/
cp producer/data/curated/works.json snapshot/data/curated/works.json
(cd snapshot && map-build-snapshot && map-validate-snapshot \
  --snapshot snapshots/cmu-engineering-publications.parquet \
  --manifest snapshots/cmu-engineering-publications.parquet.manifest.json)
```

With producer `e33cf00`, source `7fee7fc`, and the unchanged 18-row curated
catalog, the builder reproduced the released 47,696-row Parquet SHA-256
`8b935ad922aab84a80a614ab6d9807cca88f6d8c6ab7a28c8c540e39b2c70dca`.
Replacing only Donahue's cache and rerunning that builder produced 47,768 raw
rows, 270 cached profiles, and 18 curated rows, with Parquet SHA-256
`121331df665559a357c8cc76122350418eaaad6c190f9bad1eaddc3327a5b60e`.
The trusted validator accepted this local preview. All 47,145 non-Donahue rows
are unchanged; there are no duplicate `(scholar_id, author_pub_id)` keys. The
Donahue delta is exactly 551 to 623 rows, or 72 additional current
observations after 84 additions and 12 removals.

## Release dependency

This draft changes no committed snapshot or public dataset. Before rebuilding
and publishing, coordinate with producer
[PR #218](https://github.com/cmccomb/map-of-research/pull/218), which combines
the CMU-linked Scholar provenance, 43 exact graph exclusions, separately
verified historical works, and the exact prior-ID crosswalk. Sync that
producer registry and curated catalog into the source checkout, then build
and validate the new snapshot with the trusted producer package. Publishing
the old 551-row cache with the new curated catalog would duplicate the
truncated 2007 A&WMA entry.
