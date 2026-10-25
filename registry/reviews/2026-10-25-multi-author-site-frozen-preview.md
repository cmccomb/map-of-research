# Iyer and Anderson: combined frozen-cache source preview

**State: no-upload, release held.** This draft composes the 47-row [Gautam
Iyer source](../../data/author_sites/gautam-iyer-2026-10-25.json) and 42-row
[Theresa Anderson source](../../data/author_sites/theresa-anderson-2026-10-25.json)
in one schema-v7 raw snapshot. Both captures are `author_site`, not Google
Scholar. The publisher still rejects `author_site` before a Hub write; this
preview did not invoke the publisher or change the source branch, Hub dataset,
or public site.

## Frozen input and exact diff

The read-only Scholar cache was the 311 JSON files in source worktree
`8be6a2400cf72dd959c874bfcada8624540820ef` of `automation/map-snapshot`.
Their sorted filename-to-SHA-256 manifest hashes to
`52ce8e066456f877dcd22e6b65a3d1a9a19431f95c2cd64a13f03daf9aa63e77`;
all 311 file hashes were identical before and after the two local builds. Both
builds used the same producer registry, curated records, frozen cache, and
timestamp (`2026-10-25T12:15:00Z`).

| Source rows | Baseline | Iyer + Anderson |
| --- | ---: | ---: |
| Google Scholar | 51,851 | 51,851 |
| Curated verified | 28 | 28 |
| Author site | 0 | 89 |
| **Total** | **51,879** | **51,968** |

The 89 additions are exactly 47 Iyer and 42 Anderson observations. Every
preexisting Scholar and curated row had the same `(source_type, scholar_id,
author_pub_id)` key and the same value across every baseline column after
Parquet array and nullable-scalar normalization. Registry checksum, curated
file checksum, cached profile count, and cached-profile-ID checksum were
unchanged. The baseline Parquet SHA-256 was
`c2c71f5e0b22c0c0905f0769d327a44b638a78da5cb08d95f516d1d391c5b935`;
the combined Parquet SHA-256 was
`8311bc7ab35edcd836113ff174afcd34346d01314f539660b92b427980353611`.
The combined snapshot passed `validate_snapshot` with both manifests supplied
in reverse order.

The manifest's `author_site_captures` list sorts by person ID and binds each
person, filename, manifest checksum, and exact row count. Validation reloads
the pinned HTML/CV-backed source through each capture's loader and compares
every author-site row to the snapshot. Duplicate people, capture filenames,
and source-native IDs fail before a snapshot is accepted. Legacy zero- or
one-capture schema-v7 manifests without the new list still validate with their
singular capture fields.

This is source-graph preparation, not a release decision. Exact work identity,
the Anderson metadata variants, consumer provenance, and the publisher release
gate remain separate review work.
