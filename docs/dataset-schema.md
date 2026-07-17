# Dataset and artifact schema

The normalized Hugging Face config is named `publications`. It intentionally
differs from the legacy `default` config.

Each row represents one publication listed on one unique Scholar profile. A
cross-appointed faculty member still has one row per profile publication; their
site memberships are nested rather than duplicated.

| Field | Type | Meaning |
| --- | --- | --- |
| `scholar_id` | string | Public profile identifier from the registry |
| `faculty` | string | Canonical display label |
| `map_slugs` | list[string] | Machine-readable map memberships |
| `memberships` | list[struct] | Per-map slug, department, and faculty label |
| `author_pub_id` | string | Scholar's profile-publication identifier |
| `title` | string | Publication title |
| `authors` | string | Author text reported by the profile |
| `year` | nullable integer | Parsed publication year |
| `venue` | string | Best available venue text |
| `citation` | string | Original citation text when present |
| `citation_count` | non-negative integer | Citation count at collection time |
| `source_url` | string | Publication URL when present |
| `fetched_at_utc` | timestamp string | Time that profile cache was refreshed |
| `embedding` | list[float32] | 768-dimensional pinned MPNet embedding |

The publisher also writes `maps/<map-slug>.json` and `maps/manifest.json` to the
dataset repository. Map artifacts contain only the browser-facing fields and
deterministic two-dimensional PCA coordinates. Department artifacts expand the
matching membership label; `map-of-eng` uses the canonical faculty label and
one primary department per profile publication.

The raw automation-branch Parquet snapshot omits embeddings. Its manifest binds
the file checksum, schema version, registry checksum, row count, profile count,
and creation time. Publication is rejected if any of those checks fail or the
snapshot is stale.
