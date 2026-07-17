# Map of Research

`map-of-research` is the source of truth for the CMU engineering publication
maps. It collects a deliberately small number of public Google Scholar profiles,
publishes one versioned Hugging Face dataset, and builds browser-ready map
artifacts for the department sites.

The department repositories do **not** contact Google Scholar. They are static
clients of [`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications).

## Architecture

1. `registry/people.csv`, `memberships.csv`, and `maps.csv` separate stable
   identity, role history, map inclusion, and official annual-review sources.
2. A cache-first collector refreshes at most two unique profiles in a scheduled
   run. It never uses proxies or attempts to bypass a block.
3. A raw Parquet snapshot and checksum manifest are staged on the
   `automation/map-snapshot` branch.
4. A separate trusted workflow validates that snapshot, reuses existing
   embeddings, and publishes `people`, `works`, `authorships`, and
   `profile_publications` configs plus precomputed map JSON files.
5. Each `map-of-*` site fetches its JSON file from the Hub and renders it with a
   small dependency-free canvas client.

This split keeps the Hugging Face token away from the Scholar-facing process,
fetches each unique profile only once, preserves the last good cache on errors,
and makes every collection/upload auditable.

## Scholar access policy

Google Scholar does not offer bulk access and asks automated clients to respect
its robots policy. The maintained defaults are intentionally conservative:

- no search-result crawling, citation traversal, proxying, or CAPTCHA bypass;
- direct lookup of registry-listed public profile IDs only;
- at most two profiles per scheduled run, with a 90-second inter-profile delay;
- no profile refreshed more often than once per year;
- immediate stop after the first Scholar error or block;
- old data retained when a refresh fails.

Collection can be disabled without affecting the sites: they continue serving
the last successfully published dataset.

## Faculty inclusion and annual review

Maps include faculty, teaching faculty, and emeriti. Affiliate, courtesy,
visiting, former, and other excluded roles remain in the registry and dataset
for provenance but do not appear in map artifacts. Missing Scholar IDs remain
blank; they are never guessed by an automated Scholar search.

Every July, a scheduled workflow opens a review issue with each official CMU
directory, role counts, unresolved IDs, and a completion checklist. The review
is deliberately human-verified and never changes registry rows automatically.

## Dataset grain

The lossless `profile_publications` config retains every collected profile
observation and its normalized source record. `works` conservatively collapses
observations by DOI or exact normalized title and year; `authorships` makes that
derivation reversible. `people` includes registered people even when they have
no verified Scholar profile or publications.

## Local development

Use Python 3.12 or newer.

```bash
python -m pip install --editable '.[dev]'
python -m pytest
```

Plan a collection without network access:

```bash
map-collect --dry-run
```

Build and validate a raw snapshot from the current cache:

```bash
map-build-snapshot
map-validate-snapshot \
  --snapshot snapshots/cmu-engineering-publications.parquet \
  --manifest snapshots/cmu-engineering-publications.parquet.manifest.json
```

Publishing requires `HF_TOKEN` and is normally performed only by the protected
upload workflow.

## Repository roles

- `map-of-research`: registry, collection, validation, publication, map assets
- `map-of-eng`: thin all-engineering visualization
- department `map-of-*` repositories: thin filtered visualizations

See `status/README.md` for the machine-readable receipts and `SECURITY.md` for
reporting guidance. The normalized Hub and browser artifact fields are defined
in `docs/dataset-schema.md`.
