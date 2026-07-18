# Map of Research

`map-of-research` is the source of truth for the CMU engineering faculty catalog
and publication corpus. It collects a deliberately small number of public
Google Scholar profiles, publishes one versioned Hugging Face dataset, and
builds one browser-ready artifact for the unified research map.

The only visualization repository, `map-of-eng`, does **not** contact Google
Scholar. It is a static client of
[`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications).

## Architecture

1. `registry/people.csv`, `memberships.csv`, and `departments.csv` separate
   stable identity, role history, inclusion, and official annual-review sources.
2. A cache-first collector refreshes at most two unique profiles in a scheduled
   run. It never uses proxies or attempts to bypass a block.
3. A raw Parquet snapshot and checksum manifest are staged on the
   `automation/map-snapshot` branch.
4. A separate trusted workflow validates that snapshot, reuses existing
   embeddings and compatible layouts, and publishes `people`, `works`,
   `authorships`, and `profile_publications` configs plus
   `maps/publications.json`.
5. `map-of-eng` fetches that artifact and switches between the precomputed PCA
   and t-SNE full-corpus layouts in the browser; it owns no faculty roster,
   publication data, scraper, embedding model, or layout computation.

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

Collection can be disabled without affecting the site: it continues serving
the last successfully published dataset.

## Faculty inclusion and annual review

Maps include faculty, teaching faculty, and emeriti. Affiliate, courtesy,
visiting, former, and other excluded roles remain in the registry and dataset
for provenance but do not appear in mapped works. Missing Scholar IDs remain
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

Publishing requires `HF_TOKEN` locally. The protected `huggingface` GitHub
environment exposes the same credential to the upload workflow from a secret
named `HUGGINGFACE_TOKEN`.

## Repository roles

- `map-of-research`: faculty catalog, collection, validation, dataset
  publication, and the single full-corpus artifact
- `map-of-eng`: the single static interactive visualization

The legacy department-specific `map-of-*` repositories are archived. Their
department views now live as filters in `map-of-eng`.

See `status/README.md` for the machine-readable receipts and `SECURITY.md` for
reporting guidance. The normalized Hub and browser artifact fields are defined
in `docs/dataset-schema.md`.
