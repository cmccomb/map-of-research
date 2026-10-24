# Map of Research

`map-of-research` is the source of truth for the CMU faculty catalog and
publication corpus. Its normalized school hierarchy has dated faculty
rosters for Engineering, SCS, and MCS. It collects small, explicit batches of
public Google Scholar profiles and retains separately verified DOI records
for publications missing from the profile cache. It publishes one versioned
Hugging Face dataset and builds one browser-ready publication artifact. The
dataset ID retains its original Engineering name; the manifest records the
scope of each published revision.

The visualization repositories `map-of-eng` and `map-of-collaboration` do
**not** contact Google Scholar. They are static clients of
[`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications).

## Architecture

1. `registry/people.csv`, `memberships.csv`, `departments.csv`, and
   `schools.csv` separate stable identity, role history, unit inclusion,
   school hierarchy, and official annual-review sources.
2. A cache-first `scholarly` collector refreshes at most two unique profiles in
   an explicitly started run. It never uses proxies or attempts to bypass a
   block.
3. `data/curated/works.json` holds separately verified DOI observations,
   including the Tetali–Erdős and Mathematical Sciences–MSE papers, with
   source URLs, verification dates, and external coauthor identities.
   The raw Parquet snapshot and checksum manifest bind both the registry and
   curated source on the `automation/map-snapshot` branch.
4. A separate trusted workflow validates that snapshot, records a versioned
   map-eligibility decision for every observation and work, reuses existing
   embeddings and compatible layouts, assigns a reviewed topic hierarchy to
   visible regions, audits diffuse low-information regions, and publishes
   `people`, `works`, `authorships`, and
   `profile_publications` configs plus `maps/publications.json`.
5. `map-of-eng` packages that artifact and switches between the precomputed PCA
   and t-SNE layouts in the browser. `map-of-collaboration` derives a
   coauthorship graph from its faculty and work IDs. Neither site owns a faculty
   roster, scraper, or publication embedding model.

This split keeps the Hugging Face token away from the Scholar-facing process,
fetches each selected profile at most once per collection run, preserves the
last good cache on errors, and makes every collection/upload auditable.

## Scholar access policy

Google Scholar does not offer bulk access and asks automated clients to respect
its robots policy. The maintained defaults are intentionally conservative:

- no search-result crawling, citation traversal, proxying, or CAPTCHA bypass;
- direct lookup of registry-listed public profile IDs only;
- at most two profiles per explicitly started run, with a 90-second
  inter-profile delay;
- no profile refreshed more often than once per year;
- immediate stop after the first Scholar error or block;
- old data retained when a refresh fails.

Collection can be disabled without affecting the site: it continues serving
the last successfully published dataset.

GitHub-hosted Scholar collection is manual-only. A bounded diagnostic on
2026-07-18 received HTTP 403 from a hosted macOS runner, so the workflow does
not schedule repeated attempts from that IP space. The pinned `scholarly`
client has a fail-closed response guard: the first 403, 429, or CAPTCHA aborts
the run before the library's internal retry path. Local collection uses the
same limits and must also stop after any refusal.

An explicit local browser capture can bootstrap a directly linked CMU profile
when the `scholarly` client cannot access it. It uses the same two-profile cap,
90-second inter-profile gap, 200-publication default limit, and stop-on-refusal
rule. Captured rows retain their Scholar citation URLs and checksums of the
locally saved source pages; a capped profile is labeled as such. Browser
capture is not a scheduled or search-based collection path.

A browser cache marked `complete_visible_profile` is excluded from bounded
`scholarly` refreshes, including `--force`. The collector checks again before
writing so a complete capture added during a run is also preserved. A new
complete browser capture is required to update its publication rows and
coverage claim; a response capped at the collector's 200-publication default
cannot replace them.

## Update cadence

- Faculty membership is reviewed annually on July 1.
- Scholar data changes only after an explicit, successful bounded collection;
  each profile has a one-year minimum refresh interval. Small manual local
  batches may run when access permits, under the same stop-on-refusal limits;
  there is no scheduled retry.
- Curated DOI records change only after a person verifies the CMU/publisher
  evidence and updates `data/curated/works.json`. Their verification date is
  separate from the map's Scholar-profile refresh date.
- Snapshot validation and Hugging Face publication are separate steps. The
  trusted `upload.yml` workflow can be dispatched with a reviewed snapshot
  ref; a GitHub Actions commit from the collection workflow does not itself
  start the upload workflow.
- The map's daily or manually dispatched Pages build packages the newest
  immutable Hugging Face artifact at the same origin; browsers never contact
  Hugging Face for publication data.
- Visualization code deploys to GitHub Pages when site files change on `main`.

## Faculty inclusion and annual review

Maps include faculty, teaching faculty, and emeriti. Affiliate, courtesy,
visiting, former, and other excluded roles remain in the registry and dataset
for provenance but do not appear in mapped works. Missing Scholar IDs remain
blank; they are never guessed by an automated Scholar search.

Every July, a scheduled workflow opens a review issue with each official CMU
directory, role counts, unresolved IDs, and a completion checklist. The review
is deliberately human-verified and never changes registry rows automatically.
The [2026-10-24 SCS/MCS review](registry/reviews/2026-10-24.md) records the
initial school-wide directory expansion, source extracts, role boundaries,
identity decisions, and publication coverage gaps.

## Dataset grain

The lossless `profile_publications` config retains each Scholar profile
observation and each explicitly marked `curated_verified` DOI record with its
source metadata. `works` conservatively collapses observations by DOI or exact
normalized title and year; a DOI-less Scholar record matches a curated DOI only
when title and year match exactly one curated work. `authorships` makes that
derivation reversible. `people` includes registered people even when they have
no verified Scholar profile or publications. Curated records never count as
Scholar profile refreshes.

The research map excludes records that match high-confidence,
content-based rules for conference front matter, journal front matter,
organization or container records, affiliations and contact blocks,
person/citation indexes, navigation text, event labels, placeholders, garbled
index text, or titles too small to support a reliable semantic placement.
Missing year, venue, author text, DOI, citations, or source URL is never
sufficient for exclusion. The decision, policy version, and reason codes remain
on the dataset rows; excluded observations and works are not deleted. A work
stays map-eligible if any retained source observation passes the title-level
policy.

The separate reviewed-exclusion manifest names individual Scholar rows that
source review found to be nonpublications, false profile attributions, or
corrupted duplicates. Those rows stay in `profile_publications` with review
evidence. They do not supply work fields or faculty links when a valid
observation of the same work exists. See `data/curated/README.md`.

The publisher assigns every mapped work to one of 30 deterministic overview
regions and approximately 120 nested detail regions in the shared t-SNE layout.
Each receives a coverage-aware, corpus-specific phrase from its titles. A
versioned editorial catalog, checked against representative titles on
2026-07-19, turns those extracted phrases into concise public labels without
discarding the source phrase or its review status. Unmatched phrases remain
visible and are explicitly marked unreviewed. If a reviewed label collides with
another region's extracted phrase, the direct extraction keeps that label and
the reviewed region displays its own extracted phrase as unreviewed. The
`*_label_reviewed` flag means the reviewed wording was actually applied; an
exact catalog match alone is insufficient. Detail regions contain at least 50
works and are revealed by the site as visitors zoom.

A preliminary detail pass also exposes the rare catch-all region that isolated
title rules cannot identify safely. The publisher flags a region only when it
is simultaneously incoherent in embedding space, dominated by short titles,
missing many years, and mostly uncited. Inside a flagged region, a work with a
DOI, known year, substantive title, named authors, venue, and source URL stays
visible; weaker works retain the `low_information_region` exclusion. The
versioned audit records `map_region_documented_override` for each retained
exception, keeps all source rows in the dataset, and regenerates the final
topic hierarchy from visible works.

## Local development

Use Python 3.12 or newer.

```bash
python -m pip install --editable '.[collector,publisher,dev]'
python -m pytest
```

The test command enforces 100% statement and branch coverage for the package.

Plan a collection without network access:

```bash
map-collect --dry-run
```

For a small local department or school batch, use the full canonical registry
and target included, verified Scholar profiles before starting a real collection:

```bash
map-collect --dry-run --department-id math --max-profiles 2
map-collect --dry-run --school-id scs --max-profiles 2
map-collect --dry-run --school-id mcs --max-profiles 2
map-collect --dry-run --person-id person-prasadtetali-bf494329
```

`math` is the Mathematical Sciences department ID in
`registry/departments.csv`. `--department-id`, `--school-id`, and `--person-id`
are each repeatable, but cannot be mixed in one run. Targeting does not change
the profile cap, delay, annual refresh window, or stop-on-refusal rule. The
status receipt records the requested targets. Build snapshots from the full
registry, not a temporary targeted copy.

Build and validate a raw snapshot from the current cache:

```bash
map-build-snapshot
map-validate-snapshot \
  --snapshot snapshots/cmu-engineering-publications.parquet \
  --manifest snapshots/cmu-engineering-publications.parquet.manifest.json
```

`map-publish-snapshot` reads `data/curated/scholar-exclusions.json` by default;
pass `--reviewed-exclusions` when the trusted source checkout is in another
directory, as in the upload workflow. The upload receipt records the manifest
checksum and number of matched exclusions.

Publishing requires `HF_TOKEN` locally. The protected `huggingface` GitHub
environment exposes the same credential to the upload workflow from a secret
named `HUGGINGFACE_TOKEN`.

## Repository roles

- `map-of-research`: faculty catalog, collection, validation, dataset
  publication, and the single full-corpus artifact
- `map-of-eng`: static publication map
- `map-of-collaboration`: static coauthorship graph derived from the same work
  and faculty IDs

The legacy department-specific `map-of-*` repositories are archived. Their
department views now live as filters in `map-of-eng`.

See `status/README.md` for the machine-readable receipts and `SECURITY.md` for
reporting guidance. The normalized Hub and browser artifact fields are defined
in `docs/dataset-schema.md`.
