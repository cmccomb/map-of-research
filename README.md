# Map of Research

`map-of-research` is the source of truth for the CMU faculty catalog and
publication corpus. Its normalized school hierarchy has dated faculty
rosters for all seven CMU schools: Engineering, SCS, MCS, CFA, Dietrich,
Heinz, and Tepper. It collects small, explicit batches of
public Google Scholar profiles and retains separately verified work records
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
3. `data/curated/works.json` holds separately verified DOI and source-native
   observations, including the Tetali–Erdős papers and historical Donahue
   conference items, with source URLs, verification dates, and any external
   coauthor identities.
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
90-second inter-profile gap, and stop-on-refusal rule. The capture pages through
the visible profile until Scholar offers no more rows; a deliberate row cap
leaves the cache labeled `capped_partial`. A refusal preserves the prior good
cache and records the failed attempt. A completed capture is labeled
`complete_visible_profile`, which describes the visible Scholar profile, not
the researcher's entire academic history. Captured rows retain their Scholar
citation URLs and checksums of the saved source HTML and row JSON. Browser
capture is not a scheduled or search-based collection path.

A browser cache marked `complete_visible_profile` is excluded from bounded
`scholarly` refreshes, including `--force`. The collector checks again before
writing so a complete capture added during a run is also preserved. A new
complete browser capture is required to update its publication rows and
coverage claim; a response capped at the collector's 200-publication default
cannot replace them.

The trusted snapshot and upload workflows also check source coverage. The
frozen [legacy cache baseline](registry/legacy-scholar-cache-baseline.json)
allows the exact caches already present in the 2026-10-25 source snapshot to
remain available while they await recapture. A new cache, a changed cache, or
a previously excluded person newly included on the map needs a complete
visible-profile browser receipt. The check compares all cached citation IDs
and visible row text with the saved Scholar JSON and HTML, verifies their
checksums and disabled final Show More control, and compares the upload's raw
Scholar rows with those caches. A bounded collector result without that
receipt cannot enter a new published snapshot. If hosted collection or source
validation fails, its workflow commits only `status/last-collection.json` with
the `snapshot_validation` failure reason; unvalidated caches and snapshots stay
out of the source branch. The frozen baseline also keeps all 49,687 Scholar
citation IDs already present in that snapshot. After a new
complete profile is published and verified, its citation IDs should be added
to the reviewed retention baseline before a later recapture; the frozen
baseline does not track later publications automatically.

Complete capture and public release are separate. The trusted
`data/curated/scholar-profile-releases.json` decision currently holds Chien
Ho's complete 582-row source while exact attribution and context review
continues. The raw snapshot and coverage gates still validate all 582 rows.
The publisher checks the decision against the registry, exact cache checksum,
and raw citation IDs, then omits held own-source observations from public
datasets and embeddings. The roster remains visible, and publications sourced
from other faculty are unaffected. The upload receipt reports raw and
published counts and the decision-manifest checksum.

Legacy caches already at or above a requested publication limit are also
excluded from that bounded refresh, including `--force`. For smaller caches,
the collector rejects a replacement missing any previously observed Scholar
citation ID and keeps the last good cache for review. This prevents a partial
response from silently removing older publication rows. It also rejects an
incoming response that reaches the requested limit, since more Scholar rows
may remain, and refuses to refresh a malformed existing cache until its
publication list, declared row count, and citation IDs are reviewed.

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
The [CFA](registry/reviews/2026-10-24-cfa.md),
[Dietrich](registry/reviews/2026-10-24-dietrich.md),
[Heinz](registry/reviews/2026-10-24-heinz.md), and
[Tepper](registry/reviews/2026-10-24-tepper.md) reviews record the four-school
faculty expansion and its official source extracts. These rows establish
faculty identities and appointments only; they do not add publication data.

## Dataset grain

The lossless `profile_publications` config retains each Scholar profile
observation and each explicitly marked `curated_verified` record with its
source metadata. `works` conservatively collapses observations by DOI, a
curated source-native ID, or exact normalized title and year. A DOI-less
Scholar record matches a curated identity only when its title and year match
exactly one curated work. `authorships` makes that
derivation reversible. `people` includes registered people even when they have
no verified Scholar profile or publications. Curated records never count as
Scholar profile refreshes.

For a verified profile, its visible Google Scholar rows are the primary
publication source. Official CMU pages establish the profile identity. Curated
records are clearly labeled supplemental evidence for works missing from
the Scholar cache; they do not redefine whether the visible profile was fully
captured. The saved raw rows remain available when a specific attribution or
nonpublication needs review.

Exact source-backed context decisions in `data/curated/scholar-contexts.json`
identify patents, edited books, panel transcripts, and withdrawn articles
without changing captured Scholar rows. Patent publication numbers keep
same-title documents distinct. Nine reviewed patents and one edited book are
typed research-map items without coauthor edges. The exact withdrawn article
and panel transcript stay in the lossless `profile_publications` table but are
excluded from the map and collaboration graph.

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
Substantive, dated errata, corrigenda, and corrections are citable works;
reviewed exact-ID decisions handle a falsely attributed or corrupted notice.

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
visible. A DOI-less work also stays visible when a separately verified curated
record supplies a stable native identifier and dated HTTPS evidence for those
bibliographic details. An exact reviewed DOI alias may also retain a shorter
title when its Scholar citation, DOI, year, and publisher chapter URL are pinned
in the alias manifest with `allow_short_title_region_override: true`. The
current entry covers Gordon Rule's *Exchange processes* chapter. Weaker works
retain the `low_information_region` exclusion. The
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

The hosted upload workflow runs `map-validate-math-coverage` before requesting
its Hugging Face token. For every included Mathematical Sciences person with a
verified Scholar ID, it checks saved full-profile JSON and HTML evidence,
cache rows, and raw snapshot citation IDs. People without an exact verified ID
remain visible as identity gaps. The `map-publish-snapshot` CLI repeats this
check and the complete Scholar source check before any upload, including when
invoked outside the workflow. Both checks have dedicated pre-authentication
steps in the hosted upload workflow. Its
default evidence directories are `data/authors` and `data/captures` beneath
the snapshot checkout; pass `--math-cache-dir` and `--math-capture-dir` for a
different layout for the Math check, or `--scholar-cache-dir`,
`--scholar-captures-dir`, and `--scholar-baseline` for the full-source check.
Direct Python calls to the `publish_snapshot` library function are intended
for controlled previews and do not enforce these gates.

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
