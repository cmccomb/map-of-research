# Dataset and artifact schema

The Hugging Face repository contains four schema-v6 configurations. Together
they retain the collection grain while exposing a cleaner work-centric model.

## `people`

One row per stable registry identity, including people without a Scholar ID.
Fields include `person_id`, display name, Scholar ID, Scholar-ID verification
source and date, ORCID, homepage, notes, all retained memberships, included
department IDs, observation count, unique work count, and mapped work count.

## `profile_publications`

One row per publication observation on one Scholar profile. This is the
lossless analytical source. It retains the original profile-publication ID,
parsed bibliographic fields, citation count at collection time, complete
normalized source record as JSON, fetched timestamp, memberships, work match
method, title embedding, `map_eligible`, `map_exclusion_reasons`, and the quality
assessment version.

## `works`

One canonical work derived conservatively from the observations. DOI is used
when present; otherwise an exact normalized title and year are used. Undated
records without a DOI remain profile-specific to avoid unsafe merges.

The table retains field variants, all source observation IDs, profile IDs,
faculty relationships, source URLs, first and last observation timestamps,
match method, observation count, embedding, eligibility decision, exclusion
reasons, both shared full-corpus map layouts, and the assigned topic hierarchy.
`x` and `y` are the broad-structure PCA coordinates; `tsne_x` and `tsne_y` are
the local-neighborhood t-SNE coordinates after a memory-bounded 50-dimensional
PCA reduction and a final deterministic orientation. `keyword_id`, `keyword`,
`detail_keyword_id`, `detail_keyword`, and `keyword_model_version` record the
deterministic nested visible-region assignment. `map_region_audit_version` and
`map_region_outlier` retain the result of the multi-signal catch-all audit.
Excluded works retain their coordinates for reproducibility and safe layout
reuse; `map_eligible` alone controls browser-map participation.
Citation count is the maximum retained source observation, not a sum across
faculty profiles.

Canonical map fields prefer passing observations when a DOI joins passing and
excluded variants of the same work. All variants remain in their loss-aware
arrays and in `profile_publications`. The title-level policy excludes a work
from layout fitting only when every source observation is excluded; the later
region audit affects the browser artifact without discarding fitted coordinates.

## `authorships`

One reversible relationship between a canonical work and the registered person
whose Scholar profile supplied the observation. It carries the work, person,
observation and profile-publication IDs plus every retained department
membership and fetch timestamp.

## Browser artifacts

`maps/publications.json` uses schema version 6. It is the only browser artifact.
Every point represents one work and contains both precomputed full-corpus
coordinate pairs, work ID, title, author text, stable faculty and department ID
arrays, year, venue, citation count, DOI, first available source URL, and source
observation count, plus one `keyword_ids` entry per hierarchy level. The
top-level `layouts` catalog gives
each view's label, method, interpretation, coordinate fields, and version;
`default_layout_id` selects the initial view. The `keywords` catalog provides a
concise label, hierarchy level, optional parent, publication count, and centroid
in every layout for each topic region. `keyword_levels` describes the ordered
overview and detail levels. `quality_assessment_version` identifies the policy,
`keyword_model_version` identifies the labeling procedure, and
`region_audit_version` identifies the multi-signal catch-all audit. The
`excluded_work_count` reports how many faculty-linked works were retained in
the dataset but left out of the visualization.

The embedded department catalog provides titles, directory sources, annual
review dates, and publication counts. The faculty catalog includes every person
with an included faculty, teaching, or emeritus membership—even when that
person has no verified Scholar profile or mapped publications—and provides
stable identity, public identifiers, memberships, roles, and publication
counts. The complete loss-aware records remain in the four dataset configs.

The artifact reports both generation time and the oldest/newest underlying
profile refresh timestamps. The site displays source freshness rather than
implying that an artifact upload refreshed every Scholar profile.

PCA is the default because it preserves broad global variation. The t-SNE view
is provided for local neighborhood exploration; spacing between separate t-SNE
clusters must not be interpreted quantitatively. Both layouts use the same
eligible canonical work rows and title embeddings. Compatibility includes the
quality-policy layout version, so a policy change forces a complete refit rather
than mixing old and new coordinates. Compatible coordinates are reused when the
eligible corpus is unchanged so routine republishes do not move points.

Topic keywords partition the visible t-SNE landscape with deterministic
k-means and label each region from its publication titles using repeated 2- and
3-word phrases weighted by within-region coverage and corpus specificity. The
default hierarchy has 30 overview regions and approximately 120 nested detail
regions; undersized detail fragments merge into their nearest sibling. Stable
keyword IDs are derived from labels and centroids rather than k-means' arbitrary
numbering. Keywords describe visible neighborhoods and do not claim a formal
taxonomy.

## Map-eligibility policy

The policy removes high-confidence non-research records from layout computation
without deleting source data. Reason codes cover conference front matter,
publication front matter, organization or container records, affiliations or
contact blocks, person or citation indexes, navigation or interface text, event
or session labels, placeholders or document controls, garbled index text, and
underspecified titles that cannot carry a reliable semantic placement.

No single missing bibliographic field, citation count, or embedding position is
an exclusion rule. After the title-level policy, a provisional detail hierarchy
is used to audit large catch-all regions. A region is withheld only when four
independent thresholds all fail: embedding coherence, year completeness,
citation presence, and title length. This conservative conjunction prevents any
single incomplete signal from becoming a broad filter. Affected works remain in
`works` with `low_information_region`; only their browser-map participation is
disabled.
