# Dataset and artifact schema

The Hugging Face repository contains four schema-v7 configurations. Together
they retain the collection grain while exposing a cleaner work-centric model.

## School hierarchy

`registry/schools.csv` defines school IDs and public directories. Every unit in
`registry/departments.csv` has one parent `school_id`. A person or work can have
multiple derived school IDs through included department memberships. The
snapshot manifest hashes all four registry files so the publisher cannot pair
a snapshot with a different school hierarchy.

## `people`

One row per stable registry identity, including people without a Scholar ID.
Fields include `person_id`, display name, Scholar ID, Scholar-ID verification
source and date, ORCID, homepage, notes, all retained memberships, included
department IDs, raw observation count, eligible unique work count, and mapped
work count.
`included_school_ids` and school information on memberships are derived from
the validated registry; they are not a second identity or collection key.

## `profile_publications`

One row per publication observation on one Scholar profile. This is the
lossless analytical source. It retains the original profile-publication ID,
parsed bibliographic fields, citation count at collection time, complete
normalized source record as JSON, fetched timestamp, memberships, work match
method, title embedding, `map_eligible`, `map_exclusion_reasons`, and the quality
assessment version. Reviewed exclusions add a reason code, specific reason,
evidence URL, and verification date on the same retained row.

## `works`

One canonical work derived conservatively from the observations. DOI is used
when present; otherwise an exact normalized title and year are used. Undated
records without a DOI remain profile-specific to avoid unsafe merges.

The table retains field variants, contributing observation IDs, profile IDs,
faculty relationships, source URLs, first and last observation timestamps,
match method, observation count, embedding, eligibility decision, exclusion
reasons, included `school_ids`, both shared full-corpus map layouts, and the
assigned topic hierarchy.
For a work with at least one eligible observation, these fields come only from
eligible observations. An entirely excluded work remains with its source
fields and exclusion reasons, but without faculty memberships.
`has_included_source_profile` records whether such a work came from an included
roster profile without claiming that person as an author.
`x` and `y` are the broad-structure PCA coordinates; `tsne_x` and `tsne_y` are
the local-neighborhood t-SNE coordinates after a memory-bounded 50-dimensional
PCA reduction and a final deterministic orientation. `keyword_id`, `keyword`,
`detail_keyword_id`, `detail_keyword`, and `keyword_model_version` record the
deterministic nested visible-region assignment. `keyword_extracted` and
`detail_keyword_extracted` preserve the title-derived phrases behind the public
labels; the corresponding `*_label_reviewed` fields and
`topic_label_review_version` make the editorial pass auditable.
`map_region_audit_version` and `map_region_outlier` retain the result of the
multi-signal catch-all audit.
Excluded works remain in the dataset; `map_eligible` controls browser-map
participation.
Citation count is the maximum retained source observation, not a sum across
faculty profiles.

Canonical map fields use passing observations when a DOI joins passing and
excluded variants of the same work. Excluded variants remain in
`profile_publications`; they cannot change a visible work's title, year,
citation count, source URL, external coauthors, or faculty links. The title-level
policy excludes a work
from layout fitting only when every source observation is excluded; the later
region audit affects the browser artifact without discarding fitted coordinates.

## `authorships`

One relationship between an eligible observation and its registered person.
It carries the work, person, observation and profile-publication IDs plus every
retained department membership and fetch timestamp. Excluded observations
remain in `profile_publications` but do not create authorship rows.

## Browser artifacts

`maps/publications.json` uses schema version 7 with additive school metadata.
It is the only browser artifact.
Every point represents one work and contains both precomputed full-corpus
coordinate pairs, work ID, title, author text, stable faculty and department ID
arrays, derived school ID array, year, venue, citation count, DOI, first
available source URL, and source
observation count, plus one `keyword_ids` entry per hierarchy level. The
top-level `layouts` catalog gives
each view's label, method, interpretation, coordinate fields, and version;
`default_layout_id` selects the initial view. The `keywords` catalog provides a
concise label, extracted phrase, review status, hierarchy level, optional
parent, publication count, and centroid in every layout for each topic region.
`keyword_levels` describes the ordered overview and detail levels.
`quality_assessment_version` identifies the policy, `keyword_model_version`
identifies the extraction and clustering procedure,
`topic_label_review_version` identifies the manual representative-title audit,
and `region_audit_version` identifies the multi-signal catch-all audit. The
`excluded_work_count` reports how many faculty-linked works were retained in
the dataset but left out of the visualization.

The embedded school catalog provides titles, directory sources, and publication
counts. The department catalog includes each parent school ID, title, directory
source, annual review date, and publication count. The faculty catalog includes every person
with an included faculty, teaching, or emeritus membership—even when that
person has no verified Scholar profile or mapped publications—and provides
stable identity, public identifiers, memberships, roles, and publication
counts, with school IDs derived from included memberships. The complete
loss-aware records remain in the four dataset configs.

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
k-means and label each region from its publication titles using repeated
two-to-four-word phrases weighted by within-region coverage and corpus
specificity. Strongly supported longer phrases are preferred over clipped
forms. A versioned, exact-match editorial catalog then supplies the public
label. Both phrases and whether the reviewed wording was applied remain in the dataset
and browser artifact; a changed, unmatched phrase is published as-is and marked
unreviewed rather than being assigned a guess from an older cluster. The
same rule applies when a reviewed label collides with another region's direct
extraction: the direct phrase keeps the label, and the displaced reviewed
region displays its extracted phrase as unreviewed. The `*_label_reviewed`
field is true only when the reviewed wording was actually applied. The
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

`data/curated/scholar-exclusions.json` adds source-backed decisions for exact
`(scholar_id, author_pub_id)` pairs. It covers reviewed nonpublications, false
profile attributions, and corrupted duplicates with an HTTPS evidence URL and
verification date for each decision. The trusted publisher loads this file
before embedding and upload, writes its checksum in the upload receipt, and
applies the decision only to the named observation. A later valid observation
of the same work may still appear on the map, with attribution only to the
eligible source rows.

No single missing bibliographic field, citation count, or embedding position is
an exclusion rule. After the title-level policy, a provisional detail hierarchy
is used to audit large catch-all regions. A region is withheld only when four
independent thresholds all fail: embedding coherence, year completeness,
citation presence, and title length. This conservative conjunction prevents any
single incomplete signal from becoming a broad filter. Affected works remain in
`works` with `low_information_region`; only their browser-map participation is
disabled.
