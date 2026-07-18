# Dataset and artifact schema

The Hugging Face repository contains four schema-v4 configurations. Together
they retain the collection grain while exposing a cleaner work-centric model.

## `people`

One row per stable registry identity, including people without a Scholar ID.
Fields include `person_id`, display name, Scholar ID, Scholar-ID verification
source and date, ORCID, homepage, notes, all retained memberships, included
department IDs, observation count, and unique work count.

## `profile_publications`

One row per publication observation on one Scholar profile. This is the
lossless analytical source. It retains the original profile-publication ID,
parsed bibliographic fields, citation count at collection time, complete
normalized source record as JSON, fetched timestamp, memberships, work match
method, and title embedding.

## `works`

One canonical work derived conservatively from the observations. DOI is used
when present; otherwise an exact normalized title and year are used. Undated
records without a DOI remain profile-specific to avoid unsafe merges.

The table retains field variants, all source observation IDs, profile IDs,
faculty relationships, source URLs, first and last observation timestamps,
match method, observation count, embedding, and both shared full-corpus map
layouts. `x` and `y` are the broad-structure PCA coordinates; `tsne_x` and
`tsne_y` are the local-neighborhood t-SNE coordinates, oriented using the same
final PCA rotation as the previous full map. Citation count is the maximum
retained source observation, not a sum across faculty profiles.

## `authorships`

One reversible relationship between a canonical work and the registered person
whose Scholar profile supplied the observation. It carries the work, person,
observation and profile-publication IDs plus every retained department
membership and fetch timestamp.

## Browser artifacts

`maps/publications.json` uses schema version 4. It is the only browser artifact.
Every point represents one work and contains both precomputed full-corpus
coordinate pairs, work ID, title, author text, stable faculty and department ID
arrays, year, venue, citation count, DOI, first available source URL, and source
observation count. The top-level `layouts` catalog gives each view's label,
method, interpretation, coordinate fields, and version; `default_layout_id`
selects the initial view.

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
canonical work rows, title embeddings, and filters. Compatible coordinates are
reused when the corpus is unchanged so routine republishes do not move points.
