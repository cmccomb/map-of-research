# Dataset and artifact schema

The Hugging Face repository contains four schema-v3 configurations. Together
they retain the collection grain while exposing a cleaner work-centric model.

## `people`

One row per stable registry identity, including people without a Scholar ID.
Fields include `person_id`, display name, Scholar ID, ORCID, homepage, notes,
all retained memberships, included map slugs, observation count, and unique
work count.

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
match method, observation count, embedding, and the shared global map
coordinates. Citation count is the maximum retained source observation, not a
sum across faculty profiles.

## `authorships`

One reversible relationship between a canonical work and the registered person
whose Scholar profile supplied the observation. It carries the work, person,
observation and profile-publication IDs plus every retained map membership and
fetch timestamp.

## Browser artifacts

`maps/<map-slug>.json` uses schema version 2. Every point represents one work
and contains the shared global coordinates, work ID, title, author text,
associated CMU faculty array, filter group array, year, venue, citation count,
DOI, first available source URL, and source observation count.

Artifacts report both generation time and the oldest/newest underlying profile
refresh timestamps. Sites display source freshness rather than implying that
an artifact upload refreshed every Scholar profile.
