# Verified DOI work records

`works.json` holds two manually verified Prasad Tetali–Paul Erdős papers while
Google Scholar collection is refusing requests. Each row names its CMU faculty
person, exact DOI, publisher or CMU verification URLs, verification date, and
external coauthor IDs. Paul Erdős is an external coauthor, not a CMU faculty
registry member.

The snapshot builder validates this file, records its checksum in the manifest,
and retains `curated_verified` separately from `scholar` observations. Dataset
normalization merges observations with the same DOI. It also merges a later
Scholar observation lacking a DOI only when its normalized title and year
match exactly one curated DOI. A conflicting DOI remains a separate work.

The publisher computes embeddings and map layouts through the normal path for
any newly eligible works. The map artifact exposes `source_types`,
`source_verified_at`, `source_verification_urls`, and `external_coauthor_ids` on
points, plus `catalogs.external_coauthors`. The artifact's Scholar refresh date
uses only real Scholar fetches.
