# Verified work records

`works.json` holds 27 faculty observations across 17 verified DOI works and
five source-native works. These include the Tetali–Erdős papers,
Mathematics–Engineering collaborations, Donahue's historical conference
records, and Sullivan's UC San Diego dissertation. Each row names a CMU
faculty person, its DOI or stable source-native identifier, verification URLs
and date, and any external coauthor IDs. Paul Erdős is an external coauthor,
not a CMU faculty registry member.

The snapshot builder validates this file, records its checksum in the manifest,
and retains `curated_verified` separately from `scholar` observations. Dataset
normalization merges observations with the same DOI. It also merges a later
Scholar observation lacking a DOI only when its normalized title and year
match exactly one curated DOI. A conflicting DOI remains a separate work.

For a verified work with no DOI, use a stable `source_native_id` in place of
`doi`, such as an abstract or meeting-program identifier. The ID is unique per
source work, shared across faculty observations of that work, and requires an
HTTPS source URL included in `source_verification_urls`. The snapshot stores
this identifier in `source_native_id`; the normalized work and map artifact
expose `source_native_ids`. A later DOI-free Scholar observation may match a
single curated native ID by exact normalized title and year. A later verified
DOI requires updating the curated row to its DOI before DOI reconciliation.
Curated records have no current Scholar ID or Scholar fetch time.

`scholar-doi-aliases.json` records exact, reviewed version links when a Scholar
row describes a preprint and the publisher gives the same article a later DOI
and year. The publisher checks the row ID, observed title and year, source DOI,
and exclusion state before applying the alias. It preserves the raw Scholar
observation and joins any later curated DOI observation to one work. The
[Walkington review](../../registry/reviews/2026-10-25-math-walkington-version-alias.md)
documents the first link and its arXiv and publisher evidence.

The publisher computes embeddings and map layouts through the normal path for
any newly eligible works. The map artifact exposes `source_types`,
`source_verified_at`, `source_verification_urls`, and `external_coauthor_ids` on
points, plus `catalogs.external_coauthors`. The artifact's Scholar refresh date
uses only real Scholar fetches.

When a verified Google Scholar profile exists, its visible profile rows are the
primary source for that person's publications. Curated DOI observations fill
specific gaps and remain labeled `curated_verified`; an official CMU page
verifies identity or an individual DOI record, not Scholar completeness.

## Reviewed Scholar observations

`scholar-exclusions.json` records exact Scholar profile-publication IDs whose
captured rows were checked against the linked evidence. Each decision names
the profile, row, reason code, specific reason, evidence URL, and date. A review
does not edit the profile cache or raw snapshot. The publisher retains these
rows in `profile_publications` with the decision fields, but excludes them from
`authorships`, mapped work metadata, and faculty memberships. If another
eligible observation matches the same work, that observation can still map the
work. Add a row only after checking the particular observation and cited
evidence; uncertain rows stay unexcluded until reviewed.

Walkington's Scholar row `CGSMDGwAAAAJ:8AbLer7MMksC` is a reviewed
`corrupted_duplicate` of the verified DOI `10.1090/qam/1598`. Its full raw row
stays in `profile_publications`. The DOI observation supplies the mapped
authors, venue, citation, and faculty authorship. Recheck this exclusion if
Scholar repairs those displayed fields in a later capture.
