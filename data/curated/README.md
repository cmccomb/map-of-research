# Verified work records

`works.json` holds 26 faculty observations for 17 verified DOI works and four
source-identified conference records. These include the Tetali–Erdős papers,
Mathematics–Engineering collaborations, and five Donahue items recovered from
independent sources. Each row names a CMU faculty person, its DOI or stable
source-native identifier, verification URLs and date, and any external
coauthor IDs. Paul Erdős is an external coauthor, not a CMU faculty registry
member.

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

`scholar-doi-aliases.json` records exact, reviewed links between Scholar rows
and a published DOI. It distinguishes a `preprint_of` row from a
`version_of_record` row when Scholar lists both. The publisher checks each row
ID, observed title and year, source DOI, and exclusion state before applying
the alias. It preserves both raw Scholar observations and joins them to one
published work, including a matching curated DOI observation when present. See
the [Walkington review](../../registry/reviews/2026-10-25-math-walkington-version-alias.md),
the [Math version-pair review](../../registry/reviews/2026-10-25-math-scholar-version-pairs.md),
and [three further Math version pairs](../../registry/reviews/2026-10-25-math-three-more-version-pairs.md)
for the exact source evidence.

`scholar-contexts.json` is a separate exact-observation review manifest. It
records publication type and primary evidence for unusual Scholar records.
Patent work identity uses the published document number; the raw Scholar cache
remains intact. An exact context entry can set `map_eligible: false` to keep a
reviewed observation and any matched copies in raw profile data while excluding
the work from the map. Reviewed non-author contributor types do not create
coauthor edges. The trusted upload receipt records this manifest's SHA-256.

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

`scholar-profile-releases.json` records a source-bound release decision for a
complete Scholar profile whose captured rows are still under attribution review.
The current decision holds Chien Ho's 582-row cache. The raw source branch and
validated snapshot retain every row; the trusted publisher checks the exact
cache checksum, registry identity, row count, and citation IDs, then removes
this profile's own observations before embeddings and all four public dataset
tables are built. The faculty roster still lists Chien Ho with zero own-source
publications. A paper independently captured from another faculty profile is
still eligible even if its author list includes him. The upload receipt records
the release-manifest checksum, raw and published counts, and each decision.
Changing `state` to `released` requires a reviewed producer change and a
matching source cache; deleting or altering the saved source fails validation.

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
