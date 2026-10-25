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
published work, including a matching curated DOI observation when present.
The version-of-record title, year, and venue supply that work's display. See
the [Walkington review](../../registry/reviews/2026-10-25-math-walkington-version-alias.md),
the [Math version-pair review](../../registry/reviews/2026-10-25-math-scholar-version-pairs.md),
the [three further Math version pairs](../../registry/reviews/2026-10-25-math-three-more-version-pairs.md),
the [Eur/Young exact-work review](../../registry/reviews/2026-10-25-math-eur-young-exact-works.md),
and [Brandon/Kramkov exact-work review](../../registry/reviews/2026-10-25-math-brandon-kramkov-exact-works.md)
for the exact source evidence.

`scholar-metadata-corrections.json` repairs only effective display metadata for
an exact Scholar observation; the saved cache and `source_record_json` retain
the observed values. A byline repair must guard the observed authors as well as
title, year, and venue. The [Yilma review](../../registry/reviews/2026-10-25-math-yilma-whole-scholar-profile.md)
documents two publisher-verified five-author repairs.

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

`person-publication-holds.json` records a reviewed person-level visibility hold
when the saved source has only individually verified works and no complete
publication history. It binds the stable person ID, frozen snapshot SHA-256,
exact DOI, `author_pub_id`, and observation ID for every held row. The trusted
publisher refuses changed or unreviewed hold evidence. It keeps those rows in
the frozen raw snapshot while omitting that person's contributions from public
tables and maps. Other faculty observations of the same DOI remain public. The
public `people` table and map faculty catalog state “Publication history
pending full capture” for the held person.

## Reviewed Scholar observations

`scholar-profile-releases.json` records a source-bound release decision for a
complete Scholar profile whose captured rows are still under attribution review.
The raw source branch and validated snapshot retain every captured row. The
trusted publisher checks the exact cache checksum, registry identity, row
count, and citation IDs before applying the decision. A `held` profile supplies
no own-source public rows; Chien Ho's `released` 582-row profile uses a separate
positive-attribution gate, so only exact confirmed rows map. The Collins
`released` decision in this draft is conditional on merging his complete source
and exact review. The upload receipt records the release-manifest checksum, raw
and published counts, and each decision. Deleting or altering the saved source
fails validation.

`scholar-provisional-holds.json` records exact rows whose identity or output
type remains unresolved within a complete released Scholar capture. It is
separate from the exact-exclusion manifest. Each profile pins its person, Scholar
ID, source row count, and cache SHA-256. Each hold pins a capture row index,
`author_pub_id`, source citation URL, reason, and date. The hold profile also
pins the count and sorted-ID SHA-256 of reviewed exclusions that must be present
for that source. The release decision pins the hold count and sorted-ID
SHA-256. Before any upload, the publisher checks this contract, preserves the
held observations in `profile_publications`,
and rejects a held ID in any mapped work or authorship. The initial draft
[Collins review](../../registry/reviews/2026-10-25-chem-collins-provisional-holds.md)
names seven held rows from his 361-row visible capture. Remaining rows still
pass the ordinary quality and exact-exclusion checks; the hold does not certify
them individually.

The [Murshid, Faulkner, and Silva release review](../../registry/reviews/2026-10-25-chem-murshid-faulkner-silva-publisher-review.md)
pins three more complete visible Chemistry Scholar profiles (49, 5, and 41
rows). It identifies four degraded duplicate citations for exact exclusion
and four unresolved individual rows for provisional hold. The three producer
registry identities and primary Chemistry departments match the source draft
because the trusted uploader reads the registry from producer main.

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
