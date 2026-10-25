# Verified DOI work records

`works.json` holds 21 faculty observations for 16 verified DOI works: two
Prasad Tetali–Paul Erdős papers, six Mathematics observations across three
Math–MSE papers, and four observations across three David R. Owen papers (one
shared with Irene Fonseca). Six further observations across five papers link
Noel Walkington to Engineering, Ian Tice to Giovanni Leoni, and David
Kinderlehrer and Robert Suter to Engineering collaborators. One additional
record supplements Clive Newstead's Scholar profile with his 2018 CMU doctoral
thesis, using the DOI of its 2021 arXiv deposit. Another records Walkington's
chapter in place of an incorrectly attributed whole-book Scholar row. A third
supplies correct metadata for a Walkington journal paper whose Scholar row
has malformed author and venue fields. Each row names its CMU faculty person,
exact DOI, publisher or CMU verification URLs, verification date, and any
external coauthor IDs. Paul Erdős is an external coauthor, not a CMU faculty
registry member.

The snapshot builder validates this file, records its checksum in the manifest,
and retains `curated_verified` separately from `scholar` observations. Dataset
normalization merges observations with the same DOI. It also merges a later
Scholar observation lacking a DOI only when its normalized title and year
match exactly one curated DOI. A conflicting DOI remains a separate work.

`scholar-doi-aliases.json` records exact, reviewed links between Scholar rows
and a published DOI. It distinguishes a `preprint_of` row from a
`version_of_record` row when Scholar lists both. The publisher checks each row
ID, observed title and year, source DOI, and exclusion state before applying
the alias. It preserves both raw Scholar observations and joins them to one
published work. See the [Walkington review](../../registry/reviews/2026-10-25-math-walkington-version-alias.md),
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
