# Verified DOI work records

`works.json` holds 19 faculty observations for 14 verified DOI works: two
Prasad Tetali–Paul Erdős papers, six Mathematics observations across three
Math–MSE papers, and four observations across three David R. Owen papers (one
shared with Irene Fonseca). Six further observations across five papers link
Noel Walkington to Engineering, Ian Tice to Giovanni Leoni, and David
Kinderlehrer and Robert Suter to Engineering collaborators. One additional
Walkington chapter replaces an incorrectly attributed whole-book Scholar row.
Each row names its CMU faculty person, exact DOI, publisher or CMU verification
URLs, verification date, and any external coauthor IDs. Paul Erdős is an
external coauthor, not a CMU faculty registry member.

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
