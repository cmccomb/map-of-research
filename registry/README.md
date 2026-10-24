# Canonical faculty registry

The registry is intentionally normalized:

- `people.csv` owns stable person IDs, full display names, optional public
  identifiers, Scholar-ID verification sources and dates, homepages, and
  durable notes.
- `memberships.csv` owns department roles, inclusion state, legacy labels, the
  official source URL, and the date of human verification.
- `departments.csv` owns stable department IDs, their parent `school_id`, titles,
  official directory URLs, annual review dates, and the documented review policy.
- `schools.csv` owns stable school IDs, titles, and official directory URLs.

Every department must reference a school in `schools.csv`. School membership is
derived from a person's included department memberships; a cross-appointed
person keeps one `person_id` and can belong to multiple schools. The school
catalog includes Engineering, SCS, and MCS. A dated review of the official
faculty listings on 2026-10-24 added all seven SCS departments and all four
MCS departments. The reviewed pages supplied 357 SCS and 270 MCS department
observations; 346 and 212 memberships, respectively, meet the existing
faculty, teaching, or emeritus inclusion policy. After identity reconciliation,
the registry has 317 included SCS people and 211 included MCS people. The
dated source extracts and coverage limits are in
[`reviews/2026-10-24.md`](reviews/2026-10-24.md).

Faculty registration does not imply publication coverage. The source pages
directly linked 93 SCS and 44 MCS Scholar IDs. In the first CMU-wide snapshot,
only 8 included SCS and 3 included MCS people had a Scholar cache. A later
bounded local batch captured Prasad Tetali and Dejan Slepčev; its
[Math source-graph review](reviews/2026-10-24-math-source-graph.md) records
coverage and provenance separately from map publication. Two separately
verified Tetali–Erdős DOI works are retained in the curated source. Paul
Erdős is an external coauthor, not CMU faculty.
The [Bio Scholar-ID follow-up](reviews/2026-10-24-bio-scholar-ids.md) verifies
three additional IDs from named CMU expert pages; it does not add publication
observations.
The [Huaiying Zhang follow-up](reviews/2026-10-24-bio-zhang-scholar-id.md)
records another direct ID link from a CMU-hosted lab publications page.
The [Bob Murphy follow-up](reviews/2026-10-24-bio-murphy-scholar-id.md)
reconciles his Bio and Computational Biology emeritus memberships under one
Scholar-linked person.
The [three Bio profile checks](reviews/2026-10-24-bio-scholar-verified-profiles.md)
record Scholar IDs confirmed directly from named profiles with verified CMU
email addresses and matching official faculty identities.

Roles `faculty`, `teaching`, and `emeritus` must be included. Affiliate,
courtesy, visiting, former, special, adjunct, and unknown roles must be retained
but excluded. Validation rejects mismatched role and inclusion values.

Rows without a Scholar ID remain visible in the `people` dataset but are not
collected. A Scholar ID may belong to only one person. Cross-appointments are
represented by multiple memberships for the same stable `person_id` and are
collected once.

The registry retains every legacy label in `legacy_label` for provenance.
The first full-name and role review against all 11 Engineering directories was
completed on 2026-07-17. SCS and MCS were reviewed on 2026-10-24. Unresolved
identities and Scholar IDs remain blank rather than being guessed.

New Scholar IDs should record the faculty-controlled, institutional, or
self-managed public page that links the profile in `scholar_id_source_url`,
plus the human verification date in `scholar_id_verified_at`. Legacy IDs may
leave those fields blank until they are reverified.
When no such page links an ID, a directly viewed Scholar profile can be the
source if its verified CMU email, name, and affiliation or linked homepage
corroborate an official faculty listing. Record that identity evidence in a
dated review; a name-only search match is insufficient.
If the owner explicitly confirms an ID without a linking page, record the
supplied Scholar URL and confirmation date, and identify that provenance in
`notes` rather than claiming independent link verification.
