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
catalog includes Engineering and SCS. The initial SCS pilot contains four
verified S3D core faculty and two CSD research and tenure faculty. These are
selected pilot records, not a complete SCS roster; only cached Scholar profiles
contribute publication observations.

Roles `faculty`, `teaching`, and `emeritus` must be included. Affiliate,
courtesy, visiting, former, special, adjunct, and unknown roles must be retained
but excluded. Validation rejects mismatched role and inclusion values.

Rows without a Scholar ID remain visible in the `people` dataset but are not
collected. A Scholar ID may belong to only one person. Cross-appointments are
represented by multiple memberships for the same stable `person_id` and are
collected once.

The registry retains every legacy label in `legacy_label` for provenance.
The first full-name and role review against all 11 official directories was
completed on 2026-07-17. Unresolved identities and Scholar IDs remain blank
rather than being guessed.

New Scholar IDs should record the faculty-controlled, institutional, or
self-managed public page that links the profile in `scholar_id_source_url`,
plus the human verification date in `scholar_id_verified_at`. Legacy IDs may
leave those fields blank until they are reverified.
