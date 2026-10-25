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
catalog includes Engineering, SCS, MCS, CFA, Dietrich, Heinz, and Tepper. It
contains 7 top-level schools, 40 academic units or documented roster
containers, 2,001 people, and 2,355 memberships after the four-school
2026-10-24 expansion. A dated review of the official
faculty listings on 2026-10-24 added all seven SCS departments and all four
MCS departments. The reviewed pages supplied 357 SCS and 270 MCS department
observations; 346 and 212 memberships, respectively, meet the existing
faculty, teaching, or emeritus inclusion policy. After identity reconciliation,
the registry has 317 included SCS people and 211 included MCS people. The
dated source extracts and coverage limits are in
[`reviews/2026-10-24.md`](reviews/2026-10-24.md).

The four added schools were reviewed against their public faculty listings on
2026-10-24. Source observations include repeat cards and source-only staff or
students; registered memberships apply the faculty-role policy below.

| School | Units or roster containers | Source observations | Memberships | Included people | Dated review |
| --- | ---: | ---: | ---: | ---: | --- |
| College of Fine Arts | 5 | 425 | 367 | 256 | [CFA](reviews/2026-10-24-cfa.md) |
| Dietrich College | 9 | 472 | 471 | 270 | [Dietrich](reviews/2026-10-24-dietrich.md) |
| Heinz College | 3 | 309 | 282 | 83 | [Heinz](reviews/2026-10-24-heinz.md) |
| Tepper School of Business | 1 | 138 | 138 | 118 | [Tepper](reviews/2026-10-24-tepper.md) |

The five CFA rows represent its actual schools. Dietrich has seven departments
and two faculty-owning institutes. Heinz has two actual graduate schools plus
one shared college faculty container because its main directory does not
assign every person to a school. Tepper has one school-level container because
its 12 academic areas lack separate faculty rosters. The containers satisfy
the current registry schema; they are not asserted to be academic departments.
The linked reviews and source CSVs document role decisions, identities,
directly linked Scholar IDs, and unresolved directory gaps.

The primary academic home is a single **display affiliation**, not a
replacement for memberships or a claim about university HR records. Published
`primary_academic_home_type`, `_id`, `_title`, `_school_id`, `_source_url`,
`_verified_at`, and `_basis` fields distinguish departments, constituent
schools, institutes, academic areas, broader schools, and unresolved cases.
The legacy `primary_department_id` is a compatibility field only for an actual
department home; its basis reads `unresolved` for other resolved home types,
which consumers distinguish through the typed fields. A person with one
included academic unit can inherit that
verified unit; a school-level roster container is not thereby a verified
department. CMU-Africa is an academic department under Engineering for this
dataset; CMU Silicon Valley is a campus facet and cannot be a primary.
Excluded courtesy, adjunct, affiliate, and former roles cannot supply a
primary on their own. Multiple eligible units require a source-backed choice,
a clearly labeled advising-count or title inference, or an explicitly reviewed
visual tie-break. `display_tie_break_first_listed` selects the first department
in an official dual-department title only for the map display. It does not
rank university appointments or remove either membership. Other cases where
the evidence does not distinguish one remain unresolved. No alphabetical
tie-break is used.

Tepper's official faculty listings support one academic area for 23 of the 28
reviewed Scholar-ID holders. Five are listed in two areas; their explicit
Tepper school-level fallback is a broad, evidence-limited home, not a verified
department or a claim that either area leads. Heinz has constituent schools
rather than academic departments. Pedro Ferreira's School of Information
Systems and Management choice is an inference: his CMU-hosted self-profile
identifies ISM and EPP, while named PhD advisees favor Heinz over EPP 10 to 9.
All of these people's included memberships remain available to filters.

The [primary academic-home review ledger](reviews/2026-10-24-primary-departments.md)
records all 190 people needing explicit review. Across 2,001 registry
people, 1,447 included people have a display primary and 554 remain unresolved,
including all 521 excluded-only people. Of 1,480 included people, 33 remain
unresolved. Of 459 included people with a Scholar ID, all have a primary. The
[four-school review](reviews/2026-10-24-other-schools-primary.md) and
[Physics review](reviews/2026-10-24-physics-audit.md) supply later evidence.
Avigad, Sekerka, Feather, and Kamlet have explicit first-listed visual
tie-breaks; their joint appointments remain included. Publication and school
filters continue to use all included memberships.

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
completed on 2026-07-17. SCS, MCS, CFA, Dietrich, Heinz, and Tepper were
reviewed on 2026-10-24. Unresolved
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
