# Ryan C. Sullivan: source-bound release gate (2026-10-25)

The complete visible Google Scholar capture contains 250 unique rows. The
candidate cache SHA-256 is
`960b296af5cf0c9e3aab6be32844ec2fe2ad7f2a2b3e331cfc4c857f4df57fd7`.
Source draft #244 preserves the raw HTML, JSON, capture receipt, and row-level
review. A separate source child draft stages this exact cache as canonical;
neither source draft changes a published snapshot.

This producer draft applies 56 exact reviewed exclusions: the 51 decisions in
Sullivan draft #217 (35 were absent from the Collins-hold base branch), plus
five newly verified rows. It places 35 other current rows in provisional hold,
pinned by exact Scholar ID, original capture index, source URL, cache hash,
and the exact reviewed exclusion set. The 56 exclusion IDs have digest
`bc82650651779bda8ea25c141d14af00633ec8795ffd3e2f5c2aef956a0a0e21`;
the 35 held IDs have digest
`78fe11269cdda8e270e1c4ec98e1cb24ddec53b5c0a942e79acff00cc9f677ae`.
Holds preserve raw observations but prevent those observations from making
mapped works, authorships, or coauthor links. Against all 250 distinct capture
IDs at indices 1 through 250, the 56 exclusions and 35 holds are disjoint
and all 91 IDs occur in the capture. The remaining 159 rows enter the existing
quality gate; they are not represented as individually confirmed works.

Four IDs from the frozen 196-row cache are absent from the 250-row view. The
exact crosswalk binds the frozen snapshot and cache hashes, the replacement
cache hash, each old ID, and a source-backed destination:

| Old ID suffix | Disposition | Destination |
| --- | --- | --- |
| `Ehil0879vHcC` | Verified 2017 AGU abstract | Curated native `agu2017:a13j-05` |
| `OR75R8vi5nAC` | Argonne namesake dataset variant | Excluded current `XUvXOeBm_78C` |
| `OcBU2YAGkTUC` | Same Argonne dataset variant | Excluded current `XUvXOeBm_78C` |
| `PVgj2kMGcgYC` | Same EGU25 abstract | Current `j7_hQOaDUrUC` |

Five additional rows were promoted from provisional holds to exact exclusions
after primary-source review: row 99 is assigned to the Argonne Ryan C. Sullivan
in [Crossref-deposited IEEE metadata](https://api.crossref.org/works/10.1109/LGRS.2025.3546797);
row 181 is credited to Ryan Sullivan of NSF in the [AMS program](https://ams.confex.com/ams/98Annual/webprogram/meeting.html);
row 194 is an [AGU session heading](https://studio.m-anage.com/agu/fm15/webprogram/Session10991.html)
with Sullivan's authored abstract listed separately; and row 229's R. Sullivan
is assigned to Cornell Astronomy in the [publisher article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JE003102).
Row 231's exact [Lunar and Planetary Science Conference abstract](https://www.lpi.usra.edu/meetings/lpsc2007/pdf/2235.pdf)
also assigns R. Sullivan to Cornell Astronomy.

The two namesake rekeys are allowed only when the exact current target has a
`false_profile_attribution` exclusion with the same primary evidence URL. The
curated AGU destination must exist under the same person and source URL. The
EGU destination must remain unexcluded with matching title and year. No old
row is silently dropped or inserted into the current Scholar capture.

Local preflight against the saved source passed the complete receipt validator,
the exact four-ID retention gate, and the release/hold gate: 250 current Scholar
rows, four reconciled old IDs, 56 exclusions, and 35 provisional holds. The
pre-upload relationship build kept all 250 Sullivan observations in the raw
table, marked 56 excluded and 35 held, and left 159 map-eligible before
downstream quality and display filters. The provisional-hold contribution gate
found no held authorship or map-eligible work. This local preview used empty
embedding vectors and did not upload data.

This producer draft is based on `c6009698`, after Bernhard's complete profile
was published and verified on both live maps. The Bernhard baseline has 1,235
exact exclusions; this draft retains them and adds 40 novel Sullivan decisions,
for 1,275 total. It retains the 12 existing source-bound holds, including
Bernhard's one unresolved row, and adds Sullivan's 35, for 47 total. A prior
combined no-upload preview predicted 51,823 to 51,879 raw observations after
Sullivan's 196 older cache rows are replaced with 250 current rows and two
curated native works are preserved. The source drafts #244 and #252 still need
to be restacked onto the Bernhard upload receipt before a new exact-head
snapshot and relationship build can verify these counts. That snapshot must
pass the complete-source, four-ID retention, release, hold, and Math coverage
gates before the protected upload and both live-map checks.
