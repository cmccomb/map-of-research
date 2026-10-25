# Sullivan attribution and retention review — 2026-10-25

## Scope and provenance

Ryan Christopher Sullivan is registered as `person-sullivan-589c9545` with Google Scholar ID `wXSgFr4AAAAJ`. The frozen canonical cache contains 196 observations (SHA-256 `08a8a47169d020493435e7973d41781feabc11bccc2b1d22fe16c3bd66ba5996`). Source draft [#189](https://github.com/cmccomb/map-of-research/pull/189) preserves a complete 250-row visible profile capture (JSON SHA-256 `56c1f2125a674fb5261bfd0ca4a828a34dd75d08c7fdea70abc1e892498773cb`), including its final disabled Show More state. The two ID sets have 192 rows in common, four old-only IDs, and 58 new-only IDs. The exact sets and row metadata are recorded in the accompanying JSON audits.

This producer draft depends on [#123](https://github.com/cmccomb/map-of-research/pull/123), which adds the narrowly scoped `source_native_id` path for verified DOI-free curated works. It proposes two exact current-row exclusions and one source-native dissertation recovery. It does not stage a replacement canonical cache or a new snapshot. The 250-row profile has multiple namesake and quality signals; a complete visible Scholar capture alone does not establish a correct whole-person publication history.

## Exact source-backed corrections

| Scholar row | Displayed observation | Reviewed disposition |
| --- | --- | --- |
| `wXSgFr4AAAAJ:L8Ckcad2t8MC` (current row 112) | *Garden Between*, A. T. Sullivan, Pacific Lutheran University, 2008 | Exact `false_profile_attribution` exclusion. [UC San Diego eScholarship](https://escholarship.org/uc/item/89c072pv) identifies CMU Sullivan's different 2008 dissertation, *Heterogeneous chemistry of atmospheric mineral dust particles and their resulting cloud-nucleation properties*, by Ryan Christopher Sullivan. [Prather Lab alumni](https://atofms.ucsd.edu/lab-members/alumni) links its 2008 Ryan Sullivan graduate to Carnegie Mellon. Recover the real dissertation as `escholarship:qt89c072pv`. |
| `wXSgFr4AAAAJ:XUvXOeBm_78C` (current row 101) | *Global Wetland Methane Emissions derived from FLUXNET and the UpCH4 Model, 2001-2018*, displayed as 2024 | Exact `false_profile_attribution` exclusion. The [ORNL DAAC dataset citation](https://daac.ornl.gov/CMS/guides/Global_Freshwater_CH4Emissions.html) gives DOI `10.3334/ORNLDAAC/2253` and 2023. The linked [AGU Advances article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023AV000956) identifies its Ryan C. Sullivan contributor at Argonne National Laboratory and links the same dataset DOI. |

The old canonical cache uses `L8Ckcad2t8MC` for the genuine UC San Diego dissertation title; the same Scholar ID now displays an unrelated namesake work. The curated record and exact exclusion must be coordinated with the full cache review so the old cached dissertation is neither lost nor represented twice. The old-only audit below is a separate issue.

## Four prior IDs absent from the current capture

The [merged retention gate #185](https://github.com/cmccomb/map-of-research/pull/185) requires exact dispositions for disappeared prior IDs. `2026-10-25-chem-sullivan-prior-id-audit.json` pins the frozen baseline, prior row hashes, source capture hash, target IDs and metadata where known. It is a preparatory audit, **not** a gate exception. No candidate cache hash exists yet.

| Old-only ID suffix | Current evidence | Release status |
| --- | --- | --- |
| `Ehil0879vHcC` | Science.gov indexes a NASA ADS abstract for a 2017 AGU presentation with this exact N2O5/biomass-burning title and Sullivan/Jahl/Goldberger/Ahern/Thornton. A direct primary AGU or ADS record has not been recovered. The 2019 journal article is a distinct output. | **Unresolved.** Retain the hold pending a primary source-native record or another exact, reviewable disposition. |
| `OR75R8vi5nAC` and `OcBU2YAGkTUC` | Both old records cite the same [ORNL DAAC dataset](https://daac.ornl.gov/CMS/guides/Global_Freshwater_CH4Emissions.html), DOI `10.3334/ORNLDAAC/2253`; the current profile uses `XUvXOeBm_78C`. The [AGU publisher record](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023AV000956) ties its Ryan C. Sullivan to Argonne. | Proposed `false_profile_attribution_rekeyed` crosswalk, contingent on a narrow gate rule requiring the target current row's exact reviewed exclusion and evidence. |
| `PVgj2kMGcgYC` | Current `j7_hQOaDUrUC` points to the same [EGU25-7227 abstract](https://meetingorganizer.copernicus.org/EGU25/EGU25-7227.html), DOI `10.5194/egusphere-egu25-7227`; the conference page assigns Ryan Sullivan to Carnegie Mellon. | Source-backed `duplicate_rekeyed` candidate, pending the complete exact crosswalk and cache hash. |

The Donahue crosswalk in [#188](https://github.com/cmccomb/map-of-research/pull/188) supplies the exact-set, frozen-hash, prior-row-hash, and target-verification pattern. Sullivan needs an additional narrowly defined false-attribution rekey classification; the existing Donahue rule must not be weakened or treated as a Sullivan exception.

## All 58 new-only rows

`2026-10-25-chem-sullivan-new-only-triage.json` enumerates every new-only ID with its captured index, title, year, authors, venue, candidate source URL, and review state. **All 58 remain held for cache promotion:** row 101 has the exact false-attribution exclusion proposed above, while the other 57 still need per-row disposition. Its signals overlap and are leads for exact attribution review, not evidence that a row belongs to or should be removed from CMU Sullivan's publication history:

- 10 rows have a candidate primary Argonne namesake source; five display Argonne in the venue; 27 have flux, ARM, CROCUS, or carbon-program subject signals.
- Nine display an NSF award venue, two are obvious page fragments, and three display a future year (2027).
- Current row 132, *The CROCUS Measurement Strategy*, has direct [Copernicus conference evidence](https://meetingorganizer.copernicus.org/ICUC12/ICUC12-438.html) naming an Argonne Ryan Sullivan with an `anl.gov` address. Current row 141's venue itself cites the same ORNL dataset DOI as the excluded row 101.
- Current row 160 uses an author list as its Scholar title. The [University of Groningen record](https://research.rug.nl/en/publications/gap-filling-eddy-covariance-methane-fluxes-comparison-of-machine-/) identifies the underlying 2021 article, DOI `10.1016/j.agrformet.2021.108528`. Its exact person attribution and title repair remain for review.
- Current row 123, *Evolution of the photosensitized production of singlet oxygen by aqueous extracts of biomass-burning aerosol*, is displayed as an `Environmental Science: Processes & Impacts` article. Its journal name was initially mistaken for a page fragment; that triage error was corrected. No exclusion is proposed.

## Release hold

Do not promote Sullivan's 250-row cache until the 2017 abstract has a primary-backed disposition, the two ORNL old IDs have a narrow reviewed crosswalk to the exact excluded current row, the remaining 57 new-only rows have been assessed for attribution and record quality, and the frozen-to-candidate retention gate passes. Keep this producer draft separate from the source capture and from Donahue's crosswalk. No `collect.yml` dispatch is part of this review.
