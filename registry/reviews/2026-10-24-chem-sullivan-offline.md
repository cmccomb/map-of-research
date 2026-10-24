# Ryan Sullivan: offline Scholar identity and legacy-cache audit (2026-10-24)

This review used CMU pages, the existing 196-row cache, and original publisher,
university, and government records. It made **no new Google Scholar request**.
The raw cache is unchanged. Exact exclusion decisions below are proposed only
on this isolated branch; they are not a replacement for a complete current
Scholar capture or authorization to publish the source snapshot.

## Profile identity and coverage

| Evidence | What it establishes |
| --- | --- |
| [CMU Chemistry profile](https://www.cmu.edu/chemistry/people/faculty/sullivan.html) | Ryan Sullivan is professor of **Chemistry and Mechanical Engineering**, Chemistry graduate program chair, and courtesy professor of Civil and Environmental Engineering. It links Scholar `8EppB1cAAAAJ` and ORCID [`0000-0003-0701-7158`](https://orcid.org/0000-0003-0701-7158). |
| [CMU Mechanical Engineering profile](https://www.meche.engineering.cmu.edu/directory/bios/sullivan-ryan.html) | The same CMU professor is in Chemistry and Mechanical Engineering, with a CEE courtesy appointment. It links Scholar `wXSgFr4AAAAJ`. |
| [Earth System Science Data author record](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025.html) | A **different** Ryan C. Sullivan works at Argonne (`rcsullivan@anl.gov`) and has ORCID [`0000-0002-3500-1842`](https://orcid.org/0000-0002-3500-1842). The distinct ORCIDs disambiguate the identical name and middle initial. |
| [UIC Psychiatry](https://www.psych.uic.edu/event/psychdept-importance-of-neuropsychology-in-psychiatric-disorders-20240410) | Ryan **M.** Sullivan works on cannabis and neuropsychology, separately from CMU's atmospheric chemist. |

The registry currently assigns `wXSgFr4AAAAJ`. Its pending legacy cache has
`publication_count: 196`, `fetched_at_utc: 2025-03-28T12:50:46+00:00`, and
`migration_source: map-of-mech/data/Sullivan.json`. It has no headed-browser
receipt, final Show More state, or error check. The old review recorded that
the alternate Chemistry-linked ID returned HTTP 404 **then**; this audit did
not retest it. The 196 rows therefore cannot be certified as the full visible
profile. A future capture must reconcile both CMU-linked IDs and reach the
visible end under the shared Scholar collection gate.

Both departments are official CMU appointments. The evidence above does not
settle which should be displayed as Sullivan's **primary** department; that
choice stays open for the separate affiliation review.

## Exact, source-backed exclusion decisions

The IDs below are keys in the 196-row legacy cache. The reason for each is
false attribution to **CMU** Ryan Sullivan. Some Argonne rows are datasets or
technical reports that may be valid outputs for the Argonne researcher; their
format is not the reason for removing the CMU authorship link. The cache keeps
all raw rows.

| `author_pub_id` suffix | Cached record | Primary evidence and decision |
| --- | --- | --- |
| `P7Ujq4OLJYoC` | Connectome signatures of depression/cannabis | [UIC Psychiatry newsletter](https://www.psych.uic.edu/sites/default/files/2024-07/July_2024_Newsletter_0.pdf) gives the exact presentation to Ryan M. Sullivan. |
| `kh2fBNsKQNwC` | Generally Applicable Law and the Free Exercise of Religion | [Nebraska Law Review](https://digitalcommons.unl.edu/nlr/vol95/iss1/2/) names Douglas Laycock and Steven Collis, with no Sullivan author. |
| `_Re3VWB3Y0AC` | `30ebbr` | [ESSD assets](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025-assets.html) give the exact ARM data-stream record to R. C. Sullivan et al.; the [article author identity](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025.html) is Argonne. |
| `IRz6iEL74y4C` | `30ecor (a1)` | The same [ESSD assets](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025-assets.html) give this exact data-stream record to the Argonne author group. |
| `anf4URPfarAC` | AmeriFlux US-A03 ARM-AMF3-Oliktok | [AmeriFlux's US-A03 record](https://ameriflux.lbl.gov/sites/siteinfo/US-A03) names its PI and dataset author Ryan Sullivan at Argonne, with `rcsullivan@anl.gov`. |
| `0izLItjtcgwC` | AmeriFlux US-A10 ARM-NSA-Barrow | The [ESSD primary paper](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025.html) cites this exact AmeriFlux dataset and identifies its Sullivan author as Argonne. |
| `mNrWkgRL2YcC` | ARM ECORSF instrument record | [ESSD assets](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025-assets.html) list the exact ECORSF record by the Argonne author group. |
| `AHdEip9mkN0C` | Endocannabinoid serum concentrations | [UIC REWARD Lab](https://www.psych.uic.edu/research/lab-directory/reward-lab/in-the-news/dr-natania-crane-emily-jones-ryan-sullivan-rebecca-denson-and-the-reward-lab-presented-talks-and-posters-at-the-86th-annual-meeting-of-the-college-on-problems-of-drug-dependence-conference) gives the exact presentation to its R. M. Sullivan. |
| `sJsF-0ZLhtgC` | Energy Balance Bowen Ratio handbook | The [DOE ARM handbook](https://www.arm.gov/publications/tech_reports/handbooks/ebbr_handbook.pdf) names D. R. Cook and R. C. Sullivan, **both at Argonne**. Its revisions have different dates; the cached `2005` year is not used to infer identity. |
| `HeT0ZceujKMC`, `yMeIxYmEMEAC` | Two FLUXNET-CH4 title variants | The [published ESSD article](https://essd.copernicus.org/articles/13/3607/2021/index.html) identifies coauthor Ryan C. Sullivan with Argonne and ORCID `0000-0002-3500-1842`. Both cached rows denote this paper. |
| `5MTHONV0fEkC` | Grassland flux intercomparison | [Scientific Reports](https://www.nature.com/articles/s41598-024-67911-z) gives the exact title and R. C. Sullivan's Argonne affiliation. |
| `6ZxmRoH8BuwC` | Permafrost landscape history | The [University of Nebraska copy of the paper](https://digitalcommons.unl.edu/biosysengfacpub/813/) lists coauthor Ryan C. Sullivan at Argonne. |
| `cWzG1nlazyYC` | `sebs (b1)` | [ESSD assets](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025-assets.html) list the exact ARM data-stream record by the Argonne author group. |
| `NyGDZy8z5eUC` | QCECOR technical report | The [DOE ARM report](https://www.arm.gov/publications/tech_reports/doe-sc-arm-tr-223.pdf) gives R. C. Sullivan on its updated author list; the [ESSD article](https://essd.copernicus.org/articles/17/5007/2025/essd-17-5007-2025.html) cites the original 2019 report under Tang et al. and the updated 2024 version under Tao et al. This is the ARM/Argonne report, not a CMU chemistry paper. |
| `HtS1dXgVpQUC` | UpCH4 wetland methane article | The [original AGU paper hosted by USDA](https://www.fs.usda.gov/nrs/pubs/jrnl/2023/nrs_2023_mcnicol_001.pdf) assigns its Ryan C. Sullivan author to Argonne. |

That is **16** exact observations: two UIC, one law-review false author,
and 13 Argonne-associated records. The reviewed-exclusion manifest uses
`false_profile_attribution` for each one.

## Held for further review

- `tzM49s52ZIMC` (`5ebbr`), `2KloaMYe4IUC` (Eddy Correlation Flux
  Measurement System), `4X0JR2_MtJMC` (Soil Temperature and Moisture
  Profiles), `uDGL6kOW6j0C` (STAMP precipitation), and `tH6gc1N1XXoC`
  (AmeriFlux supplement) look like data-stream or supplemental records. Their
  exact source authorship and the map's treatment of citable datasets need
  reconciliation before exclusion.
- `OcBU2YAGkTUC` and `OR75R8vi5nAC` are two title variants of an
  [ORNL DAAC dataset](https://daac.ornl.gov/CMS/guides/Global_Freshwater_CH4Emissions.html).
  Its dataset type is established, but that alone does not settle the CMU
  authorship decision for this Google Scholar-based corpus.
- `gVv57TyPmFsC` concerns a Beverton-Holt equation. A separate
  [URI mathematician named Ryan Sullivan](https://digitalcommons.uri.edu/oa_diss/1676/)
  works on that subject, but an exact primary record for this cached title was
  not found. `0CzhzZyukY4C` and `BJbdYPG6LGMC` also need exact author
  evidence before being assigned to the Argonne namesake.
- `YOwf2qJgpHMC`, `bFI3QPDXJZMC`, `pyW8ca7W8N0C`, and
  `F9fV5C73w3QC` are apparent imported author text, title fragments, or
  journal boilerplate. They were not excluded here because the exact raw
  citation needs a stronger source-level reconciliation.

Next capture: use the shared Scholar lane guard, open only the assigned CMU
linked profile in a headed browser, expand to disabled Show More, check for
visible errors, and preserve raw HTML/JSON and a receipt. A successful capture
can supersede the legacy cache while these exact-ID decisions are reviewed
against the new source rows.
