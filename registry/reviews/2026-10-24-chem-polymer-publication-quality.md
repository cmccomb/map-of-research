# Chemistry complete-profile publication quality review (2026-10-24)

These decisions concern the staged 114-row Sydlik and 112-row Noonan Scholar captures. They do not alter the raw capture. Exact `author_pub_id` values allow a later curator to retain provenance while excluding a nonpublication or duplicate from the collaboration graph.

## Verified exclusions to stage with release

| Scholar owner | `author_pub_id` | Decision | Primary evidence |
| --- | --- | --- | --- |
| Sydlik | `rQ79IvMAAAAJ:4OULZ7Gr8RgC` | Exclude NSF award record | [NSF award 1905665](https://api.nsf.gov/services/v1/awards/1905665.json) matches the title, PI and institution. |
| Sydlik | `rQ79IvMAAAAJ:cFHS6HbyZ2cC` | Exclude NSF award record | [NSF award 2318652](https://api.nsf.gov/services/v1/awards/2318652.json) matches the title, Sydlik as PI and Noonan among co-PIs. |
| Noonan | `wmP8_2AAAAAJ:xtRiw3GOFMkC` | Exclude NSF award record | [NSF award 1455136](https://api.nsf.gov/services/v1/awards/1455136.json) matches title and Noonan as PI. |
| Noonan | `wmP8_2AAAAAJ:NhqRSupF_l8C` | Exclude NSF award record | [NSF award 1809658](https://api.nsf.gov/services/v1/awards/1809658.json) matches title and Noonan as PI. |
| Noonan | `wmP8_2AAAAAJ:CHSYGLWDkRkC` | Exclude NSF award record | [NSF award 2109065](https://api.nsf.gov/services/v1/awards/2109065.json) matches title and Noonan as PI. |
| Noonan | `wmP8_2AAAAAJ:1sJd4Hv_s6UC` | Exclude NSF award record | [NSF award 2404367](https://api.nsf.gov/services/v1/awards/2404367.json) matches title, Noonan as PI, Peteanu and Kowalewski as co-PIs. |
| Noonan | `wmP8_2AAAAAJ:tOudhMTPpwUC` | Exclude NSF award record | Same [NSF award 2318652](https://api.nsf.gov/services/v1/awards/2318652.json), shared with Sydlik. |
| Noonan | `wmP8_2AAAAAJ:zA6iFVUQeVQC` | Exclude dataset as separate publication | [Cornell eCommons record](https://ecommons.cornell.edu/entities/publication/bda05268-8a12-463d-85af-63df2038a7e7) classifies it as dataset and links the associated article. |
| Noonan | `wmP8_2AAAAAJ:XiSMed-E-HIC` | Exclude dataset as separate publication | [Cornell eCommons record](https://ecommons.cornell.edu/entities/publication/e7e1913d-3ee9-4342-80cb-95263484044e) classifies it as dataset and links the associated article. |
| Noonan | `wmP8_2AAAAAJ:l7t_Zn2s7bgC` | Exclude corrupted duplicate | It repeats the complete title and year of `wmP8_2AAAAAJ:WbkHhVStYXYC` but omits author metadata. The [ACS journal page](https://pubs.acs.org/doi/10.1021/jacs.6c11419) confirms the one 2026 article. |
| Noonan | `wmP8_2AAAAAJ:u_35RYKgDlwC` | Exclude corrupted duplicate | Undated “Chemistry—A European” fragment repeats the authors of the complete 2023 row `wmP8_2AAAAAJ:ZHo1McVdvXMC`. The [journal article](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202300477) confirms the title, venue, DOI, Noonan, and Kowalewski. |

## Retain or defer

- **Retain** `wmP8_2AAAAAJ:WbkHhVStYXYC`: the [ACS article page](https://pubs.acs.org/doi/10.1021/jacs.6c11419) explicitly lists Kevin Noonan as a coauthor. The staged Scholar row abbreviates the author list before his name; that is not a false attribution.
- `wmP8_2AAAAAJ:K3LRdlH-MEoC` is an ACS URL string with no author or year, already excluded by the generic quality rule. An explicit row exclusion would be redundant.
- **Retain** Noonan `wmP8_2AAAAAJ:ZHo1McVdvXMC` and Kowalewski `6VKJW4kAAAAJ:zCSUwVk65WsC`, the complete 2023 article rows on each profile. The [journal article](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202300477) lists both authors. The corresponding undated journal-title fragments are excluded by exact ID while the real coauthorship remains.
- **Defer** Sydlik `rQ79IvMAAAAJ:OU6Ihb5iCvQC` (editorial masthead), `:UxriW0iASnsC` (apparently garbled duplicate of the 2012 MIT thesis `:W7OEmFMy1HYC`), and Noonan `wmP8_2AAAAAJ:Tiz5es2fbqcC` (2025 reviewer list), `:KlAtU1dfN6UC` (journal frontmatter). The available primary pages have not yet resolved these exact rows sufficiently for exclusion. The complete 2012 thesis row should remain pending any separate policy decision about theses.

## Tomasz Kowalewski: newly captured 351-row profile

Nine visible Scholar rows are **NSF award records, not research publications**. Exact official award entries confirm the titles and grant investigators:

| `author_pub_id` suffix | NSF award | Primary record |
| --- | --- | --- |
| `hsZV8lGYWTMC` | 1809658 | [NSF](https://api.nsf.gov/services/v1/awards/1809658.json) |
| `-mN3Mh-tlDkC` | 2404367 | [NSF](https://api.nsf.gov/services/v1/awards/2404367.json) |
| `L1USKYWJimsC` | 2109065 | [NSF](https://api.nsf.gov/services/v1/awards/2109065.json) |
| `XUvXOeBm_78C` | 2117523 | [NSF](https://api.nsf.gov/services/v1/awards/2117523.json) |
| `1yWc8FF-_SYC` | 0521079 | [NSF](https://api.nsf.gov/services/v1/awards/0521079.json) |
| `prdVHNxh-e8C` | 9871874 | [NSF](https://api.nsf.gov/services/v1/awards/9871874.json) |
| `kw52XkFRtyQC` | 0304508 | [NSF](https://api.nsf.gov/services/v1/awards/0304508.json) |
| `yxmsSjX2EkcC` | 0210247 | [NSF](https://api.nsf.gov/services/v1/awards/0210247.json) |
| `Br1UauaknNIC` | 0824188 | [NSF](https://api.nsf.gov/services/v1/awards/0824188.json) |

`6VKJW4kAAAAJ:1taIhTC69MYC` is labeled “Supporting Information” with no year. The [ACS article](https://pubs.acs.org/doi/10.1021/acsaem.3c02822) lists Tomasz as a coauthor and provides a separate Supporting Information section; retain the main 2024 article `6VKJW4kAAAAJ:v1_lew4L6wgC` and exclude the standalone supporting file.

Two high-citation rows that initially seemed like namesake errors are **real Tomasz coauthorships**. The [2004 JBC article](https://www.sciencedirect.com/science/article/pii/S0021925820771966) explicitly lists Tomasz Kowalewski after the Scholar display truncation; the [2020 ACS Catalysis accepted manuscript](https://par.nsf.gov/servlets/purl/10159101) lists *both* Jakub and Tomasz Kowalewski. Retain `6VKJW4kAAAAJ:Y0pCki6q_DkC` and `6VKJW4kAAAAJ:sJsF-0ZLhtgC`. Other truncated author strings should not be excluded solely because the rendered Scholar row ends with an ellipsis.

The short row `6VKJW4kAAAAJ:43bX7VzcjpAC` (“Chemistry—A European”) is an undated fragment with the same author string as the complete 2023 row `6VKJW4kAAAAJ:zCSUwVk65WsC`. The [journal article](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202300477) confirms the complete article, so the fragment is excluded as an exact corrupted duplicate. Several ACS meeting abstracts, patents, theses, and repeated titles are intentionally left for a later output-type policy review because they can be legitimate scholarly outputs or distinct versions.

## Carrie McDonough: newly captured 55-row profile

The two undated SETAC meeting abstracts and undated PFAS talk are retained pending output-type policy, as are dated journal papers and the USGS Open-File Report. No clear namesake or grant-award row was found in this first-pass review. The short “Tackling PFAS Complexity with HRMS and Bioanalytical Techniques” row has no venue and needs source review before it contributes a shared-work edge.

## Isaac Garcia-Bosch and Maria Kurnikova: newly captured profiles

- Isaac `-HzcsYQAAAAJ:SeFeTyx0c_EC` is [NSF award 2153109](https://api.nsf.gov/services/v1/awards/2153109.json), not a paper. The exact award title and CMU PI match the staged Scholar row.
- Maria `IggMbacAAAAJ:hkOj_22Ku90C` is [NSF award 1818213](https://api.nsf.gov/services/v1/awards/1818213.json), not a paper. The exact award title and CMU PI match the staged Scholar row.
- Other short or editorial-looking rows remain in raw captures and are not excluded without source-backed decisions.

## Yisong Guo: newly captured 182-row profile

The two undated, venue-blank rows below explicitly describe supplementary files. Each profile also contains the complete parent article, and the publisher page lists Supporting Information under the same DOI. Preserve the raw rows, exclude these exact files as separate mapped works, and retain both complete article rows.

| Standalone file | Retained article | Publisher evidence |
| --- | --- | --- |
| `xfTeH7YAAAAJ:lvd772isFD0C` | `xfTeH7YAAAAJ:_FM0Bhl9EiAC` (2019) | [Inorganic Chemistry DOI 10.1021/acs.inorgchem.9b01951](https://pubs.acs.org/doi/abs/10.1021/acs.inorgchem.9b01951) |
| `xfTeH7YAAAAJ:fbc8zXXH2BUC` | `xfTeH7YAAAAJ:edDO8Oi4QzsC` (2018) | [JACS DOI 10.1021/jacs.8b10077](https://pubs.acs.org/doi/abs/10.1021/jacs.8b10077) |
