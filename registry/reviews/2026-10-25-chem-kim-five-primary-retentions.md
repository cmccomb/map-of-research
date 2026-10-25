# Hyung J. Kim: five source-backed positive observations

This follow-up reviews five previously held IDs in the complete 918-row
Google Scholar capture `bC4aq-4AAAAJ`. The paired source JSON remains
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`
(SHA-256). Each decision is guarded by the exact Scholar ID, displayed title,
and year in `rebuild_chem_kim_918_review.py`. No new Scholar request or raw-row
change was made.

| Capture row and exact ID | Primary evidence | Decision |
| --- | --- | --- |
| 480 `bC4aq-4AAAAJ:R3hNpaxXUhUC` | [AIP publisher-deposited DOI record](https://api.crossref.org/works/10.1063%2F1.473606) for *Journal of Chemical Physics* 106, 5979–5989 (1997) names **Hyung J. Kim** at CMU Chemistry with Jonggu Jeon; the Scholar title, year, venue, and pages agree. | Positive CMU attribution. |
| 587 `bC4aq-4AAAAJ:u0Mu_IsstPMC` | [ACS original article](https://pubs.acs.org/doi/abs/10.1021/jasms.1c00346) names **Hyung J. Kim** at CMU Chemistry and links ORCID `0000-0003-4334-1879`; its 2022 volume 33, pages 521–529 match the Scholar row, whose displayed title is truncated. | Positive CMU attribution. |
| 643 `bC4aq-4AAAAJ:hMod-77fHWUC` | [ACS issue record](https://pubs.acs.org/toc/jpccck/116/34) lists the distinct published correction by **Hyung J. Kim**, DOI `10.1021/jp307086f`, *J. Phys. Chem. C* 116, 18574–18575 (2012). The [publisher-deposited DOI record](https://api.crossref.org/works/10.1021%2Fjp307086f) gives his CMU Chemistry affiliation. | Positive CMU attribution; retain as a published correction distinct from the 2011 parent article. |
| 709 `bC4aq-4AAAAJ:abG-DnoFyZgC` | [RSC correction](https://pubs.rsc.org/en/content/articlelanding/2017/ra/c7ra90031g), DOI `10.1039/C7RA90031G`, names **Hyung J. Kim** at CMU Chemistry; 2017 volume 7, page 13876 matches Scholar's truncated title and venue. | Positive CMU attribution; retain the separate correction record. |
| 796 `bC4aq-4AAAAJ:4fKUyHm3Qg0C` | [RSC correction](https://pubs.rsc.org/en/content/articlelanding/2017/cp/c7cp90081c), DOI `10.1039/C7CP90081C`, names **Hyung J. Kim** at CMU Chemistry; 2017 volume 19, page 11709 matches Scholar. | Positive CMU attribution; retain the separate correction record. |

The RSC pages identify both 2017 outputs as corrections with their own DOIs.
They should not be collapsed into the parent research articles, nor excluded
merely because the title starts with “Correction.”

After this review, the three disjoint exact-ID sets cover all 918 observations:
388 exclusions, 114 positive person signals, and 416 unresolved attribution or
output holds. The sorted-ID, newline-separated SHA-256 digests, with no trailing
newline, are respectively
`83ff33c73b4adb3fee0b608d6b1cb472946b7c269f87c0295cdd29fcac4b2a70`,
`a7fa4ee3c74d12e25cd0e17e011710efbc62e61ac3bfb50f13d6f2f22f599a3a`,
and `3a4989c98340494bc2c1854c6fb42e124554354d50bd38fd3229cac9adc5f939`.
The producer exclusion manifest is unchanged. Kim still has no canonical cache;
the 416 held IDs prevent a complete-profile release until independently
resolved and pinned in the source-bound hold gate.

Other high-citation medical, accelerator, and biology rows were screened in
this pass. Their available primary records used initials without resolving the
specific Kim identity, so they remain held. No topic-based exclusion was made.
