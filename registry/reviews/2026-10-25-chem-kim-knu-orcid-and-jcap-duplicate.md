# Hyung J. Kim: one KNU ORCID attribution and one repeated journal row

These two decisions use the same complete 918-observation Google Scholar
capture `bC4aq-4AAAAJ`, JSON SHA-256
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`.
The paired redacted HTML SHA-256 is
`a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94`.
No Scholar request or raw-row change was made.

| Capture row and exact ID | Evidence | Decision |
| --- | --- | --- |
| 88 `bC4aq-4AAAAJ:oi2SiIJ9l4AC` | [Elsevier's deposited DOI record](https://api.crossref.org/works/10.1016%2Fj.jallcom.2018.03.266) names the same leading authors, *Journal of Alloys and Compounds* 749, pages 197–204 (2018), and gives its H. J. Kim author ORCID `0000-0001-9787-4684`. [Wiley identifies that ORCID](https://onlinelibrary.wiley.com/doi/abs/10.1002/crat.202000074) as HongJoo Kim of KNU Physics; [KNU's faculty page](https://research.knu.ac.kr/en/persons/hong-joo-kim/) confirms the separate person. Scholar truncates the title, but its year, venue, pages, leading authors, and exact citation ID agree. | `false_profile_attribution`; 92 displayed citations. |
| 793 `bC4aq-4AAAAJ:VnuxuLaQPLMC` | Scholar repeats row 336 `bC4aq-4AAAAJ:gmHTDCtJMcoC` with the same title, 2019 JCAP issue, and leading coauthors, while omitting article number 048 and adding the collaboration as a leading author. [Helsinki's institutional record](https://researchportal.helsinki.fi/en/publications/cosine-100-and-damalibra-phase2-in-wimp-effective-models/) and the [author manuscript](https://eprints.whiterose.ac.uk/id/eprint/145362/8/1904.00128.pdf) identify one 2019 journal article, DOI `10.1088/1475-7516/2019/06/048`. | `corrupted_duplicate`; row 336 remains raw and held for person attribution. |

The rebuild pins row 88's Scholar ID, title, year, venue, and leading authors;
it pins both row IDs, title, year, issue, leading authors, and article-number
contrast for the duplicate. The two additions bring the disjoint screen sets
to **390 exact exclusions**, **114 positive person signals**, and **414 held
observations**, totaling all **918** raw observations. Their sorted-ID,
newline-separated SHA-256 digests, with no trailing newline, are respectively
`03bfaaa3a0b64b6a4ba2154bb4901f1fb751f4e315ab70f12b9cd7686b9599c1`,
`a7fa4ee3c74d12e25cd0e17e011710efbc62e61ac3bfb50f13d6f2f22f599a3a`,
and `68ca9640346030697e42eda0875782eb996db46ade4d6cc983f97a95ac90109c`.

Kim remains candidate-only with no canonical cache or public map release.
Neither article similarity nor affiliation alone is used to resolve the
remaining KNU-looking, medical, accelerator, or conference rows.
