# Two further KNU attributions in the complete Kim Scholar capture

The unchanged 918-row Scholar capture for CMU Chemistry professor Hyung J. Kim
(`bC4aq-4AAAAJ`) contains two more publications of **Hong Joo Kim**, a
Kyungpook National University Physics professor. These decisions identify
specific Scholar observations; they do not authorize a Kim canonical cache or
a map release.

| Capture row and exact Scholar ID | Source check | Decision |
| --- | --- | --- |
| 148 `bC4aq-4AAAAJ:RfUwGJFMQ-0C` | The [Elsevier article](https://www.sciencedirect.com/science/article/abs/pii/S0022231315001696) supplies the exact 2015 title, *Journal of Luminescence* 164, pages 86–89, and DOI `10.1016/j.jlumin.2015.03.026`. That title, year, and DOI appear in [Hong Joo Kim's separate ORCID work record](https://pub.orcid.org/v3.0/0000-0001-9787-4684/work/153825059). A [US patent](https://patents.google.com/patent/US10126436B2/en) names Hong Joo Kim, Gul Rooh, Hwanbae Park, and Sunghwan Kim as inventors and identifies the same 2015 article by H. J. Kim and colleagues as their prior disclosure. These are the four authors in the Scholar row. | `false_profile_attribution` |
| 267 `bC4aq-4AAAAJ:foquWX3nUaYC` | [KNU's article record](https://research.knu.ac.kr/en/publications/luminescence-and-scintillation-properties-of-a-cebrsub3sub-single-2/) gives the 2008 *IEEE Transactions on Nuclear Science* article, DOI `10.1109/TNS.2008.920253`, volume 55(3), pages 1221–1224. Its H. J. Kim author link resolves to [Hong Joo Kim's KNU Physics profile](https://research.knu.ac.kr/en/persons/hong-joo-kim/). Scholar drops the `3` subscript from `CeBr3`, but title, seven-author byline, venue, year, and pages match. | `false_profile_attribution` |

## One genuine paper indexed twice

Capture rows 312 (`bC4aq-4AAAAJ:738O_yMBCRsC`) and 797
(`bC4aq-4AAAAJ:9Nmd_mFXekcC`) have identical titles and 2017 years. Row 312
has *Journal of Molecular Structure* 1146, pages 203–212, while row 797 has no
venue or citation count and a malformed fragment of the byline. [Elsevier's
original article](https://www.sciencedirect.com/science/article/pii/S0022286017307585)
names **Hyung J. Kim** in full and gives exactly one paper with DOI
`10.1016/j.molstruc.2017.05.138`. Retain row 312 as the genuine publication;
mark only row 797 `corrupted_duplicate`. This is an output-level decision,
not a false person attribution.

The rebuild pins each Scholar ID, captured title, and year before applying its
decision. The producer manifest now has **387 exact Kim exclusions**: eight
earlier decisions and 379 proposed decisions. The ledger has 99 exact CMU
ORCID title/year signals, 10 other primary-positive retentions, and **422
held observations**. All three excluded observations remain in the raw
Scholar source. The capture, canonical cache status, graph snapshot, and live
maps are unchanged.
