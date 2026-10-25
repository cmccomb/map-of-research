# Yisong Guo exact duplicate and derivative decisions — 2026-10-24

The [complete 182-row capture review](https://github.com/cmccomb/map-of-research/pull/69) compares each candidate with its retained main work. These 15 exact Scholar observations would otherwise add a second work or a misleading coauthor set. The raw Scholar rows remain untouched; the manifest excludes only these observation IDs from map authorship edges.

| Excluded Scholar observation | Retained research work | Primary evidence |
| --- | --- | --- |
| `xfTeH7YAAAAJ:2tRrZ1ZAMYUC` (Duplicate article variant) | `xfTeH7YAAAAJ:hFOr9nPyWt4C` | [Source](https://pubmed.ncbi.nlm.nih.gov/25275608/) |
| `xfTeH7YAAAAJ:gKiMpY-AVTkC` (Duplicate article variant) | `xfTeH7YAAAAJ:IjCSPb-OGe4C` | [Source](https://pubmed.ncbi.nlm.nih.gov/21500788/) |
| `xfTeH7YAAAAJ:4xDN1ZYqzskC` (Duplicate article variant) | `xfTeH7YAAAAJ:PVgj2kMGcgYC` | [Source](https://pubmed.ncbi.nlm.nih.gov/39899716/) |
| `xfTeH7YAAAAJ:1lhNe0rCu4AC` (Duplicate article variant) | `xfTeH7YAAAAJ:0CzhzZyukY4C` | [Source](https://pubmed.ncbi.nlm.nih.gov/39699233/) |
| `xfTeH7YAAAAJ:ALROH1vI_8AC` (Duplicate article variant) | `xfTeH7YAAAAJ:sJsF-0ZLhtgC` | [Source](https://pubmed.ncbi.nlm.nih.gov/39471288/) |
| `xfTeH7YAAAAJ:7wO8s98CvbsC` (Duplicate article variant) | `xfTeH7YAAAAJ:4X0JR2_MtJMC` | [Source](https://pubmed.ncbi.nlm.nih.gov/39442618/) |
| `xfTeH7YAAAAJ:CB2v5VPnA5kC` (Duplicate article variant) | `xfTeH7YAAAAJ:tH6gc1N1XXoC` | [Source](https://pubmed.ncbi.nlm.nih.gov/38806591/) |
| `xfTeH7YAAAAJ:5MTHONV0fEkC` (Duplicate article variant) | `xfTeH7YAAAAJ:tH6gc1N1XXoC` | [Source](https://pubmed.ncbi.nlm.nih.gov/38806591/) |
| `xfTeH7YAAAAJ:yMeIxYmEMEAC` (Duplicate article variant) | `xfTeH7YAAAAJ:LO7wyVUgiFcC` | [Source](https://pubmed.ncbi.nlm.nih.gov/32017545/) |
| `xfTeH7YAAAAJ:QIV2ME_5wuYC` (Duplicate article variant) | `xfTeH7YAAAAJ:3fE2CSJIrl8C` | [Source](https://pubmed.ncbi.nlm.nih.gov/23249091/) |
| `xfTeH7YAAAAJ:a3BOlSfXSfwC` (Duplicate article variant) | `xfTeH7YAAAAJ:W7OEmFMy1HYC` | [Source](https://pubs.rsc.org/en/content/articlelanding/2012/sc/c2sc01044e) |
| `xfTeH7YAAAAJ:SpbeaW3--B0C` (Edited volume) | `xfTeH7YAAAAJ:YohjEiUPhakC` | [De Gruyter volume](https://www.degruyterbrill.com/serial/isc-b/html?lang=en), [Guo group chapter record](https://www.chem.cmu.edu/groups/guo/pub.html) |
| `xfTeH7YAAAAJ:IRz6iEL74y4C` (Edited volume or cover-only feature) | `xfTeH7YAAAAJ:9c2xU6iGI7YC` | [Source](https://chemistry-europe.onlinelibrary.wiley.com/toc/18673899/2019/11/6) |
| `xfTeH7YAAAAJ:ClCfbGk0d_YC` (Edited volume or cover-only feature) | `xfTeH7YAAAAJ:XoXfffV-tXoC` | [Source](https://onlinelibrary.wiley.com/toc/15213773/57/49) |
| `xfTeH7YAAAAJ:FPJr55Dyh1AC` (Edited volume or cover-only feature) | `xfTeH7YAAAAJ:XoXfffV-tXoC` | [Source](https://onlinelibrary.wiley.com/toc/15213757/130/49) |

The producer manifest records the specific reason and verification date for each row. The original work identities remain eligible. The 2023 *Scientific Reports* title variants (`CB2v5VPnA5kC` and `5MTHONV0fEkC`) already collapse with each other by normalized title/year, but remain separate from the 2024 final article; both are excluded exactly.

The two separately reviewed supporting-file rows are already in the manifest from PR #62. All three dated Guo correction notices (`xfTeH7YAAAAJ:jU7OWUQzBzMC`, `xfTeH7YAAAAJ:kw52XkFRtyQC`, `xfTeH7YAAAAJ:WC23djZS0W4C`) remain eligible under producer `main`'s established citable-correction rule. The [RSC correction](https://pubs.rsc.org/en/content/articlepdf/2023/sc/d3sc90183a) adds an omitted coauthor, Zachary Martinez; excluding the notice without reconciling the parent article would lose that correction evidence. The genuinely coauthored [retracted Nature article](https://www.nature.com/articles/s41586-021-03527-x) stays in the historical graph, with retraction status to be surfaced separately. Conference abstracts and ambiguous thesis-like entries are also unchanged pending an exact attribution or output-type decision.
