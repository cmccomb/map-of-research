# Diana Parno Scholar observation review, 2026-10-24

[Source carve #131](https://github.com/cmccomb/map-of-research/pull/131), derived from [draft #87](https://github.com/cmccomb/map-of-research/pull/87), preserves Diana Parno's complete 156-row visible Scholar capture, saved HTML/JSON, cache, and receipt. This producer draft excludes only the four exact observations below. The raw source rows remain intact. Parno has one included Physics academic membership, so the registry derives Physics as her primary display home.

Each exact ID occurs once in the saved raw JSON and once in the converted author cache. The complete IDs and reason codes are in [`data/curated/scholar-exclusions.json`](../../data/curated/scholar-exclusions.json).

| Capture row | ID suffix | Decision and primary evidence |
| ---: | --- | --- |
| 90 | `2KloaMYe4IUC` | Scholar imported the journal title *Advances in High Energy Physics* as a work title. [Publisher-deposited DOI metadata](https://api.crossref.org/works/10.1155%2F2008%2F982341) identify the article *Hybrid Charmonium and the rho-pi Puzzle*; retain complete row 96. |
| 130 | `ns9cj8rnVeAC` | The captured Japanese text is a related-links label, while the [Paleontological Society of Japan issue listing](https://www.palaeo-soc-japan.jp/pr/2015-1.html) attributes the matching pages to unrelated authors. Parno is not an author. |
| 146 | `z_wVstp3MssC` | A copyright/title-page fragment of [Ying-Ting Lin's University of Washington dissertation](https://hdl.handle.net/1773/45253), not a Parno-authored work. Parno served on the committee. |
| 149 | `VL0QpB8kHFEC` | [Jefferson Lab's 2008 Hall A report](https://hallaweb.jlab.org/publications/AnnualReports/AnnualReport2008.pdf) labels this section a collaboration member list, not a research publication. |

The other captured rows remain eligible or pending exact source review. Same-title pairs, blank venues, and meeting abstracts are preserved as grouped review flags; none is excluded wholesale. The 706 existing producer exclusions are unchanged, and the manifest contains 710 entries with these four additions. If the source slice is ingested after this producer review, its 156 raw observations yield 152 eligible observations before any general title-quality filtering or cross-profile work matching. This draft does not build, upload, or publish a dataset.
