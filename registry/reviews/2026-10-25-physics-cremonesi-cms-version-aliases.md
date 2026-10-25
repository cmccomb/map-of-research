# Cremonesi CMS original and revised title observations

[Source PR #156](https://github.com/cmccomb/map-of-research/pull/156) preserves the complete 1,963-row Matteo Cremonesi Scholar capture and its exact citation IDs. These five saved rows describe two CMS studies, each released under an original title and later under a revised journal title. This review joins their work identity through the published DOI. It retains every source observation and does not edit the captured titles, years, venues, HTML, or author cache.

| Capture row and exact citation suffix | Observed form | Reviewed relationship | Published DOI |
| --- | --- | --- | --- |
| 1958 `:UBSsHvkqiq4C` | Original HIN-18-015 title; blank year and venue | `preprint_of` | `10.1103/PhysRevC.103.014902` |
| 695 `:RJNGbXJAtMsC` | Revised title; 2021 *Physical Review C* 103, 014902 | `version_of_record` | `10.1103/PhysRevC.103.014902` |
| 1959 `:QjNCP7ux8QYC` | Original BPH-13-001 “Measurement” title; blank year and venue | `preprint_of` | `10.1103/PhysRevLett.124.162002` |
| 584 `:jYy9R3AkpWgC` | Revised “Constraints” title; 2019 arXiv:1912.07706 | `preprint_of` | `10.1103/PhysRevLett.124.162002` |
| 1349 `:2U2whq44yacC` | Revised title; 2020 *Physical Review Letters* 124(16) | `version_of_record` | `10.1103/PhysRevLett.124.162002` |

The [original HIN-18-015 arXiv PDF](https://arxiv.org/pdf/1905.09935v1) has row 1958's title and ordered Sirunyan–Tumasyan–Adam–Ambrogi–Asilar–Bergauer prefix and lists M. Cremonesi in the collaboration. [arXiv v2](https://arxiv.org/abs/1905.09935v2) has row 695's revised title. The [official CMS HIN-18-015 page](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-18-015/index.html) links arXiv:1905.09935 to the 2021 *Physical Review C* article and DOI.

The [original BPH-13-001 arXiv PDF](https://arxiv.org/pdf/1912.07706v1) has row 1959's “Measurement” title and ordered Sirunyan–Tumasyan–Adam–Ambrogi–Bergauer prefix and lists M. Cremonesi. [arXiv v3](https://arxiv.org/abs/1912.07706v3) has the later “Constraints ... versus” title used by rows 584 and 1349. The [official CMS BPH-13-001 page](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-13-001/index.html) links arXiv:1912.07706 to the 2020 *Physical Review Letters* article and DOI. The two preprint rows use one arXiv identifier but preserve different real version titles; the DOI aliases assign them one map work instead of discarding either observation.

Each alias guards the exact saved `scholar_id:author_pub_id`, literal title, and observed year, including `null` for the two yearless rows. The `version_of_record` rows supply the canonical journal year and venue. The current source review still has other map-eligible exact IDs under `map_quality_hold=true`; this producer draft must remain unmerged until the source and all other hold decisions are ready for a trusted combined release. A producer-only merge would make the live uploader require Scholar rows absent from the current snapshot.
