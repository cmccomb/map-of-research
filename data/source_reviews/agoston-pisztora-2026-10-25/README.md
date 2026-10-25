# Ágoston Pisztora: held Math publication-source review

Reviewed 2026-10-25 for `person-agostonpisztora-810e0d1f`. This folder is **source evidence only**. Every candidate and related observation has `release_state=held` and `graph_eligible=false`. It changes no Scholar ID, registry assignment, publication snapshot, Hub dataset, or public graph.

## What is pinned

- `observed-11.csv` preserves the ten records in the earlier audit (eight journal articles, one ICM proceedings contribution, and one separately typed ETH doctoral thesis) and adds one **distinct, unpublished 2000 CMU CNA preprint**. Thus the previously reported minimum of **nine published research works** remains nine. The thesis and preprint are separate genres; neither is silently turned into a journal article.
- `related-held-5.csv` records two likely preprint manifestations of known journal families (1999 CNA Wulff and 2000 Orsay phase coexistence), a 2003 arXiv manuscript possibly related to the 2002 ICM proceedings, a 2001 JSM talk potentially related to the CNA shortest-path manuscript, and a 2004 Oberwolfach talk report. Their version relationships and genre inclusion remain held. These five rows must not create five extra publications.
- `raw/` captures CMU's [faculty page](https://www.cmu.edu/math/people/faculty/pisztora.html), its personal directory listing, the [CNA author search](https://www.math.cmu.edu/CNA/Publications/publications.php), both [2000 CNA](https://www.math.cmu.edu/CNA/Publications/publication_auto.php?cnacode=00-CNA-001) and [1999 CNA](https://www.math.cmu.edu/CNA/Publications/publication_auto.php?cnacode=99-CNA-012) details, seven Crossref DOI metadata records, the ETH thesis DataCite record, [Numdam's 2001 article](https://www.numdam.org/item/AIHPB_2001__37_6_643_0/), [arXiv's ICM manuscript](https://arxiv.org/abs/math/0304371), the [JSM abstract](https://ww2.amstat.org/meetings/jsm/2001/index.cfm?abstractid=300170&fuseaction=abstract_details), and [Cerf's coauthor list](https://www.imo.universite-paris-saclay.fr/~raphael.cerf/articles.html). `manifest.json` pins the bytes; `validate_capture.py` checks hashes, counts, release holds, DOI metadata, and the new CNA attribution.
- `cna-preprint-source-hashes.json` records SHA-256 for CMU's [full PDF](https://www.math.cmu.edu/CNA/Publications/publications2000/papers/00-CNA-001.pdf) and [TeX source](https://www.math.cmu.edu/CNA/Publications/publications2000/001abs/00-CNA-001.tex), without checking the manuscript text into this repository. The source starts with a January 16, 2000 version comment. Its title page names Agoston Pisztora and his CMU Mathematical Sciences affiliation. The CMU CNA search indexes it as `00-CNA-001` and lists `99-CNA-012` Wulff separately.

The earlier local audit and its ten-row CSV remain unchanged at `outputs/agoston-pisztora-source-audit-2026-10-25.md` and `outputs/agoston-pisztora-primary-source-works-2026-10-25.csv` in the task workspace. Their SHA-256 values are recorded in `manifest.json`. The earlier audit's “nine research works plus thesis” count was accurate for *published* records found then; this review adds the previously missed unpublished CNA manuscript.

## Release gate

CMU labels its four citations **Select Publications**; it does not claim a complete bibliography. The personal directory lists no CV. The account-coverage review has `no_verified_scholar_id`, and bounded searches found no reliable exact Google Scholar profile. A secondary ten-row index duplicates three 1996 works and omits known primary records, so it cannot certify completeness. Additional unpublished, conference, or published work may still exist. **Do not turn this inventory into a whole-researcher graph import.** A verified Scholar account or a complete researcher-maintained bibliography, followed by version and genre reconciliation, is needed before any release decision.

To check the pinned review:

```sh
python3 data/source_reviews/agoston-pisztora-2026-10-25/validate_capture.py
```
