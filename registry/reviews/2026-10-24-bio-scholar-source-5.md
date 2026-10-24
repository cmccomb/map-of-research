# Biological Sciences Scholar source review, Gordon Rule — 2026-10-24

This offline review covers all **142 visible rows** in the saved Gordon S. Rule Google Scholar profile (`n6flaAsAAAAJ`). The raw JSON and sanitized HTML are paired at `work/scholar-bootstrap/gordon-rule-visible-142-2026-10-24.{json,html}`. Both contain 142 publication rows; all 142 citation IDs are unique, the final **Show more** button is present but disabled, and the capture reports no page error or challenge. These observations establish a complete *visible profile at capture time*, not every scholarly output Rule may have authored.

| Raw artifact | SHA-256 |
| --- | --- |
| JSON | `544f627fa6b2baa4b762fe57198d492778aa87041f2e66f4d2362b8c46da945d` |
| Sanitized HTML | `028a218885e6667ad785e7aa29af171c7c846bc3cfaa81e0c8a19db717f9a33d` |

[CMU identifies Gordon S. Rule as professor and head of Biological Sciences](https://www.cmu.edu/bio/people/faculty/rule.html), supporting Bio as primary department. The profile-to-person Scholar ID binding remains in a separate identity review; this publication review does not edit registry membership or primary-department rows. The 18 exact `(scholar_id, author_pub_id)` decisions below are in [`data/curated/scholar-exclusions.json`](../../data/curated/scholar-exclusions.json). They affect map eligibility only. All 142 raw Scholar rows remain in the separate capture lane, and this PR does not copy caches or publish a snapshot.

## Nonpublications: seven NSF awards

The following rows carry NSF award metadata, and the [official NSF award API](https://www.nsf.gov/funding/award-search) confirms their exact titles, award numbers, and investigators. An award record is not itself an authored publication.

| Citation suffix | Award | Primary record |
| --- | ---: | --- |
| `tS2w5q8j5-wC` | 2112872 | [NSF](https://api.nsf.gov/services/v1/awards/2112872.json) |
| `NhqRSupF_l8C` | 0942572 | [NSF](https://api.nsf.gov/services/v1/awards/0942572.json) |
| `vRqMK49ujn8C` | 0837734 | [NSF](https://api.nsf.gov/services/v1/awards/0837734.json) |
| `LPZeul_q3PIC` | 0821296 | [NSF](https://api.nsf.gov/services/v1/awards/0821296.json) |
| `fQNAKQ3IYiAC` | 9305002 | [NSF](https://api.nsf.gov/services/v1/awards/9305002.json) |
| `5Ul4iDaHHb8C` | 9217013 | [NSF](https://api.nsf.gov/services/v1/awards/9217013.json) |
| `8AbLer7MMksC` | 9005374 | [NSF](https://api.nsf.gov/services/v1/awards/9005374.json) |

## Corrupted duplicates: nine rows

- `B3FOqHPlNUQC`, `4fKUyHm3Qg0C`, and `2osOgNQ5qMEC` are affiliation or abstract fragments with the same author set and opening page as the retained *Specificities of Germ Line Antibodies* chapter `_xSYboBqXhAC`. [Springer identifies the single chapter, its authors, pages 367–376, and abstract](https://link.springer.com/chapter/10.1007/978-1-4684-5745-2_24). The 2013 ebook metadata on one fragment is not a second authored chapter.
- `sSrBHYA8nusC` and `pyW8ca7W8N0C` are an address and sentence fragment with the same 1987 proceedings, authors, and starting page as retained chapter `WbkHhVStYXYC`. [CMU's Chien Ho publication list](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf) identifies the complete Rule–Pratt–Ho chapter on pages 3–20.
- `hqOjcs7Dif8C` prepends the journal section header to the complete 2001 Der p 2 article retained as `YsMSGLbcyi4C`. [The original Journal of Biological Chemistry article](https://www.sciencedirect.com/science/article/pii/S002192581946242X) displays the heading and true title together on pages 9359–9365.
- `geHnlv5EZngC` abbreviates the complete 2023 *Protein Science* title and omits its volume/article number. It repeats `738O_yMBCRsC`, the one Rule coauthored article DOI `10.1002/pro.4770` in the [published bibliographic record](https://pubmed.ncbi.nlm.nih.gov/37632831/).
- `l7t_Zn2s7bgC` and `u9iWguZQMMsC` repeat the same 2018 thymidylate-kinase title with blank venues and partial author metadata. The retained `u_35RYKgDlwC` has the [published article's four authors, *Biochemistry* 57:2868–2875, and DOI `10.1021/acs.biochem.8b00162`](https://pubmed.ncbi.nlm.nih.gov/29684273/). [CMU's Rule page](https://www.cmu.edu/bio/people/faculty/rule.html) also lists that one article.

## False profile attributions: two rows

- `p2g8aNsByqUC` carries the Polish title *Kokpity menedżerskie jako narzędzie monitorowania efektów kształcenia studentów* but mismatched 2013 workshop authors and venue. [The journal publisher](https://www.e-mentor.edu.pl/artykul/index/numer/60/id/1184) identifies the actual 2015 article and its three authors, none Gordon Rule.
- `tOudhMTPpwUC` is only the address *Pittsburgh, Pennsylvania 15213* and names Mark S. Fox and Norman Sadeh. [CMU's archived scheduling manuscript](https://iiif.library.cmu.edu/file/Newell_box00099_fld07543_doc0001/Newell_box00099_fld07543_doc0001.pdf) contains that address on its title page and identifies Fox and Sadeh as its authors.

## Retained observations and limits

The record `XiSMed-E-HIC` abbreviates its authors to Wu and Yang, but [CMU's Ho publication list](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf) names Gordon Rule as a coauthor of that distinct 1988 *Structure and Expression* chapter, so it remains eligible. The 1987 *Biochemistry* paper, 1986 *Biophysical Journal* article, 1987 membrane-protein proceedings chapter, 1988 chapter, 1983 conference abstract, and 1986 thesis are separate dated forms despite overlapping D-lactate dehydrogenase wording. The 2006 NMR book and its individually titled chapters, later conference abstracts, and the 2017 thymidylate-kinase patent are likewise retained. The one-word 1992 `Immunotherapy` row has insufficient metadata to support an exact exclusion; it remains a raw Scholar observation pending stronger evidence.
