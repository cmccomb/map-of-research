# Subha R. Das Scholar attribution review — 2026-10-24

CMU Chemistry's [faculty page](https://www.cmu.edu/chemistry/people/faculty/das.html) links Google Scholar profile `FrF1YXQAAAAJ`. A headed Chrome capture reached the disabled **Show More** control at 670 visible records with no visible Scholar error. The unmodified HTML, browser JSON, canonical cache, and capture receipt are staged on `codex/chem-das-snapshot` at `83ca419`; they have not been promoted to the map.

This review adds 76 exact `author_pub_id` decisions to `data/curated/scholar-exclusions.json`. The producer keeps the raw observation and marks the reviewed record ineligible for the map. The remaining 594 records have **not** been certified as publications of CMU's Subha R. Das.

| Decision group | Count | Evidence used |
| --- | ---: | --- |
| Subir Das, mathematician | 27 | [IIT (BHU) faculty publication list](https://www.iitbhu.ac.in/dept/mat/people/sdasapm), with matching titles and author context |
| CSIR-CGCRI optical-fiber group | 13 | [Shyamal Das's institutional publication list](https://www.cgcri.res.in/research/research-divisions/fiber-optics-and-photonics/dr-shyamal-das/), [CGCRI's book-chapter record](https://cgcri.csircentral.net/698/), [Anirban Dhar's institutional list](https://www.cgcri.res.in/wp-content/uploads/2021/publication_list/AnirbanDhar_publist.pdf), and the exact paper or collaborating institution linked for each remaining row |
| Other distinct authors or affiliations | 10 | Exact paper pages or PDFs from IIT Guwahati, PubMed, Cureus, the Indian Journal of Veterinary Anatomy, the European Journal of Anatomy, Advances in Research, and original plant and anatomy studies; each manifest entry links its individual evidence |
| Journal/series/contents titles | 4 | Elsevier, ISSN records, and the published contents page identify these observations as containers or indexes rather than authored research papers |
| Sanatan Das, mathematician | 19 | [University of Gour Banga's official faculty publication directory](https://www.ugb.ac.in/couch/uploads/file/pdf2025/Directory%20of%20Faculty%20Publication_Final.pdf), [his institutional CV](https://www.ugb.ac.in/couch/uploads/file/cv_professor-sanatan-das.pdf), and exact publisher or original-paper pages for three rows |
| Srijit Das, anatomist | 3 | [Sultan Qaboos University's author-linked publication records](https://squ.elsevierpure.com/en/persons/srijit-das/) and the [published UKM Anatomy case report](https://www.clinicaterapeutica.it/download/367/fascicolo-3/7963/166-03-04_case_report_kosai.pdf) |

The two institutional lists were compared with the captured Scholar titles using normalized title prefixes, then each candidate was checked against its source line and Scholar row ID. The 27 mathematical entries include a few punctuation, spelling, or `WITHDRAWN` prefix variations confirmed by author and venue context; the remainder match the first six normalized title words. The 13 optical-fiber entries were checked against the named scientist's list, a CGCRI collaborator's bibliography, CGCRI's own repository, or the original paper's institutional affiliation. No decision was made solely because a subject seemed outside CMU Chemistry.

The Gour Banga directory identifies Sanatan Das as a Mathematics professor and lists his papers and one book chapter. The 19 new exact-ID decisions are rows 70 (`BrOSOlqYqPUC`), 88 (`HJSXoJQnj-YC`), 118 (`EPG8bYD4jVwC`), 120 (`DrR-2ekChdkC`), 150 (`YsrPvlHIBpEC`), 166 (`kWvqk_afx_IC`), 168 (`raTqNPD5sRQC`), 200 (`wvYxNZNCP7wC`), 217 (`WAzi4Gm8nLoC`), 227 (`1Ye0OR6EYb4C`), 257 (`ce2CqMG-AY4C`), 287 (`kF1pexMAQbMC`), 288 (`XUvXOeBm_78C`), 294 (`-jrNzM816MMC`), 299 (`TlpoogIpr_IC`), 338 (`JTqpx9DYBaYC`), 429 (`-95Q15plzcUC`), 475 (`mKu_rENv82IC`), and 541 (`O0nohqN1r9EC`), all prefixed `FrF1YXQAAAAJ:`. Rows 70, 166, 200, 287, and 299 have a title or author-list difference in Scholar; the available year, coauthor, venue, or page evidence reconciles them with the institutional bibliography. Row 338 is supported by [the publisher's explicit Gour Banga affiliation](https://yjes.researchcommons.org/yjes/vol14/iss1/1/), row 475 by [the publisher's explicit Gour Banga affiliation](https://www.jafmonline.net/article_1746.html), and row 541 by [the original paper linked from Vidyasagar's migrated repository item](https://ir.vidyasagar.ac.in/xml/handle/123456789/802). The original PDF explicitly affiliates its `S. Das` author with Vidyasagar University.

An independent source-link check found that the exact [row 51 journal PDF](https://www.entomoljournal.com/archives/2017/vol5issue3/PartV/5-3-45-507.pdf) and [row 103 original paper PDF](https://cio.repositorioinstitucional.mx/jspui/bitstream/1002/648/1/silicon%20nano-particles%2C%20yuri.pdf) both load and display the cited author identity or affiliation in a standard browser request. Vidyasagar moved row 541's paper to the `/xml/handle/` repository path above. For row 558's 2008 journal-title record, the [print ISSN record](https://portal.issn.org/resource/ISSN/0970-3268) is the period-appropriate title evidence; the newer online ISSN begins in 2018.

Rows 18 (`3bvyWxjaHKcC`) and 79 (`3MwzVuqRcA0C`) are linked to Srijit Das in Sultan Qaboos University's exact article records. Row 101 (`eFf2swCANGcC`) is the published case report that affiliates S. Das with UKM Anatomy and names Srijit Das in its disclosure. All three IDs have the same `FrF1YXQAAAAJ:` prefix.

Row 78 (`FrF1YXQAAAAJ:a2necdfpwlEC`) is additionally excluded because [the original 2015 anatomy paper](https://www.clinicaterapeutica.it/download/368/fascicolo-2/7957/166-02-04_case_report_hansdak_arora.pdf) specifically affiliates its `S. Das` author with Universiti Kebangsaan Malaysia Anatomy. The other Arora/Mehta/Rath anatomy rows were left unexcluded without equivalent row-level primary evidence.

## Open attribution questions

Six of the original 25 suspect observations remain unexcluded because the available primary record does not establish the identity of the `S. Das` author strongly enough:

| Scholar row | Short title | Current evidence gap |
| ---: | --- | --- |
| 13 | Colorectal carcinoma in young adults | [PubMed](https://pubmed.ncbi.nlm.nih.gov/20128837/) identifies the Kolkata surgery setting but only initials for S. Das. |
| 19 | Speaker verification | The 1971 publication date predates CMU's listed degrees, but the [CMU profile](https://www.cmu.edu/chemistry/people/faculty/das.html) cannot identify the 1971 author. |
| 35 | Simple renal cyst | [PubMed](https://pubmed.ncbi.nlm.nih.gov/9567602/) gives the first author's North Bengal affiliation, not the S. Das author's full identity. |
| 47 | Mullite formation from kaolinite | The [CGCRI record](https://cgcri.csircentral.net/829/) names S. Das by initial only. |
| 48 | Koro epidemic in India | The 1983 date is suspicious, but the article's S. Das identity has not been resolved. |
| 54 | Kadaknath fowl thesis | The [G. B. Pant veterinary department](https://gbpuat.ac.in/colleges/COV/D4/index.html) lists the thesis title but not a clearly resolved full author name. |

Additional rows in the 670-record capture may need review; this exclusion set is a bounded source-backed pass, not a complete attribution audit.

The broader title scan also flagged row 408 (a 2023 journal index), row 646 (`OF CONTENTS NUMBER`), and row 668 (a title apparently spliced from a fiber-laser contents page). They have no manifest decisions in this pass because the exact imported item has not been reconciled against a primary source.

Row 523 (`FrF1YXQAAAAJ:uVUOdF_882EC`, 2011, *Hall effects on oscillatory Couette flow*) also remains unexcluded. Gour Banga's directory lists a **different** 2013 paper whose title starts with the same words; its venue and author list differ from the 2011 Scholar row. A direct source that resolves the 2011 `S. Das` is still needed. Other medical and anatomy records may share the Srijit Das name, but this pass did not infer their identity from coauthors or subject alone.
