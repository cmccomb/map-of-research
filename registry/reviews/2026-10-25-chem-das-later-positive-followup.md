# Subha R. Das: later exact positive attributions and two held edge cases

This offline pass checked 16 remaining ORCID title/year signals against CMU, original publisher metadata, NLM, and original article pages. Fourteen exact publication IDs have direct positive attribution. Two remain held despite identity clues because the specific Scholar record or metadata is unresolved. Scholar was not opened, and all 670 raw rows remain preserved.

| Capture row | Exact citation ID | Output type | Source for positive attribution |
| ---: | --- | --- | --- |
| 73 | `FrF1YXQAAAAJ:86PQX7AUzd4C` | research article | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 74 | `FrF1YXQAAAAJ:hMsQuOkrut0C` | research article | [Original or CMU source](https://api.crossref.org/works/10.1021/prechem.3c00042) |
| 81 | `FrF1YXQAAAAJ:qUcmZB5y_30C` | research article | [Original or CMU source](https://pubmed.ncbi.nlm.nih.gov/25222520/) |
| 96 | `FrF1YXQAAAAJ:b1wdh0AR-JQC` | research article | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 113 | `FrF1YXQAAAAJ:dshw04ExmUIC` | research article | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 133 | `FrF1YXQAAAAJ:IjCSPb-OGe4C` | research article | [Original or CMU source](https://api.crossref.org/works/10.5101/nbe.v2i3.p165-170) |
| 135 | `FrF1YXQAAAAJ:dhFuZR0502QC` | research article | [Original or CMU source](https://www.chem.cmu.edu/groups/das/) |
| 140 | `FrF1YXQAAAAJ:zLWjf1WUPmwC` | research communication | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 141 | `FrF1YXQAAAAJ:pqnbT2bcN3wC` | research article | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 159 | `FrF1YXQAAAAJ:mWEH9CqjF64C` | research article | [Original or CMU source](https://api.crossref.org/works/10.1039/D5DD00335K) |
| 240 | `FrF1YXQAAAAJ:jFemdcug13IC` | research article | [Original or CMU source](https://api.crossref.org/works/10.1021/acs.biomac.5c00838) |
| 242 | `FrF1YXQAAAAJ:M7yex6snE4oC` | research article | [Original or CMU source](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1408248/full) |
| 347 | `FrF1YXQAAAAJ:eq2jaN3J8jMC` | book chapter | [Original or CMU source](https://www.cmu.edu/chemistry/people/faculty/das.html) |
| 370 | `FrF1YXQAAAAJ:B2rIPIGFPLEC` | research article | [Original or CMU source](https://api.crossref.org/works/10.1021/acsabm.6c01051) |

Row 242 is a genuine CMU Subha R. Das paper despite its unexpected phytochemical subject and truncated Scholar byline: the [original Frontiers article](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1408248/full) spells out his name and CMU affiliation. Row 347 is a book chapter and must retain that output type.

**Row 330 remains held for metadata adjudication.** CMU’s [faculty bibliography](https://www.cmu.edu/chemistry/people/faculty/das.html) attributes “RNA Conjugations and Ligations for RNA Nanotechnology” to Paredes and Das as a 2013 chapter at pages 197–211. The [CRC Press deposited chapter record](https://api.crossref.org/works/10.1201%2Fb15152-18) gives the same title in *RNA Nanotechnology and Therapeutics* dated 2013-07-09 at pages 215–230. Scholar’s saved row says 2019. The identity is supported, but the exact edition, year, and pages need reconciliation before publication. Its ID stays in the proposal’s held array.

**Row 535 remains held for record identity and output policy.** The saved row names “Click chemistry for RNA: Labeling and ligation” in a 2010 ACS meeting abstract volume. The [2009 Rustbelt RNA program](https://rustbeltrna.org/2009/program.php) confirms Das and Paredes presented this exact title, but it does not prove the particular 2010 ACS abstract record. Do not merge it with the full 2011 ChemBioChem paper at row 2.

At this pass, the provisional exact-positive proposal contained 49 IDs, with 188 exact exclusions and 433 held IDs. The later [UKM Anatomy follow-up](2026-10-25-chem-das-next-ukm-evidence.md) updates that partition to 49/199/422. The trusted upload manifest still holds all 670 rows and does not publish this incomplete subset.
