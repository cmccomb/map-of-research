"""Rebuild the exact-ID Hyung Kim screening ledger from the saved Scholar capture.

The source capture is in draft source PR #118. This script makes no Scholar
request. ORCID comparison inputs are the dated, compact public API snapshot
beside this script. Run with --capture PATH --write, then --capture PATH --check.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCHOLAR_ID = "bC4aq-4AAAAJ"
KNU_ORCID = "0000-0001-9787-4684"
CMU_ORCID = "0000-0003-4334-1879"
BASELINE_EXCLUDED_INDICES = {3, 14, 22, 34, 50, 226, 378, 695}
LEDGER = HERE / "2026-10-24-chem-kim-918-screen.csv"
EVIDENCE = HERE / "2026-10-24-chem-kim-orcid-evidence.json"
MANIFEST = ROOT / "data/curated/scholar-exclusions.json"

# Each linked primary author record names Hyung Joon Kim, not CMU Hyung J. Kim.
NEUROSCIENCE = {
    2: "https://www.nature.com/articles/nature12647",
    4: "https://www.salk.edu/news-release/new-stem-cell-research-points-to-early-indicators-of-schizophrenia/",
    5: "https://mbel.snu.ac.kr/publication/biological-applications-of-microfluidic-gradient-devices/",
    17: "https://www.nature.com/articles/ncb1916",
    30: "https://pubmed.ncbi.nlm.nih.gov/29727680/",
    37: "https://mbel.snu.ac.kr/en/publication/generation-of-stable-complex-gradients-across-two-dimensional-surfaces-and-three-dimensional-gels/",
    42: "https://journals.biologists.com/dev/article/142/12/2213/46760/In-vitro-myelin-formation-using-embryonic-stem",
    48: "https://pmc.ncbi.nlm.nih.gov/articles/PMC3131323/",
    58: "https://pubmed.ncbi.nlm.nih.gov/27387650/",
    61: "https://pubs.rsc.org/en/content/articlehtml/2013/lc/c2lc41081h",
    74: "https://www.nature.com/articles/nn.3246",
    110: "https://pubs.rsc.org/en/Content/ArticleLanding/2010/LC/B927153H",
    125: "https://pubmed.ncbi.nlm.nih.gov/22302320/",
    141: "https://pubmed.ncbi.nlm.nih.gov/18314199/",
    165: "https://snu.elsevierpure.com/en/publications/modeling-neural-circuit-blood-brain-barrier-and-myelination-on-a-/",
}

# Exact capture rows with a primary work record identifying a different Kim.
# The expected title/year prevents an index-only decision if the raw capture
# changes. KNU's author-linked records resolve H. J. Kim to Hong Joo Kim;
# publisher/author copies also identify the H. J. author with KNU Physics.
OTHER_PRIMARY = {
    99: (
        "Identification of Salmonella enterica serovar Typhimurium using "
        "specific PCR primers obtained by comparative genomics in "
        "Salmonella serovars",
        2006,
        "Hyun-Joong Kim",
        "10.4315/0362-028X-69.7.1653",
        "https://foodsci.oregonstate.edu/parks-lab/publications/2008-2016",
        "Oregon State University coauthor-lab publication list",
        "The Oregon State University coauthor lab lists this exact Journal "
        "of Food Protection article, volume 69 issue 7, pages 1653–1661, "
        "under the full name Hyun-Joong Kim with matching coauthors.",
    ),
    161: (
        "Brown color characteristics and antioxidizing activity of doenjang extracts",
        2002,
        "Hyun Jung Kim",
        "",
        "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09673546",
        "DBpia journal article record for the Korean Society of Food Cookery Science",
        "The journal-platform record for this exact volume 18 issue 6, "
        "pages 644–654 article names first author Hyun Jung Kim of "
        "Yonsei University Food and Nutrition.",
    ),
    142: (
        "Reclamation of abandoned coal mine waste in Korea using lime cake by-products",
        2006,
        "Hee Joung Kim",
        "10.1007/s10230-006-0137-z",
        "https://pure.korea.ac.kr/en/publications/"
        "reclamation-of-abandoned-coal-mine-waste-in-korea-using-lime-cake/",
        "Korea University full-author article record",
        "Korea University's record for the exact Mine Water and the "
        "Environment article, volume 25 issue 4, pages 227–232, "
        "names its final author in full as Hee Joung Kim.",
    ),
    175: (
        "(): New Tl-based Elpasolite Scintillation Material",
        2016,
        "Hong Joo Kim",
        "10.1109/TNS.2016.2530822",
        "https://research.knu.ac.kr/en/publications/"
        "tlsub2subliyclsub6sub-cesup3sup-new-tl-based-elpasolite-scintilla/",
        "KNU author-linked IEEE article record",
        "KNU identifies the malformed Scholar title's matching 2016 IEEE "
        "article, volume 63 issue 2, pages 439–442; its H. J. Kim "
        "author link resolves to KNU Physics professor Hong Joo Kim.",
    ),
    88: (
        "Investigations of optical and luminescence features of Sm3+ "
        "doped Li2O-MO-B2O3 (M= Mg/Ca/Sr/Ba) glasses mixed with different "
        "modifier oxides as an orange light emitting\xa0…",
        2018,
        "Hong Joo Kim",
        "10.1016/j.jallcom.2018.03.266",
        "https://api.crossref.org/works/10.1016%2Fj.jallcom.2018.03.266",
        "Elsevier publisher-deposited DOI record with author ORCID",
        "The publisher's exact article has the same leading authors, 2018 "
        "volume 749 and pages 197–204, and assigns its H. J. Kim author "
        "ORCID 0000-0001-9787-4684, which Wiley identifies as KNU Physics "
        "professor Hong Joo Kim.",
    ),
    148: (
        "Luminescence and scintillation properties of the new Ce-doped "
        "Tl2LiGdCl6 single crystals",
        2015,
        "Hong Joo Kim",
        "10.1016/j.jlumin.2015.03.026",
        "https://patents.google.com/patent/US10126436B2/en",
        "US patent's inventor and prior-disclosure records; KNU-linked ORCID DOI",
        "The KNU Physics professor's separate ORCID record gives the exact "
        "article DOI, title, and year. A US patent names Hong Joo Kim and "
        "the same three coauthors as inventors and identifies their exact "
        "2015 article as an inventor prior disclosure.",
    ),
    267: (
        "Luminescence and Scintillation Properties of a CeBr Single Crystal",
        2008,
        "Hong Joo Kim",
        "10.1109/TNS.2008.920253",
        "https://research.knu.ac.kr/en/publications/"
        "luminescence-and-scintillation-properties-of-a-cebrsub3sub-single-2/",
        "KNU institutional article; H. J. Kim author linked to Hong Joo Kim",
        "KNU's exact IEEE article links its H. J. Kim author directly to "
        "Physics professor Hong Joo Kim.",
    ),
    109: (
        "Search for low-mass dark matter with CsI (Tl) crystal detectors",
        2014,
        "Hong Joo Kim",
        "10.1103/PhysRevD.90.052006",
        "https://research.knu.ac.kr/en/publications/search-for-low-mass-dark-matter-with-csitl-crystal-detectors/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    174: (
        "The COSINE-100 data acquisition system",
        2018,
        "Hong Joo Kim",
        "10.1088/1748-0221/13/09/P09006",
        "https://research.knu.ac.kr/en/publications/the-cosine-100-data-acquisition-system/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    235: (
        "Measurement of the cosmic muon annual and diurnal flux variation "
        "with the COSINE-100 detector",
        2021,
        "Hong Joo Kim",
        "10.1088/1475-7516/2021/02/013",
        "https://research.knu.ac.kr/en/publications/measurement-of-the-cosmic-muon-annual-and-diurnal-flux-variation-/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    241: (
        "The COSINE-100 liquid scintillator veto system",
        2021,
        "Hong Joo Kim",
        "10.1016/j.nima.2021.165431",
        "https://research.knu.ac.kr/en/publications/the-cosine-100-liquid-scintillator-veto-system/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    62: (
        "AMoRE experiment: a search for neutrinoless double beta decay "
        "of 100Mo isotope with 40Ca100MoO4 cryogenic scintillation detector",
        2012,
        "Hong Joo Kim",
        "10.1088/1742-6596/375/1/042023",
        "https://research.knu.ac.kr/en/publications/amore-experiment-a-search-for-neutrinoless-double-beta-decay-of-s/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    93: (
        "CaMoO/sub 4/scintillation crystal for the search of/sup 100/Mo "
        "double beta decay",
        2005,
        "Hong Joo Kim",
        "10.1109/TNS.2005.852678",
        "https://research.knu.ac.kr/en/publications/camoosub4sub-scintillation-crystal-for-the-search-ofsup100supmo-d/",
        "KNU institutional article; H. J. Kim linked to Hong Joo Kim",
        "The KNU publication record links the exact article's H. J. Kim "
        "author to its separate Hong Joo Kim Physics profile.",
    ),
    6: (
        "Decentralized nonlinear model predictive control of multiple flying robots",
        2003,
        "H. Jin Kim",
        "",
        "https://people.eecs.berkeley.edu/~sastry/sastry-pubs/Postdocs/ShimDavid/ShimDecentralized.pdf",
        "UC Berkeley author-hosted conference paper",
        "The paper names H. Jin Kim of UC Berkeley EECS as its coauthor.",
    ),
    10: (
        "New limits on interactions between weakly interacting massive "
        "particles and nucleons obtained with CsI (Tl) crystal detectors",
        2012,
        "Hong Joo Kim",
        "10.1103/PhysRevLett.108.181301",
        "https://arxiv.org/pdf/1204.2646",
        "author-supplied paper; KNU Physics affiliation",
        "The author-supplied paper assigns H. J. Kim to Kyungpook "
        "National University Physics.",
    ),
    11: (
        "Limits on WIMP-nucleon interactions with CsI (Tl) crystal detectors",
        2007,
        "Hong Joo Kim",
        "10.1103/PhysRevLett.99.091301",
        "https://arxiv.org/pdf/0704.0423",
        "author-supplied paper; KNU Physics affiliation",
        "The author-supplied paper assigns H. J. Kim to Kyungpook "
        "National University Physics.",
    ),
    19: (
        "Search for a dark matter-induced annual modulation signal in "
        "NaI (Tl) with the COSINE-100 experiment",
        2019,
        "Hong Joo Kim",
        "10.1103/PhysRevLett.123.031302",
        "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.031302",
        "APS article; KNU Physics affiliation",
        "The APS article assigns H. J. Kim to Kyungpook National University Physics.",
    ),
    29: (
        "Development of CaMoO4 crystal scintillators for a double beta "
        "decay experiment with Mo100",
        2008,
        "Hong Joo Kim",
        "10.1016/j.nima.2007.10.038",
        "https://arxiv.org/pdf/0707.1428",
        "author-supplied paper; KNU Physics affiliation",
        "The author-supplied paper assigns H. J. Kim to Kyungpook "
        "National University Physics; the journal version appeared in 2008.",
    ),
    53: (
        "First limit on WIMP cross section with low background CsI (Tℓ) "
        "crystal detector",
        2006,
        "Hong Joo Kim",
        "10.1016/j.physletb.2005.12.035",
        "https://arxiv.org/pdf/astro-ph/0509080",
        "author-supplied paper; KNU Physics affiliation",
        "The author-supplied paper assigns H. J. Kim to Kyungpook "
        "National University Physics; the journal version appeared in 2006.",
    ),
    75: (
        "Three-year annual modulation search with COSINE-100",
        2022,
        "Hong Joo Kim",
        "10.1103/PhysRevD.106.052005",
        "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.052005",
        "APS article; KNU Physics affiliation",
        "The APS article assigns H. J. Kim to Kyungpook National University Physics.",
    ),
    96: (
        "Background model for the NaI (Tl) crystals in COSINE-100",
        2018,
        "Hong Joo Kim",
        "10.1140/epjc/s10052-018-5970-2",
        "https://research.knu.ac.kr/en/publications/background-model-for-the-naitl-crystals-in-cosine-100/",
        "KNU institutional article; author-linked record",
        "The KNU article links its H. J. Kim author to that university "
        "and gives the exact DOI.",
    ),
    107: (
        "Neutrino-Less Double Beta Decay Experiment Using Ca MoO "
        "Scintillation Crystals",
        2010,
        "Hong Joo Kim",
        "10.1109/TNS.2010.2043264",
        "https://research.knu.ac.kr/en/publications/neutrino-less-double-beta-decay-experiment-using-ca-sup100supmoo-/",
        "KNU institutional article; author-linked record",
        "The KNU article links its H. J. Kim author directly to Hong Joo "
        "Kim's profile and gives the exact DOI.",
    ),
    156: (
        "Muon detector for the COSINE-100 experiment",
        2018,
        "Hong Joo Kim",
        "10.1088/1748-0221/13/02/T02007",
        "https://research.knu.ac.kr/en/publications/muon-detector-for-the-cosine-100-experiment/",
        "KNU institutional article; author-linked record",
        "The KNU article links its H. J. Kim author directly to Hong Joo "
        "Kim's profile and gives the exact DOI.",
    ),
    163: (
        "Searching for low-mass dark matter via the Migdal effect in COSINE-100",
        2022,
        "Hong Joo Kim",
        "10.1103/PhysRevD.105.042006",
        "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.042006",
        "APS article; KNU Physics affiliation",
        "The APS article assigns H. J. Kim to Kyungpook National University Physics.",
    ),
    234: (
        "Stable adaptive control with online learning",
        2004,
        "H. Jin Kim",
        "",
        "https://papers.neurips.cc/paper/2623-stable-adaptive-control-with-online-learning.pdf",
        "NeurIPS conference paper",
        "The conference paper names H. Jin Kim of Seoul National "
        "University as its coauthor.",
    ),
    68: (
        "Electrospraying electrospun nanofiber segments into injectable "
        "microspheres for potential cell delivery",
        2018,
        "Hyung Joon Kim",
        "10.1021/acsami.8b06386",
        "https://pubs.acs.org/doi/10.1021/acsami.8b06386",
        "ACS publisher full-author article",
        "The ACS author list names Hyung Joon Kim at the University of Nebraska.",
    ),
    92: (
        "Three-dimensional objects consisting of hierarchically assembled "
        "nanofibers with controlled alignments for regenerative medicine",
        2019,
        "Hyung Joon Kim",
        "10.1021/acs.nanolett.9b00217",
        "https://pubs.acs.org/doi/10.1021/acs.nanolett.9b00217",
        "ACS publisher full-author article",
        "The ACS article names Hyung Joon Kim of the University of Nebraska "
        "as its coauthor.",
    ),
    123: (
        "Establishment of a human iPSC-and nanofiber-based microphysiological "
        "blood–brain barrier system",
        2018,
        "Hyung Joon Kim",
        "10.1021/acsami.8b03962",
        "https://pubs.acs.org/doi/10.1021/acsami.8b03962",
        "ACS publisher full-author article",
        "The ACS article names Hyung Joon Kim of the University of Nebraska "
        "as its coauthor.",
    ),
    153: (
        "REST regulates non–cell-autonomous neuronal differentiation and "
        "maturation of neural progenitor cells via secretogranin II",
        2015,
        "Hyung Joon Kim",
        "10.1523/JNEUROSCI.4286-14.2015",
        "https://www.jneurosci.org/content/jneuro/35/44/14872.full.pdf",
        "Journal of Neuroscience full-author article",
        "The journal's article names Hyung Joon Kim of the Salk Institute "
        "as first author.",
    ),
    173: (
        "Characterisation of tungstate and molybdate crystals ABO4 "
        "(A= Ca, Sr, Zn, Cd; B= W, Mo) for luminescence lifetime cryothermometry",
        2018,
        "Hong Joo Kim",
        "10.1016/j.mtla.2018.09.039",
        "https://research.knu.ac.kr/en/publications/characterisation-of-tungstate-and-molybdate-crystals-abosub4sub-a/",
        "KNU institutional article; author-linked record",
        "The KNU article links H. J. Kim directly to physics professor Hong Joo Kim.",
    ),
    180: (
        "Monitoring the differentiation and migration patterns of neural "
        "cells derived from human embryonic stem cells using a microfluidic "
        "culture system",
        2014,
        "Hyung Joon Kim",
        "10.14348/molcells.2014.0137",
        "https://pubmed.ncbi.nlm.nih.gov/24938227/",
        "PubMed full-author article record",
        "The complete indexed author list names Hyung Joon Kim, not the "
        "CMU Chemistry professor.",
    ),
    339: (
        "Compartmental culture of embryonic stem cell-derived neurons "
        "in microfluidic devices for use in axonal biology",
        2010,
        "Hyung Joon Kim",
        "10.1007/s10529-010-0280-2",
        "https://pubmed.ncbi.nlm.nih.gov/20424889/",
        "PubMed full-author article record",
        "The complete indexed author list names Hyung Joon Kim, not the "
        "CMU Chemistry professor.",
    ),
    389: (
        "Quantitative analysis of CNS axon regeneration using a "
        "microfluidic neuron culture device",
        2008,
        "Hyung Joon Kim",
        "",
        "https://mbel.snu.ac.kr/publication/quantitative-analysis-of-cns-axon-regeneration-using-a-microfluidic-neuron-culture-device/",
        "Seoul National University lab publication record",
        "The producing laboratory's exact publication record names Hyung Joon Kim.",
    ),
    831: (
        "In vitro assay for quantitative analysis of axonal regeneration "
        "in the spinal cord injury using microfluidic platform",
        2009,
        "Hyung Joon Kim",
        "",
        "https://www.rsc.org/binaries/LOC/2009/Pdf/631-W71F.pdf",
        "Royal Society of Chemistry conference paper",
        "The conference paper names Hyung Joon Kim of UC Irvine as first author.",
    ),
    349: (
        "Design and fabrication of the double-sided silicon microstrip sensor",
        2006,
        "Hong Joo Kim",
        "",
        "https://research.knu.ac.kr/en/publications/design-and-fabrication-of-the-double-sided-silicon-microstrip-sen/",
        "KNU institutional article; author-linked record",
        "The KNU article links its H. J. Kim author directly to physics "
        "professor Hong Joo Kim's institutional profile.",
    ),
    590: (
        "Scintillation crystal growth at the CUP",
        2019,
        "Hong Joo Kim",
        "",
        "https://research.knu.ac.kr/en/publications/scintillation-crystal-growth-at-the-cup/",
        "KNU institutional conference article; author-linked record",
        "The KNU conference article links its H. J. Kim author directly to "
        "physics professor Hong Joo Kim's institutional profile.",
    ),
    638: (
        "Study on structure and luminescence properties of LaBMoO6:Dy3+ "
        "phosphor for photonic material applications",
        2018,
        "Hong Joo Kim",
        "",
        "https://research.knu.ac.kr/en/publications/study-on-structure-and-luminescence-properties-of-labmoosub6sub-d/",
        "KNU institutional article; full-name author record",
        "The KNU article names Hong Joo Kim in its full author record "
        "and links him to his physics faculty profile.",
    ),
    546: (
        "Czochralski growth and scintillation properties of Li6LuxY1− x "
        "(BO3) 3: Ce3+ single crystals",
        2016,
        "Hong Joo Kim",
        "10.1016/j.nima.2015.10.008",
        "https://research.knu.ac.kr/en/publications/czochralski-growth-and-scintillation-properties-of-lisub6sublusub/",
        "KNU institutional article; author-linked record",
        "The KNU article links H. J. Kim to physics professor Hong Joo Kim; "
        "KNU's October 2015 online date and the Scholar 2016 year refer "
        "to the same article, printed in January 2016.",
    ),
    605: (
        "Scintillation characteristics of an undoped CsI crystal at "
        "low-temperature for dark matter search",
        2025,
        "Hong Joo Kim",
        "10.1016/j.astropartphys.2025.103150",
        "https://research.knu.ac.kr/en/publications/scintillation-characteristics-of-an-undoped-csi-crystal-at-low-te-2/",
        "KNU institutional article; author-linked record",
        "The KNU article links H. J. Kim directly to physics professor "
        "Hong Joo Kim and matches the 2025 journal publication.",
    ),
    746: (
        "Development of a Dual Cryogenic Detection System for the "
        "Forbidden Non-unique -Decay Spectrum Study",
        2024,
        "Hong Joo Kim",
        "10.1007/s10909-024-03139-5",
        "https://research.knu.ac.kr/en/publications/development-of-a-dual-cryogenic-detection-system-for-the-forbidde/",
        "KNU institutional article; author-linked record",
        "The KNU article links H. J. Kim directly to physics professor "
        "Hong Joo Kim; the Scholar title omits the beta glyph but its "
        "authors, journal, volume, pages, and year match this article.",
    ),
    12: (
        "Cerebral microbleeds are regionally associated with intracerebral hemorrhage",
        2004,
        "Ho Kim",
        "10.1212/01.wnl.0000101463.50798.0d",
        "https://healthstat.snu.ac.kr/hokim/papers.html",
        "Seoul National University author's publication list",
        "SNU researcher Ho Kim lists this exact Neurology article, "
        "volume 62, pages 72–76, among his own publications; the "
        "Scholar H. Kim byline is his.",
    ),
    38: (
        "Comparative analysis of the spatial distribution and severity "
        "of cerebral microbleeds and old lacunes",
        2004,
        "Ho Kim",
        "",
        "https://healthstat.snu.ac.kr/hokim/papers.html",
        "Seoul National University author's publication list",
        "SNU researcher Ho Kim lists this exact Journal of Neurology, "
        "Neurosurgery & Psychiatry article, volume 75, pages 423–427, "
        "under his full name.",
    ),
    40: (
        "Left ventricular hypertrophy is associated with cerebral "
        "microbleeds in hypertensive patients",
        2004,
        "Ho Kim",
        "10.1212/01.wnl.0000132525.36804.a1",
        "https://healthstat.snu.ac.kr/hokim/papers.html",
        "Seoul National University author's publication list",
        "SNU researcher Ho Kim lists this exact Neurology article, "
        "volume 63, pages 16–21, among his own publications; the "
        "Scholar H. Kim byline is his.",
    ),
    106: (
        "Augmentation of NAD+ by NQO1 attenuates cisplatin-mediated hearing impairment",
        2014,
        "Hyung-Jin Kim",
        "10.1038/cddis.2014.255",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4611728/",
        "journal full-text author note",
        "The journal article identifies its first author in full as "
        "Hyung-Jin Kim of Wonkwang University School of Medicine.",
    ),
    81: (
        "High sensitive micro-cantilever sensor and fabricating method thereof",
        2003,
        "Hyung-Joon Kim",
        "",
        "https://patents.google.com/patent/US20030032293A1/en",
        "published US patent application and inventor assignment",
        "The exact US application 10/209,866 names Hyung-Joon Kim of "
        "Korea Institute of Science and Technology as an inventor; "
        "the published assignment also spells out his full name.",
    ),
    85: (
        "Homocysteine, folate and pregnancy outcomes",
        2012,
        "Hai Joong Kim",
        "10.3109/01443615.2012.693984",
        "https://pure.korea.ac.kr/en/publications/homocysteine-folate-and-pregnancy-outcomes/",
        "Korea University institutional full-author article record",
        "The Korea University record names this exact article's H. J. Kim "
        "author in full as Hai Joong Kim.",
    ),
}

# In addition to title and year, freeze each newly reviewed Scholar observation
# ID so a reordered or replaced capture cannot inherit a person decision.
OTHER_PRIMARY_ID_GUARDS = {
    99: "bC4aq-4AAAAJ:cB__R-XWw9UC",
    161: "bC4aq-4AAAAJ:YlPif8NxrbYC",
    142: "bC4aq-4AAAAJ:VOx2b1Wkg3QC",
    175: "bC4aq-4AAAAJ:qsWQJNntlusC",
    88: "bC4aq-4AAAAJ:oi2SiIJ9l4AC",
    148: "bC4aq-4AAAAJ:RfUwGJFMQ-0C",
    267: "bC4aq-4AAAAJ:foquWX3nUaYC",
    109: "bC4aq-4AAAAJ:sbeIDTyQOFgC",
    174: "bC4aq-4AAAAJ:bVQMTfhMCi4C",
    235: "bC4aq-4AAAAJ:M0j1y4EgrScC",
    241: "bC4aq-4AAAAJ:cSdaV2aYdYsC",
}

# CMU links this ORCID to Hyung J. Kim, but the ResearcherID-imported work
# assertion does not independently resolve the initials-only APS author.
CMU_ORCID_DOI_HOLD = {
    1: (
        "Mass spectrum of chiral ten-dimensional N=2 supergravity on",
        1985,
        "10.1103/PhysRevD.32.389",
        "https://pub.orcid.org/v3.0/0000-0003-4334-1879/work/28834334",
    ),
}

# CMU-linked bibliographies or a publisher article identify these older works.
CMU_PRIMARY = {
    9: (
        "Nanoporous carbon supercapacitors in an ionic liquid: "
        "a computer simulation study",
        2010,
        "https://api.crossref.org/works/10.1021%2Fnn901916m",
        "ACS publisher-deposited full Hyung J. Kim byline, ACS Nano 4:2345-2355; "
        "CMU Kim Group bibliography corroborates the exact work",
    ),
    15: (
        "Molecular interactions of a Cu-based metal–organic framework with "
        "a confined imidazolium-based ionic liquid: a combined density "
        "functional theory and experimental vibrational\xa0…",
        2016,
        "https://www.cmu.edu/chemistry/people/faculty/kim.html",
        "CMU faculty publication list; matching authors, venue, and pages",
    ),
    16: (
        "Equilibrium and nonequilibrium solvation and solute electronic "
        "structure. I. Formulation",
        1990,
        "https://api.crossref.org/works/10.1063%2F1.459665",
        "AIP publisher-deposited Crossref record; full Hyung J. Kim byline",
    ),
    21: (
        "Equilibrium and nonequilibrium solvation and solute electronic "
        "structure. III. Quantum theory",
        1992,
        "https://api.crossref.org/works/10.1063%2F1.462752",
        "AIP publisher-deposited Crossref record; full Hyung J. Kim byline",
    ),
    23: (
        "Solvation of carbon nanotubes in a room-temperature ionic liquid",
        2009,
        "https://api.crossref.org/works/10.1021%2Fnn900195b",
        "ACS publisher-deposited full Hyung J. Kim byline, ACS Nano 3:1693-1702; "
        "CMU Kim Group bibliography corroborates the exact work",
    ),
    41: (
        "Graphene-based supercapacitors: A computer simulation study",
        2011,
        "https://api.crossref.org/works/10.1021%2Fjp203458b",
        "ACS publisher-deposited full Hyung J. Kim byline, J. Phys. Chem. C "
        "115:23574-23583; CMU Kim Group bibliography corroborates the exact work",
    ),
    44: (
        "A molecular dynamics computer simulation study of room-temperature "
        "ionic liquids. II. Equilibrium and nonequilibrium solvation dynamics",
        2005,
        "https://www.chem.cmu.edu/groups/kim/pub.html",
        "CMU Kim Group publication list",
    ),
    49: (
        "Molecular interactions in 1-ethyl-3-methylimidazolium acetate ion "
        "pair: a density functional study",
        2009,
        "https://api.crossref.org/works/10.1021%2Fjp907394v",
        "ACS publisher-deposited full Hyung J. Kim byline, J. Phys. Chem. A "
        "113:10397-10404; CMU Kim Group bibliography corroborates the exact work",
    ),
    78: (
        "Graphene-based supercapacitors in the parallel-plate electrode "
        "configuration: Ionic liquids versus organic electrolytes",
        2011,
        "https://pubs.rsc.org/en/content/articlelanding/2012/fd/c1fd00086a",
        "RSC original article with CMU author affiliation",
    ),
    79: (
        "Equilibrium and nonequilibrium solvation and solute electronic "
        "structure. II. Strong coupling limit",
        1990,
        "https://api.crossref.org/works/10.1063%2F1.459666",
        "AIP publisher-deposited Crossref record; full Hyung J. Kim byline "
        "and Colorado affiliation",
    ),
    89: (
        "Solvation, solute rotation and vibration relaxation, and "
        "electron-transfer reactions in room-temperature ionic liquids",
        2007,
        "https://api.crossref.org/works/10.1021%2Far700061r",
        "ACS publisher-deposited full Hyung J. Kim byline, Acc. Chem. Res. "
        "40:1130-1137; CMU Kim Group bibliography corroborates the exact work",
    ),
    104: (
        "Excited state intramolecular charge transfer rates of "
        "p-dirnethylaminobenzonitrile (DMABN) in solution: a "
        "two-dimensional dynamics perspective",
        1997,
        "https://experts.colorado.edu/vitas/106076.pdf",
        "Coauthor James T. Hynes's Colorado CV; same journal and pages, "
        "with the chemical name corrected",
    ),
    111: (
        "Free energy and dynamics of electron-transfer reactions in a room "
        "temperature ionic liquid",
        2007,
        "https://api.crossref.org/works/10.1021%2Fjp0703859",
        "ACS publisher-deposited full Hyung J. Kim byline, J. Phys. Chem. B "
        "111:4510-4519; CMU Kim Group bibliography corroborates the exact work",
    ),
    145: (
        "Fragility, Stokes–Einstein violation, and correlated local excitations "
        "in a coarse-grained model of an ionic liquid",
        2010,
        "https://api.crossref.org/works/10.1039%2FB921725H",
        "RSC publisher-deposited full Hyung J. Kim byline, PCCP 12:2001; "
        "CMU Kim Group bibliography confirms pages 2001-2010",
    ),
    155: (
        "Electronically adiabatic reaction field approach to solvation. "
        "I. Theoretical formulation via multipole expansion in a "
        "fluctuating cavity",
        1996,
        "https://api.crossref.org/works/10.1063%2F1.472531",
        "AIP publisher-deposited Crossref record; full Hyung J. Kim "
        "byline and CMU Chemistry affiliation",
    ),
    166: (
        "Effects of solute electronic polarizability on solvation in a "
        "room-temperature ionic liquid",
        2007,
        "https://api.crossref.org/works/10.1021%2Fjp067316z",
        "ACS publisher-deposited full Hyung J. Kim byline, J. Phys. Chem. B "
        "111:4920-4925; CMU Kim Group bibliography corroborates the exact work",
    ),
    220: (
        "Electronically adiabatic reaction field approach to solvation. "
        "II. Solvent effects on electronic spectra",
        1996,
        "https://api.crossref.org/works/10.1063%2F1.471831",
        "AIP publisher-deposited Crossref record; full Hyung J. Kim "
        "byline and CMU Chemistry affiliation",
    ),
    479: (
        "Temperature-and pressure-dependence of the outer-sphere "
        "reorganization free energy for electron transfer reactions: "
        "a continuum approach",
        2006,
        "https://www.chem.cmu.edu/groups/kim/pub.html",
        "CMU Kim Group publication list",
    ),
    480: (
        "Electrostriction effects on electron transfer reactions in solution. "
        "I. Adiabatic regime",
        1997,
        "https://api.crossref.org/works/10.1063%2F1.473606",
        "AIP publisher-deposited DOI record; full Hyung J. Kim byline and "
        "CMU Chemistry affiliation",
    ),
    587: (
        "Mass Spectrometry of Au10(4-tert-butylbenzenethiolate)10 "
        "Nanoclusters Using Superconducting Tunnel Junction Cryodetection "
        "Reveals Distinct Metastable\xa0…",
        2022,
        "https://pubs.acs.org/doi/abs/10.1021/jasms.1c00346",
        "ACS article; full Hyung J. Kim byline, CMU Chemistry affiliation, "
        "and linked CMU ORCID",
    ),
    643: (
        "Correction to “Graphene-Based Supercapacitors: A Computer Simulation Study”",
        2012,
        "https://pubs.acs.org/toc/jpccck/116/34",
        "ACS published correction DOI 10.1021/jp307086f; full Hyung J. "
        "Kim byline, CMU Chemistry affiliation, and separate pages",
    ),
    709: (
        "Correction: A mechanistic insight into the organocatalytic "
        "properties of imidazolium-based ionic liquids and a positive "
        "co-solvent effect on cellulose modification reactions in\xa0…",
        2017,
        "https://pubs.rsc.org/en/content/articlelanding/2017/ra/c7ra90031g",
        "RSC published correction DOI 10.1039/C7RA90031G; full Hyung "
        "J. Kim byline, CMU Chemistry affiliation, and separate page",
    ),
    796: (
        "Correction: A molecular dynamics study of the ionic liquid, choline acetate",
        2017,
        "https://pubs.rsc.org/en/content/articlelanding/2017/cp/c7cp90081c",
        "RSC published correction DOI 10.1039/C7CP90081C; full Hyung "
        "J. Kim byline, CMU Chemistry affiliation, and separate page",
    ),
    # CMU Kim Group entries checked against publisher-deposited DOI metadata.
    63: (
        "Dielectric relaxation, ion conductivity, solvent rotation, and "
        "solvation dynamics in a room-temperature ionic liquid",
        2008,
        "https://api.crossref.org/works/10.1021%2Fjp802595r",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. B 112:11028-11038 agree",
    ),
    86: (
        "Electronic structure and normal vibrations of the "
        "1-ethyl-3-methylimidazolium ethyl sulfate ion pair",
        2011,
        "https://api.crossref.org/works/10.1021%2Fjp1122322",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. A 115:3551-3558 agree",
    ),
    170: (
        "On the photoabsorption spectroscopy of water",
        2000,
        "https://api.crossref.org/works/10.1021%2Fjp992582g",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and 2000 print issue, J. Phys. Chem. A 104:45-52, agree",
    ),
    232: (
        "Adiabatic electron transfer in a room-temperature ionic liquid: "
        "Reaction dynamics and kinetics",
        2009,
        "https://api.crossref.org/works/10.1021%2Fjp9065407",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. B 113:12964-12972 agree",
    ),
    254: (
        "Free Energies of Electron Transfer Reactions in Polarizable, "
        "Nondipolar, Quadrupolar Solvents",
        2000,
        "https://api.crossref.org/works/10.1021%2Fjp002146q",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. A 104:9812-9815 agree",
    ),
    273: (
        "MD Study of Solvation in the Mixture of a Room-Temperature Ionic "
        "Liquid and CO2",
        2010,
        "https://api.crossref.org/works/10.1021%2Fjp105021b",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. B 114:10160-10170 agree",
    ),
    281: (
        "Carbon nanotubes in benzene: internal and external solvation",
        2011,
        "https://api.crossref.org/works/10.1039%2FC0CP01845G",
        "RSC full Hyung J. Kim byline and PCCP 13:3969 first page; "
        "CMU group list confirms pages 3969-3978",
    ),
    468: (
        "MD Study of SN1 Reactivity of 2-Chloro-2-methylpropane in the "
        "Room-Temperature Ionic Liquid 1-Ethyl-3-methylimidazolium "
        "Hexafluorophosphate",
        2008,
        "https://api.crossref.org/works/10.1021%2Fjp710128p",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Phys. Chem. B 112:2637-2643 agree",
    ),
    687: (
        "Electron-transfer reactions in supercritical water",
        2008,
        "https://api.crossref.org/works/10.1021%2Fjp075270j",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and 2008 print issue, J. Phys. Chem. B 112:585-594, agree",
    ),
    723: (
        "Solubility of Water in a Benzene− Cyclohexane Mixture",
        2006,
        "https://api.crossref.org/works/10.1021%2Fjp053578%2B",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and 2006 print issue, J. Phys. Chem. A 110:429-435, agree",
    ),
    # CMU group entries checked against AIP, PLOS, and BKCS DOI records.
    25: (
        "Solvation in molecular ionic liquids",
        2003,
        "https://api.crossref.org/works/10.1063%2F1.1611875",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 119:6411-6414 agree",
    ),
    67: (
        "A molecular dynamics computer simulation study of room-temperature "
        "ionic liquids. I. Equilibrium solvation structure and free energetics",
        2005,
        "https://api.crossref.org/works/10.1063%2F1.1819317",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 122:044510 agree",
    ),
    214: (
        "Rotational dynamics of a diatomic solute in the room-temperature "
        "ionic liquid 1-ethyl-3-methylimidazolium hexafluorophosphate",
        2006,
        "https://api.crossref.org/works/10.1063%2F1.2232303",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 125:061102 agree",
    ),
    297: (
        "Solvation in supercritical water",
        2006,
        "https://api.crossref.org/works/10.1063%2F1.2194012",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 124:204504 agree",
    ),
    348: (
        "1∕ f spectrum and memory function analysis of solvation dynamics "
        "in a room-temperature ionic liquid",
        2008,
        "https://api.crossref.org/works/10.1063%2F1.2911927",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 128:174504 agree",
    ),
    352: (
        "Dielectric relaxation of hot water",
        2000,
        "https://api.crossref.org/works/10.1063%2F1.1315991",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 113:6025-6028 agree",
    ),
    377: (
        "Vibrational energy relaxation of a diatomic molecule in a "
        "room-temperature ionic liquid",
        2006,
        "https://api.crossref.org/works/10.1063%2F1.2206579",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 125:024507 agree",
    ),
    405: (
        "Conformational dynamics and ligand binding in the multi-domain protein PDC109",
        2010,
        "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009180",
        "PLOS original article distinguishes Hyun Jin Kim from Hyung J. Kim, "
        "names CMU Chemistry and hjkim@cmu.edu; CMU group list agrees",
    ),
    517: (
        "On the temperature and pressure dependences of cavities in the "
        "dielectric continuum picture",
        2005,
        "https://api.crossref.org/works/10.1063%2F1.1948375",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 123:014504 agree",
    ),
    596: (
        "Solvation of a small metal-binding peptide in room-temperature ionic liquids",
        2012,
        "https://api.crossref.org/works/10.5012%2Fbkcs.2012.33.11.3601",
        "BKCS DOI record names Hyung J. Kim in full; CMU group list "
        "confirms Bull. Korean Chem. Soc. 33:3601-3606",
    ),
    691: (
        "Free energy, entropy and volume of activation for electron "
        "transfer reactions in a polar solvent",
        2006,
        "https://api.crossref.org/works/10.1063%2F1.2217941",
        "AIP full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU group list and J. Chem. Phys. 125:011101 agree",
    ),
    # Recent CMU faculty bibliography entries checked against DOI registrations.
    7: (
        "Gold Nanoclusters Promote Electrocatalytic Water Oxidation at the "
        "Nanocluster/CoSe2 Interface",
        2017,
        "https://api.crossref.org/works/10.1021%2Fjacs.6b12529",
        "ACS full Hyung J. Kim byline, CMU Chemistry affiliation and "
        "his ORCID; CMU faculty list and JACS 139:1077-1080 agree",
    ),
    52: (
        "Molecular dynamics study of water flow across multiple layers "
        "of pristine, oxidized, and mixed regions of graphene oxide",
        2017,
        "https://api.crossref.org/works/10.1021%2Facsnano.6b08538",
        "ACS full Hyung J. Kim byline, CMU Chemistry affiliation and "
        "his ORCID; CMU faculty list and ACS Nano 11:2187-2193 agree",
    ),
    64: (
        "Influence of Water on the Chemistry and Structure of the "
        "Metal–Organic Framework Cu3(btc)2",
        2016,
        "https://api.crossref.org/works/10.1021%2Facs.jpcc.6b02906",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and J. Phys. Chem. C 120:17323-17333 agree",
    ),
    71: (
        "Shuttling single metal atom into and out of a metal nanoparticle",
        2017,
        "https://api.crossref.org/works/10.1038%2Fs41467-017-00939-0",
        "Nature-deposited full Hyung J. Kim byline and ORCID; CMU faculty "
        "list links the exact DOI and Nat. Commun. 8:848",
    ),
    149: (
        "Computer simulation study of graphene oxide supercapacitors: "
        "charge screening mechanism",
        2016,
        "https://api.crossref.org/works/10.1021%2Facs.jpclett.6b00202",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and J. Phys. Chem. Lett. 7:1180-1186 agree",
    ),
    158: (
        "A mechanistic insight into the organocatalytic properties of "
        "imidazolium-based ionic liquids and a positive co-solvent effect "
        "on cellulose modification reactions in an ionic liquid",
        2017,
        "https://api.crossref.org/works/10.1039%2Fc6ra28659c",
        "RSC full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and RSC Adv. 7:9423-9430 agree",
    ),
    236: (
        "CO2 capture in ionic liquid 1-alkyl-3-methylimidazolium acetate: "
        "a concerted mechanism without carbene",
        2017,
        "https://api.crossref.org/works/10.1039%2Fc6cp06556b",
        "RSC full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and PCCP 19:1361-1368 agree",
    ),
    261: (
        "Molecular dynamics study of water flow across multiple layers "
        "of pristine, oxidized, and mixed regions of graphene oxide: "
        "effect of graphene oxide layer-to-layer distance",
        2017,
        "https://api.crossref.org/works/10.1021%2Facs.jpcc.7b06063",
        "ACS full Hyung J. Kim byline, CMU Chemistry affiliation and "
        "his ORCID; CMU faculty list and J. Phys. Chem. C 121:23659-23668 agree",
    ),
    375: (
        "Theoretical study of interactions of a Li+(CF3SO2)2N− ion pair "
        "with CR3(OCR2CR2)nOCR3 (R = H or F)",
        2016,
        "https://api.crossref.org/works/10.1039%2Fc6cp00139d",
        "RSC full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and PCCP 18:6754-6762 agree",
    ),
    383: (
        "Spectroscopic and MD study of dynamic and structural "
        "heterogeneities in ionic liquids",
        2017,
        "https://api.crossref.org/works/10.1021%2Facs.jpcb.6b10678",
        "ACS full Hyung J. Kim byline and CMU Chemistry affiliation; "
        "CMU faculty list and J. Phys. Chem. B 121:1100-1107 agree",
    ),
}

# Guard newly resolved source rows with their immutable Scholar citation IDs.
CMU_PRIMARY_ID_GUARDS = {
    9: "bC4aq-4AAAAJ:d1gkVwhDpl0C",
    23: "bC4aq-4AAAAJ:Tyk-4Ss8FVUC",
    41: "bC4aq-4AAAAJ:kNdYIx-mwKoC",
    49: "bC4aq-4AAAAJ:LkGwnXOMwfcC",
    89: "bC4aq-4AAAAJ:W7OEmFMy1HYC",
    111: "bC4aq-4AAAAJ:_FxGoFyzp5QC",
    145: "bC4aq-4AAAAJ:9ZlFYXVOiuMC",
    166: "bC4aq-4AAAAJ:3fE2CSJIrl8C",
    480: "bC4aq-4AAAAJ:R3hNpaxXUhUC",
    587: "bC4aq-4AAAAJ:u0Mu_IsstPMC",
    643: "bC4aq-4AAAAJ:hMod-77fHWUC",
    709: "bC4aq-4AAAAJ:abG-DnoFyZgC",
    796: "bC4aq-4AAAAJ:4fKUyHm3Qg0C",
    63: "bC4aq-4AAAAJ:Y0pCki6q_DkC",
    86: "bC4aq-4AAAAJ:ULOm3_A8WrAC",
    170: "bC4aq-4AAAAJ:0EnyYjriUFMC",
    232: "bC4aq-4AAAAJ:mVmsd5A6BfQC",
    254: "bC4aq-4AAAAJ:Zph67rFs4hoC",
    273: "bC4aq-4AAAAJ:IWHjjKOFINEC",
    281: "bC4aq-4AAAAJ:HDshCWvjkbEC",
    468: "bC4aq-4AAAAJ:TQgYirikUcIC",
    687: "bC4aq-4AAAAJ:isC4tDSrTZIC",
    723: "bC4aq-4AAAAJ:blknAaTinKkC",
    25: "bC4aq-4AAAAJ:u-x6o8ySG0sC",
    67: "bC4aq-4AAAAJ:UeHWp8X0CEIC",
    214: "bC4aq-4AAAAJ:aqlVkmm33-oC",
    297: "bC4aq-4AAAAJ:4JMBOYKVnBMC",
    348: "bC4aq-4AAAAJ:hFOr9nPyWt4C",
    352: "bC4aq-4AAAAJ:L8Ckcad2t8MC",
    377: "bC4aq-4AAAAJ:mB3voiENLucC",
    405: "bC4aq-4AAAAJ:M3NEmzRMIkIC",
    517: "bC4aq-4AAAAJ:_Qo2XoVZTnwC",
    596: "bC4aq-4AAAAJ:k_IJM867U9cC",
    691: "bC4aq-4AAAAJ:maZDTaKrznsC",
    7: "bC4aq-4AAAAJ:b0M2c_1WBrUC",
    52: "bC4aq-4AAAAJ:dshw04ExmUIC",
    64: "bC4aq-4AAAAJ:P5F9QuxV20EC",
    71: "bC4aq-4AAAAJ:eflP2zaiRacC",
    149: "bC4aq-4AAAAJ:SeFeTyx0c_EC",
    158: "bC4aq-4AAAAJ:bFI3QPDXJZMC",
    236: "bC4aq-4AAAAJ:D03iK_w7-QYC",
    261: "bC4aq-4AAAAJ:B3FOqHPlNUQC",
    375: "bC4aq-4AAAAJ:EUQCXRtRnyEC",
    383: "bC4aq-4AAAAJ:nb7KW1ujOQ8C",
}

CMU_GROUP_PUBLISHER_DOI_GUARDS = {
    9: "10.1021/nn901916m",
    23: "10.1021/nn900195b",
    41: "10.1021/jp203458b",
    49: "10.1021/jp907394v",
    89: "10.1021/ar700061r",
    111: "10.1021/jp0703859",
    145: "10.1039/b921725h",
    166: "10.1021/jp067316z",
    63: "10.1021/jp802595r",
    86: "10.1021/jp1122322",
    170: "10.1021/jp992582g",
    232: "10.1021/jp9065407",
    254: "10.1021/jp002146q",
    273: "10.1021/jp105021b",
    281: "10.1039/c0cp01845g",
    468: "10.1021/jp710128p",
    687: "10.1021/jp075270j",
    723: "10.1021/jp053578+",
    25: "10.1063/1.1611875",
    67: "10.1063/1.1819317",
    214: "10.1063/1.2232303",
    297: "10.1063/1.2194012",
    348: "10.1063/1.2911927",
    352: "10.1063/1.1315991",
    377: "10.1063/1.2206579",
    405: "10.1371/journal.pone.0009180",
    517: "10.1063/1.1948375",
    596: "10.5012/bkcs.2012.33.11.3601",
    691: "10.1063/1.2217941",
    7: "10.1021/jacs.6b12529",
    52: "10.1021/acsnano.6b08538",
    64: "10.1021/acs.jpcc.6b02906",
    71: "10.1038/s41467-017-00939-0",
    149: "10.1021/acs.jpclett.6b00202",
    158: "10.1039/c6ra28659c",
    236: "10.1039/c6cp06556b",
    261: "10.1021/acs.jpcc.7b06063",
    375: "10.1039/c6cp00139d",
    383: "10.1021/acs.jpcb.6b10678",
}

# The later Scholar row has a broken superscript title and no page number.
# The publisher's one 1985 article at PRD 32, 389 is row 1; its person
# attribution remains held separately from this duplicate-work decision.
CORRUPTED_DUPLICATE_PRIMARY = {
    793: (
        "COSINE-100 and DAMA/LIBRA-phase2 in WIMP effective models",
        2019,
        "https://researchportal.helsinki.fi/en/publications/"
        "cosine-100-and-damalibra-phase2-in-wimp-effective-models/",
        "bC4aq-4AAAAJ:gmHTDCtJMcoC",
    ),
    795: (
        "Muon detector for the COSINE-100 experiment",
        2018,
        "https://research.knu.ac.kr/en/publications/muon-detector-for-the-cosine-100-experiment/",
        "bC4aq-4AAAAJ:hNSvKAmkeYkC",
    ),
    797: (
        "Synthesis, conductivity, and vibrational spectroscopy of "
        "tetraphenylphosphonium bis (trifluoromethanesulfonyl) imide",
        2017,
        "https://www.sciencedirect.com/science/article/pii/S0022286017307585",
        "bC4aq-4AAAAJ:738O_yMBCRsC",
    ),
    907: (
        "Mass spectrum of chiral ten-dimensional N= 2 supergravity on S/sup 5",
        1985,
        "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.32.389",
        "bC4aq-4AAAAJ:SP6oXDckpogC",
    ),
}

ACS_PARENT_URL = "https://pubs.acs.org/doi/10.1021/acs.jpclett.6b00202"
ACS_PARENT_ID = f"{SCHOLAR_ID}:SeFeTyx0c_EC"
CORRECTION_PREFIX = re.compile(
    r"(?i)^(correction|erratum|corrigendum|comment on|reply to)"
)


def normalized_title(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value)).lower()
    value = unicodedata.normalize("NFKD", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def scholar_pub_id(row: dict) -> str:
    match = re.search(r"citation_for_view=([^&#]+)", row["citation_url"])
    if not match:
        raise ValueError(f"Missing citation ID for row {row['index']}")
    return match.group(1)


def orcid_index(profile: dict) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for work in profile["works"]:
        key = normalized_title(work["title"])
        if key:
            result[key].append(work)
    return result


def decision(
    pub_id: str,
    reason_code: str,
    reason: str,
    url: str,
    *,
    verified_at: str = "2026-10-24",
) -> dict:
    return {
        "scholar_id": SCHOLAR_ID,
        "author_pub_id": pub_id,
        "reason_code": reason_code,
        "reason": reason,
        "evidence_url": url,
        "verified_at": verified_at,
    }


def build(capture: dict, evidence: dict) -> tuple[list[dict], list[dict]]:
    rows = capture["works"]
    if len(rows) != 918 or capture["more_remain"]:
        raise ValueError("Expected completed 918-row Kim capture")
    profiles = {item["orcid"]: item for item in evidence["profiles"]}
    cmu = orcid_index(profiles[CMU_ORCID])
    knu = orcid_index(profiles[KNU_ORCID])
    duplicate_titles = defaultdict(list)
    for row in rows:
        title_key = normalized_title(row["title"])
        if title_key:
            duplicate_titles[title_key].append(row["index"])
    additions = []
    screening = []
    for row in rows:
        index, title, year = row["index"], row["title"], row["year"]
        pub_id = scholar_pub_id(row)
        key = normalized_title(title)
        cmu_hits, knu_hits = cmu.get(key, []), knu.get(key, [])
        knu_exact = next((w for w in knu_hits if w["year"] == year and w["doi"]), None)
        knu_year_only = next((w for w in knu_hits if w["year"] == year), None)
        cmu_exact = next((w for w in cmu_hits if w["year"] == year), None)
        if cmu_hits and (knu_exact or index in NEUROSCIENCE or index in OTHER_PRIMARY):
            raise ValueError(f"Conflicting person evidence at capture row {index}")
        doi, evidence_url, evidence_source = "", "", ""
        if index in BASELINE_EXCLUDED_INDICES:
            status = "already_excluded"
        elif knu_exact:
            status = "proposed_other_person_knu_orcid_title_year_record_doi"
            doi = knu_exact["doi"]
            evidence_source = knu_exact["source_name"]
            evidence_url = (
                f"https://pub.orcid.org/v3.0/{KNU_ORCID}/work/{knu_exact['put_code']}"
            )
            additions.append(
                decision(
                    pub_id,
                    "false_profile_attribution",
                    (
                        f"The Scholar observation's {year} title '{title}' "
                        "matches Kyungpook Physics professor Hong Joo Kim's "
                        f"separate ORCID work record, which supplies DOI {doi}; "
                        "the H. J. Kim on this observation is not CMU Chemistry "
                        "professor Hyung J. Kim."
                    ),
                    evidence_url,
                )
            )
        elif index in NEUROSCIENCE:
            status = "proposed_other_person_hyung_joon_primary"
            evidence_source = "primary author record"
            evidence_url = NEUROSCIENCE[index]
            additions.append(
                decision(
                    pub_id,
                    "false_profile_attribution",
                    (
                        f"The primary author record for the {year} work "
                        f"'{title}' names "
                        "Hyung Joon Kim, not CMU Chemistry professor Hyung J. Kim."
                    ),
                    evidence_url,
                )
            )
        elif index in OTHER_PRIMARY:
            (
                expected_title,
                expected_year,
                other_person,
                doi,
                evidence_url,
                evidence_source,
                basis,
            ) = OTHER_PRIMARY[index]
            if (
                title != expected_title
                or year != str(expected_year)
                or (
                    index in OTHER_PRIMARY_ID_GUARDS
                    and pub_id != OTHER_PRIMARY_ID_GUARDS[index]
                )
                or (
                    index == 88
                    and (
                        row["venue"]
                        != "Journal of Alloys and Compounds 749, 197-204, 2018"
                        or not all(
                            author in row["authors"]
                            for author in ("K Kirdsiri", "RR Ramakrishna", "B Damdee")
                        )
                    )
                )
                or (
                    index == 99
                    and (
                        row["venue"]
                        != "Journal of food protection 69 (7), 1653-1661, 2006"
                        or row["authors"]
                        != "HJ Kim, SH Park, TH Lee, BH Nahm, YH Chung, KH Seo, HY Kim"
                    )
                )
                or (
                    index == 161
                    and (
                        row["venue"]
                        != "Korean J Soc Food Cookery Sci 18 (6), 644-654, 2002"
                        or row["authors"] != "HJ Kim, KH Sohn, SH Chae, TK Kwak, SK Yim"
                    )
                )
                or (
                    index == 142
                    and (
                        row["venue"]
                        != "Mine Water and the Environment 25 (4), 227-232, 2006"
                        or row["authors"]
                        != "JE Yang, JG Skousen, YS Ok, KY Yoo, HJ Kim"
                    )
                )
                or (
                    index == 175
                    and (
                        row["venue"]
                        != "IEEE Transactions on Nuclear Science 63 (2), 439-442, 2016"
                        or row["authors"] != "HJ Kim, G Rooh, H Park, S Kim"
                    )
                )
            ):
                raise ValueError(f"Primary-source match changed at capture row {index}")
            status = "proposed_other_person_primary_work"
            additions.append(
                decision(
                    pub_id,
                    "false_profile_attribution",
                    (
                        f"{basis} This exact {year} Scholar observation "
                        f"'{title}' belongs to {other_person}, not CMU Chemistry "
                        f"professor Hyung J. Kim."
                        + (f" Article DOI {doi}." if doi else "")
                    ),
                    evidence_url,
                    verified_at=(
                        "2026-10-25"
                        if index in OTHER_PRIMARY_ID_GUARDS
                        else "2026-10-24"
                    ),
                )
            )
        elif index == 917:
            status = "proposed_acs_supporting_information_nonpublication"
            evidence_source = "ACS parent article"
            evidence_url = ACS_PARENT_URL
            doi = "10.1021/acs.jpclett.6b00202"
            additions.append(
                decision(
                    pub_id,
                    "nonpublication",
                    (
                        "This undated Scholar row is the Supporting Information "
                        f"attached to the 2016 ACS paper DOI {doi}, already "
                        f"represented by {ACS_PARENT_ID}; "
                        "it is not a separate publication."
                    ),
                    evidence_url,
                )
            )
        elif index in CORRUPTED_DUPLICATE_PRIMARY:
            (
                expected_title,
                expected_year,
                evidence_url,
                retained_pub_id,
            ) = CORRUPTED_DUPLICATE_PRIMARY[index]
            if index == 907:
                valid_duplicate = (
                    row["authors"] == rows[0]["authors"]
                    and "32 (2)" in row["venue"]
                    and scholar_pub_id(rows[0]) == retained_pub_id
                )
            elif index == 793:
                retained = rows[335]
                valid_duplicate = (
                    pub_id == f"{SCHOLAR_ID}:VnuxuLaQPLMC"
                    and title == retained["title"]
                    and year == retained["year"]
                    and row["venue"]
                    == "Journal of Cosmology and Astroparticle Physics 2019 (06), 2019"
                    and retained["venue"]
                    == (
                        "Journal of Cosmology and Astroparticle Physics "
                        "2019 (06), 048-048, 2019"
                    )
                    and all(
                        author in row["authors"] and author in retained["authors"]
                        for author in ("G Adhikari", "P Adhikari", "N Carlin", "S Choi")
                    )
                    and scholar_pub_id(retained) == retained_pub_id
                )
            elif index == 795:
                retained = rows[155]
                valid_duplicate = (
                    pub_id == f"{SCHOLAR_ID}:pUxgyZctzPYC"
                    and title == retained["title"]
                    and year == retained["year"]
                    and row["venue"] == "Journal of Instrumentation 13 (02), 2018"
                    and retained["venue"].startswith(
                        "Journal of Instrumentation 13 (02), T02007"
                    )
                    and all(
                        author in row["authors"] and author in retained["authors"]
                        for author in ("H Prihtiadi", "G Adhikari", "P Adhikari")
                    )
                    and scholar_pub_id(retained) == retained_pub_id
                )
            else:
                valid_duplicate = (
                    index == 797
                    and title == rows[311]["title"]
                    and year == rows[311]["year"]
                    and not row["venue"]
                    and not row["citation_count"]
                    and "HJ Kim" in row["authors"]
                    and scholar_pub_id(rows[311]) == retained_pub_id
                )
            if (
                title != expected_title
                or year != str(expected_year)
                or not valid_duplicate
            ):
                raise ValueError(f"Source duplicate match changed at row {index}")
            status = "proposed_corrupted_duplicate_primary"
            evidence_source = {
                793: "Helsinki institutional article record",
                795: "KNU institutional article",
                797: "Elsevier original article",
                907: "APS original article",
            }[index]
            additions.append(
                decision(
                    pub_id,
                    "corrupted_duplicate",
                    (
                        "The initials, coauthors, 1985 year, Physical Review D "
                        "volume 32 issue 2, and malformed S/sup 5 title point to "
                        "the one APS article represented by Scholar observation "
                        f"{retained_pub_id}; this row is a broken duplicate."
                        if index == 907
                        else (
                            "This Scholar row repeats the 2018 Journal of "
                            "Instrumentation title, volume 13 issue 2, and leading "
                            "coauthors of observation "
                            f"{retained_pub_id}; the KNU institutional record "
                            "identifies one article, T02007, whose H. J. Kim "
                            "author links to Hong Joo Kim. This row is a "
                            "redundant citation observation."
                            if index == 795
                            else (
                                "This Scholar row repeats the 2019 JCAP title, issue, "
                                "and leading coauthors of observation "
                                f"{retained_pub_id} but omits article number 048. "
                                "The institutional record identifies one journal "
                                "article, DOI 10.1088/1475-7516/2019/06/048. "
                                "This row is a redundant citation observation."
                                if index == 793
                                else "The blank-venue Scholar row repeats the "
                                "exact title "
                                "and year of the complete 2017 journal observation "
                                f"{retained_pub_id}; Elsevier publishes one article at "
                                "DOI 10.1016/j.molstruc.2017.05.138 with the full "
                                "Hyung J. Kim byline. This row is a broken duplicate."
                            )
                        )
                    ),
                    evidence_url,
                    verified_at="2026-10-25"
                    if index in {793, 795, 797}
                    else "2026-10-24",
                )
            )
        elif index in CMU_ORCID_DOI_HOLD:
            expected_title, expected_year, doi, evidence_url = CMU_ORCID_DOI_HOLD[index]
            if title != expected_title or year != str(expected_year):
                raise ValueError(f"CMU ORCID DOI match changed at capture row {index}")
            status = "cmu_orcid_doi_identity_hold"
            evidence_source = (
                "CMU-linked ORCID work; ResearcherID import; author unresolved"
            )
        elif index in CMU_PRIMARY:
            (
                expected_title,
                expected_year,
                evidence_url,
                evidence_source,
            ) = CMU_PRIMARY[index]
            if (
                title != expected_title
                or year != str(expected_year)
                or (
                    index in CMU_PRIMARY_ID_GUARDS
                    and pub_id != CMU_PRIMARY_ID_GUARDS[index]
                )
            ):
                raise ValueError(f"CMU primary match changed at capture row {index}")
            if index in CMU_GROUP_PUBLISHER_DOI_GUARDS:
                doi = CMU_GROUP_PUBLISHER_DOI_GUARDS[index]
                if not cmu_exact or cmu_exact["doi"].lower() != doi:
                    raise ValueError(f"CMU group publisher DOI changed at row {index}")
            status = "cmu_primary_work_retained"
        elif cmu_exact:
            status = "cmu_orcid_exact_title_year_retained"
            doi = cmu_exact["doi"]
            evidence_source = cmu_exact["source_name"]
            evidence_url = (
                f"https://pub.orcid.org/v3.0/{CMU_ORCID}/work/{cmu_exact['put_code']}"
            )
        elif cmu_hits:
            status = "cmu_orcid_title_only_year_check"
            doi = cmu_hits[0]["doi"]
            evidence_source = cmu_hits[0]["source_name"]
            evidence_url = (
                f"https://pub.orcid.org/v3.0/{CMU_ORCID}/work/{cmu_hits[0]['put_code']}"
            )
        elif knu_year_only:
            status = "knu_orcid_exact_title_year_missing_doi_hold"
            evidence_source = knu_year_only["source_name"]
            evidence_url = f"https://pub.orcid.org/v3.0/{KNU_ORCID}/work/{knu_year_only['put_code']}"
        elif knu_hits:
            status = "knu_orcid_title_only_year_check"
            doi = knu_hits[0]["doi"]
            evidence_source = knu_hits[0]["source_name"]
            evidence_url = (
                f"https://pub.orcid.org/v3.0/{KNU_ORCID}/work/{knu_hits[0]['put_code']}"
            )
        else:
            status = "unresolved_attribution_hold"
        duplicates = duplicate_titles[key] if key else []
        flags = [
            name
            for name, flagged in (
                ("blank_author", not row["authors"]),
                ("blank_year", not year),
                ("blank_venue", not row["venue"]),
                ("repeated_title_review", len(duplicates) > 1),
                (
                    "supporting_information_review",
                    title.lower().startswith("supporting information"),
                ),
                ("correction_review", bool(CORRECTION_PREFIX.match(title))),
            )
            if flagged
        ]
        screening.append(
            {
                "capture_index": index,
                "author_pub_id": pub_id,
                "source_title": title,
                "source_authors": row["authors"],
                "source_year": year,
                "source_venue": row["venue"],
                "citation_count": row["citation_count"],
                "status": status,
                "matched_doi": doi,
                "evidence_url": evidence_url,
                "evidence_source": evidence_source,
                "duplicate_title_indices": ";".join(map(str, duplicates))
                if len(duplicates) > 1
                else "",
                "source_flags": ";".join(flags),
            }
        )
    counts = Counter(item["status"] for item in screening)
    expected = {
        "already_excluded": 8,
        "cmu_orcid_exact_title_year_retained": 60,
        "cmu_orcid_doi_identity_hold": 1,
        "cmu_primary_work_retained": 54,
        "knu_orcid_exact_title_year_missing_doi_hold": 2,
        "knu_orcid_title_only_year_check": 2,
        "proposed_other_person_hyung_joon_primary": 15,
        "proposed_other_person_knu_orcid_title_year_record_doi": 320,
        "proposed_other_person_primary_work": 46,
        "proposed_acs_supporting_information_nonpublication": 1,
        "proposed_corrupted_duplicate_primary": 4,
        "unresolved_attribution_hold": 405,
    }
    if counts != expected:
        raise ValueError(f"Screening counts changed: {dict(counts)}")
    addition_ids = [item["author_pub_id"] for item in additions]
    capture_ids = [item["author_pub_id"] for item in screening]
    if len(set(addition_ids)) != 386 or len(set(capture_ids)) != 918:
        raise ValueError("Duplicate Scholar IDs in additions or capture")
    if not set(addition_ids) <= set(capture_ids):
        raise ValueError("Proposed ID absent from capture")
    return additions, screening


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, required=True, help="Raw JSON from source PR #118"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Append missing exact decisions and write ledger",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Compare manifest and ledger to rebuilt decisions",
    )
    args = parser.parse_args()
    additions, screening = build(
        json.loads(args.capture.read_text()), json.loads(EVIDENCE.read_text())
    )
    manifest = json.loads(MANIFEST.read_text())
    existing = {item["author_pub_id"]: item for item in manifest["exclusions"]}
    baseline = {
        scholar_pub_id(row)
        for row in json.loads(args.capture.read_text())["works"]
        if row["index"] in BASELINE_EXCLUDED_INDICES
    }
    if not baseline <= set(existing):
        raise ValueError("The eight baseline Kim decisions from PR #119 are missing")
    for item in additions:
        current = existing.get(item["author_pub_id"])
        if current is not None and current != item:
            raise ValueError(f"Conflicting decision for {item['author_pub_id']}")
    if args.write:
        missing = [item for item in additions if item["author_pub_id"] not in existing]
        insert_at = next(
            i
            for i, item in enumerate(manifest["exclusions"])
            if item["scholar_id"] == "0e1jwkkAAAAJ"
        )
        manifest["exclusions"][insert_at:insert_at] = missing
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        with LEDGER.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(screening[0]))
            writer.writeheader()
            writer.writerows(screening)
        print(
            f"Added {len(missing)} exact decisions; wrote {len(screening)} ledger rows"
        )
    else:
        if any(item["author_pub_id"] not in existing for item in additions):
            raise ValueError("Missing proposed decision from manifest")
        with LEDGER.open(newline="") as handle:
            observed = list(csv.DictReader(handle))
        if observed != [
            {key: str(value) for key, value in row.items()} for row in screening
        ]:
            raise ValueError("Screening ledger differs from rebuilt capture comparison")
        print(f"Verified {len(additions)} decisions and {len(screening)} ledger rows")


if __name__ == "__main__":
    main()
