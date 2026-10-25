"""Rebuild the exact-ID Hyung Kim screening ledger from the saved Scholar capture.

The source capture is in merged source PR #118. This script makes no Scholar
request and writes only the inert Kim exclusions proposal, never the canonical
producer manifest. ORCID comparison inputs are the dated, compact public API
snapshot beside this script. Run with --capture PATH --write, then --capture
PATH --check.
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
SCHOLAR_ID = "bC4aq-4AAAAJ"
KNU_ORCID = "0000-0001-9787-4684"
CMU_ORCID = "0000-0003-4334-1879"
BASELINE_EXCLUDED_INDICES = {3, 14, 22, 34, 50, 226, 378, 695}
LEDGER = HERE / "2026-10-24-chem-kim-918-screen.csv"
EVIDENCE = HERE / "2026-10-24-chem-kim-orcid-evidence.json"
MANIFEST = HERE / "proposals/scholar-exclusions.kim-proposal.json"

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
    197: "bC4aq-4AAAAJ:WqliGbK-hY8C",
    448: "bC4aq-4AAAAJ:BrmTIyaxlBUC",
    513: "bC4aq-4AAAAJ:1yQoGdGgb4wC",
    620: "bC4aq-4AAAAJ:dQ2og3OwTAUC",
    688: "bC4aq-4AAAAJ:hkOj_22Ku90C",
    722: "bC4aq-4AAAAJ:ZuybSZzF8UAC",
    854: "bC4aq-4AAAAJ:SdhP9T11ey4C",
    205: "bC4aq-4AAAAJ:Bg7qf7VwUHIC",
    206: "bC4aq-4AAAAJ:u-coK7KVo8oC",
    237: "bC4aq-4AAAAJ:gVv57TyPmFsC",
    283: "bC4aq-4AAAAJ:gKiMpY-AVTkC",
    318: "bC4aq-4AAAAJ:uJ-U7cs_P_0C",
    319: "bC4aq-4AAAAJ:4MWp96NkSFoC",
    438: "bC4aq-4AAAAJ:i2xiXl-TujoC",
    28: "bC4aq-4AAAAJ:kvJssbFybhEC",
    117: "bC4aq-4AAAAJ:AFmTUeZ1pmEC",
    150: "bC4aq-4AAAAJ:2KloaMYe4IUC",
    195: "bC4aq-4AAAAJ:tKAzc9rXhukC",
    208: "bC4aq-4AAAAJ:aDl3D7KC1E4C",
    190: "bC4aq-4AAAAJ:FiytvqdAVhgC",
    245: "bC4aq-4AAAAJ:NDuN12AVoxsC",
    262: "bC4aq-4AAAAJ:7wO8s98CvbsC",
    266: "bC4aq-4AAAAJ:SpPTWFSNUtQC",
    274: "bC4aq-4AAAAJ:DBa1UEJaJKAC",
    295: "bC4aq-4AAAAJ:9shLKfS_uJEC",
    714: "bC4aq-4AAAAJ:c1e4I3QdEKYC",
    816: "bC4aq-4AAAAJ:OP4eGU-M3BUC",
    811: "bC4aq-4AAAAJ:LhH-TYMQEocC",
    812: "bC4aq-4AAAAJ:r_AWSJRzSzQC",
    122: "bC4aq-4AAAAJ:WZBGuue-350C",
    196: "bC4aq-4AAAAJ:vDijr-p_gm4C",
    272: "bC4aq-4AAAAJ:M7yex6snE4oC",
    289: "bC4aq-4AAAAJ:nrtMV_XWKgEC",
    716: "bC4aq-4AAAAJ:Mojj43d5GZwC",
    407: "bC4aq-4AAAAJ:1qzjygNMrQYC",
    446: "bC4aq-4AAAAJ:HtEfBTGE9r8C",
    457: "bC4aq-4AAAAJ:uc_IGeMz5qoC",
    556: "bC4aq-4AAAAJ:epqYDVWIO7EC",
    580: "bC4aq-4AAAAJ:yB1At4FlUx8C",
    654: "bC4aq-4AAAAJ:EkHepimYqZsC",
    152: "bC4aq-4AAAAJ:x2hKVfJWtf0C",
    230: "bC4aq-4AAAAJ:ZysSsiWj_g4C",
    247: "bC4aq-4AAAAJ:4vMrXwiscB8C",
    250: "bC4aq-4AAAAJ:QtA78RmWg5MC",
    251: "bC4aq-4AAAAJ:NXYAu82O0W8C",
    253: "bC4aq-4AAAAJ:AvfA0Oy_GE0C",
    270: "bC4aq-4AAAAJ:xii_ZKWM4-0C",
    296: "bC4aq-4AAAAJ:Ehil0879vHcC",
    411: "bC4aq-4AAAAJ:uLbwQdceFCQC",
    412: "bC4aq-4AAAAJ:HIFyuExEbWQC",
    434: "bC4aq-4AAAAJ:fFSKOagxvKUC",
    280: "bC4aq-4AAAAJ:n3vGvpFsckwC",
    324: "bC4aq-4AAAAJ:TeJ9juy8vcMC",
    371: "bC4aq-4AAAAJ:Vr2j17o0sqMC",
    423: "bC4aq-4AAAAJ:cOfwuRB03ygC",
    129: "bC4aq-4AAAAJ:GUYAmugLYisC",
    442: "bC4aq-4AAAAJ:urP0JZOBBUsC",
    447: "bC4aq-4AAAAJ:jtI9f0ekYq0C",
}

OTHER_PRIMARY.update(
    {
        197: (
            "Double stochastic resonance peaks in systems with dynamic "
            "phase transitions",
            2001,
            "Hyun Jin Kim",
            "10.1209/epl/i2001-00524-7",
            "https://arxiv.org/abs/cond-mat/0108288",
            "Original author-submitted arXiv record linked to EPL article",
            "The original preprint names the H. J. Kim coauthor in full as "
            "Hyun Jin Kim and links the exact Europhysics Letters DOI, volume "
            "56, and first page 333.",
        ),
        448: (
            "Dynamic transition and resonance in current-driven arrays of "
            "Josephson junctions",
            2002,
            "Hyun Jin Kim",
            "10.1103/PhysRevB.66.024511",
            "https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.024511",
            "Original American Physical Society article",
            "The publisher names Hyun Jin Kim in full at Seoul National "
            "University Physics in the exact 2002 Physical Review B article "
            "66, 024511.",
        ),
    }
)

HYOJOON_KAPRAL_PRIMARY = {
    513: (
        "Proton and deuteron transfer reactions in molecular nanoclusters",
        2008,
        "Hyojoon Kim",
        "10.1002/cphc.200700709",
        "https://api.crossref.org/works/10.1002%2Fcphc.200700709",
        "Wiley publisher-deposited full-author DOI record",
        "The publisher identifies Hyojoon Kim, not CMU Hyung J. Kim, with "
        "Raymond Kapral in the exact ChemPhysChem 9:470-474 article.",
    ),
    620: (
        "Quantum equilibrium structure for strong nonadiabatic coupling: "
        "Reaction rate enhancement",
        2007,
        "Hyojoon Kim",
        "10.1016/j.cplett.2007.04.039",
        "https://api.crossref.org/works/10.1016%2Fj.cplett.2007.04.039",
        "Elsevier publisher-deposited full-author DOI record",
        "The publisher identifies Hyojoon Kim with Raymond Kapral in the "
        "exact Chemical Physics Letters 440:215-220 article.",
    ),
    688: (
        "Quantum reaction rates and sampling of quantum equilibrium structure",
        2007,
        "Hyojoon Kim",
        "10.1063/1.2805071",
        "https://api.crossref.org/works/10.1063%2F1.2805071",
        "AIP publisher-deposited full-author DOI record",
        "The publisher identifies Hyojoon Kim with Raymond Kapral in the "
        "exact Journal of Chemical Physics 127:226101 article.",
    ),
    722: (
        "Quantum–Classical Reaction Rate Theory",
        2007,
        "Hyojoon Kim",
        "10.1007/978-3-540-34460-5_13",
        "https://web.donga.ac.kr/hkim/publications.htm",
        "Hyojoon Kim's Dong-A University bibliography; original Springer chapter",
        "His university bibliography gives full Hyojoon Kim with Hanna and "
        "Kapral on the exact chapter, book, and pages 295-319; Springer "
        "confirms the exact DOI/year and H. Kim byline at Toronto Chemistry.",
    ),
    854: (
        "Simulation of Quantum-Classical Dynamics by Surface-Hopping Trajectories",
        2007,
        "Hyojoon Kim",
        "10.1090/crmp/041/08",
        "https://api.crossref.org/works/10.1090%2Fcrmp%2F041%2F08",
        "American Mathematical Society publisher-deposited full-author DOI record",
        "The publisher names Hyojoon Kim with Raymond Kapral in the exact "
        "2007 CRM Proceedings chapter, pages 131-143; his Dong-A University "
        "bibliography corroborates.",
    ),
}
OTHER_PRIMARY.update(HYOJOON_KAPRAL_PRIMARY)

HYUNG_JOON_NEURO_PRIMARY = {
    205: (
        "Reprogramming axonal behavior by axon-specific viral transduction",
        2012,
        "Hyung Joon Kim",
        "10.1038/gt.2011.217",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3426632/",
        "Original author manuscript with full-name byline",
        "The author-deposited Gene Therapy manuscript identifies its H. J. "
        "Kim coauthor as Hyung Joon Kim, linked to Seoul National University "
        "Mechanical and Aerospace Engineering, on the exact 2012 article.",
    ),
    206: (
        "Novel microfluidic platform for culturing neurons: culturing and "
        "biochemical analysis of neuronal components",
        2009,
        "Hyung Joon Kim",
        "10.1002/biot.200900159",
        "https://mbel.snu.ac.kr/publication/novel-microfluidic-platform-for-"
        "culturing-neurons-culturing-and-biochemical-analysis-of-neuronal-components/",
        "Seoul National University coauthor lab's exact publication record",
        "The Seoul National University lab lists Hyung Joon Kim in full "
        "with Park, Byun, Ryu, and Jeon on the exact 2009 Biotechnology "
        "Journal 4:1573-1577 article.",
    ),
    237: (
        "Quantitative analysis of axonal transport by using compartmentalized "
        "and surface micropatterned culture of neurons",
        2012,
        "Hyung Joon Kim",
        "10.1021/cn3000026",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3643179/",
        "Original ACS article with full-name byline and UCI affiliation",
        "The ACS article names Hyung Joon Kim at UC Irvine Biomedical "
        "Engineering with the same six coauthors in ACS Chemical "
        "Neuroscience 3:433-438.",
    ),
    283: (
        "Shear stress effect on transfection of neurons cultured in "
        "microfluidic devices",
        2009,
        "Hyung Joon Kim",
        "10.1166/jnn.2009.1769",
        "https://mbel.snu.ac.kr/publication/shear-stress-effect-on-"
        "transfection-of-neurons-cultured-in-microfluidic-devices/",
        "Seoul National University coauthor lab's exact publication record",
        "The Seoul National University lab names Hyung Joon Kim in full "
        "with Shin, Sim, and Jeon in the exact 2009 Journal of Nanoscience "
        "and Nanotechnology 9:7330-7335 article.",
    ),
    318: (
        "Yeast G-proteins mediate directional sensing and polarization "
        "behaviors in response to changes in pheromone gradient direction",
        2013,
        "Hyung Joon Kim",
        "10.1091/mbc.E12-10-0739",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3571874/",
        "Original Molecular Biology of the Cell article with full-name byline",
        "The original article names Hyung Joon Kim at UC Irvine Biomedical "
        "Engineering with Moore, Tanaka, Jeon, and Yi in volume 24:521-534. "
        "Its online date was 2012 and print year 2013.",
    ),
    319: (
        "Chemical genetic-mediated spatial regulation of protein expression "
        "in neurons reveals an axonal function for WldS",
        2012,
        "Hyung Joon Kim",
        "10.1016/j.chembiol.2012.01.012",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3292772/",
        "Original author manuscript with full-name byline and UCI affiliation",
        "The author-deposited Chemistry & Biology article names Hyung Joon "
        "Kim at UC Irvine Biomedical Engineering with Cohen, Ghosh, Jeon, "
        "and Jaffrey in volume 19:179-187.",
    ),
    438: (
        "Kinase pathway inhibition restores PSD95 induction in neurons "
        "lacking fragile X mental retardation protein",
        2019,
        "Hyung Joon Kim",
        "10.1073/pnas.1812056116",
        "https://www.pnas.org/content/pnas/116/24/12007.full.pdf",
        "Original PNAS article full-name byline and SNU affiliation",
        "The original PNAS paper names Hyung Joon Kim at Seoul National "
        "University Mechanical and Aerospace Engineering in the exact "
        "2019 volume 116:12007-12012 article.",
    ),
}
OTHER_PRIMARY.update(HYUNG_JOON_NEURO_PRIMARY)

HYOUNG_JUHN_KIST_FUEL_CELL_PRIMARY = {
    150: (
        "Performance of proton-exchange membrane fuel cells using the "
        "catalyst-gradient electrode technique",
        2007,
        "Hyoung-Juhn Kim",
        "10.1016/j.jpowsour.2006.12.076",
        "https://www.h2cu.org/drupaluni/sites/default/files/evento_allegati/"
        "Evento20051214_EFC05_FinalProgram.pdf",
        "Original 2005 fuel-cell conference program; exact KAIST article record",
        "The exact journal record lists H. J. Kim at KIST with Prasanna, Cho, "
        "Oh, Lim, and Hong. The 2005 EFC05 program names Hyoung-Juhn Kim "
        "at KIST with those same five coauthors; KENTECH independently "
        "records his 2004-2022 KIST appointment. This is a full-name "
        "same-team identity inference, not a full-name byline on the "
        "exact 2007 article.",
    ),
    195: (
        "Effects of platinum loading on performance of proton-exchange "
        "membrane fuel cells using surface-modified Nafion® membranes",
        2006,
        "Hyoung-Juhn Kim",
        "10.1016/j.jpowsour.2006.01.071",
        "https://www.h2cu.org/drupaluni/sites/default/files/evento_allegati/"
        "Evento20051214_EFC05_FinalProgram.pdf",
        "Original 2005 fuel-cell conference program; exact KAIST article record",
        "The exact journal record lists H. J. Kim at KIST with Prasanna, Cho, "
        "Lim, Oh, and Hong. The 2005 EFC05 program names Hyoung-Juhn Kim "
        "at KIST with those same five coauthors; KENTECH independently "
        "records his 2004-2022 KIST appointment. This is a full-name "
        "same-team identity inference, not a full-name byline on the "
        "exact 2006 article.",
    ),
    208: (
        "Operation of a proton-exchange membrane fuel cell under "
        "non-humidified conditions using thin cast Nafion membranes with "
        "different gas-diffusion media",
        2006,
        "Hyoung Juhn Kim",
        "10.1016/j.jpowsour.2005.06.023",
        "https://pure.kaist.ac.kr/en/publications/operation-of-a-proton-"
        "exchange-membrane-fuel-cell-under-non-humid-2/",
        "KAIST full-name 2007 follow-up; exact 2006 KAIST article record",
        "The exact 2006 article lists H. J. Kim at KIST with Vengatesan, "
        "Cho, Ha, Oh, Hong, and Lim. KAIST expands the H. J. Kim byline "
        "to Hyoung Juhn Kim in this team's directly related 2007 follow-up; "
        "KENTECH records his 2004-2022 KIST appointment. This is a "
        "full-name same-team identity inference, not a full-name byline "
        "on the exact 2006 article.",
    ),
}
OTHER_PRIMARY.update(HYOUNG_JUHN_KIST_FUEL_CELL_PRIMARY)

KNU_SCINTILLATOR_PERSON_LINK_PRIMARY = {
    190: (
        "Growth and characterization of isotopically enriched 40Ca100MoO4 "
        "single crystals for rare event search experiments",
        2011,
        "10.1002/crat.201100364",
        "growth-and-characterization-of-isotopically-enriched-sup40supca-s/",
    ),
    245: (
        "Temperature dependence of scintillation properties of SrMoO4",
        2015,
        "10.1016/j.nima.2015.04.018",
        "temperature-dependence-of-scintillation-properties-of-srmoosub4su/",
    ),
    262: (
        "Heat and Light Measurement of a  Crystal for the AMoRE Double "
        "Beta Decay Experiment",
        2016,
        "10.1109/TNS.2015.2493529",
        "heat-and-light-measurement-of-a-sup40supcasup100supmoosub4sub-cry/",
    ),
    266: (
        "Study on Crystal Growth and Scintillation Characteristics of Cs LiCeCl",
        2010,
        "10.1109/TNS.2009.2037903",
        "study-on-crystal-growth-and-scintillation-characteristics-of-cssu/",
    ),
    274: (
        "Scintillator-based detectors for dark matter searches I",
        2010,
        "10.1088/1367-2630/12/7/075003",
        "scintillator-based-detectors-for-dark-matter-searches-i/",
    ),
    295: (
        "Scintillation Properties and Internal Background Study of CaMoO "
        "Crystal Scintillators for Neutrino-Less Double Beta Decay Search",
        2012,
        "10.1109/TNS.2012.2200908",
        "scintillation-properties-and-internal-background-study-of-sup40su/",
    ),
    714: (
        "Czochralski growth and scintillation properties of bismuth "
        "germanium silicon oxide (BGSO) single crystals",
        2011,
        "10.1109/NSSMIC.2011.6154635",
        "czochralski-growth-and-scintillation-properties-of-bismuth-german/",
    ),
    816: (
        "Scintillation properties of Rb2LiGdBr6: Ce3+ single crystals",
        2013,
        "10.1109/NSSMIC.2013.6829671",
        "scintillation-properties-of-rbsub2subligdbrsub6sub-ce-sup3sup-sin/",
    ),
}
for index, (title, year, doi, url_slug) in KNU_SCINTILLATOR_PERSON_LINK_PRIMARY.items():
    OTHER_PRIMARY[index] = (
        title,
        year,
        "Hong Joo Kim",
        doi,
        f"https://research.knu.ac.kr/en/publications/{url_slug}",
        "KNU exact DOI work-to-person institutional link",
        "KNU's exact-title/DOI work record links its H. J. Kim byline "
        "directly to the institutional faculty profile for Hong Joo Kim, "
        "professor of Physics. The Scholar title may omit isotope or "
        "chemical formula superscripts, but its DOI, year, and other "
        "authors identify this exact output.",
    )

OTHER_PRIMARY.update(
    {
        811: (
            "Erratum to: Growth and scintillation characterizations of "
            "SrMoO4 single crystals",
            2014,
            "Hong Joo Kim",
            "10.3938/jkps.64.333",
            "https://link.springer.com/article/10.3938/jkps.64.333",
            "Original Springer erratum with corresponding-author address",
            "Springer identifies this 2014 volume 64, page 333 output as an "
            "erratum and assigns H. J. Kim to KNU Physics with the "
            "corresponding address hongjoo@knu.ac.kr. KNU's exact erratum "
            "record links that author to Hong Joo Kim. This is a real "
            "correction record, but it belongs to the KNU namesake.",
        ),
        812: (
            "Growth and scintillation characterizations of SrMoO4 single "
            "crystals (vol 63, pg 2018, 2013)",
            2014,
            "Hong Joo Kim",
            "10.3938/jkps.64.333",
            "https://link.springer.com/article/10.3938/jkps.64.333",
            "Original Springer erratum with corresponding-author address",
            "This alternate Scholar title is a second rendering of the "
            "same 2014 volume 64, page 333 erratum, not the 2013 "
            "volume 63, page 2018 research article. Springer assigns "
            "H. J. Kim to KNU Physics and prints the corresponding address "
            "hongjoo@knu.ac.kr; KNU links him to Hong Joo Kim.",
        ),
    }
)
KNU_ERRATUM_VENUES = {
    811: "Journal of the Korean Physical Society 64 (2), 333-333, 2014",
    812: "JOURNAL OF THE KOREAN PHYSICAL SOCIETY 64 (2), 333-333, 2014",
}

OTHER_PRIMARY.update(
    {
        28: (
            "Identification of two epidermal growth factor-sensitive tyrosine "
            "phosphorylation sites of phospholipase C-gamma in intact HSC-1 "
            "cells.",
            1990,
            "Hyun Kim",
            "10.1016/S0021-9258(19)39685-1",
            "https://www.researchgate.net/publication/348269868_Identification_of_"
            "two_epidermal_growth_factor-sensitive_tyrosine_phosphorylation_"
            "sites_of_phospholipase_C-gamma_in_intact_HSC-1_cells",
            "Original Journal of Biological Chemistry article scan",
            "The original article prints Hyun Kim as the fourth author, with "
            "Vanderbilt/NIH laboratory affiliations, in Journal of Biological "
            "Chemistry 265:3944-3948.",
        ),
        117: (
            "An overview of KSTAR results",
            2013,
            "Haejin Kim",
            "10.1088/0029-5515/53/10/104005",
            "https://jobs.iterkorea.org/thesis?amount=10&mid=a20302000000&"
            "pageNum=4&pblYear=2013",
            "Korea Institute of Fusion Energy's exact article record",
            "The fusion institute lists Haejin Kim on the exact 2013 KSTAR "
            "article. The publisher's H. J. Kim byline is assigned to its "
            "National Fusion Research Institute affiliation; a contemporary "
            "original KSTAR paper identifies H. J. Kim there with the "
            "haejin@nfri.re.kr address and overlapping coauthors.",
        ),
    }
)

# Original US patent/application front pages expand the initials of five
# unrelated inventors. Keep patent and application observations separate:
# rows 122 and 272 are two distinct filings in one LG Innotek family.
PATENT_FULL_INVENTOR_PRIMARY = {
    122: (
        "Lighting module and lighting apparatus including the same",
        2012,
        "Hyong Jin Kim",
        "",
        "https://patentimages.storage.googleapis.com/b2/ef/49/bd6c2ccdc060a1/US8231258.pdf",
        "Original US Patent 8,231,258 B2 front page",
        "The exact 2012 US Patent 8,231,258 names seventh inventor Hyong "
        "Jin Kim in full with the same LG Innotek inventor team as the "
        "Scholar row.",
    ),
    196: (
        "Cantilever sensor and fabrication method thereof",
        2003,
        "Hyung Joon Kim",
        "",
        "https://patentimages.storage.googleapis.com/0d/4e/77/ab94825f832e1a/US20030224551A1.pdf",
        "Original US application 10/448,188 front page",
        "The exact 2003 application 10/448,188 names second inventor Hyung "
        "Joon Kim in full with Tae Song Kim, Jeong Hoon Lee, and Ji Yoon "
        "Kang, matching the Scholar initials and application number.",
    ),
    272: (
        "Lighting module and lighting apparatus including the same",
        2012,
        "Hyong Jin Kim",
        "",
        "https://patentimages.storage.googleapis.com/f9/41/88/0f0fdeef979ec3/US20120224392A1.pdf",
        "Original US application 13/471,967 front page",
        "The exact 2012 application 13/471,967 names seventh inventor "
        "Hyong Jin Kim in full with the same LG Innotek inventor team "
        "as the Scholar row.",
    ),
    289: (
        "Method for the preparation of polyester by use of composite catalyst",
        1998,
        "Hyung Joon Kim",
        "",
        "https://patentimages.storage.googleapis.com/45/1e/79/8c467892858e36/US5714570.pdf",
        "Original US Patent 5,714,570 front page",
        "The exact 1998 US Patent 5,714,570 names fourth inventor Hyung "
        "Joon Kim in full with Chung Yup Kim, Hyun Nam Cho, and Hoi Wan "
        "Yoo, matching the Scholar initials and patent number.",
    ),
    716: (
        "Computer program product and system comprising community management method",
        2010,
        "Hyeonsook Kim",
        "",
        "https://patentimages.storage.googleapis.com/82/10/09/3193665cfec3f0/US7849136.pdf",
        "Original US Patent 7,849,136 B2 front page",
        "The exact 2010 US Patent 7,849,136 names second inventor "
        "Hyeonsook Kim in full with Weduke Cho, Soondong Kim, Yunju "
        "Shim, and Dongsoon Choi; its Scholar byline abbreviates her as H. Kim.",
    ),
}
OTHER_PRIMARY.update(PATENT_FULL_INVENTOR_PRIMARY)
PATENT_VENUES = {
    122: "US Patent 8,231,258, 2012",
    196: "US Patent App. 10/448,188, 2003",
    272: "US Patent App. 13/471,967, 2012",
    289: "US Patent 5,714,570, 1998",
    716: "US Patent 7,849,136, 2010",
}

# These exact IEEE/Springer/Informa works expand the saved Scholar initials.
# IEEE and Informa deposits give full authors, while Springer also supplies
# the KIST affiliation on its original article page.
KIST_COMMUNITY_FULL_AUTHOR_PRIMARY = {
    407: (
        "Community manager: A dynamic collaboration solution on "
        "heterogeneous environment",
        2006,
        "Hyeonsook Kim",
        "10.1109/PERSER.2006.1652205",
        "https://api.crossref.org/works/10.1109%2FPERSER.2006.1652205",
        "IEEE publisher-deposited full-author DOI record",
        "The IEEE deposit expands the first H. Kim author to Hyeonsook "
        "Kim and matches the exact 2006 conference title, pages 39-46, "
        "and four other coauthors in the Scholar row.",
    ),
    446: (
        "Resonance properties and mass sensitivity of monolithic "
        "microcantilever sensors actuated by piezoelectric PZT thick film",
        2006,
        "Hyung Joon Kim",
        "10.1007/s10832-006-6290-8",
        "https://link.springer.com/article/10.1007/s10832-006-6290-8",
        "Original Springer article with full-name byline and KIST affiliation",
        "Springer names third author Hyung Joon Kim at KIST Microsystem "
        "Research Center in the exact 2006 Journal of Electroceramics "
        "17:565-572 article.",
    ),
    457: (
        "Fabrication and resonant behavior of PZT thick film cantilever for BioChip",
        2002,
        "Hyung Joon Kim",
        "10.1080/743817676",
        "https://api.crossref.org/works/10.1080%2F743817676",
        "Informa publisher-deposited full-author DOI record",
        "The exact Integrated Ferroelectrics 50:11-20 article names first "
        "author Hyung Joon Kim at KIST with Yong-Bum Kim, Ji-Yoon Kang, "
        "and Tae Song Kim.",
    ),
    556: (
        "Electrical properties of 3-component piezoelectric thick films "
        "by screen printing method",
        2001,
        "Hyung-Joon Kim",
        "10.1080/10584580108012813",
        "https://api.crossref.org/works/10.1080%2F10584580108012813",
        "Informa publisher-deposited full-author DOI record",
        "The exact Integrated Ferroelectrics 41:107-117 article names "
        "second author Hyung-Joon Kim with the same five-author team; "
        "Yonsei University's publication record independently expands "
        "the same exact output to Hyung Joon Kim.",
    ),
    580: (
        "Preparation of diffuser type micropump using screen-printed "
        "PZT-PCW thick films",
        2002,
        "Hyung-Joon Kim",
        "10.1080/743817664",
        "https://api.crossref.org/works/10.1080%2F743817664",
        "Informa publisher-deposited full-author DOI record",
        "The exact Integrated Ferroelectrics 50:61-70 article names "
        "second author Hyung-Joon Kim with the same five-author team; "
        "Yonsei University's publication record independently expands "
        "the same exact output to Hyung Joon Kim.",
    ),
    654: (
        "Biological element detection sensor application of micromachined "
        "PZT thick film cantilever",
        2003,
        "Hyung Joon Kim",
        "10.1109/ICSENS.2003.1279104",
        "https://api.crossref.org/works/10.1109%2FICSENS.2003.1279104",
        "IEEE publisher-deposited full-author DOI record",
        "The IEEE Sensors 2003 proceedings deposit expands first author "
        "H. J. Kim to Hyung Joon Kim and matches pages 1054-1058 and "
        "Yong Bum Kim, Jaebum Park, and Tae Song Kim.",
    ),
}
OTHER_PRIMARY.update(KIST_COMMUNITY_FULL_AUTHOR_PRIMARY)
KIST_COMMUNITY_VENUES = {
    407: "2006 ACS/IEEE International Conference on Pervasive Services, 39-46, 2006",
    446: "Journal of electroceramics 17 (2), 565-572, 2006",
    457: "Integrated Ferroelectrics 50 (1), 11-20, 2002",
    556: "Integrated Ferroelectrics 41 (1-4), 107-117, 2001",
    580: "Integrated Ferroelectrics 50 (1), 61-70, 2002",
    654: "SENSORS, 2003 IEEE 2, 1054-1058, 2003",
}

# Exact original work records expand the captured H. J. initials to other people.
NEXT_FULL_AUTHOR_PRIMARY = {
    152: (
        "Diffusion tensor imaging of normal‐appearing white matter in patients "
        "with neuromyelitis optica spectrum disorder and multiple sclerosis",
        2017,
        "Ho Jin Kim",
        "10.1111/ene.13321",
        "https://onlinelibrary.wiley.com/doi/10.1111/ene.13321",
        "Wiley original article author ORCID and ORCID person record",
        "The exact European Journal of Neurology 24(7):966-973 article "
        "links its H. J. Kim author at National Cancer Center Neurology "
        "to ORCID 0000-0002-8672-8419, whose public person record names "
        "Ho Jin Kim.",
    ),
    230: (
        "Widespread cortical thinning in patients with neuromyelitis "
        "optica spectrum disorder",
        2016,
        "Ho Jin Kim",
        "10.1111/ene.13011",
        "https://onlinelibrary.wiley.com/doi/10.1111/ene.13011",
        "Wiley original corresponding-author address and NCC identity page",
        "The exact European Journal of Neurology 23(7):1165-1173 article "
        "identifies its H. J. Kim corresponding author in National Cancer "
        "Center Neurology at hojinkim@ncc.re.kr; NCC's faculty page "
        "identifies that address as Ho Jin Kim.",
    ),
    247: (
        "Treatment of abandoned coal mine discharged waters using lime wastes",
        2007,
        "Hee Joung Kim",
        "10.1007/BF02913923",
        "https://pure.uos.ac.kr/en/publications/"
        "treatment-of-abandoned-coal-mine-discharged-waters-using-lime-was/",
        "University of Seoul exact-output full-author record",
        "The University of Seoul lists second author Hee Joung Kim on the "
        "exact Geosciences Journal 11(2):111-114 article; the Springer "
        "DOI deposit independently gives the same full byline.",
    ),
    250: (
        "Korean Endocrine Society, Committee for Clinical Practice Guidelines",
        2017,
        "Hae Jin Kim",
        "10.3803/EnM.2017.32.2.200",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC5503865/",
        "original Korean Endocrine Society full-author guideline article",
        "The saved Scholar title misparses the corporate author as a title; "
        "its venue names the exact Clinical Guidelines for the Management "
        "of Adrenal Incidentaloma article. The original Endocrinology and "
        "Metabolism 32(2):200-218 byline identifies eighth author "
        "Hae Jin Kim at Ajou University.",
    ),
    251: (
        "Normal‐appearing white matter demyelination in neuromyelitis "
        "optica spectrum disorder",
        2017,
        "Ho Jin Kim",
        "10.1111/ene.13266",
        "https://onlinelibrary.wiley.com/doi/10.1111/ene.13266",
        "Wiley original corresponding-author address and NCC exact publication",
        "The exact European Journal of Neurology 24(4):652-658 article "
        "identifies its H. J. Kim corresponding author at "
        "hojinkim@ncc.re.kr; NCC's Ho Jin Kim faculty bibliography lists "
        "the exact title, year, volume, and pages.",
    ),
    253: (
        "Drop impact analysis of plate-type fuel assembly in research reactor",
        2014,
        "Hyun-Jung Kim",
        "10.5516/NET.09.2013.103",
        "https://api.crossref.org/works/10.5516%2FNET.09.2013.103",
        "Korean Nuclear Society publisher-deposited full-author DOI record",
        "The exact Nuclear Engineering and Technology 46(4):529-540 "
        "publisher deposit names first author Hyun-Jung Kim with the "
        "matching five-author team and DOI.",
    ),
    270: (
        "Outcome of pregnancies after onset of the neuromyelitis optica "
        "spectrum disorder",
        2020,
        "Ho Jin Kim",
        "10.1111/ene.14274",
        "https://onlinelibrary.wiley.com/doi/10.1111/ene.14274",
        "Wiley original article author ORCID and ORCID person record",
        "The exact European Journal of Neurology 27(8):1546-1555 article "
        "links its H. J. Kim author at National Cancer Center Neurology "
        "to ORCID 0000-0002-8672-8419, whose public person record names "
        "Ho Jin Kim.",
    ),
    296: (
        "InAs critical-point energies at 22 K from spectroscopic ellipsometry",
        2010,
        "HyeJung Kim",
        "10.1063/1.3506497",
        "https://api.crossref.org/works/10.1063%2F1.3506497",
        "AIP publisher-deposited full-author DOI record",
        "The exact Applied Physics Letters 97(17):171912 DOI deposit "
        "expands seventh author H. J. Kim to HyeJung Kim, with the "
        "matching eight-author team.",
    ),
    411: (
        "Novel ELANE gene mutation in a Korean girl with severe congenital neutropenia",
        2011,
        "Hee-Jin Kim",
        "10.3346/jkms.2011.26.12.1646",
        "https://jkms.org/DOIx.php?id=10.3346/jkms.2011.26.12.1646",
        "original Journal of Korean Medical Science full byline",
        "The exact Journal of Korean Medical Science 26(12):1646-1649 "
        "article names second author Hee-Jin Kim at Samsung Medical "
        "Center, Sungkyunkwan University.",
    ),
    412: (
        "Assessment of water pollution by discharge of abandoned mines",
        2005,
        "Hee Joung Kim",
        "",
        "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/"
        "ciSereArtiView.kci?sereArticleSearchBean.artiId=ART000974760",
        "KCI original society exact-output full-author record",
        "The exact Journal of Soil and Groundwater Environment 10(5):25-36 "
        "article names Hee Joung Kim (Kim Hwi-Jung) among its complete "
        "authors at Kangwon National University.",
    ),
    434: (
        "Phytoremediation of heavy metal contaminated soils using transgenic plants",
        2004,
        "Hee Joung Kim",
        "",
        "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/"
        "ciSereArtiView.kci?sereArticleSearchBean.artiId=ART000930334",
        "KCI original society exact-output full-author record",
        "The exact Korean Journal of Soil Science and Fertilizer "
        "37(6):396-406 article names sixth author Hee Joung Kim "
        "(Kim Hwi-Jung) at Kangwon National University.",
    ),
}
OTHER_PRIMARY.update(NEXT_FULL_AUTHOR_PRIMARY)
NEXT_FULL_AUTHOR_VENUES = {
    152: "European Journal of Neurology 24 (7), 966-973, 2017",
    230: "European journal of neurology 23 (7), 1165-1173, 2016",
    247: "Geosciences Journal 11 (2), 111-114, 2007",
    250: "Clinical guidelines for the management of adrenal incidentaloma. "
    "Endocrinol\xa0…, 2017",
    251: "European journal of neurology 24 (4), 652-658, 2017",
    253: "Nuclear engineering and technology 46 (4), 529-540, 2014",
    270: "European Journal of Neurology 27 (8), 1546-1555, 2020",
    296: "Applied Physics Letters 97 (17), 2010",
    411: "Journal of Korean medical science 26 (12), 1646, 2011",
    412: "Journal of Soil and Groundwater Environment 10 (5), 25-36, 2005",
    434: "Korean Journal of Soil Science and Fertilizer 37 (6), 2004",
}

FOLLOWUP_FULL_AUTHOR_PRIMARY = {
    280: (
        "Characteristics of a stilbene scintillation crystal in a neutron spectrometer",
        2013,
        "Hong Joo Kim",
        "10.1016/j.radmeas.2013.01.004",
        "https://research.knu.ac.kr/en/publications/characteristics-of-a-"
        "stilbene-scintillation-crystal-in-a-neutron-/",
        "KNU exact-output record linked to Hong Joo Kim person authority",
        "Kyungpook National University's exact Radiation Measurements "
        "58:133-137 output links its H. J. Kim author to Hong Joo Kim's "
        "KNU Physics person record.",
    ),
    324: (
        "Altered distributions in circulating follicular helper and follicular "
        "regulatory T cells accountable for imbalanced cytokine production "
        "in multiple sclerosis",
        2021,
        "Ho Jin Kim",
        "10.1111/cei.13596",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8209573/",
        "original Clinical and Experimental Immunology author correspondence",
        "The exact Clinical and Experimental Immunology 205(1):75-88 article "
        "identifies H. J. Kim as National Cancer Center Neurology head "
        "at hojinkim@ncc.re.kr. NCC's faculty page names that person "
        "Ho Jin Kim and lists this exact publication.",
    ),
    371: (
        "Emotion dysregulation in idiopathic rapid eye movement sleep "
        "behavior disorder",
        2020,
        "Han-Joon Kim",
        "10.1093/sleep/zsz224",
        "https://academic.oup.com/sleep/article-abstract/43/2/zsz224/5573800",
        "original Oxford University Press full-author article",
        "The exact SLEEP 43(2):zsz224 article names ninth author "
        "Han-Joon Kim at Seoul National University Hospital; the "
        "original publisher PDF confirms its full byline.",
    ),
    423: (
        "Characteristics of complex light modulation through an "
        "amplitude-phase double-layer spatial light modulator",
        2017,
        "Hwi Kim",
        "10.1364/OE.25.003469",
        "https://pure.korea.ac.kr/en/publications/characteristics-of-complex-"
        "light-modulation-through-an-amplitude-/",
        "Korea University exact-output full-author record",
        "The exact Optics Express 25(4):3469-3480 article names ninth "
        "author Hwi Kim; Optica's issue table of contents independently "
        "prints the same full author and title.",
    ),
}
OTHER_PRIMARY.update(FOLLOWUP_FULL_AUTHOR_PRIMARY)
FOLLOWUP_FULL_AUTHOR_VENUES = {
    280: "Radiation measurements 58, 133-137, 2013",
    324: "Clinical & Experimental Immunology 205 (1), 75-88, 2021",
    371: "Sleep 43 (2), zsz224, 2020",
    423: "Optics express 25 (4), 3469-3480, 2017",
}

NEXT_EXACT_AUTHOR_PRIMARY = {
    129: (
        "Prevalence of multiple sclerosis in Korea",
        2010,
        "Ho Jin Kim",
        "10.1212/WNL.0b013e3181f88191",
        "https://www.neurology.org/doi/10.1212/WNL.0b013e3181f88191",
        "original Neurology article's full-name correspondence",
        "The exact Neurology 75:1432-1438 article names H. J. Kim in the "
        "byline and identifies correspondent Ho Jin Kim at National Cancer "
        "Center and hojinkim@ncc.re.kr. The NLM author and collaborator "
        "record corroborates his full name.",
    ),
    442: (
        "Tumefactive demyelination, an uncommon form of tacrolimus neurotoxicity",
        2011,
        "Ho Jin Kim",
        "10.1212/WNL.0b013e31820c30af",
        "https://www.neurology.org/doi/10.1212/WNL.0b013e31820c30af",
        "original Neurology article's full-name correspondence",
        "The exact Neurology 76:672-674 article identifies its last "
        "H. J. Kim author as Dr. Ho Jin Kim, National Cancer Center "
        "Neurology correspondent at hojinkim@ncc.re.kr.",
    ),
    447: (
        "Decentralized nonlinear model predictive control of multiple "
        "flying robots in dynamic environments",
        2003,
        "H. Jin Kim",
        "10.1109/CDC.2003.1271710",
        "https://people.eecs.berkeley.edu/~sastry/pubs/PDFs%20of%20Pubs2000-2005/"
        "Publications%20of%20Postdocs/ShimDavid/ShimDecentralized2003.pdf",
        "Berkeley original IEEE paper, KAIST exact-output record, and Sastry CV",
        "The original 42nd IEEE CDC paper at pages 3621-3626 spells its "
        "second author H. Jin Kim at Berkeley EECS; KAIST's exact-output "
        "record independently identifies him. Coauthor Shankar Sastry's "
        "CV gives the saved longer title for that same 2003 three-author, "
        "six-page conference work.",
    ),
}
OTHER_PRIMARY.update(NEXT_EXACT_AUTHOR_PRIMARY)
NEXT_EXACT_AUTHOR_VENUES = {
    129: "Neurology 75 (16), 1432-1438, 2010",
    442: "Neurology 76 (7), 672-674, 2011",
    447: "IEEE Conference on Decision and Control, 5, 2003",
}

# These malformed scintillator titles resolve through KNU's exact DOI work
# pages; the arXiv and conference renderings resolve to identified KNU works
# through publisher or collaboration version links. Each H. J. Kim byline
# belongs to the KNU Physics professor Hong Joo Kim, not CMU Chemistry.
NEXT_KNU_VERSION_PRIMARY = {
    388: (
        "Scintillation Characteristics of the SrCl Single Crystal for the "
        "Neutrinoless /EC Decay Search",
        2008,
        "Hong Joo Kim",
        "10.1109/TNS.2008.920252",
        "https://research.knu.ac.kr/en/publications/"
        "scintillation-characteristics-of-the-srclsub2sub-single-crystal-f/",
        "KNU exact DOI article and author-linked person record",
        "KNU's IEEE 55(3):1445-1448 article links the third H. J. Kim "
        "author to Hong Joo Kim's Physics page; Scholar loses SrCl2 and "
        "beta-plus formula symbols.",
    ),
    431: (
        "Large Volume CaMoO Scintillation Crystals",
        2008,
        "Hongjoo Kim",
        "10.1109/TNS.2008.920428",
        "https://research.knu.ac.kr/en/publications/"
        "large-volume-camoosub4sub-scintillation-crystals/",
        "KNU exact DOI article with full-name author and linked person record",
        "KNU's exact IEEE 55(3):1473-1475 record names author Hongjoo Kim "
        "in full and links him to its Physics person page; Scholar drops "
        "the CaMoO4 formula's subscript.",
    ),
    498: (
        "Scintillation Characterization of RbLiCeClSingle Crystal",
        2012,
        "Hong Joo Kim",
        "10.1109/TNS.2012.2200907",
        "https://research.knu.ac.kr/en/publications/"
        "scintillation-characterization-of-rb-sub2sublicecl-sub6sub-single/",
        "KNU exact DOI article and author-linked person record",
        "KNU's 59(5):2248-2251 article links its H. J. Kim author to the "
        "Hong Joo Kim Physics person record; the Scholar title has lost "
        "the Rb2LiCeCl6 subscripts.",
    ),
    548: (
        "Study on scintillation characteristics of BGSO single crystals",
        2013,
        "Hong Joo Kim",
        "10.1109/TNS.2013.2283878",
        "https://research.knu.ac.kr/en/publications/"
        "study-on-scintillation-characteristics-of-bgso-single-crystals/",
        "KNU exact DOI article and author-linked person record",
        "KNU's IEEE 61(1):323-327 article links its H. J. Kim author to "
        "the Hong Joo Kim Physics person record; Scholar labels this "
        "2013 while KNU dates the matching volume and pages to February 2014.",
    ),
    699: (
        "A Study on  Phonon-Scintillation Detection with MMC Readouts for "
        "a Neutrinoless Double Beta Decay Search",
        2022,
        "Hong Joo Kim",
        "10.1007/s10909-022-02824-7",
        "https://research.knu.ac.kr/en/publications/"
        "a-study-on-pbmoo-sub4sub-phonon-scintillation-detection-with-mmc-/",
        "KNU original DOI article and author-linked person record",
        "The original KNU PbMoO4 article links its fourth H. J. Kim "
        "author to Hong Joo Kim. Scholar drops the chemical formula; "
        "the exact 2022 title, author order, and publisher DOI agree.",
    ),
    734: (
        "AMoRE experiment: a search for neutrinoless double beta decay of "
        "{sup 100} Mo isotope with {sup 40} Ca {sup 100} MoO {sub 4} "
        "cryogenic scintillation detector",
        2012,
        "Hong Joo Kim",
        "10.1088/1742-6596/375/1/042023",
        "https://research.knu.ac.kr/en/publications/"
        "amore-experiment-a-search-for-neutrinoless-double-beta-decay-of-s/",
        "KNU exact DOI conference article and author-linked person record",
        "The markup-heavy Scholar rendering repeats excluded observation "
        "bC4aq-4AAAAJ:MDX3w3dAD3YC: J. Phys. Conf. Ser. 375, article "
        "042023. KNU links its H. J. Kim author to Hong Joo Kim.",
    ),
    751: (
        "Design and Initial Performance of the COSINE-100 Experiment",
        2017,
        "Hong Joo Kim",
        "10.1140/epjc/s10052-018-5590-x",
        "https://research.knu.ac.kr/en/publications/"
        "initial-performance-of-the-cosine-100-experiment/",
        "KNU exact DOI article and Yale collaboration version link",
        "Yale's COSINE-100 publication list uses this saved title and links "
        "arXiv:1710.05299 to the 2018 EPJC article; arXiv lists that DOI, "
        "and the exact KNU journal record links H. J. Kim to Hong Joo Kim.",
    ),
    787: (
        "Searching for low-mass dark matter via Migdal effect in COSINE-100 [preprint]",
        2021,
        "Hong Joo Kim",
        "10.1103/PhysRevD.105.042006",
        "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.042006",
        "original APS article and arXiv-to-journal version link",
        "The arXiv:2110.05806 preprint gives this exact title and 2021 "
        "submission and explicitly identifies the 2022 APS journal version. "
        "APS assigns its H. J. Kim author to KNU Physics, not CMU Chemistry.",
    ),
    789: (
        "Measurement of the cosmic muon annual and diurnal flux variation "
        "with the COSINE-100 detector",
        2020,
        "Hong Joo Kim",
        "10.1088/1475-7516/2021/02/013",
        "https://research.knu.ac.kr/en/publications/"
        "measurement-of-the-cosmic-muon-annual-and-diurnal-flux-variation-/",
        "KNU exact DOI article and arXiv-to-journal version link",
        "The exact arXiv:2005.13672 preprint gives this 2020 title and "
        "links the 2021 JCAP DOI; KNU's article links its H. J. Kim author "
        "to Hong Joo Kim's Physics record.",
    ),
    803: (
        "Crystal growth and scintillation properties of Li {sub 6} Lu "
        "{sub x} Gd {sub 1− x}(BO {sub 3}){sub 3}: Ce {sup 3+} single crystals",
        2014,
        "Hong Joo Kim",
        "10.1016/j.nima.2014.07.040",
        "https://research.knu.ac.kr/en/publications/"
        "crystal-growth-and-scintillation-properties-of-lisub6sublu-subxsu/",
        "KNU exact DOI article and author-linked person record",
        "The markup-heavy Scholar rendering repeats already excluded "
        "observation bC4aq-4AAAAJ:Hck25ST_3aIC: NIM A 764:268-272. "
        "KNU links its H. J. Kim author to Hong Joo Kim.",
    ),
    840: (
        "Scintillation properties of Ba {sub 0.20} Sr {sub 0.80} Cl {sub 2} "
        "single crystal",
        2008,
        "Hong Joo Kim",
        "10.1080/00223131.2008.10875918",
        "https://research.knu.ac.kr/en/publications/"
        "scintillation-properties-of-basub020subsrsub080subclsub2sub-singl/",
        "KNU exact DOI article and author-linked person record",
        "The formula-markup Scholar row repeats already excluded observation "
        "bC4aq-4AAAAJ:2l5NCbZemmgC: J. Nucl. Sci. Technol. 45:568-571. "
        "KNU links its H. J. Kim author to Hong Joo Kim.",
    ),
}
OTHER_PRIMARY.update(NEXT_KNU_VERSION_PRIMARY)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        388: "bC4aq-4AAAAJ:sszUF3NjhM4C",
        431: "bC4aq-4AAAAJ:kVjdVfd2voEC",
        498: "bC4aq-4AAAAJ:AXkvAH5U_nMC",
        548: "bC4aq-4AAAAJ:yxmsSjX2EkcC",
        699: "bC4aq-4AAAAJ:RVqaWcrwK10C",
        734: "bC4aq-4AAAAJ:as0KMg8qHbkC",
        751: "bC4aq-4AAAAJ:ShjGdcaqzI0C",
        787: "bC4aq-4AAAAJ:s1ouQE5r0WUC",
        789: "bC4aq-4AAAAJ:C-Rn0OCouf8C",
        803: "bC4aq-4AAAAJ:q0uBw5dMOAkC",
        840: "bC4aq-4AAAAJ:eO3_k5sD8BwC",
    }
)
NEXT_KNU_VERSION_VENUES = {
    388: "IEEE Transactions on Nuclear Science 55 (3), 1445-1448, 2008",
    431: "IEEE Transactions on nuclear Science 55 (3), 1473-1475, 2008",
    498: "IEEE Transactions on Nuclear Science 59 (5), 2248-2251, 2012",
    548: "IEEE Transactions on Nuclear Science 61 (1), 323-327, 2013",
    699: "Journal of Low Temperature Physics 209 (3), 409-416, 2022",
    734: "Journal of Physics. Conference Series (Online) 375, 2012",
    751: "",
    787: "arXiv, 2021",
    789: "arXiv, 2020",
    803: "Nuclear Instruments and Methods in Physics Research. Section A, "
    "Accelerators\xa0…, 2014",
    840: "",
}
NEXT_KNU_VERSION_AUTHORS = {
    388: "G Rooh, H Kang, HJ Kim, H Park, SH Doh",
    431: "MV Korzhik, VN Kornoukhov, OV Missevitch, AA Fedorov, AN Annenkov, ...",
    498: "G Rooh, HJ Kim, H Park, S Kim",
    548: "H Jiang, G Rooh, HJ Kim, JM Lee, YJ Lee, S Kim",
    699: "HL Kim, JH So, YH Kim, HJ Kim, SS Nagorny, SR Kim, YD Kim, MH Lee, ...",
    734: "H Bhang, JH Choi, S Choi, KB Kim, SK Kim, RS Boiko, DM Chernyak, ...",
    751: "G Adhikari, P Adhikari, EB de Souza, N Carlin, S Choi, WQ Choi, ...",
    787: "G Adhikari, N Carlin, JJ Choi, S Choi, AC Ezeribe, LE Franca, C Ha, ...",
    789: "C Collaboration, H Prihtiadi, G Adhikari, EBD Souza, N Carlin, JJ Choi, ...",
    803: "U Fawad, G Rooh, HJ Kim, H Park, S Kim, H Jiang",
    840: "J Kim, H Kang, HJ Kim, H Park, SH Doh, S Kim",
}

# Four alternate Scholar renderings repeat DOI-identical works already
# excluded by the KNU ORCID comparison. Three institutional article pages link
# their H. J. Kim author directly to Hong Joo Kim's KNU Physics person record;
# an original conference abstract supplies his matching KNU email for the fourth.
FOLLOWUP_KNU_DUPLICATE_PRIMARY = {
    666: (
        "Proton, UV, and X-ray Induced Luminescence in [Tb. sup. 3+] Doped "
        "Lu [Gd. sub. 2][Ga. sub. 2][Al. sub. 3][O. sub. 12] Phosphors",
        2020,
        "Hong Joo Kim",
        "10.3390/cryst10090844",
        "https://research.knu.ac.kr/en/publications/"
        "proton-uv-and-x-ray-induced-luminescence-in-tbsup3sup-doped-lugds/",
        "KNU exact DOI article and author-linked person record",
        "KNU's Crystals 10(9):844 record links H. J. Kim to Hong Joo Kim. "
        "This literal-markup rendering has the same seven authors and "
        "source work as already excluded Scholar row 474.",
    ),
    804: (
        "Luminescent investigations of Li {sub 6} Lu (BO {sub 3}){sub 3}: "
        "Tb {sup 3+}, Dy {sup 3+} phosphors",
        2014,
        "Hong Joo Kim",
        "10.1016/j.jallcom.2014.05.001",
        "https://research.knu.ac.kr/en/publications/"
        "luminescent-investigations-of-lisub6sublubosub3sub-sub3subtbsup3s/",
        "KNU exact DOI article and author-linked person record",
        "KNU's Journal of Alloys and Compounds 610:281-287 record links "
        "H. J. Kim to Hong Joo Kim. This literal-markup rendering has the "
        "same five authors and source work as already excluded row 231.",
    ),
    810: (
        "Up-and downconversion luminescence properties of [Nd. sup. 3+] "
        "ions doped in [Bi. sub. 2][O. sub. 3]-BaO-[B. sub. 2][O. sub. 3] "
        "glass system.",
        2014,
        "Hong Joo Kim",
        "10.1155/2014/751973",
        "https://research.knu.ac.kr/en/publications/"
        "up-and-downconversion-luminescence-properties-of-ndsup3sup-ions-d/",
        "KNU exact DOI article and author-linked person record",
        "KNU's Advances in Materials Science and Engineering article 751973 "
        "links H. J. Kim to Hong Joo Kim. This literal-markup rendering has "
        "the same author sequence and source work as already excluded row 346.",
    ),
    826: (
        "Scintillation properties of Cs {sub 2} LiGdCl {sub 6}: Ce {sup 3+}",
        2010,
        "Hong Joo Kim",
        "10.1016/j.radmeas.2009.10.018",
        "https://www.ifj.edu.pl/conf/lumdetr/Book_of_Abstracts_Lumdetr2009.pdf",
        "original exact-title conference abstract; KNU ORCID exact journal DOI",
        "The original 2009 LumDeTr abstract for this exact title lists "
        "H. J. Kim at KNU Physics with correspondence hongjoo@knu.ac.kr, "
        "matching KNU's Hong Joo Kim person record. This 2010 Radiation "
        "Measurements 45 journal rendering has the same three journal "
        "authors and source work as already excluded row 499, whose KNU "
        "ORCID work record supplies this exact DOI. The earlier abstract "
        "has two additional coauthors and is identity evidence, not the "
        "journal article being excluded.",
    ),
}
OTHER_PRIMARY.update(FOLLOWUP_KNU_DUPLICATE_PRIMARY)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        666: "bC4aq-4AAAAJ:h168fVGZblEC",
        804: "bC4aq-4AAAAJ:vVJNg6_NJEsC",
        810: "bC4aq-4AAAAJ:-7ulzOJl1JYC",
        826: "bC4aq-4AAAAJ:JH5k92_tO-AC",
    }
)
FOLLOWUP_KNU_DUPLICATE_VENUES = {
    666: "Crystals 10 (9), 2020",
    804: "Journal of Alloys and Compounds 610, 2014",
    810: "Advances in Materials Science and Engineering, NA-NA, 2014",
    826: "Radiation Measurements 45, 2010",
}
FOLLOWUP_KNU_DUPLICATE_AUTHORS = {
    666: "U Fawad, HJ Kim, I Gul, M Khan, S Tahir, T Jamal, W Muhammad",
    804: "U Fawad, M Oh, H Park, S Kim, HJ Kim",
    810: "R Ruamnikhom, P Limsuwan, M Horprathum, N Chanthima, HJ Kim, ...",
    826: "G Rooh, HJ Kim, S Kim",
}

# The original Nature pages expand the exact COSINE bylines to Hongjoo Kim
# at Kyungpook Physics. These two captured titles are unique, not inferred
# from adjacent initials-only collaboration papers.
COSINE_FULLNAME_PRIMARY = {
    325: (
        "Author Correction: An experiment to search for dark-matter "
        "interactions using sodium iodide detectors",
        2019,
        "Hongjoo Kim",
        "10.1038/s41586-019-0890-3",
        "https://www.nature.com/articles/s41586-019-0890-3",
        "Nature exact correction and complete COSINE-100 collaborator list",
        "Nature identifies this exact correction to the 2018 detector paper "
        "as a 2019 Author Correction, volume 566 page E2, and expands "
        "the collaboration's H. J. Kim to Hongjoo Kim of Kyungpook "
        "National University. The captured venue's page 1 is inaccurate; "
        "this is also a correction rather than a new research article.",
    ),
    366: (
        "Upgrading the COSINE-100 experiment for enhanced sensitivity "
        "to low-mass dark matter detection",
        2025,
        "Hongjoo Kim",
        "10.1038/s42005-025-02067-4",
        "https://www.nature.com/articles/s42005-025-02067-4",
        "Communications Physics original full byline and affiliation",
        "The exact Communications Physics 8:135 article prints Hongjoo "
        "Kim in its full author list and assigns him to the Department "
        "of Physics, Kyungpook National University, not CMU Chemistry.",
    ),
}
OTHER_PRIMARY.update(COSINE_FULLNAME_PRIMARY)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        325: "bC4aq-4AAAAJ:mgoTDWlsYNUC",
        366: "bC4aq-4AAAAJ:8JTMrWI6FdcC",
    }
)
COSINE_FULLNAME_VENUES = {
    325: "Nature 566, 1, 2019",
    366: "Communications Physics 8 (1), 135, 2025",
}
COSINE_FULLNAME_AUTHORS = {
    325: "G Adhikari",
    366: "D Lee, JY Cho, CH Ha, E Jeon, H Kim, J Kim, K Kim, SH Kim, SK Kim, ...",
}

# These three exact DOI records identify the initials-only author through
# institutional author links, a full university byline, or publisher-linked
# ORCID person records. Similar collaborations without that identity link
# remain held.
KNU_COSINE_AMORE_PRIMARY = {
    162: (
        "Improved Limit on Neutrinoless Double Beta Decay of  from AMoRE-I",
        2025,
        "Hong Joo Kim",
        "10.1103/PhysRevLett.134.082501",
        "https://api.crossref.org/works/10.1103%2FPhysRevLett.134.082501",
        "APS publisher-deposited exact author ORCID and KNU person record",
        "APS's deposited record for Physical Review Letters 134:082501 "
        "assigns its H. J. Kim author ORCID 0000-0001-9787-4684. The "
        "ORCID person record names HongJoo Kim, whom KNU identifies as a "
        "Physics professor; CMU's Hyung J. Kim has a different ORCID. "
        "The same byline lists Ho-Jong Kim as a separate author.",
    ),
    353: (
        "Measurements of low-energy nuclear recoil quenching factors for Na "
        "and I recoils in the NaI (Tl) scintillator",
        2024,
        "Hong Joo Kim",
        "10.1103/PhysRevC.110.014614",
        "https://research.knu.ac.kr/en/publications/measurements-of-low-energy-nuclear-recoil-quenching-factors-for-n/",
        "Kyungpook National University exact DOI author-to-person link",
        "The Kyungpook National University record for Physical Review C "
        "110:014614 links its H. J. Kim author directly to Hong Joo Kim's "
        "Physics faculty page. He is distinct from CMU Chemistry's Hyung J. Kim.",
    ),
    408: (
        "Limits on WIMP dark matter with NaI (Tl) crystals in three years "
        "of COSINE-100 data",
        2025,
        "Hongjoo Kim",
        "10.1103/77p2-rnbw",
        "https://scholar.korea.ac.kr/item/c10ee6c6-a308-44a3-a039-13d2f39d10c1",
        "Korea University exact DOI complete full-name byline and APS affiliation",
        "Korea University's record of the exact Physical Review Letters "
        "135:231001 article and DOI names Hongjoo Kim in the complete "
        "byline. The APS original identifies H. J. Kim's affiliation as "
        "Kyungpook National University Physics, not CMU Chemistry.",
    ),
    519: (
        "Radioassay of the materials for AMoRE-II experiment",
        2024,
        "Ho-Jong Kim and HongJoo Kim",
        "10.3389/fphy.2024.1362209",
        "https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1362209/full",
        "Frontiers original byline with two author-linked ORCID identities",
        "The Frontiers article for volume 12:1362209 lists two H. J. Kim "
        "authors and links them to ORCID 0000-0002-8265-5283 (Ho-Jong Kim) "
        "and 0000-0001-9787-4684 (HongJoo Kim). Neither is CMU Chemistry "
        "professor Hyung J. Kim.",
    ),
}
OTHER_PRIMARY.update(KNU_COSINE_AMORE_PRIMARY)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        162: "bC4aq-4AAAAJ:qSd0DAb9jMoC",
        353: "bC4aq-4AAAAJ:RgznTc0nqo4C",
        408: "bC4aq-4AAAAJ:e0LTWoPxLYMC",
        519: "bC4aq-4AAAAJ:oXKBmVzQOggC",
    }
)
KNU_COSINE_AMORE_VENUES = {
    162: "Physical review letters 134 (8), 082501, 2025",
    353: "Physical Review C 110 (1), 014614, 2024",
    408: "Physical review letters 135 (23), 231001, 2025",
    519: "Frontiers in Physics 12, 1362209, 2024",
}
KNU_COSINE_AMORE_AUTHORS = {
    162: "A Agrawal, VV Alenkov, P Aryal, J Beyer, B Bhandari, RS Boiko, K Boonin, ...",
    353: "SH Lee, HW Joo, HJ Kim, KW Kim, SK Kim, YD Kim, YJ Ko, HS Lee, ...",
    408: "GH Yu, N Carlin, JY Cho, JJ Choi, S Choi, AC Ezeribe, LE França, C Ha, ...",
    519: "A Agrawal, VV Alenkov, P Aryal, H Bae, J Beyer, B Bhandari, RS Boiko, ...",
}

HYOJOON_SOLE_AUTHOR_PRIMARY = {
    537: (
        "Relationship between Kubo-transformed and classical correlation functions",
        2007,
        "Hyojoon Kim",
        "10.1016/j.cplett.2007.01.027",
        "https://api.crossref.org/works/10.1016%2Fj.cplett.2007.01.027",
        "Elsevier publisher-deposited sole full-author DOI record",
        "The publisher-deposited record names sole author Hyojoon Kim on the "
        "exact Chemical Physics Letters 436:111-115 article. His Dong-A "
        "University personal bibliography independently lists the exact "
        "title, year, volume, and pages. He is not CMU Chemistry professor "
        "Hyung J. Kim.",
    )
}
OTHER_PRIMARY.update(HYOJOON_SOLE_AUTHOR_PRIMARY)
OTHER_PRIMARY_ID_GUARDS[537] = "bC4aq-4AAAAJ:PR6Y55bgFSsC"
HYOJOON_SOLE_AUTHOR_VENUE = "Chemical physics letters 436 (1-3), 111-115, 2007"
HYOJOON_SOLE_AUTHOR_AUTHORS = "H Kim"

HYUN_JUNG_ORCID_PRIMARY = {
    24: (
        "Second-line chemotherapy versus supportive cancer treatment in "
        "advanced gastric cancer: a meta-analysis",
        2013,
        "Hyun Jung Kim",
        "10.1093/annonc/mdt351",
        "https://pub.orcid.org/v3.0/0000-0003-2018-2385/works",
        "Hyun Jung Kim ORCID exact DOI and original-journal affiliation",
        "The Annals of Oncology original article assigns its H. J. Kim author "
        "to the Department of Preventive Medicine, Korea University College "
        "of Medicine. ORCID 0000-0003-2018-2385 names Hyun Jung Kim and "
        "lists this exact title, 2013 journal article, and DOI; the ORCID "
        "record also gives Korea University employment and education in "
        "Preventive Medicine. This is not CMU Chemistry professor Hyung J. Kim.",
    )
}
OTHER_PRIMARY.update(HYUN_JUNG_ORCID_PRIMARY)
OTHER_PRIMARY_ID_GUARDS[24] = "bC4aq-4AAAAJ:jRIwE-1ttnoC"
HYUN_JUNG_ORCID_VENUE = "Annals of Oncology 24 (11), 2850-2854, 2013"
HYUN_JUNG_ORCID_AUTHORS = (
    "HS Kim, HJ Kim, SY Kim, TY Kim, KW Lee, SK Baek, MH Ryu, BH Nam, ..."
)

# The Electrochemical Society deposited the full names for this exact meeting
# abstract. Its H. J. Kim is Hyoung-Juhn Kim, a separate fuel-cell researcher.
ECS_OTHER_PERSON_PRIMARY = {
    857: (
        "Effects of Gradient Loading of Cathode Catalyst on the fuel Cell Performance",
        2006,
        "Hyoung-Juhn Kim",
        "10.1149/ma2005-03/1/33",
        "https://api.crossref.org/works/10.1149%2Fma2005-03%2F1%2F33",
        "Electrochemical Society publisher-deposited meeting abstract",
        "The society's exact abstract has the same six authors in order, 2006 "
        "publication date and abstract number 33, and expands its H. J. Kim "
        "author to Hyoung-Juhn Kim, not CMU Chemistry's Hyung J. Kim.",
    )
}
OTHER_PRIMARY.update(ECS_OTHER_PERSON_PRIMARY)
OTHER_PRIMARY_ID_GUARDS[857] = "bC4aq-4AAAAJ:KUbvn5osdkgC"
ECS_OTHER_PERSON_VENUE = "Electrochemical Society Meeting Abstracts PBFC-2, 33-33, 2006"
ECS_OTHER_PERSON_AUTHORS = "E Cho, M Prasanna, HJ Kim, TH Lim, IH Oh, SA Hong"

# The original Korean Society of Applied Entomology pages for these exact
# articles spell the H. J. Kim author's name 김현주. The Korean national
# citation index assigns that author to the Yeongnam crop-research station;
# a Korean academic repository romanizes the same crop-research author as
# Hyun-Joo Kim. These are exact article identifications, not exclusions from
# subject area or from initials alone.
ENTOMOLOGY_OTHER_PERSON_PRIMARY = {
    286: (
        "Effects of food combinations of leguminous seeds on nymphal "
        "development, adult longevity and oviposition of bean bug, "
        "Riptortus clavatus Thunberg",
        2004,
        "Hyun-Joo Kim (김현주)",
        "",
        "https://entomology2.or.kr/journal/article.php?Array=&code=10874",
        "Korean Society of Applied Entomology original article",
        "The society's exact volume 43 issue 2, pages 123–127 article "
        "names its Kim coauthor 김현주; the Korean national citation index "
        "assigns that author to the Yeongnam crop-research station.",
    ),
    341: (
        "Insecticide susceptibility in the different larva of tabacco "
        "cutworm, Spodoptera litura Fabricius (Lepidoptera: Noctuidae) "
        "collected in the soybean fields of Milyang, Korea",
        2003,
        "Hyun-Joo Kim (김현주)",
        "",
        "https://www.entomology2.or.kr/journal/article.php?Array=&code=10834",
        "Korean Society of Applied Entomology original article",
        "The society's exact volume 42 issue 3, pages 225–231 article "
        "names its Kim coauthor 김현주; the Korean national citation index "
        "assigns that author to the Yeongnam agricultural experiment station.",
    ),
}
OTHER_PRIMARY.update(ENTOMOLOGY_OTHER_PERSON_PRIMARY)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        286: "bC4aq-4AAAAJ:9tXw7Op4-u0C",
        341: "bC4aq-4AAAAJ:TiIbgCYny7sC",
    }
)
ENTOMOLOGY_OTHER_PERSON_VENUES = {
    286: "Korean Journal of Applied Entomology 43 (2), 2004",
    341: "Korean Journal of Applied Entomology 42 (3), 2003",
}
ENTOMOLOGY_OTHER_PERSON_AUTHORS = {
    286: "SD Bae, HJ Kim, HJ Cho, JK Park, JK Jung",
    341: "SD Bae, HJ Kim, BR Choi, YH Song",
}

# The corresponding author's own CAU energy-lab bibliography expands the
# H. J. Kim on this exact Elsevier article to Hyoung-Juhn Kim. Unlike a
# same-topic team inference, the entry identifies this work by title, journal,
# year, volume, and pages, with the same eight-author order.
DECAL_OTHER_PERSON_PRIMARY = {
    356: (
        "Fabrication of MEA with hydrocarbon based membranes using low "
        "temperature decal method for DMFC",
        2010,
        "Hyoung-Juhn Kim",
        "10.1016/j.ijhydene.2010.03.023",
        "https://cauenpl.wixsite.com/sookilkimlab/general-5",
        "Original coauthor lab bibliography with expanded exact-work byline",
        "Soo-Kil Kim's CAU energy-lab bibliography identifies the exact "
        "International Journal of Hydrogen Energy 35:5647–5655 article "
        "and expands its fourth author to Hyoung-Juhn Kim, not CMU "
        "Chemistry professor Hyung J. Kim.",
    ),
}
OTHER_PRIMARY.update(DECAL_OTHER_PERSON_PRIMARY)
OTHER_PRIMARY_ID_GUARDS[356] = "bC4aq-4AAAAJ:bz8QjSJIRt4C"
DECAL_OTHER_PERSON_VENUE = (
    "International journal of hydrogen energy 35 (11), 5647-5655, 2010"
)
DECAL_OTHER_PERSON_AUTHORS = (
    "NN Krishnan, J Prabhuram, YT Hong, HJ Kim, K Yoon, HY Ha, TH Lim, ..."
)

# Complete original-journal author lists for these exact outputs do not contain
# CMU professor Hyung J. Kim, even when another author's initials overlap.
ABSENT_AUTHOR_PRIMARY = {
    394: (
        "bC4aq-4AAAAJ:ndLnGcHYRF0C",
        "Establishing boundary conditions considering influence factors of "
        "the room equipped with a ceiling radiant cooling panel",
        2020,
        "10.3390/en13071684",
        "https://www.mdpi.com/1996-1073/13/7/1684",
        "MDPI original complete author list",
        "The original Energies 13(7):1684 article names all six authors "
        "in full: Sang-Hoon Park, Dong-Woo Kim, Goo-Sang Joe, "
        "Seong-Ryong Ryu, Myoung-Souk Yeo, and Kwang-Woo Kim. "
        "CMU Chemistry professor Hyung J. Kim is absent from its byline.",
    ),
    396: (
        "bC4aq-4AAAAJ:BUYA1_V_uYcC",
        "Analysis on the post-irradiation examination of the HANARO "
        "miniplate-1 irradiation test for kijang research reactor",
        2017,
        "10.1016/j.net.2017.04.003",
        "https://www.kns.org/uploads/TOC2.pdf",
        "Korean Nuclear Society original journal table of contents",
        "The society's complete full-name byline for the exact Nuclear "
        "Engineering and Technology 49(5):1044–1062 article identifies "
        "its fifth author as Heemoon Kim and eleventh as Hyun Jung Kim. "
        "CMU Chemistry professor Hyung J. Kim is absent.",
    ),
    90: (
        "bC4aq-4AAAAJ:kuK5TVdYjLIC",
        "Reduced transforming growth factor-beta type II receptor (TGF-beta "
        "RII) expression in adenocarcinoma of the lung.",
        1999,
        "",
        "https://pubmed.ncbi.nlm.nih.gov/10226558/",
        "NLM complete original-journal author list",
        "The complete ten-author NLM record for this exact 1999 Anticancer "
        "Research 19(1A):301-306 article lists W. S. Kim, C. Park, "
        "Y. S. Jung, H. S. Kim, J. Han, C. H. Park, K. Kim, J. Kim, "
        "Y. M. Shim, and K. Park. No H. J. Kim is an author, so the "
        "Scholar citation cannot belong to CMU Chemistry professor "
        "Hyung J. Kim.",
    ),
    70: (
        "bC4aq-4AAAAJ:WzTVkKNmPSkC",
        "Overview of KSTAR initial operation",
        2011,
        "10.1088/0029-5515/51/9/094006",
        "https://api.crossref.org/works/10.1088/0029-5515/51/9/094006",
        "IOP publisher-deposited complete author list",
        "The IOP publisher-deposited complete 147-author list for the exact "
        "2011 Nuclear Fusion 51:094006 article has no H. J. Kim. Its Kim "
        "authors have other initials, so this Scholar citation cannot "
        "belong to CMU Chemistry professor Hyung J. Kim.",
    ),
    203: (
        "bC4aq-4AAAAJ:JdL-Xu2nR38C",
        "Measurement of the gamma ray background in the Davis cavern at "
        "the Sanford Underground Research Facility",
        2020,
        "10.1016/j.astropartphys.2019.102391",
        "https://api.crossref.org/works/10.1016%2Fj.astropartphys.2019.102391",
        "Elsevier publisher-deposited complete author list",
        "The Elsevier publisher-deposited complete 167-author list for the "
        "exact Astroparticle Physics 116:102391 article contains only "
        "W. T. Kim among Kim authors and no H. J. Kim. The original "
        "LUX-ZEPLIN accepted manuscript deposited at UCL agrees.",
    ),
    403: (
        "bC4aq-4AAAAJ:kz9GbA2Ns4gC",
        "The optimal effect site concentration of remifentanil in combination "
        "with intravenous midazolam and topical lidocaine for awake "
        "fibreoptic nasotracheal intubation in patients\xa0…",
        2012,
        "",
        "https://www.minervamedica.it/en/journals/minerva-anestesiologica/"
        "article.php?cod=R02Y2012N05A0521",
        "Minerva Medica original complete article byline",
        "The original Minerva Anestesiologica 78(5):521-526 article "
        "lists J. W. Song, Y. L. Kwak, J. W. Lee, C. H. Chang, "
        "H. S. Kim, and Y. H. Shim. It has no H. J. Kim author; the "
        "saved Scholar title is truncated but its year, journal, pages, "
        "and six-author order match this one article.",
    ),
}

ABSENT_AUTHOR_VENUES = {
    203: "Astroparticle physics 116, 102391, 2020",
    403: "Minerva anestesiologica 78 (5), 521-526, 2012",
    394: "Energies 13 (7), 1684, 2020",
    396: "Nuclear Engineering and Technology 49 (5), 1044-1062, 2017",
}
ABSENT_AUTHOR_AUTHORS = {
    394: "SH Park, DW Kim, GS Joe, SR Ryu, MS Yeo, KW Kim",
    396: "JM Park, YW Tahk, YJ Jeong, KH Lee, H Kim, YH Jung, BO Yoo, YG Jin, ...",
}

# Four previously held observations have exact publisher or institutional
# author records. Preserve the full saved Scholar fingerprint for each one;
# an index-only or initials-only match cannot inherit these decisions.
EXACT_FOLLOWUP_CAPTURE_GUARDS = {
    511: (
        "bC4aq-4AAAAJ:KbBQZpvPDL4C",
        "Evaluation of leaching potential of heavy metals from bottom ashes "
        "generated in coal-fired power plants in Korea",
        "D Park, H Choi, NC Woo, H Kim, D Chung",
        "2013",
        "Journal of Soil and Groundwater Environment 18 (7), 32-40, 2013",
    ),
    637: (
        "bC4aq-4AAAAJ:E8ajGqO0XoUC",
        "Nanowire length and diameter dependent electrical and optical "
        "properties of transparent conductive silver nanowires films",
        "J Hwang, HS Kim, JW Roh",
        "2018",
        "International Journal of Nanotechnology 15 (6-7), 630-637, 2018",
    ),
    717: (
        "bC4aq-4AAAAJ:iKswqCX-FLkC",
        "Study of the Scintillation Properties and of the Proton-Induced "
        "Radiation Damage in the K2CeCl5 Single Crystal",
        "G Rooh, H Kang, J Moon, K Kim",
        "2009",
        "Journal of the Korean Physical Society, 2093-2097, 2009",
    ),
    775: (
        "bC4aq-4AAAAJ:V3AGJWp-ZtQC",
        "HLA-DRB1 and DQB1 gene polymorphism is associated with disease "
        "severity, recurrence and drug resistance of pulmonary "
        "tuberculosis in Koreans",
        "HS Kim, MH Park, EY Song, HJ Park, SY Kwon, SK Han, YS Shim",
        "2003",
        "Human Immunology 64 (10), S139, 2003",
    ),
}
ABSENT_AUTHOR_PRIMARY.update(
    {
        637: (
            EXACT_FOLLOWUP_CAPTURE_GUARDS[637][0],
            EXACT_FOLLOWUP_CAPTURE_GUARDS[637][1],
            2018,
            "10.1504/IJNT.2018.096353",
            "https://www.inderscience.com/info/inarticletoc.php?"
            "issue=6%2F7&jcode=ijnt&vol=15&year=2018",
            "Inderscience original issue and complete three-author byline",
            "Inderscience's exact International Journal of Nanotechnology "
            "15(6/7):630-637 article names Hyun-Sik Kim, not CMU Chemistry "
            "professor Hyung J. Kim.",
        ),
        775: (
            EXACT_FOLLOWUP_CAPTURE_GUARDS[775][0],
            EXACT_FOLLOWUP_CAPTURE_GUARDS[775][1],
            2003,
            "10.1016/j.humimm.2003.08.261",
            "https://api.crossref.org/works/10.1016%2Fj.humimm.2003.08.261",
            "Elsevier publisher-deposited complete seven-author DOI byline",
            "Elsevier's complete seven-author byline for the exact Human "
            "Immunology 64(10):S139 abstract includes H. S. Kim and H. J. "
            "Park, but no H. J. Kim. This Scholar observation is not a work "
            "by CMU Chemistry professor Hyung J. Kim.",
        ),
    }
)
OTHER_PRIMARY.update(
    {
        511: (
            EXACT_FOLLOWUP_CAPTURE_GUARDS[511][1],
            2013,
            "Hee Joung Kim",
            "10.7857/JSGE.2013.18.7.032",
            "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/"
            "ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001843871",
            "Korean Citation Index exact full-author and affiliation record; "
            "publisher-deposited DOI byline",
            "The Korean Citation Index names the exact 2013 Journal of Soil "
            "and Groundwater Environment 18(7):32-40 article's H. Kim author "
            "as Hee Joung Kim at Kangwon University. The publisher-deposited "
            "DOI record independently names Heejoung Kim in the same five-author "
            "byline, distinct from CMU Chemistry professor Hyung J. Kim.",
        ),
        717: (
            EXACT_FOLLOWUP_CAPTURE_GUARDS[717][1],
            2009,
            "Hong Joo Kim",
            "10.3938/jkps.54.2093",
            "https://research.knu.ac.kr/en/publications/"
            "study-of-the-scintillation-properties-and-of-the-proton-induced-r",
            "Kyungpook National University exact article author-to-person link",
            "KNU's record matches the 2009 Journal of the Korean Physical Society "
            "54:2093-2097 article and links its H. J. Kim author to Physics "
            "professor Hong Joo Kim, distinct from CMU Chemistry professor "
            "Hyung J. Kim.",
        ),
    }
)
OTHER_PRIMARY_ID_GUARDS.update(
    {
        511: EXACT_FOLLOWUP_CAPTURE_GUARDS[511][0],
        717: EXACT_FOLLOWUP_CAPTURE_GUARDS[717][0],
    }
)

# Three held observations now have exact work-to-HongJoo Kim links. The 2016
# IEEE DOI is on his public ORCID works list; Springer prints his ORCID on the
# 2025 article; KNU links the 2005 paper's H. J. Kim byline to his person page.
KNU_THREE_EXACT_GUARDS = {
    374: (
        "bC4aq-4AAAAJ:R22Rs3tN8aoC",
        "A Study of Radioactive Contamination of  Crystals for the AMoRE Experiment",
        "JY Lee, V Alenkov, L Ali, J Beyer, R Bibi, RS Boiko, K Boonin, O Buzanov, ...",
        "2016",
        "IEEE Transactions on Nuclear Science 63 (2), 543-547, 2016",
    ),
    379: (
        "bC4aq-4AAAAJ:dMpQl7XwOw4C",
        "Projected background and sensitivity of AMoRE-II",
        "A Agrawal, VV Alenkov, P Aryal, J Beyer, B Bhandari, RS Boiko, K Boonin, ...",
        "2025",
        "The European Physical Journal C 85 (1), 9, 2025",
    ),
    390: (
        "bC4aq-4AAAAJ:mel-f30kHHgC",
        "Inhibition of Cs137 contamination in cesium iodide",
        "YD Kim, IS Hahn, MJ Hwang, L Jin, WG Kang, HJ Kim, SC Kim, SK Kim, ...",
        "2005",
        "Nuclear Instruments and Methods in Physics Research Section A: "
        "Accelerators\xa0…, 2005",
    ),
}
EXACT_FOLLOWUP_CAPTURE_GUARDS.update(KNU_THREE_EXACT_GUARDS)
OTHER_PRIMARY.update(
    {
        374: (
            KNU_THREE_EXACT_GUARDS[374][1],
            2016,
            "HongJoo Kim",
            "10.1109/TNS.2016.2530828",
            "https://pub.orcid.org/v3.0/0000-0001-9787-4684/works",
            "HongJoo Kim public ORCID person and exact IEEE DOI work records",
            "The public ORCID person record names HongJoo Kim, and its works "
            "list carries the exact 2016 IEEE DOI 10.1109/TNS.2016.2530828 "
            "as a self-linked work. The Scholar title loses the crystal formula, "
            "but journal, volume, pages, year, and leading authors match.",
        ),
        379: (
            KNU_THREE_EXACT_GUARDS[379][1],
            2025,
            "HongJoo Kim",
            "10.1140/epjc/s10052-024-13516-9",
            "https://link.springer.com/article/10.1140/epjc/s10052-024-13516-9",
            "Springer exact article H. J. Kim-to-ORCID author link",
            "Springer's exact 2025 article links H. J. Kim to ORCID "
            "0000-0001-9787-4684, whose public person record names HongJoo Kim.",
        ),
        390: (
            KNU_THREE_EXACT_GUARDS[390][1],
            2005,
            "Hong Joo Kim",
            "10.1016/j.nima.2005.06.080",
            "https://research.knu.ac.kr/en/publications/"
            "inhibition-of-sup137supcs-contamination-in-cesium-iodide/",
            "KNU exact Elsevier article H. J. Kim-to-person link",
            "KNU's exact DOI, volume 552, pages 456-462 article links its "
            "H. J. Kim author to KNU Physics professor Hong Joo Kim.",
        ),
    }
)
OTHER_PRIMARY_ID_GUARDS.update(
    {index: guard[0] for index, guard in KNU_THREE_EXACT_GUARDS.items()}
)

# Original publisher or university records resolve four more exact works to
# named researchers other than CMU Chemistry's Hyung J. Kim.
NEXT_FOUR_EXACT_GUARDS = {
    420: (
        "bC4aq-4AAAAJ:aIdbFUkbNIkC",
        "The  crystal: a crystal scintillator for dark matter search experiment",
        "IR Pandey, HJ Kim, HS Lee, YD Kim, MH Lee, VD Grigorieva, VN Shlegel",
        "2018",
        "The European Physical Journal C 78 (11), 973, 2018",
    ),
    430: (
        "bC4aq-4AAAAJ:anf4URPfarAC",
        "Optical properties of GaN by using ellipsometry and a band calculation",
        "TJ Kim, JS Byun, YD Kim, YC Chang, HJ Kim",
        "2008",
        "Journal of the Korean Physical Society 53 (3), 1575-1579, 2008",
    ),
    432: (
        "bC4aq-4AAAAJ:edDO8Oi4QzsC",
        "Large Size CdWO4 Crystal for Energetic X- and γ-Ray Detection",
        "HJ Kim, HD Kang, H Park, S Doh, SH Kim, SJ Kang",
        "2008",
        "Journal of Nuclear Science and Technology 45 (sup5), 356-359, 2008",
    ),
    435: (
        "bC4aq-4AAAAJ:F9HO9s0W2bwC",
        "Background study of the AMoRE-pilot experiment",
        "A Agrawal, VV Alenkov, P Aryal, J Beyer, B Bhandari, RS Boiko, K Boonin, ...",
        "2024",
        "Astroparticle Physics 162, 102991, 2024",
    ),
}
EXACT_FOLLOWUP_CAPTURE_GUARDS.update(NEXT_FOUR_EXACT_GUARDS)
OTHER_PRIMARY.update(
    {
        420: (
            NEXT_FOUR_EXACT_GUARDS[420][1],
            2018,
            "HongJoo Kim",
            "10.1140/epjc/s10052-018-6462-0",
            "https://link.springer.com/article/10.1140/epjc/s10052-018-6462-0",
            "Springer exact article H. J. Kim-to-ORCID author link",
            "Springer's original EPJC 78:973 article restores the Na2W2O7 "
            "formula lost by Scholar and links H. J. Kim to ORCID "
            "0000-0001-9787-4684, whose public person record names HongJoo Kim.",
        ),
        430: (
            NEXT_FOUR_EXACT_GUARDS[430][1],
            2008,
            "Hye Jung Kim",
            "10.3938/jkps.53.1575",
            "https://khu.elsevierpure.com/en/publications/"
            "optical-properties-of-gan-by-using-ellipsometry-and-a-band-calcul-2/",
            "Kyung Hee University exact full-author journal record",
            "Kyung Hee University's exact Journal of the Korean Physical Society "
            "53(3):1575-1579 record names the fifth author Hye Jung Kim in full.",
        ),
        432: (
            NEXT_FOUR_EXACT_GUARDS[432][1],
            2008,
            "Hong Joo Kim",
            "10.1080/00223131.2008.10875862",
            "https://research.knu.ac.kr/en/publications/"
            "large-size-cdwosub4sub-crystal-for-energetic-x-and-%CE%B3-ray-detectio/",
            "KNU exact journal article H. J. Kim-to-person link",
            "KNU's exact Journal of Nuclear Science and Technology 45:356-359 "
            "article links its first H. J. Kim author to Physics professor "
            "Hong Joo Kim.",
        ),
        435: (
            NEXT_FOUR_EXACT_GUARDS[435][1],
            2024,
            "Hong Joo Kim",
            "10.1016/j.astropartphys.2024.102991",
            "https://research.knu.ac.kr/en/publications/"
            "background-study-of-the-amore-pilot-experiment/",
            "KNU exact Elsevier article H. J. Kim-to-person link",
            "KNU's exact Astroparticle Physics 162:102991 article links its "
            "H. J. Kim author to Physics professor Hong Joo Kim, distinct "
            "from another Ho Jong Kim author in the same collaboration list.",
        ),
    }
)
OTHER_PRIMARY_ID_GUARDS.update(
    {index: guard[0] for index, guard in NEXT_FOUR_EXACT_GUARDS.items()}
)

# Six more saved observations resolve through exact institutional, journal,
# NLM, or named-person ORCID records. Four neighboring initials-only records
# remain held: their affiliations and subjects do not identify an individual.
MID_SIX_EXACT_GUARDS = {
    467: (
        "bC4aq-4AAAAJ:Ej9njvOgR2oC",
        "Progress of kstar 5-ghz lower hybrid current drive system",
        "S Park, YS Bae, JH Kim, H Do, HT Kim, KM Kim, HK Kim, HJ Kim, WS Han, ...",
        "2013",
        "Fusion Science and Technology 63 (1), 49-58, 2013",
    ),
    469: (
        "bC4aq-4AAAAJ:b1wdh0AR-JQC",
        "Diterpenes of Biota orientalis leaves",
        "SH Sung, KA Koo, HK Lim, HS Lee, JH Cho, HS Kim, YC Kim",
        "1998",
        "Korean Journal of Pharmacognosy 29 (4), 347-352, 1998",
    ),
    476: (
        "bC4aq-4AAAAJ:LPtt_HFRSbwC",
        "Cerium-Doped  Scintillator for X-Ray and -Ray Detection",
        "G Rooh, HJ Kim, H Park, S Kim, H Jiang",
        "2013",
        "IEEE Transactions on Nuclear Science 61 (1), 397-401, 2013",
    ),
    490: (
        "bC4aq-4AAAAJ:lbI08cpqPnQC",
        "Phonon–Scintillation Detection Systems with MMC Readout",
        "HL Kim, HJ Kim, I Kim, SR Kim, YD Kim, YH Kim, DH Kwon, JA Jeon, ...",
        "2020",
        "Journal of Low Temperature Physics 199 (3), 1082-1088, 2020",
    ),
    491: (
        "bC4aq-4AAAAJ:U5uP8zs9lfgC",
        "Clinical characteristics of pediatric cutaneous lupus erythematosus: "
        "experience from a tertiary referral center in Korea",
        "SK Lee, J Baek, JY Roh, HJ Kim",
        "2019",
        "Lupus 28 (7), 888-892, 2019",
    ),
    494: (
        "bC4aq-4AAAAJ:_Ybze24A_UAC",
        "Is there any age cutoff to treat elderly patients with head and neck "
        "cancer? Comparing with septuagenarians and octogenarians",
        "H Kim, SD Kim, YJ Shim, SY Lee, MW Sung, KH Kim, JH Hah",
        "2016",
        "Journal of Korean Medical Science 31 (8), 1300-1306, 2016",
    ),
}
EXACT_FOLLOWUP_CAPTURE_GUARDS.update(MID_SIX_EXACT_GUARDS)
OTHER_PRIMARY.update(
    {
        467: (
            MID_SIX_EXACT_GUARDS[467][1],
            2013,
            "Haejin Kim",
            "10.13182/FST12-493",
            "https://jobs.iterkorea.org/thesis?amount=10&mid=a20302000000"
            "&pageNum=4&pblYear=2012",
            "Korea Institute of Fusion Energy exact journal record and "
            "publisher-deposited DOI metadata",
            "The institute spells the eighth author Kim Haejin. Its title, "
            "Fusion Science and Technology venue, and first eight author "
            "positions match Crossref DOI 10.13182/FST12-493, published in "
            "volume 63(1):49-58 in January 2013. The institute displays "
            "August 2012, but gives the same journal work and author order.",
        ),
        476: (
            MID_SIX_EXACT_GUARDS[476][1],
            2013,
            "HongJoo Kim",
            "10.1109/TNS.2013.2283882",
            "https://pub.orcid.org/v3.0/0000-0001-9787-4684/work/153825465",
            "HongJoo Kim public ORCID exact IEEE DOI work record",
            "The exact IEEE Transactions on Nuclear Science 61(1):397-401 "
            "article is a self-linked DOI work on HongJoo Kim's ORCID. The "
            "DOI carries 2013, while the journal issue is dated 2014 and "
            "the Scholar observation displays 2013; the title formula is "
            "missing from Scholar's rendering.",
        ),
        490: (
            MID_SIX_EXACT_GUARDS[490][1],
            2020,
            "HongJoo Kim",
            "10.1007/s10909-019-02291-7",
            "https://pub.orcid.org/v3.0/0000-0001-9787-4684/work/153825252",
            "HongJoo Kim public ORCID exact Springer DOI work record",
            "The exact 2020 Journal of Low Temperature Physics 199:1082-1088 "
            "article is a self-linked DOI work on HongJoo Kim's ORCID. "
            "Scholar omits Li2MoO4 from the title, but the remaining title, "
            "journal, pages, year, and leading authors match.",
        ),
        491: (
            MID_SIX_EXACT_GUARDS[491][1],
            2019,
            "Hee Joo Kim",
            "10.1177/0961203319851568",
            "https://pubmed.ncbi.nlm.nih.gov/31112454/",
            "NLM exact article author-to-ORCID link and public ORCID person name",
            "The NLM XML attaches ORCID 0000-0003-1585-8184 to this exact "
            "Lupus article's H. J. Kim author; the public ORCID person "
            "record names Hee Joo Kim, distinct from CMU Hyung J. Kim.",
        ),
    }
)
OTHER_PRIMARY_ID_GUARDS.update(
    {index: MID_SIX_EXACT_GUARDS[index][0] for index in (467, 476, 490, 491)}
)
ABSENT_AUTHOR_PRIMARY.update(
    {
        469: (
            MID_SIX_EXACT_GUARDS[469][0],
            MID_SIX_EXACT_GUARDS[469][1],
            1998,
            "",
            "https://koreantk.com/ktkp2014/thesis/"
            "thesis-view.view?ctrlNo=HKSOBF-1998-v29n4-347",
            "Korean traditional knowledge article record with complete names",
            "The exact Korean Journal of Pharmacognosy 29(4):347-352 article "
            "names Hye-Soo Kim and Young-Choong Kim in its complete seven "
            "person byline; no Hyung J. Kim or H. J. Kim authored it.",
        ),
        494: (
            MID_SIX_EXACT_GUARDS[494][0],
            MID_SIX_EXACT_GUARDS[494][1],
            2016,
            "10.3346/jkms.2016.31.8.1300",
            "https://jkms.org/Synapse/Data/PDFData/0063JKMS/jkms-31-1300.pdf",
            "Journal of Korean Medical Science original full-author PDF",
            "The exact Journal of Korean Medical Science 31:1300-1306 article "
            "names first author Heejin Kim and its complete byline has no "
            "Hyung J. Kim or H. J. Kim.",
        ),
    }
)
ABSENT_AUTHOR_VENUES.update(
    {index: MID_SIX_EXACT_GUARDS[index][4] for index in (469, 494)}
)
ABSENT_AUTHOR_AUTHORS.update(
    {index: MID_SIX_EXACT_GUARDS[index][2] for index in (469, 494)}
)

# Nine additional KNU records link the exact H. J. Kim byline to the separate
# Hong Joo Kim person page. Guard the entire saved Scholar row before adding
# any proposed exclusion; nearby initials-only and collaboration rows stay held.
KNU_NINE_EXACT_GUARDS = {
    508: (
        "bC4aq-4AAAAJ:owLR8QvbtFgC",
        "Measurement of the quenching and channeling effects in a CsI crystal "
        "used for a WIMP search",
        "JH Lee, GB Kim, IS Seong, BH Kim, JH Kim, J Li, JW Park, JK Lee, ...",
        "2015",
        "Nuclear Instruments and Methods in Physics Research Section A: "
        "Accelerators\xa0…, 2015",
    ),
    512: (
        "bC4aq-4AAAAJ:buQ7SEKw-1sC",
        "Low energy fast events from radon progenies at the surface of a "
        "CsI (Tℓ) scintillator",
        "SC Kim, H Bhang, JH Choi, WG Kang, HJ Kim, KW Kim, SK Kim, YD Kim, ...",
        "2012",
        "Astroparticle Physics 35 (12), 781-784, 2012",
    ),
    516: (
        "bC4aq-4AAAAJ:1lhNe0rCu4AC",
        "Performance of a large volume liquid scintillation detector for the "
        "measurement of fast neutrons",
        "JJ Zhu, KJ Kang, YJ Li, JW Kwak, H Bhang, SY Kim, SC Kim, SK Kim, ...",
        "2005",
        "J. Kor. Phys. Soc 47, 202, 2005",
    ),
    529: (
        "bC4aq-4AAAAJ:s9ia6_kGH2AC",
        "Neutron calibration facility with an Am-Be source for pulse shape "
        "discrimination measurement of CsI (Tl) crystals",
        "HS Lee, H Bhang, JH Choi, S Choi, IS Hahn, EJ Jeon, HW Joo, WG Kang, ...",
        "2014",
        "Journal of Instrumentation 9 (11), P11015-P11015, 2014",
    ),
    533: (
        "bC4aq-4AAAAJ:CaZNVDsoPx4C",
        "A study of CaMoO4 crystals for the AMoRE Experiment",
        "JH So, HJ Kim, VV Alenkov, AN Annenkov, H Bhang, RS Boiko, ...",
        "2012",
        "2012 IEEE Nuclear Science Symposium and Medical Imaging Conference "
        "Record\xa0…, 2012",
    ),
    571: (
        "bC4aq-4AAAAJ:6yz0xqPARnAC",
        "Simulation and measurement of quenching and channeling effects in "
        "CsI (Tl) for dark matter search",
        "JH Lee, H Bhang, JH Choi, WG Kang, BH Kim, HJ Kim, KW Kim, SC Kim, ...",
        "2012",
        "IEEE Transactions on Nuclear Science 59 (5), 2346-2353, 2012",
    ),
    572: (
        "bC4aq-4AAAAJ:ALROH1vI_8AC",
        "Crystal Growth and Scintillation Properties of",
        "S Kim, G Rooh, HJ Kim, W Kim, U Hong",
        "2010",
        "IEEE Transactions on Nuclear Science 57 (3), 1251-1254, 2010",
    ),
    573: (
        "bC4aq-4AAAAJ:BrOSOlqYqPUC",
        "The Growth and Scintillation Characteristics of CsI:CO Single Crystals",
        "SJ Ha, H Kang, H Park, HJ Kim, S Kim, SH Doh, SJ Kang",
        "2009",
        "IEEE Transactions on Nuclear Science 56 (3), 998-1001, 2009",
    ),
    647: (
        "bC4aq-4AAAAJ:U3qCfcK-7lkC",
        "Development of a neutron tagger module using a digital pulse shape "
        "discrimination method",
        "HJ Kim, EJ Kim, SY Kim",
        "2008",
        "2008 IEEE Nuclear Science Symposium Conference Record, 2917-2919, 2008",
    ),
}
EXACT_FOLLOWUP_CAPTURE_GUARDS.update(KNU_NINE_EXACT_GUARDS)
KNU_NINE_PUBLICATIONS = {
    508: (
        "10.1016/j.nima.2015.02.011",
        "measurement-of-the-quenching-and-channeling-effects-in-a-csi-crys/",
    ),
    512: (
        "10.1016/j.astropartphys.2012.03.005",
        "low-energy-fast-events-from-radon-progenies-at-the-surface-of-a-c/",
    ),
    516: (
        "",
        "performance-of-a-large-volume-liquid-scintillation-detector-for-t/",
    ),
    529: (
        "10.1088/1748-0221/9/11/P11015",
        "neutron-calibration-facility-with-an-am-be-source-for-pulse-shape/",
    ),
    533: (
        "10.1109/NSSMIC.2012.6551459",
        "a-study-of-camoo4-crystals-for-the-amore-experiment/",
    ),
    571: (
        "10.1109/TNS.2012.2189019",
        "simulation-and-measurement-of-quenching-and-channeling-effects-in/",
    ),
    572: (
        "10.1109/TNS.2010.2041789",
        "crystal-growth-and-scintillation-properties-of-cssub2subnacebr-su/",
    ),
    573: (
        "10.1109/TNS.2009.2014063",
        "the-growth-and-scintillation-characteristics-of-csicosub3sub-sing/",
    ),
    647: (
        "10.1109/NSSMIC.2008.4774976",
        "development-of-a-neutron-tagger-module-using-a-digital-pulse-shap/",
    ),
}
for index, (doi, slug) in KNU_NINE_PUBLICATIONS.items():
    guard = KNU_NINE_EXACT_GUARDS[index]
    OTHER_PRIMARY[index] = (
        guard[1],
        int(guard[3]),
        "Hong Joo Kim",
        doi,
        f"https://research.knu.ac.kr/en/publications/{slug}",
        "KNU exact publication with H. J. Kim person link",
        "KNU's exact article links its H. J. Kim author directly to "
        "https://research.knu.ac.kr/en/persons/hong-joo-kim/, the "
        "Physics professor Hong Joo Kim; saved title, year, byline, "
        "and journal or conference details identify the same output.",
    )
OTHER_PRIMARY_ID_GUARDS.update(
    {index: guard[0] for index, guard in KNU_NINE_EXACT_GUARDS.items()}
)

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
    13: (
        "Tailoring the Electronic and Catalytic Properties of Au25 "
        "Nanoclusters via Ligand Engineering",
        2016,
        "https://api.crossref.org/works/10.1021%2Facsnano.6b03964",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; ACS Nano 7998-8005 (2016)",
    ),
    18: (
        "Revisiting the aqueous solutions of dimethyl sulfoxide by "
        "spectroscopy in the mid-and near-infrared: experiments and "
        "Car–Parrinello simulations",
        2015,
        "https://api.crossref.org/works/10.1021%2Facs.jpcb.5b09196",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry B 14780-14789 "
        "(2015)",
    ),
    20: (
        "Silicon nanoparticles with surface nitrogen: 90% quantum yield "
        "with narrow luminescence bandwidth and the ligand structure based"
        " energy law",
        2016,
        "https://api.crossref.org/works/10.1021%2Facsnano.6b03113",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; ACS Nano 8385-8393 (2016)",
    ),
    33: (
        "Chiral Ag23 nanocluster with open shell electronic structure and "
        "helical face-centered cubic framework",
        2018,
        "https://www.nature.com/articles/s41467-018-03136-9",
        "Original Nature article names Hyung J. Kim and matching CMU "
        "faculty ORCID; Nature Communications 9:744",
    ),
    51: (
        "Molecular structure and interactions in the ionic liquid "
        "1-ethyl-3-methylimidazolium bis (trifluoromethylsulfonyl) imide",
        2014,
        "https://api.crossref.org/works/10.1021%2Fjp502124y",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry A 2547-2557 (2014)",
    ),
    57: (
        "Experimental and Mechanistic Understanding of Aldehyde "
        "Hydrogenation Using Au25 Nanoclusters with Lewis Acids: Unique "
        "Sites for Catalytic Reactions",
        2015,
        "https://api.crossref.org/works/10.1021%2Fjacs.5b07716",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; Journal of the American Chemical Society 14295-14304"
        " (2015)",
    ),
    80: (
        "Effect of magnesium on calcium and oxalate ion binding",
        2013,
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3883082/",
        "Original publisher manuscript names Hyung J. Kim, PhD, CMU "
        "Chemistry, distinct from Hyunjin Kim; J. Endourol. 27:1487-1492",
    ),
    94: (
        "Spectroscopic and dielectric properties of liquid water: A "
        "molecular dynamics simulation study",
        1998,
        "https://api.crossref.org/works/10.1063%2F1.477102",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 4911-4919 (1998)",
    ),
    98: (
        "Graphene oxide supercapacitors: a computer simulation study",
        2014,
        "https://api.crossref.org/works/10.1021%2Fjp5072583",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry C 18472-18480 "
        "(2014)",
    ),
    154: (
        "Generalized molecular mechanics including quantum electronic "
        "structure variation of polar solvents. II. A molecular dynamics "
        "simulation study of water",
        1998,
        "https://api.crossref.org/works/10.1063%2F1.475725",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 3286-3295 (1998)",
    ),
    167: (
        "Molecular dynamics simulation study of polarizable solute "
        "solvation in water. 1. Equilibrium solvent structure and solute "
        "rotational dynamics",
        1996,
        "https://api.crossref.org/works/10.1021%2Fjp952286d",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry 1392-1405 (1996)",
    ),
    194: (
        "Molecular-like Transformation from PhSe-Protected Au25 to Au23 "
        "Nanocluster and Its Application",
        2017,
        "https://api.crossref.org/works/10.1021%2Facs.chemmater.7b00058",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; Chemistry of Materials 3055-3061 (2017)",
    ),
    227: (
        "Molecular dynamics simulation study of water near critical "
        "conditions. I. Structure and solvation free energetics",
        1999,
        "https://api.crossref.org/works/10.1063%2F1.478929",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 9646-9655 (1999)",
    ),
    243: (
        "Gold–Palladium Nanoalloys Supported by Graphene Oxide and "
        "Lamellar TiO2 for Direct Synthesis of Hydrogen Peroxide",
        2018,
        "https://api.crossref.org/works/10.1021%2Facsami.8b17342",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; ACS Applied Materials & Interfaces 40599-40607 "
        "(2018)",
    ),
    246: (
        "Dielectric relaxation and solvation dynamics in a "
        "room-temperature ionic liquid: temperature dependence",
        2013,
        "https://api.crossref.org/works/10.1021%2Fjp406353j",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry B 11743-11752 "
        "(2013)",
    ),
    249: (
        "Silica-encapsulated gold nanoclusters for efficient acetylene "
        "hydrogenation to ethylene",
        2019,
        "https://api.crossref.org/works/10.1021%2Facsanm.9b00384",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; ACS Applied Nano Materials 2999-3006 (2019)",
    ),
    255: (
        "Generalized molecular mechanics including quantum electronic "
        "structure variation of polar solvents. I. Theoretical formulation"
        " via a truncated adiabatic basis set description",
        1998,
        "https://api.crossref.org/works/10.1063%2F1.475724",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 3277-3285 (1998)",
    ),
    317: (
        "On the structural stability of ionic liquid–IRMOF composites: A "
        "computational study",
        2015,
        "https://api.crossref.org/works/10.1039%2Fc4cp02428a",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; Physical Chemistry Chemical Physics 6248-6254 (2015)",
    ),
    322: (
        "The effect of a laser field on electron transfer in metal "
        "complexes: Quantum degrees of freedom",
        1995,
        "https://api.crossref.org/works/10.1063%2F1.470530",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 5461-5469 (1995)",
    ),
    326: (
        "A molecular dynamics study of the ionic liquid, choline acetate",
        2016,
        "https://api.crossref.org/works/10.1039%2Fc6cp01031h",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; Physical Chemistry Chemical Physics 14850-14858 "
        "(2016)",
    ),
    329: (
        "Optical Kerr effect spectroscopy of liquid water: Role of "
        "fluctuating electronic polarizability",
        1997,
        "https://api.crossref.org/works/10.1021%2Fjp9727207",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry B 10994-10999 "
        "(1997)",
    ),
    342: (
        "Effects of solute electronic structure variation on photon echo spectroscopy",
        1996,
        "https://api.crossref.org/works/10.1021%2Fjp961001%2B",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry 16451-16456 (1996)",
    ),
    359: (
        "Molecular dynamics simulation study of water near critical "
        "conditions. II. Dynamics and spectroscopy",
        1999,
        "https://api.crossref.org/works/10.1063%2F1.478952",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 9656-9665 (1999)",
    ),
    397: (
        "Removal of confined ionic liquid from a metal organic framework "
        "by extraction with molecular solvents",
        2017,
        "https://api.crossref.org/works/10.1021%2Facs.jpcc.7b02289",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry C 10577-10586 "
        "(2017)",
    ),
    449: (
        "Infrared Spectroscopy of Li+ Solvation in EmimBF4 and in "
        "Propylene Carbonate: Ab Initio Molecular Dynamics and Experiment",
        2022,
        "https://api.crossref.org/works/10.1021%2Facs.jpcb.2c06326",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry B 9643-9662 (2022)",
    ),
    465: (
        "MD study of Stokes shifts in ionic liquids: Temperature dependence",
        2016,
        "https://api.crossref.org/works/10.1021%2Facs.jpcb.6b00979",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Physical Chemistry B 4644-4653 (2016)",
    ),
    585: (
        "Quantum tunneling in an anharmonic classical bath. Enhanced "
        "kinetic isotope effects in an Arrhenius region",
        1995,
        "https://api.crossref.org/works/10.1063%2F1.468984",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Journal of Chemical Physics 7838-7849 (1995)",
    ),
    698: (
        "Spectroscopic and Chemical Properties of Ionic Liquids: Computational Study",
        2023,
        "https://api.crossref.org/works/10.1002%2Ftcr.202300075",
        "Publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; The Chemical Record e202300075 (2023)",
    ),
    132: (
        "Molecular mechanism for the activation of Au25 (SCH2CH2Ph) 18 "
        "nanoclusters by imidazolium-based ionic liquids for catalysis",
        2016,
        "https://api.crossref.org/works/10.1016%2Fj.jcat.2016.01.011",
        "Elsevier publisher-deposited full Hyung J. Kim byline; CMU faculty "
        "list identifies the same Journal of Catalysis 337:72-79 work",
    ),
    287: (
        "Excited-state charge transfer dynamics of p-dimethylaminobenzonitrile "
        "in quadrupolar solvents",
        2002,
        "https://api.crossref.org/works/10.1021%2Fjp012991o",
        "ACS publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; CMU Kim Group lists J. Phys. Chem. A 106:2322-2327",
    ),
    298: (
        "A continuum theory of solvation in quadrupolar solvents. I. Formulation",
        2003,
        "https://api.crossref.org/works/10.1063%2F1.1605376",
        "AIP publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; CMU Kim Group lists JCP 119:8606-8625",
    ),
    306: (
        "A continuum theory of solvation in quadrupolar solvents. II. "
        "Solvation free energetics, dynamics, and solvatochromism",
        2003,
        "https://api.crossref.org/works/10.1063%2F1.1605377",
        "AIP publisher-deposited full Hyung J. Kim byline and CMU Chemistry "
        "affiliation; CMU Kim Group lists JCP 119:8626-8635",
    ),
    321: (
        "A continuum reaction field theory of polarizable, nondipolar, "
        "quadrupolar solvents: Ab Initio study of equilibrium solvation in benzene",
        2001,
        "https://api.crossref.org/works/10.1023%2Fa%3A1012715929287",
        "Springer publisher-deposited full Hyung J. Kim byline; CMU Kim Group "
        "lists J. Solution Chem. 30:849-860 and the same article",
    ),
    363: (
        "Dielectric relaxation of the ionic liquid 1-ethyl-3-methylimidazolium "
        "ethyl sulfate: microwave and far-IR properties",
        2017,
        "https://pubs.acs.org/doi/10.1021/acs.jpcb.7b00160",
        "Original ACS article names Hyung J. Kim, CMU Chemistry, matching "
        "ORCID and 121:4845-4852; CMU faculty list has a final-page typo (4851)",
    ),
    418: (
        "Vibrational spectroscopy of imidazolium-based ionic liquids: A "
        "combined MD/DFT study",
        2019,
        "https://www.sciencedirect.com/science/article/pii/S016773221931339X",
        "Original Elsevier article distinctly names Hyunjin Kim and Hyung J. "
        "Kim; CMU faculty list links exact DOI, with a middle-author order variant",
    ),
    421: (
        "Heterogeneous dynamics of ionic liquids: A four-point time "
        "correlation function approach",
        2018,
        "https://api.crossref.org/works/10.1063%2F1.5016501",
        "AIP publisher-deposited full Hyung J. Kim byline, CMU Chemistry "
        "affiliation and matching ORCID; CMU faculty DOI agrees",
    ),
    488: (
        "Dicationic-type quaternary ammonium salts as candidates of "
        "desiccants for an air-conditioning system",
        2021,
        "https://pubs.acs.org/doi/10.1021/acssuschemeng.1c04971",
        "Original ACS article names Hyung J. Kim, CMU Chemistry, matching "
        "ORCID and 9:14502-14514; CMU faculty DOI agrees",
    ),
    527: (
        "Excitation-energy dependence of solvation dynamics in "
        "room-temperature ionic liquids",
        2016,
        "https://api.crossref.org/works/10.1063%2F1.4955211",
        "AIP publisher-deposited byline distinguishes Daekeon Kim from "
        "Hyung J. Kim, who has CMU Chemistry affiliation; CMU faculty DOI agrees",
    ),
    591: (
        "CS2 capture in the ionic liquid 1-alkyl-3-methylimidazolium acetate: "
        "reaction mechanism and free energetics",
        2018,
        "https://api.crossref.org/works/10.1039%2FC8CP01724G",
        "RSC publisher-deposited full Hyung J. Kim byline, CMU Chemistry "
        "affiliation and matching ORCID; CMU faculty DOI agrees",
    ),
    671: (
        "Theoretical study of alkylsulfonic acids: force-field development "
        "and molecular dynamics simulations",
        2018,
        "https://api.crossref.org/works/10.1021%2Facs.jpcb.8b07736",
        "ACS publisher-deposited full Hyung J. Kim byline, CMU Chemistry "
        "affiliation and matching ORCID; CMU faculty DOI agrees",
    ),
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


# Exact source checks for the last 20 CMU ORCID title/year signals.
CMU_REMAINING_TWENTY_PRIMARY = {
    27: (
        "A theoretical model for SN1 ionic dissociation in solution. 1. "
        "Activation free energetics and transition-state structure",
        1992,
        "https://api.crossref.org/works/10.1021%2Fja00052a055",
        "Colorado Hynes CV item 122 names full Hyung J. Kim; ACS publisher DOI, "
        "JACS 114:10508-10528, and CMU faculty Colorado appointment agree",
    ),
    35: (
        "Free energies of electron transfer",
        1992,
        "https://api.crossref.org/works/10.1021%2Fj100183a047",
        "Colorado Hynes CV item 127 names full Hyung J. Kim; ACS publisher DOI "
        "and J. Phys. Chem. 96:1748-1753 agree",
    ),
    55: (
        "Dynamics of twisted intramolecular charge transfer complexes in polar "
        "solvents",
        1994,
        "https://api.crossref.org/works/10.1016%2F0167-7322%2894%2900744-6",
        "Colorado Hynes CV item 145 names full Hyung J. Kim; Elsevier publisher "
        "DOI and J. Mol. Liquids 60:161-200 agree",
    ),
    83: (
        "Role of solvent electronic polarization in electron-transfer processes",
        1990,
        "https://api.crossref.org/works/10.1021%2Fj100370a004",
        "Colorado Hynes CV item 109 names full Hyung J. Kim; ACS publisher DOI "
        "and J. Phys. Chem. 94:2736-2740 agree",
    ),
    108: (
        "A theoretical model for SN1 ionic dissociations in solution. 3. Analysis"
        " of tert-butyl halides",
        1993,
        "https://api.crossref.org/works/10.1021%2Fja00071a038",
        "Colorado Hynes CV item 139 names full Hyung J. Kim; ACS publisher DOI "
        "and JACS 115:8248-8262 agree",
    ),
    112: (
        "A theoretical model for SN1 ionic dissociation in solution. 2. "
        "Nonequilibrium solvation reaction path and reaction rate",
        1992,
        "https://api.crossref.org/works/10.1021%2Fja00052a056",
        "Colorado Hynes CV item 123 names full Hyung J. Kim; ACS publisher DOI "
        "and JACS 114:10528-10537 agree",
    ),
    119: (
        "Role of solute electronic polarizability in solvation dynamics",
        1995,
        "https://api.crossref.org/works/10.1021%2Fj100025a003",
        "ACS publisher DOI names full Hyung J. Kim with Bursulaya and Zichi; the "
        "same exact trio has CMU Chemistry affiliation in confirmed 1996 DOI "
        "10.1021/jp952286d",
    ),
    133: (
        "Understanding the mechanism of CO2 capture by 1,3 di-substituted "
        "imidazolium acetate based ionic liquids",
        2016,
        "https://api.crossref.org/works/10.1039%2Fc5cp05713b",
        "RSC publisher DOI assigns Hyung Kim to CMU Chemistry with CMU coauthors "
        "Yan and Dhumal; official CMU ORCID imports the exact DOI; middle initial"
        " omitted in article metadata",
    ),
    140: (
        "Perfusion analysis using dynamic arterial spin labeling (DASL)",
        1999,
        "https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291522-2594%2819990"
        "2%2941%3A2%3C299%3A%3AAID-MRM13%3E3.0.CO%3B2-R",
        "Original Wiley article names full Hyung J. Kim at CMU Chemistry, 1999 "
        "pages 299-308; publisher canonical DOI corrects a malformed ORCID "
        "variant",
    ),
    151: (
        "Twisted intramolecular charge transfer dynamics in polar solvents",
        1994,
        "https://api.crossref.org/works/10.1016%2F1010-6030%2894%2902018-3",
        "Colorado Hynes CV item 146 names full Hyung J. Kim; Elsevier publisher "
        "DOI and J. Photochem. Photobiol. A 82:67-79 agree",
    ),
    181: (
        "A simple basis set approach to solute electronic structure and free "
        "energy in solution",
        1993,
        "https://api.crossref.org/works/10.1021%2Fj100111a001",
        "Colorado Hynes CV item 136 names full Hyung J. Kim; ACS original DOI "
        "establishes pages 1723-1728, correcting CV last-page typo",
    ),
    185: (
        "Inducing coherent oscillations in the electron transfer dynamics of a "
        "strongly dissipative system with pulsed monochromatic light",
        1995,
        "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.75.3649",
        "Original APS article assigns H. J. Kim to CMU Chemistry; official CMU "
        "ORCID imports exact DOI, PRL 75:3649-3652",
    ),
    211: (
        "Influence of methyl and propyl groups on the vibrational spectra of two "
        "imidazolium ionic liquids and their non-ionic precursors",
        2017,
        "https://scholarscommons.fgcu.edu/esploro/outputs/journalArticle/Influence-o"
        "f-methyl-and-propyl-groups/99383431964006570",
        "Publisher DOI names full Hyung J. Kim; FGCU coauthor institutional "
        "record assigns him to CMU Chemistry at same exact DOI and pages",
    ),
    228: (
        "Smoluchowski fluctuation theory of dielectric relaxation",
        1991,
        "https://api.crossref.org/works/10.1063%2F1.460002",
        "AIP publisher-deposited full Hyung J. Kim at SUNY Stony Brook, exact "
        "DOI/pages; CMU biography confirms his Stony Brook chemistry PhD; CMU "
        "ORCID imports exact work",
    ),
    312: (
        "Synthesis, conductivity, and vibrational spectroscopy of "
        "tetraphenylphosphonium bis (trifluoromethanesulfonyl) imide",
        2017,
        "https://scholarscommons.fgcu.edu/esploro/outputs/journalArticle/Synthesis-c"
        "onductivity-and-vibrational-spectroscopy-of/99383432059606570",
        "Publisher DOI names full Hyung J. Kim; FGCU coauthor institutional "
        "record assigns KIAS, his documented concurrent appointment; duplicate "
        "Scholar row 797 already excluded",
    ),
    343: (
        "Solute electronic structure and solvation in chemical reactions in solution",
        1993,
        "https://api.crossref.org/works/10.1016%2F0167-7322%2893%2980047-y",
        "Colorado Hynes CV item 134 names full Hyung J. Kim; Elsevier publisher "
        "DOI and J. Mol. Liquids 57:53-73 agree",
    ),
    470: (
        "Equilibrium and nonequilibrium solvation and solute electronic structure",
        1990,
        "https://api.crossref.org/works/10.1002%2Fqua.560382480",
        "Colorado Hynes CV item 115 and Wiley publisher DOI name full Hyung J. "
        "Kim; International Journal of Quantum Chemistry 38(S24):821-833 is "
        "distinct from JCP series",
    ),
    663: (
        "Smoluchowski fluctuation theory of ion transport in solutions",
        1988,
        "https://api.crossref.org/works/10.1063%2F1.454979",
        "AIP publisher-deposited full Hyung J. Kim at SUNY Stony Brook, exact "
        "DOI/pages in 1988; CMU biography confirms his Stony Brook chemistry PhD "
        "that year; CMU ORCID imports exact work",
    ),
    712: (
        "Computational Study of Structure, Dynamics and Spectroscopy of Ionic Liquids",
        2014,
        "https://api.crossref.org/works/10.1149%2F06404.0119ecst",
        "ECS publisher DOI names full Hyung J. Kim separately from Hyunjin Kim, "
        "with known CMU coauthors Dhumal and Shim; ECS meeting contents and CMU "
        "ORCID corroborate",
    ),
    729: (
        "Deconvolution of conformational equilibria in methimazolium-based ionic "
        "liquid ion pair: Infrared spectroscopic and computational study",
        2018,
        "https://scholarscommons.fgcu.edu/esploro/outputs/journalArticle/Deconvoluti"
        "on-of-conformational-equilibria-in-methimazolium-based/99383432060606570",
        "Publisher DOI names full Hyung J. Kim; FGCU coauthor institutional "
        "record assigns him to CMU Chemistry at same exact DOI and pages",
    ),
}
CMU_PRIMARY.update(CMU_REMAINING_TWENTY_PRIMARY)

CMU_CHAPTERS_AND_PROCEEDINGS_PRIMARY = {
    725: (
        "Generalized Molecular Mechanics Including Quantum Electronic Structure "
        "Variation of Polar Solvents: An Overview",
        1998,
        "https://pubs.acs.org/doi/10.1021/bk-1998-0712.ch011",
        "Original ACS book chapter names full Hyung J. Kim at CMU Chemistry, "
        "the exact title, coauthors, 1998 publication date, and pages 172-187",
    ),
    778: (
        "Environmental dynamics and electron transfer reactions",
        1990,
        "https://link.springer.com/chapter/10.1007/978-94-009-0489-7_12",
        "Original Springer conference chapter names full Hyung J. Kim with "
        "Colorado Chemistry affiliation, exact title, 1990 year, and pages 133-148",
    ),
    817: (
        "Free Energetics and Kinetics of Charge Transfer and Shift Reactions in "
        "Room‐Temperature Ionic Liquids",
        2013,
        "https://books.rsc.org/books/edited-volume/1070/chapter-abstract/827847/"
        "Free-Energetics-and-Kinetics-of-Charge-Transfer",
        "Original RSC chapter names full Hyung J. Kim at CMU Chemistry, "
        "exact title, 2013 publication date, DOI, and pages 260-282",
    ),
    888: (
        "A truncated adiabatic basis-set approach to solvent electronic "
        "structure variations",
        1999,
        "https://api.crossref.org/works/10.1063%2F1.1301533",
        "AIP publisher-deposited conference DOI names full Hyung J. Kim, "
        "exact title, all coauthors, 1999 year, and pages 281-300",
    ),
    897: (
        "Computer simulation study of electronic spectroscopy in water [3273-14]",
        1998,
        "https://api.crossref.org/works/10.1117%2F12.306113",
        "SPIE publisher-deposited proceedings DOI names full Hyung J. Kim, "
        "exact base title, same three coauthors, and 1998 publication; "
        "the saved Scholar page range differs from publisher pages 90-99",
    ),
    903: (
        "Electronic Structure and Chemical Reactions in Solution",
        1994,
        "https://link.springer.com/chapter/10.1007/978-94-011-0786-0_21",
        "Original Springer chapter names full Hyung J. Kim and his CMU "
        "Chemistry present address, exact title, 1994 year, and pages 289-309",
    ),
    904: (
        "Solvent Dynamics and Charge Transfer Reactions",
        1992,
        "https://api.crossref.org/works/10.1016%2Fb978-0-444-89191-4.50010-0",
        "Elsevier book contents and publisher-deposited chapter DOI name full "
        "Hyung J. Kim, exact chapter title, 1992 year, and pages 39-56; "
        "Colorado coauthor Hynes CV item 129 corroborates",
    ),
}
CMU_PRIMARY.update(CMU_CHAPTERS_AND_PROCEEDINGS_PRIMARY)

# The original society program and publisher-deposited DOI name the CMU
# chemist in this published Biophysical Journal meeting abstract.
CMU_NEXT_PUBLISHER_FULLNAME_PRIMARY = {
    802: (
        "Gestation of a Glu Plasminogen Supra Fold via Molecular Dynamics Simulation",
        2015,
        "https://api.crossref.org/works/10.1016%2Fj.bpj.2014.11.1162",
        "Elsevier publisher-deposited full Hyung J. Kim byline; original "
        "Biophysical Society 2015 program poster 1052-Pos Board B3 names "
        "Hyunjin Kim, Hyung J. Kim, and Miguel Llinás; CMU Kim faculty "
        "page documents this plasminogen/Llinás collaboration and CMU Kim "
        "group page lists Hyunjin Kim; Biophysical Journal 108(2):210a is a "
        "published meeting abstract, not a full article",
    ),
}
CMU_PRIMARY.update(CMU_NEXT_PUBLISHER_FULLNAME_PRIMARY)
CMU_NEXT_PUBLISHER_VENUES = {
    802: "Biophysical Journal 108 (2), 210a, 2015",
}

# ECS's exact published abstract names full Hyung J. Kim and the same CMU
# collaborators, with an independent DOI and meeting abstract number.
CMU_ECS_MEETING_PRIMARY = {
    798: (
        "Mechanism of CO2 Capture in 1-Alkyl-3-Methylimidazolium Acetate: "
        "Is N-Heterocyclic Carbene Involved?",
        2016,
        "https://api.crossref.org/works/10.1149%2Fma2016-02%2F47%2F3437",
        "Electrochemical Society publisher-deposited full Hyung J. Kim "
        "byline with Fangyong Yan and Nilesh R. Dhumal; exact 2016 title "
        "and abstract number 3437. This is a published meeting abstract.",
    )
}
CMU_PRIMARY.update(CMU_ECS_MEETING_PRIMARY)
CMU_NEXT_PUBLISHER_VENUES[798] = (
    "Electrochemical Society Meeting Abstracts 230, 3437-3437, 2016"
)
CMU_ECS_MEETING_AUTHORS = "HJ Kim, F Yan, NR Dhumal"

# Guard newly resolved source rows with their immutable Scholar citation IDs.
CMU_PRIMARY_ID_GUARDS = {
    13: "bC4aq-4AAAAJ:f2IySw72cVMC",
    18: "bC4aq-4AAAAJ:g5m5HwL7SMYC",
    20: "bC4aq-4AAAAJ:pyW8ca7W8N0C",
    33: "bC4aq-4AAAAJ:dTyEYWd-f8wC",
    51: "bC4aq-4AAAAJ:e5wmG9Sq2KIC",
    57: "bC4aq-4AAAAJ:2P1L_qKh6hAC",
    80: "bC4aq-4AAAAJ:JV2RwH3_ST0C",
    94: "bC4aq-4AAAAJ:eQOLeE2rZwMC",
    98: "bC4aq-4AAAAJ:j3f4tGmQtD8C",
    154: "bC4aq-4AAAAJ:hqOjcs7Dif8C",
    167: "bC4aq-4AAAAJ:5nxA0vEk-isC",
    194: "bC4aq-4AAAAJ:NhqRSupF_l8C",
    227: "bC4aq-4AAAAJ:qxL8FJ1GzNcC",
    243: "bC4aq-4AAAAJ:FPJr55Dyh1AC",
    246: "bC4aq-4AAAAJ:iH-uZ7U-co4C",
    249: "bC4aq-4AAAAJ:48xauSegjOkC",
    255: "bC4aq-4AAAAJ:M3ejUd6NZC8C",
    317: "bC4aq-4AAAAJ:RGFaLdJalmkC",
    322: "bC4aq-4AAAAJ:dhFuZR0502QC",
    326: "bC4aq-4AAAAJ:yD5IFk8b50cC",
    329: "bC4aq-4AAAAJ:7PzlFSSx8tAC",
    342: "bC4aq-4AAAAJ:ZeXyd9-uunAC",
    359: "bC4aq-4AAAAJ:QIV2ME_5wuYC",
    397: "bC4aq-4AAAAJ:p2g8aNsByqUC",
    449: "bC4aq-4AAAAJ:8gBurD7jEYQC",
    465: "bC4aq-4AAAAJ:HoB7MX3m0LUC",
    585: "bC4aq-4AAAAJ:bEWYMUwI8FkC",
    698: "bC4aq-4AAAAJ:o-PowTg_VKEC",
    132: "bC4aq-4AAAAJ:pqnbT2bcN3wC",
    287: "bC4aq-4AAAAJ:4TOpqqG69KYC",
    298: "bC4aq-4AAAAJ:qUcmZB5y_30C",
    306: "bC4aq-4AAAAJ:hC7cP41nSMkC",
    321: "bC4aq-4AAAAJ:Wp0gIr-vW9MC",
    363: "bC4aq-4AAAAJ:WbkHhVStYXYC",
    418: "bC4aq-4AAAAJ:mlAyqtXpCwEC",
    421: "bC4aq-4AAAAJ:9vf0nzSNQJEC",
    488: "bC4aq-4AAAAJ:3BvdIg-l-ZAC",
    527: "bC4aq-4AAAAJ:KxtntwgDAa4C",
    591: "bC4aq-4AAAAJ:UHK10RUVsp4C",
    671: "bC4aq-4AAAAJ:JQOojiI6XY0C",
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

CMU_PRIMARY_ID_GUARDS.update(
    {
        27: "bC4aq-4AAAAJ:9yKSN-GCB0IC",
        35: "bC4aq-4AAAAJ:2osOgNQ5qMEC",
        55: "bC4aq-4AAAAJ:WF5omc3nYNoC",
        83: "bC4aq-4AAAAJ:zYLM7Y9cAGgC",
        108: "bC4aq-4AAAAJ:ufrVoPGSRksC",
        112: "bC4aq-4AAAAJ:YsMSGLbcyi4C",
        119: "bC4aq-4AAAAJ:Se3iqnhoufwC",
        133: "bC4aq-4AAAAJ:xtRiw3GOFMkC",
        140: "bC4aq-4AAAAJ:J_g5lzvAfSwC",
        151: "bC4aq-4AAAAJ:8k81kl-MbHgC",
        181: "bC4aq-4AAAAJ:UebtZRa9Y70C",
        185: "bC4aq-4AAAAJ:MXK_kJrjxJIC",
        211: "bC4aq-4AAAAJ:1sJd4Hv_s6UC",
        228: "bC4aq-4AAAAJ:KlAtU1dfN6UC",
        312: "bC4aq-4AAAAJ:738O_yMBCRsC",
        343: "bC4aq-4AAAAJ:-f6ydRqryjwC",
        470: "bC4aq-4AAAAJ:TFP_iSt0sucC",
        663: "bC4aq-4AAAAJ:a0OBvERweLwC",
        712: "bC4aq-4AAAAJ:NaGl4SEjCO4C",
        729: "bC4aq-4AAAAJ:W5xh706n7nkC",
    }
)

CMU_PRIMARY_ID_GUARDS.update(
    {
        725: "bC4aq-4AAAAJ:NMxIlDl6LWMC",
        778: "bC4aq-4AAAAJ:GnPB-g6toBAC",
        817: "bC4aq-4AAAAJ:tOudhMTPpwUC",
        888: "bC4aq-4AAAAJ:O3NaXMp0MMsC",
        897: "bC4aq-4AAAAJ:bnK-pcrLprsC",
        903: "bC4aq-4AAAAJ:YFjsv_pBGBYC",
        904: "bC4aq-4AAAAJ:ns9cj8rnVeAC",
    }
)

CMU_PRIMARY_ID_GUARDS.update(
    {
        802: "bC4aq-4AAAAJ:sSrBHYA8nusC",
        798: "bC4aq-4AAAAJ:CHSYGLWDkRkC",
    }
)

CMU_GROUP_PUBLISHER_DOI_GUARDS = {
    13: "10.1021/acsnano.6b03964",
    18: "10.1021/acs.jpcb.5b09196",
    20: "10.1021/acsnano.6b03113",
    33: "10.1038/s41467-018-03136-9",
    51: "10.1021/jp502124y",
    57: "10.1021/jacs.5b07716",
    80: "10.1089/end.2013.0173",
    94: "10.1063/1.477102",
    98: "10.1021/jp5072583",
    154: "10.1063/1.475725",
    167: "10.1021/jp952286d",
    194: "10.1021/acs.chemmater.7b00058",
    227: "10.1063/1.478929",
    243: "10.1021/acsami.8b17342",
    246: "10.1021/jp406353j",
    249: "10.1021/acsanm.9b00384",
    255: "10.1063/1.475724",
    317: "10.1039/c4cp02428a",
    322: "10.1063/1.470530",
    326: "10.1039/c6cp01031h",
    329: "10.1021/jp9727207",
    342: "10.1021/jp961001+",
    359: "10.1063/1.478952",
    397: "10.1021/acs.jpcc.7b02289",
    449: "10.1021/acs.jpcb.2c06326",
    465: "10.1021/acs.jpcb.6b00979",
    585: "10.1063/1.468984",
    698: "10.1002/tcr.202300075",
    132: "10.1016/j.jcat.2016.01.011",
    287: "10.1021/jp012991o",
    298: "10.1063/1.1605376",
    306: "10.1063/1.1605377",
    321: "10.1023/a:1012715929287",
    363: "10.1021/acs.jpcb.7b00160",
    418: "10.1016/j.molliq.2019.111282",
    421: "10.1063/1.5016501",
    488: "10.1021/acssuschemeng.1c04971",
    527: "10.1063/1.4955211",
    591: "10.1039/c8cp01724g",
    671: "10.1021/acs.jpcb.8b07736",
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

CMU_GROUP_PUBLISHER_DOI_GUARDS.update(
    {
        27: "10.1021/ja00052a055",
        35: "10.1021/j100183a047",
        55: "10.1016/0167-7322(94)00744-6",
        83: "10.1021/j100370a004",
        108: "10.1021/ja00071a038",
        112: "10.1021/ja00052a056",
        119: "10.1021/j100025a003",
        133: "10.1039/c5cp05713b",
        140: "10.1002/(sici)1522-2594(199902)41:2&lt;299::aid-mrm13&gt;3.3.co;2-i",
        151: "10.1016/1010-6030(94)02018-3",
        181: "10.1021/j100111a001",
        185: "10.1103/physrevlett.75.3649",
        211: "10.1016/j.molstruc.2017.01.008",
        228: "10.1063/1.460002",
        312: "10.1016/j.molstruc.2017.05.138",
        343: "10.1016/0167-7322(93)80047-y",
        663: "10.1063/1.454979",
        712: "10.1149/06404.0119ecst",
        729: "10.1016/j.molliq.2018.06.050",
    }
)

# Publisher canonical identifiers correct ORCID omissions or malformed variants.
CMU_PRIMARY_CANONICAL_DOI_OVERRIDE = {
    140: "10.1002/(sici)1522-2594(199902)41:2<299::aid-mrm13>3.0.co;2-r",
    470: "10.1002/qua.560382480",
    725: "10.1021/bk-1998-0712.ch011",
    778: "10.1007/978-94-009-0489-7_12",
    817: "10.1039/9781849737753-00260",
    888: "10.1063/1.1301533",
    897: "10.1117/12.306113",
    802: "10.1016/j.bpj.2014.11.1162",
    798: "10.1149/ma2016-02/47/3437",
    903: "10.1007/978-94-011-0786-0_21",
    904: "10.1016/b978-0-444-89191-4.50010-0",
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

# Three late, uncited Scholar observations repeat already confirmed CMU works.
# Their source titles have a category prefix or a second SPIE rendering, but
# publisher DOI, author sequence, year, volume, and pages identify one work.
CMU_CONFIRMED_DUPLICATE_PRIMARY = {
    892: (
        "bC4aq-4AAAAJ:AXPGKjj_ei8C",
        "Condensed Phase Dynamics, Structure, and Thermodynamics: Spectroscopy, "
        "Reactions, and Relaxation-Molecular dynamics simulation study of water "
        "near critical conditions. II\xa0…",
        "1999",
        "Journal of Chemical Physics 110 (19), 9656-9665, 1999",
        "BD Bursulaya, HJ Kim",
        359,
        "bC4aq-4AAAAJ:QIV2ME_5wuYC",
        "Molecular dynamics simulation study of water near critical conditions. "
        "II. Dynamics and spectroscopy",
        "10.1063/1.478952",
        "https://api.crossref.org/works/10.1063%2F1.478952",
    ),
    893: (
        "bC4aq-4AAAAJ:BqipwSGYUEgC",
        "Computer simulation study of electronic spectroscopy in water",
        "1998",
        "Laser Techniques for Condensed-Phase and Biological Systems 3273, 90, 1998",
        "MHJ Kim, MBD Bursulaya, MJ Jeon, MDA Zichi",
        897,
        "bC4aq-4AAAAJ:bnK-pcrLprsC",
        "Computer simulation study of electronic spectroscopy in water [3273-14]",
        "10.1117/12.306113",
        "https://api.crossref.org/works/10.1117%2F12.306113",
    ),
    896: (
        "bC4aq-4AAAAJ:XiVPGOgt02cC",
        "Condensed Phase Dynamics, Structure, and Thermodynamics: Spectroscopy, "
        "Reactions, and Relaxation-Spectroscopic and dielectric properties of "
        "liquid water: A molecular dynamics\xa0…",
        "1998",
        "Journal of Chemical Physics 109 (12), 4911-4919, 1998",
        "BD Bursulaya, HJ Kim",
        94,
        "bC4aq-4AAAAJ:eQOLeE2rZwMC",
        "Spectroscopic and dielectric properties of liquid water: A molecular "
        "dynamics simulation study",
        "10.1063/1.477102",
        "https://api.crossref.org/works/10.1063%2F1.477102",
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
    rows_by_index = {row["index"]: row for row in rows}
    if len(rows_by_index) != 918:
        raise ValueError("Duplicate source capture row indices")
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
        if (
            index in EXACT_FOLLOWUP_CAPTURE_GUARDS
            and (
                pub_id,
                title,
                row["authors"],
                year,
                row["venue"],
            )
            != EXACT_FOLLOWUP_CAPTURE_GUARDS[index]
        ):
            raise ValueError(f"Exact follow-up source changed at capture row {index}")
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
        elif index in ABSENT_AUTHOR_PRIMARY:
            (
                expected_id,
                expected_title,
                expected_year,
                doi,
                evidence_url,
                evidence_source,
                reason,
            ) = ABSENT_AUTHOR_PRIMARY[index]
            if (
                pub_id != expected_id
                or title != expected_title
                or year != str(expected_year)
                or (
                    index in ABSENT_AUTHOR_VENUES
                    and row["venue"] != ABSENT_AUTHOR_VENUES[index]
                )
                or (
                    index in ABSENT_AUTHOR_AUTHORS
                    and row["authors"] != ABSENT_AUTHOR_AUTHORS[index]
                )
            ):
                raise ValueError(
                    f"Complete-author match changed at capture row {index}"
                )
            status = "proposed_no_target_author_primary_work"
            additions.append(
                decision(
                    pub_id,
                    "false_profile_attribution",
                    reason,
                    evidence_url,
                    verified_at="2026-10-25",
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
                    index in KNU_ERRATUM_VENUES
                    and row["venue"] != KNU_ERRATUM_VENUES[index]
                )
                or (index in PATENT_VENUES and row["venue"] != PATENT_VENUES[index])
                or (
                    index in KIST_COMMUNITY_VENUES
                    and row["venue"] != KIST_COMMUNITY_VENUES[index]
                )
                or (
                    index in NEXT_FULL_AUTHOR_VENUES
                    and row["venue"] != NEXT_FULL_AUTHOR_VENUES[index]
                )
                or (
                    index in FOLLOWUP_FULL_AUTHOR_VENUES
                    and row["venue"] != FOLLOWUP_FULL_AUTHOR_VENUES[index]
                )
                or (
                    index in NEXT_EXACT_AUTHOR_VENUES
                    and row["venue"] != NEXT_EXACT_AUTHOR_VENUES[index]
                )
                or (
                    index in NEXT_KNU_VERSION_VENUES
                    and row["venue"] != NEXT_KNU_VERSION_VENUES[index]
                )
                or (
                    index in NEXT_KNU_VERSION_AUTHORS
                    and row["authors"] != NEXT_KNU_VERSION_AUTHORS[index]
                )
                or (
                    index in FOLLOWUP_KNU_DUPLICATE_VENUES
                    and row["venue"] != FOLLOWUP_KNU_DUPLICATE_VENUES[index]
                )
                or (
                    index in FOLLOWUP_KNU_DUPLICATE_AUTHORS
                    and row["authors"] != FOLLOWUP_KNU_DUPLICATE_AUTHORS[index]
                )
                or (
                    index in ENTOMOLOGY_OTHER_PERSON_VENUES
                    and row["venue"] != ENTOMOLOGY_OTHER_PERSON_VENUES[index]
                )
                or (
                    index in ENTOMOLOGY_OTHER_PERSON_AUTHORS
                    and row["authors"] != ENTOMOLOGY_OTHER_PERSON_AUTHORS[index]
                )
                or (
                    index == 356
                    and (
                        row["venue"] != DECAL_OTHER_PERSON_VENUE
                        or row["authors"] != DECAL_OTHER_PERSON_AUTHORS
                    )
                )
                or (
                    index in COSINE_FULLNAME_VENUES
                    and row["venue"] != COSINE_FULLNAME_VENUES[index]
                )
                or (
                    index in COSINE_FULLNAME_AUTHORS
                    and row["authors"] != COSINE_FULLNAME_AUTHORS[index]
                )
                or (
                    index in KNU_COSINE_AMORE_VENUES
                    and row["venue"] != KNU_COSINE_AMORE_VENUES[index]
                )
                or (
                    index in KNU_COSINE_AMORE_AUTHORS
                    and row["authors"] != KNU_COSINE_AMORE_AUTHORS[index]
                )
                or (
                    index == 537
                    and (
                        row["venue"] != HYOJOON_SOLE_AUTHOR_VENUE
                        or row["authors"] != HYOJOON_SOLE_AUTHOR_AUTHORS
                    )
                )
                or (
                    index == 24
                    and (
                        row["venue"] != HYUN_JUNG_ORCID_VENUE
                        or row["authors"] != HYUN_JUNG_ORCID_AUTHORS
                    )
                )
                or (
                    index == 857
                    and (
                        row["venue"] != ECS_OTHER_PERSON_VENUE
                        or row["authors"] != ECS_OTHER_PERSON_AUTHORS
                    )
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
        elif index in CMU_CONFIRMED_DUPLICATE_PRIMARY:
            (
                expected_id,
                expected_title,
                expected_year,
                expected_venue,
                expected_authors,
                retained_index,
                retained_pub_id,
                retained_title,
                doi,
                evidence_url,
            ) = CMU_CONFIRMED_DUPLICATE_PRIMARY[index]
            retained = rows_by_index[retained_index]
            retained_doi = CMU_PRIMARY_CANONICAL_DOI_OVERRIDE.get(
                retained_index, CMU_GROUP_PUBLISHER_DOI_GUARDS.get(retained_index)
            )
            if (
                pub_id != expected_id
                or title != expected_title
                or year != expected_year
                or row["venue"] != expected_venue
                or row["authors"] != expected_authors
                or row["citation_count"]
                or scholar_pub_id(retained) != retained_pub_id
                or retained["title"] != retained_title
                or retained["year"] != expected_year
                or retained_index not in CMU_PRIMARY
                or retained_doi != doi
            ):
                raise ValueError(f"CMU publisher duplicate changed at row {index}")
            status = "proposed_corrupted_duplicate_primary"
            evidence_source = "publisher-deposited DOI and full author record"
            additions.append(
                decision(
                    pub_id,
                    "corrupted_duplicate",
                    (
                        f"This uncited Scholar observation repeats confirmed "
                        f"observation {retained_pub_id}: publisher DOI {doi} "
                        "gives one work with matching author sequence, year, "
                        "volume, and starting page. The publisher names "
                        "Hyung J. Kim; this "
                        "second Scholar ID is a redundant rendering."
                    ),
                    evidence_url,
                    verified_at="2026-10-25",
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
                or (
                    index in CMU_NEXT_PUBLISHER_VENUES
                    and row["venue"] != CMU_NEXT_PUBLISHER_VENUES[index]
                )
                or (index == 798 and row["authors"] != CMU_ECS_MEETING_AUTHORS)
            ):
                raise ValueError(f"CMU primary match changed at capture row {index}")
            if index in CMU_GROUP_PUBLISHER_DOI_GUARDS:
                doi = CMU_GROUP_PUBLISHER_DOI_GUARDS[index]
                if not cmu_exact or cmu_exact["doi"].lower() != doi:
                    raise ValueError(f"CMU group publisher DOI changed at row {index}")
            if index in CMU_PRIMARY_CANONICAL_DOI_OVERRIDE:
                doi = CMU_PRIMARY_CANONICAL_DOI_OVERRIDE[index]
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
        "cmu_orcid_doi_identity_hold": 1,
        "cmu_primary_work_retained": 123,
        "knu_orcid_exact_title_year_missing_doi_hold": 2,
        "knu_orcid_title_only_year_check": 1,
        "proposed_other_person_hyung_joon_primary": 15,
        "proposed_other_person_knu_orcid_title_year_record_doi": 320,
        "proposed_other_person_primary_work": 153,
        "proposed_no_target_author_primary_work": 10,
        "proposed_acs_supporting_information_nonpublication": 1,
        "proposed_corrupted_duplicate_primary": 7,
        "unresolved_attribution_hold": 277,
    }
    if counts != expected:
        raise ValueError(f"Screening counts changed: {dict(counts)}")
    addition_ids = [item["author_pub_id"] for item in additions]
    capture_ids = [item["author_pub_id"] for item in screening]
    if len(set(addition_ids)) != 506 or len(set(capture_ids)) != 918:
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
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("exclusions"), list)
        or any(item.get("scholar_id") != SCHOLAR_ID for item in manifest["exclusions"])
    ):
        raise ValueError("Kim proposal manifest contains invalid or foreign decisions")
    existing = {item["author_pub_id"]: item for item in manifest["exclusions"]}
    baseline = {
        scholar_pub_id(row)
        for row in json.loads(args.capture.read_text())["works"]
        if row["index"] in BASELINE_EXCLUDED_INDICES
    }
    if not baseline <= set(existing):
        raise ValueError("The eight baseline Kim decisions from PR #119 are missing")
    expected_ids = baseline | {item["author_pub_id"] for item in additions}
    if set(existing) - expected_ids:
        raise ValueError("Unreviewed decision in Kim proposal manifest")
    for item in additions:
        current = existing.get(item["author_pub_id"])
        if current is not None and current != item:
            raise ValueError(f"Conflicting decision for {item['author_pub_id']}")
    if args.write:
        missing = [item for item in additions if item["author_pub_id"] not in existing]
        manifest["exclusions"].extend(missing)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        with LEDGER.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(screening[0]))
            writer.writeheader()
            writer.writerows(screening)
        print(
            f"Added {len(missing)} exact decisions; wrote {len(screening)} ledger rows"
        )
    else:
        if set(existing) != expected_ids:
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
