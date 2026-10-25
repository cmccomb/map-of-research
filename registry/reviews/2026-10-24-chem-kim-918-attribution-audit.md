# Hyung J. Kim: complete-profile attribution screen, draft

Reviewed 2026-10-24 against the complete visible Google Scholar capture for CMU Chemistry professor Hyung J. Kim (`bC4aq-4AAAAJ`) in source PR #118. The paired capture has **918 unique citation IDs**, no remaining Show More rows, no challenge, and no page error. Its JSON SHA-256 is `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`; its HTML SHA-256 is `a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94`. The raw capture is preserved unchanged. This producer PR adds exact observation decisions and a screen; **it does not promote a Kim canonical cache or publish Kim in either map**.

The [918-row ledger](2026-10-24-chem-kim-918-screen.csv) records each capture index, Scholar observation ID, title, displayed authors/year/venue/citations, comparison result, evidence URL, DOI, and data-quality flags. [The rebuild script](rebuild_chem_kim_918_review.py) compares the saved Scholar capture against a [dated compact ORCID API evidence snapshot](2026-10-24-chem-kim-orcid-evidence.json). It performs no Google Scholar request. The ORCID snapshot preserves each public work's title, year, DOI, ORCID put-code, and data contributor, plus SHA-256 hashes of the full ORCID API responses used to derive it.

## Identity anchors and exact decisions

[CMU's faculty page](https://www.cmu.edu/chemistry/people/faculty/kim.html) links Hyung Kim to ORCID [`0000-0003-4334-1879`](https://orcid.org/0000-0003-4334-1879). [Kyungpook National University](https://research.knu.ac.kr/en/persons/hong-joo-kim/) identifies **Hong Joo Kim** as a Physics professor; [Wiley's article author record](https://onlinelibrary.wiley.com/doi/abs/10.1002/crat.202000074) links that KNU professor to a separate ORCID [`0000-0001-9787-4684`](https://orcid.org/0000-0001-9787-4684). Exact normalized title, displayed year, and DOI matched 323 Scholar rows to the latter ORCID's works; three of those rows were already excluded in producer PR #119. The other **320 exact Scholar IDs** are proposed as `false_profile_attribution` here. The 320 matching ORCID records were supplied by Scopus–Elsevier (319) or AAAS–Science (one); an imported ORCID work is strong but not infallible identity evidence, so these decisions remain draft pending independent review. No topic or author-initials rule is applied to unmatched rows.

Another **15 exact IDs** have a separate primary author record naming **Hyung Joon Kim**, rather than CMU's Hyung J. Kim. For example, [Nature's multiple-sclerosis article](https://www.nature.com/articles/nature12647), [PubMed's full author record for a CA3-neuron article](https://pubmed.ncbi.nlm.nih.gov/29727680/), and [Seoul National University's neural-circuit record](https://snu.elsevierpure.com/en/publications/modeling-neural-circuit-blood-brain-barrier-and-myelination-on-a-/) identify that other researcher. All 15 exact evidence links are in the ledger and manifest. **Seoul National University's record corrects a prior source error:** the [Pittsburgh Quantum Institute page for CMU Hyung Kim](https://www.pqi.org/hyung-kim) lists the neural-circuit paper, but its primary author record names Hyung Joon Kim. PQI's publication list must never be used as sole identity proof.

Capture row **917** is an undated “Supporting Information” record. [ACS attaches it to the 2016 parent paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.6b00202), which is already captured as row **149**, `bC4aq-4AAAAJ:SeFeTyx0c_EC`. The exact row 917 ID is proposed as `nonpublication`; the published parent remains retained. Thus this PR proposes **336 new exact decisions**: 320 KNU, 15 Hyung Joon, and one ACS attachment. Together with #119's eight earlier decisions, Kim has **344** excluded observations in the producer manifest.

| Screen result | Rows | Release interpretation |
| --- | ---: | --- |
| Eight exact exclusions from #119 | 8 | Already reviewed. |
| New KNU title, year, and DOI matches | 320 | Draft exact false-attribution decisions. |
| New Hyung Joon primary-author matches | 15 | Draft exact false-attribution decisions. |
| ACS supporting information | 1 | Draft exact nonpublication decision. |
| CMU ORCID exact title and year | 100 | Positive retention signal; still subject to output-level review. |
| CMU ORCID title only, differing year | 1 | Hold for year/source reconciliation. |
| KNU ORCID title and year, DOI absent | 5 | Hold for independent identity check. |
| KNU ORCID title only, differing year | 5 | Hold for year/source reconciliation. |
| No specific identity match | 463 | Hold for publisher/author identity review. |
| **Total** | **918** | **No canonical cache or live release.** |

The **474 held rows** are not assumed correct or false. They include highly cited works in chemistry, physics, neuroscience, medicine, engineering, and other fields; the subject alone is not evidence. Row 1, the 1985 supergravity paper, remains held despite its appearance on PQI; its author identity still needs an independent check. The 100 CMU ORCID title/year matches are evidence for retention, not proof that a complete publication history has been cleaned.

## Nonpublication and duplicate screen

The ledger flags **21 blank venues, 13 blank years, one blank author list, one supporting-information title, five correction-like titles, and eight repeated normalized-title pairs (16 rows)**. Flags overlap, and only row 917 has a new exclusion backed by a parent publisher page. The repeated-title pairs are capture indices `122/272`, `156/795`, `235/789`, `312/797`, `336/793`, `413/744`, `779/866`, and `864/865`. Some are already foreign-author candidates; others may be distinct preprint/article, patent-family, or bibliographic renderings. No automatic duplicate exclusion follows from title equality. Published corrections remain publications unless a specific publisher record shows otherwise.

## Rebuild and release gate

With #118's JSON capture checked out locally, run:

```sh
python3 registry/reviews/rebuild_chem_kim_918_review.py \
  --capture /path/to/kim-visible-918-2026-10-24.json --check
```

The check rebuilds all 918 status rows, verifies all 336 proposed IDs and decision metadata in the manifest, and compares the ledger byte values after CSV parsing. The saved ORCID API snapshot was retrieved 2026-10-24; the underlying full API response SHA-256 hashes are `dc611e3e71332b7476ed358293b228a0af23dad44755c0ca76f0c18138d40c05` (CMU) and `97703c1f3fe77dd3477645250b35c2316c8e0061b4ed35123ba33d7d6f543c91` (KNU). Future API changes should prompt a new dated comparison, not a silent regeneration of these decisions.

**Live release remains unsafe.** The raw Scholar capture is complete as a capture, but 474 rows still lack a defensible attribution decision, and the KNU ORCID import deserves independent spot checks. Keep Kim candidate-only until the uncertain records are reconciled and the full retained set receives output-level review. This draft changes no canonical Kim cache, graph snapshot, or public page.
