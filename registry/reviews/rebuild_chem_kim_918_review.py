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


def decision(pub_id: str, reason_code: str, reason: str, url: str) -> dict:
    return {
        "scholar_id": SCHOLAR_ID,
        "author_pub_id": pub_id,
        "reason_code": reason_code,
        "reason": reason,
        "evidence_url": url,
        "verified_at": "2026-10-24",
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
            if title != expected_title or year != str(expected_year):
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
        "cmu_orcid_exact_title_year_retained": 100,
        "cmu_orcid_title_only_year_check": 1,
        "knu_orcid_exact_title_year_missing_doi_hold": 2,
        "knu_orcid_title_only_year_check": 5,
        "proposed_other_person_hyung_joon_primary": 15,
        "proposed_other_person_knu_orcid_title_year_record_doi": 320,
        "proposed_other_person_primary_work": 15,
        "proposed_acs_supporting_information_nonpublication": 1,
        "unresolved_attribution_hold": 451,
    }
    if counts != expected:
        raise ValueError(f"Screening counts changed: {dict(counts)}")
    addition_ids = [item["author_pub_id"] for item in additions]
    capture_ids = [item["author_pub_id"] for item in screening]
    if len(set(addition_ids)) != 351 or len(set(capture_ids)) != 918:
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
