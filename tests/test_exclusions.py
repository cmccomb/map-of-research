import json
from pathlib import Path

import pandas
import pytest

from map_of_research.dataset import build_dataset_tables
from map_of_research.exclusions import (
    DEFAULT_EXCLUSIONS_PATH,
    ReviewedExclusion,
    load_exclusions,
)
from map_of_research.publisher import build_map_artifact
from map_of_research.registry import load_registry
from map_of_research.snapshot import _load_curated_rows
from tests.registry_helpers import write_registry


def _write_manifest(path: Path, entries: list[object], *, version: int = 1) -> Path:
    path.write_text(json.dumps({"schema_version": version, "exclusions": entries}))
    return path


def _entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "reason_code": "false_profile_attribution",
        "reason": "Primary author list excludes the profile owner.",
        "evidence_url": "https://example.test/paper",
        "verified_at": "2026-10-24",
        **changes,
    }


def test_checked_in_review_manifest_is_complete_and_keyed() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    assert len(exclusions) >= 990
    assert sum(key[0] == "ILambBMAAAAJ" for key in exclusions) == 43
    nesrine = "IXO9K4UAAAAJ"
    assert {
        key[1].split(":")[-1]: entry.reason_code
        for key, entry in exclusions.items()
        if key[0] == nesrine
    } == {
        "_kc_bZDykSQC": "nonpublication",
        "ULOm3_A8WrAC": "nonpublication",
    }
    for retained in (
        "MXK_kJrjxJIC",
        "5nxA0vEk-isC",
        "YsMSGLbcyi4C",
        "Zph67rFs4hoC",
    ):
        assert (nesrine, f"{nesrine}:{retained}") not in exclusions

    assert sum(key[0] == "L0nXO5EAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "FRtd8_QAAAAJ" for key in exclusions) == 19
    ihab = "xHKK05MAAAAJ"
    ihab_codes = {
        "ULOm3_A8WrAC": "corrupted_duplicate",
        "_kc_bZDykSQC": "corrupted_duplicate",
        "_FxGoFyzp5QC": "corrupted_duplicate",
        "_Qo2XoVZTnwC": "nonpublication",
        "mB3voiENLucC": "false_profile_attribution",
    }
    assert {
        key[1].split(":")[-1]: entry.reason_code
        for key, entry in exclusions.items()
        if key[0] == ihab
    } == ihab_codes
    for retained in ("eQOLeE2rZwMC", "u-x6o8ySG0sC", "YOwf2qJgpHMC", "4TOpqqG69KYC"):
        assert (ihab, f"{ihab}:{retained}") not in exclusions
    mohamed = "JDo1onIAAAAJ"
    assert {
        key[1].split(":")[-1]: entry.reason_code
        for key, entry in exclusions.items()
        if key[0] == mohamed
    } == {
        "Zph67rFs4hoC": "corrupted_duplicate",
        "3fE2CSJIrl8C": "nonpublication",
        "MXK_kJrjxJIC": "nonpublication",
    }
    for retained in ("Y0pCki6q_DkC", "WF5omc3nYNoC", "5nxA0vEk-isC"):
        assert (mohamed, f"{mohamed}:{retained}") not in exclusions
    assert sum(key[0] == "JbSQsXAAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "8AqWr_4AAAAJ" for key in exclusions) == 20
    assert sum(key[0] == "_VnEj3gAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "VzAGH3wAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "cAiedtsAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "wJ_gQLoAAAAJ" for key in exclusions) == 77
    assert sum(key[0] == "LibSWRMAAAAJ" for key in exclusions) == 23
    assert all(
        exclusions[("cAiedtsAAAAJ", f"cAiedtsAAAAJ:{suffix}")].reason_code
        == "nonpublication"
        for suffix in (
            "e5wmG9Sq2KIC",
            "TQgYirikUcIC",
            "HDshCWvjkbEC",
            "R3hNpaxXUhUC",
        )
    )
    assert ("cAiedtsAAAAJ", "cAiedtsAAAAJ:qxL8FJ1GzNcC") not in exclusions
    assert (
        exclusions[("VzAGH3wAAAAJ", "VzAGH3wAAAAJ:ns9cj8rnVeAC")].reason_code
        == "nonpublication"
    )
    assert ("VzAGH3wAAAAJ", "VzAGH3wAAAAJ:iH-uZ7U-co4C") not in exclusions
    assert sum(key[0] == "TSkli5EAAAAJ" for key in exclusions) == 139
    assert sum(key[0] == "BWNUifwAAAAJ" for key in exclusions) == 13
    assert sum(key[0] == "ySlRYXoAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "28yZqbcAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "RPkJ39IAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "n6flaAsAAAAJ" for key in exclusions) == 18
    assert sum(key[0] == "VQfvIfQAAAAJ" for key in exclusions) == 16
    assert sum(key[0] == "yssfFIsAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "8Vj5Ps4AAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "gJbEoq8AAAAJ" for key in exclusions) == 6
    assert sum(key[0] == "CGSMDGwAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "ctYLFRIAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "5TCNs50AAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "CaQRJ8IAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "7ICaR4EAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "FmZaK1wAAAAJ" for key in exclusions) == 22
    assert sum(key[0] == "tWQsy4UAAAAJ" for key in exclusions) == 11
    assert sum(key[0] == "CRCzglkAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "neJ3NIUAAAAJ" for key in exclusions) == 34
    assert sum(key[0] == "JV3ryMgAAAAJ" for key in exclusions) == 21
    assert sum(key[0] == "0tIcPvMAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "SHxT0AEAAAAJ" for key in exclusions) == 10
    assert sum(key[0] == "FkS0PMEAAAAJ" for key in exclusions) == 8
    assert sum(key[0] == "W9-dYLcAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "6ArMETUAAAAJ" for key in exclusions) == 18
    assert sum(key[0] == "_IfiZD4AAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "FXqaMl4AAAAJ" for key in exclusions) == 6
    assert sum(key[0] == "T3nx5tgAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "x_xzlvoAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "RnA6G0UAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "qQLlBH4AAAAJ" for key in exclusions) == 37
    assert sum(key[0] == "bKMKO_AAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "MbRIDjgAAAAJ" for key in exclusions) == 9
    assert sum(key[0] == "wXSgFr4AAAAJ" for key in exclusions) == 16
    assert sum(key[0] == "rQ79IvMAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "wmP8_2AAAAAJ" for key in exclusions) == 9
    assert sum(key[0] == "6VKJW4kAAAAJ" for key in exclusions) == 11
    assert sum(key[0] == "-HzcsYQAAAAJ" for key in exclusions) == 7
    assert sum(key[0] == "IggMbacAAAAJ" for key in exclusions) == 10
    assert sum(key[0] == "xfTeH7YAAAAJ" for key in exclusions) == 17
    assert sum(key[0] == "HMSZExQAAAAJ" for key in exclusions) == 36
    assert sum(key[0] == "EqKRQ4YAAAAJ" for key in exclusions) == 80
    assert sum(key[0] == "H8qYeA0AAAAJ" for key in exclusions) == 10
    assert sum(key[0] == "tfBFyTMAAAAJ" for key in exclusions) == 25
    assert sum(key[0] == "9lFYMIkAAAAJ" for key in exclusions) == 31
    assert sum(key[0] == "A0ISJPUAAAAJ" for key in exclusions) == 27
    assert sum(key[0] == "P_AmD6MAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "uHqbF7sAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "eVxdemsAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "q1g_W5sAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "0Nj-v7sAAAAJ" for key in exclusions) == 20
    assert sum(key[0] == "uOCVq1cAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "ZEHOKUkAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "jrRg4PAAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "79uTAy8AAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "AcW2EXwAAAAJ" for key in exclusions) == 7
    assert sum(key[0] == "kcXOB1MAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "jcb81yAAAAAJ" for key in exclusions) == 13
    assert sum(key[0] == "wex9ft0AAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "6pPZKd0AAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "grTfBeAAAAAJ" for key in exclusions) == 16
    assert sum(key[0] == "lKAvvmoAAAAJ" for key in exclusions) == 2
    assert {decision.reason_code for decision in exclusions.values()} == {
        "nonpublication",
        "false_profile_attribution",
        "corrupted_duplicate",
    }


def test_natalie_mcguier_only_malformed_second_review_is_excluded() -> None:
    scholar_id = "2Lz_LuUAAAAJ"
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    actual = {
        author_pub_id: decision.reason_code
        for (sid, author_pub_id), decision in exclusions.items()
        if sid == scholar_id
    }
    assert actual == {f"{scholar_id}:IjCSPb-OGe4C": "corrupted_duplicate"}
    assert (scholar_id, f"{scholar_id}:u-x6o8ySG0sC") not in exclusions


def test_tina_kahniashvili_exact_reviewed_exclusions() -> None:
    scholar_id = "au_cLfoAAAAJ"
    expected = {
        "UHK10RUVsp4C": "corrupted_duplicate",
        "k_IJM867U9cC": "false_profile_attribution",
        "sSrBHYA8nusC": "corrupted_duplicate",
        "xtRiw3GOFMkC": "nonpublication",
        "HDshCWvjkbEC": "false_profile_attribution",
        "M3NEmzRMIkIC": "nonpublication",
        "e5wmG9Sq2KIC": "nonpublication",
        "olpn-zPbct0C": "nonpublication",
        "WqliGbK-hY8C": "nonpublication",
        "5awf1xo2G04C": "nonpublication",
        "SdhP9T11ey4C": "nonpublication",
        "eq2jaN3J8jMC": "nonpublication",
        "g5m5HwL7SMYC": "corrupted_duplicate",
        "blknAaTinKkC": "nonpublication",
        "N5tVd3kTz84C": "corrupted_duplicate",
    }
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    actual = {
        author_pub_id.removeprefix(f"{scholar_id}:"): decision.reason_code
        for (sid, author_pub_id), decision in exclusions.items()
        if sid == scholar_id
    }
    assert actual == expected
    assert (scholar_id, f"{scholar_id}:VL0QpB8kHFEC") not in exclusions
    assert (scholar_id, f"{scholar_id}:TFP_iSt0sucC") not in exclusions


def test_breivik_five_reviewed_exclusions_preserve_existing_decisions() -> None:
    scholar_id = "AcW2EXwAAAAJ"
    expected = {
        "K3LRdlH-MEoC": "false_profile_attribution",
        "uWQEDVKXjbEC": "false_profile_attribution",
        "q3oQSFYPqjQC": "nonpublication",
        "CHSYGLWDkRkC": "nonpublication",
        "0EnyYjriUFMC": "corrupted_duplicate",
        "u-x6o8ySG0sC": "corrupted_duplicate",
        "qUcmZB5y_30C": "nonpublication",
    }
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    actual = {
        author_pub_id.removeprefix(f"{scholar_id}:"): decision.reason_code
        for (sid, author_pub_id), decision in exclusions.items()
        if sid == scholar_id
    }
    assert actual == expected
    for retained in (
        "Tyk-4Ss8FVUC",  # 2020 journal article behind row 112.
        "RYcK_YlVTxYC",  # Journal article behind row 84 dataset.
        "P5F9QuxV20EC",  # Published paper after row 99 proposal.
        "roLk4NBRz8UC",  # Numbered AAS abstract behind row 117.
        "p2g8aNsByqUC",  # Faithful arXiv preprint observation.
        "ufrVoPGSRksC",  # 2018 manuscript, distinct from meeting abstract.
        "d1gkVwhDpl0C",  # Citable software; output policy unresolved.
    ):
        assert (scholar_id, f"{scholar_id}:{retained}") not in exclusions


def test_jarvik_exact_review_excludes_only_source_verified_rows() -> None:
    scholar_id = "grTfBeAAAAAJ"
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    expected = {
        "u9iWguZQMMsC": "nonpublication",
        "2osOgNQ5qMEC": "nonpublication",
        "Tiz5es2fbqcC": "false_profile_attribution",
        "9yKSN-GCB0IC": "nonpublication",
        "XiSMed-E-HIC": "nonpublication",
        "Se3iqnhoufwC": "false_profile_attribution",
        "IjCSPb-OGe4C": "corrupted_duplicate",
        "Zph67rFs4hoC": "corrupted_duplicate",
        "WbkHhVStYXYC": "nonpublication",
        "1sJd4Hv_s6UC": "nonpublication",
        "EUQCXRtRnyEC": "corrupted_duplicate",
        "xtRiw3GOFMkC": "nonpublication",
        "P5F9QuxV20EC": "nonpublication",
        "CHSYGLWDkRkC": "nonpublication",
        "uWQEDVKXjbEC": "corrupted_duplicate",
        "L8Ckcad2t8MC": "nonpublication",
    }
    actual = {
        author_pub_id.removeprefix(f"{scholar_id}:"): decision.reason_code
        for (sid, author_pub_id), decision in exclusions.items()
        if sid == scholar_id
    }
    assert actual == expected

    retained = {
        "g5m5HwL7SMYC",  # Original article behind excluded supplement.
        "TQgYirikUcIC",  # Original article behind corrupted OCR duplicate.
        "hMod-77fHWUC",  # Original article behind another corrupted duplicate.
        "M05iB0D1s5AC",  # Original PMSG article behind two duplicate rows.
        "p2g8aNsByqUC",  # Distinct 2008 correction, not false attribution.
        "mB3voiENLucC",  # Distinct 2003 correction, not a duplicate paper.
        "u5HHmVD_uO8C",  # Unresolved venue and attribution.
        "abG-DnoFyZgC",  # Malformed title, no demonstrated duplicate.
    }
    assert retained.isdisjoint(actual)


def test_adviti_naik_review_excludes_only_two_source_verified_duplicate_rows() -> None:
    scholar_id = "lKAvvmoAAAAJ"
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    expected = {"ZeXyd9-uunAC", "YsMSGLbcyi4C"}
    actual = {
        author_pub_id.removeprefix(f"{scholar_id}:")
        for sid, author_pub_id in exclusions
        if sid == scholar_id
    }
    assert actual == expected
    assert all(
        exclusions[(scholar_id, f"{scholar_id}:{suffix}")].reason_code
        == "corrupted_duplicate"
        for suffix in expected
    )
    for retained in (
        "ufrVoPGSRksC",  # Complete 2019 Frontiers article.
        "d1gkVwhDpl0C",  # Complete 2014 PLOS article.
        "IjCSPb-OGe4C",  # Separate 2013 dissertation.
        "0EnyYjriUFMC",  # Separate 2021 bioRxiv version.
        "u5HHmVD_uO8C",  # Yearless conference abstract, not disproved.
    ):
        assert (scholar_id, f"{scholar_id}:{retained}") not in exclusions


def test_physics_review_excludes_only_exact_children_and_keeps_retained_works() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    for scholar_id, children, retained in (
        (
            "wJ_gQLoAAAAJ",
            ("6AKj_8xLoccC",),
            "lgwcVrK6X84C",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("qjuL_XCUnM8C", "W91e3rS6dHEC"),
            "9qwnyNWB-oEC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("gTud6kY3p2wC", "S0RksyIsHSIC"),
            "P1qO8dLd1z8C",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("q22ZieUsQXoC", "5uu-MYFt2EcC"),
            "E7VqQtBCVmcC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("vYYylRVofzEC", "csTP5AdqY_wC"),
            "VN7nJs4JPk0C",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("T-Bu49-jKQgC", "-XtG2q47PdUC"),
            "2P1L_qKh6hAC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("Yp96WNAcdNsC", "ve7iT2ZEuL4C"),
            "q3oQSFYPqjQC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("aOEnwurrEPcC", "TiLqlu47W2oC"),
            "C-Rn0OCouf8C",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("IF0FbId8XVwC",),
            "kvJssbFybhEC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("6Dd5luMImnEC",),
            "GwaQhVSQhKEC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("1DdUqAzPfXgC",),
            "USX2HHcnDqcC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("X0ywoi6J0UgC",),
            "OcBU2YAGkTUC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("1L_zwHEQtZoC",),
            "VOx2b1Wkg3QC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("afsF9h1fxg0C",),
            "GUYAmugLYisC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("besYCffB3V8C",),
            "8p-ueveQw4wC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("lDOOmgye57wC",),
            "DvbUhBKpgQUC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("cIAfX1pVE14C",),
            "1sJd4Hv_s6UC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("XEKcsfOMjpkC",),
            "u76CbRzuV8AC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("uQdrZ3qaPmcC",),
            "IyxfXMz2bNAC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("ey9_mCyZ-VYC",),
            "9shLKfS_uJEC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("-mA2wk0MEBkC",),
            "5Klqo5HVOaoC",
        ),
        (
            "wJ_gQLoAAAAJ",
            ("CXI6bF9CpJ4C", "XeQYkT-08pIC"),
            "It3QlvBKHbMC",
        ),
        ("wJ_gQLoAAAAJ", ("jvrRlaHg2sAC",), "9pM33mqn1YgC"),
        ("wJ_gQLoAAAAJ", ("zBs93F57jOAC",), "cK4Rrx0J3m0C"),
        ("wJ_gQLoAAAAJ", ("YEBtnu9hOHUC",), "09LM3QYkMKUC"),
        ("wJ_gQLoAAAAJ", ("zTJoPluU4X4C",), "urP0JZOBBUsC"),
        ("wJ_gQLoAAAAJ", ("IX653JsL2_EC",), "kF1pexMAQbMC"),
        ("wJ_gQLoAAAAJ", ("kxd3qP2_5uAC",), "ziOE8S1-AIUC"),
        ("wJ_gQLoAAAAJ", ("_IwLajd9sWMC",), "Tyk-4Ss8FVUC"),
        ("wJ_gQLoAAAAJ", ("o4Qvs5Y5TLQC",), "MLfJN-KU85MC"),
        ("wJ_gQLoAAAAJ", ("lu5AzqVF4qYC",), "GUYAmugLYisC"),
        ("wJ_gQLoAAAAJ", ("PPrs6GlykQcC",), "uWQEDVKXjbEC"),
        ("wJ_gQLoAAAAJ", ("t7izwRedFcYC",), "cWzG1nlazyYC"),
        ("wJ_gQLoAAAAJ", ("ZyBjE3zI-i8C",), "cFHS6HbyZ2cC"),
        (
            "wJ_gQLoAAAAJ",
            ("6hPiuOuOds4C", "Gs1AF5H0x94C"),
            "Ggqy4mf-avwC",
        ),
        ("wJ_gQLoAAAAJ", ("FtLLHOsGVzkC",), "4X0JR2_MtJMC"),
        ("wJ_gQLoAAAAJ", ("AdGrMoNItN4C",), "PffFD9Wx8kQC"),
        ("wJ_gQLoAAAAJ", ("X41XOdD1uaEC",), "nPT8s1NX_-sC"),
        ("wJ_gQLoAAAAJ", ("eJjLl3UG7CkC",), "F2UWTTQJPOcC"),
        ("wJ_gQLoAAAAJ", ("rCuJAniQoLoC",), "1DsIQWDZLl8C"),
        ("wJ_gQLoAAAAJ", ("QZWLLlSfqgYC",), "yB1At4FlUx8C"),
        ("wJ_gQLoAAAAJ", ("AbQWx2m_oG8C",), "bkKuixW_xMkC"),
        ("wJ_gQLoAAAAJ", ("PQ6BjWB6CiEC",), "zJ9_mGRecVUC"),
        ("wJ_gQLoAAAAJ", ("cvOuzwNwsG0C",), "p39txVt3toMC"),
        ("wJ_gQLoAAAAJ", ("1l3MdapXzAoC",), "KNjnJ3z-R6IC"),
        ("wJ_gQLoAAAAJ", ("OA9cGvo6324C",), "WqliGbK-hY8C"),
        ("wJ_gQLoAAAAJ", ("HmWvvuJwtxYC",), "bVQMTfhMCi4C"),
        ("wJ_gQLoAAAAJ", ("ddB7do2jUx8C",), "HtS1dXgVpQUC"),
        ("wJ_gQLoAAAAJ", ("kmu2AJ0NrTYC",), "KxtntwgDAa4C"),
        ("wJ_gQLoAAAAJ", ("8EvVLpklxGMC",), "N2osnxbUuuUC"),
        ("wJ_gQLoAAAAJ", ("rwEhk56xNqMC",), "-_dYPAW6P2MC"),
        ("wJ_gQLoAAAAJ", ("3I1Fh14WnZcC",), "EYYDruWGBe4C"),
        ("wJ_gQLoAAAAJ", ("PcT55Ow6fAIC",), "3NQIlFlcGxIC"),
        ("wJ_gQLoAAAAJ", ("qt-6tCTBDsQC",), "ZzlSgRqYykMC"),
        ("wJ_gQLoAAAAJ", ("ZpfHopiqs50C",), "pqnbT2bcN3wC"),
        ("wJ_gQLoAAAAJ", ("sdbE4nmAF2cC",), "0Kh4an1R61UC"),
        ("wJ_gQLoAAAAJ", ("4ENpXO9GHvQC",), "OP4eGU-M3BUC"),
        ("wJ_gQLoAAAAJ", ("KJj4oxv5wnIC",), "6VlyvFCUEfcC"),
        ("wJ_gQLoAAAAJ", ("ugwUUJJV9MUC",), "HP8h_xu6YbIC"),
        ("wJ_gQLoAAAAJ", ("UrmhOAI2Q4QC",), "t4WtaE3RIIAC"),
        ("wJ_gQLoAAAAJ", ("0-VSGmPhd2EC",), "TjyTfcaXIY8C"),
        (
            "wJ_gQLoAAAAJ",
            ("tATOuK1XnKgC", "XFIFPR1yFx4C"),
            "_-FgD5s_Z1cC",
        ),
        (
            "LibSWRMAAAAJ",
            ("i91s68tWr-MC",),
            "2vr6o8x5NLkC",
        ),
        (
            "LibSWRMAAAAJ",
            ("tz746QTLzJkC",),
            "hqOjcs7Dif8C",
        ),
        (
            "LibSWRMAAAAJ",
            ("1_W9tMSvGuwC",),
            "P7Ujq4OLJYoC",
        ),
        (
            "LibSWRMAAAAJ",
            ("L24QuVWYgZ0C",),
            "IjCSPb-OGe4C",
        ),
        (
            "LibSWRMAAAAJ",
            ("hHIA4WEVY-EC", "gFrPXmx1TSsC"),
            "hkOj_22Ku90C",
        ),
        (
            "LibSWRMAAAAJ",
            ("ymY9cBF3mdcC",),
            "a0OBvERweLwC",
        ),
        (
            "LibSWRMAAAAJ",
            ("M0leSnx2MbUC", "IWHjjKOFINEC"),
            "ZqE1mSdD_DYC",
        ),
        (
            "LibSWRMAAAAJ",
            ("hNSvKAmkeYkC",),
            "kNdYIx-mwKoC",
        ),
    ):
        for child in children:
            assert exclusions[(scholar_id, f"{scholar_id}:{child}")].reason_code == (
                "corrupted_duplicate"
            )
        assert (scholar_id, f"{scholar_id}:{retained}") not in exclusions
    assert ("wJ_gQLoAAAAJ", "wJ_gQLoAAAAJ:zxzvZ9-XIW8C") not in exclusions
    for suffix in (
        "U0iAMwwPxtsC",
        "ODE9OILHJdcC",
        "LK8CI43ZvvMC",
        "86PQX7AUzd4C",
        "LO7wyVUgiFcC",
        "i2xiXl-TujoC",
        "edDO8Oi4QzsC",
        "F9fV5C73w3QC",
    ):
        assert exclusions[("LibSWRMAAAAJ", f"LibSWRMAAAAJ:{suffix}")].reason_code == (
            "nonpublication"
        )
    assert ("LibSWRMAAAAJ", "LibSWRMAAAAJ:SWgZeABleR0C") not in exclusions
    assert ("wJ_gQLoAAAAJ", "wJ_gQLoAAAAJ:y4Au4brZMfoC") not in exclusions
    assert ("LibSWRMAAAAJ", "LibSWRMAAAAJ:iyewoVqAXLQC") not in exclusions


def test_math_qatar_duplicate_review_preserves_canonical_and_joint_work() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    for scholar_id, excluded_id, retained_id in (
        ("yssfFIsAAAAJ", "W7OEmFMy1HYC", "LkGwnXOMwfcC"),
        ("8Vj5Ps4AAAAJ", "W7OEmFMy1HYC", "zYLM7Y9cAGgC"),
    ):
        assert exclusions[(scholar_id, f"{scholar_id}:{excluded_id}")].reason_code == (
            "corrupted_duplicate"
        )
        assert (scholar_id, f"{scholar_id}:{retained_id}") not in exclusions
    # CMU Qatar independently credits Demirkoparan despite Scholar's short author field.
    assert ("yssfFIsAAAAJ", "yssfFIsAAAAJ:mB3voiENLucC") not in exclusions


def test_walkington_corrupted_scholar_row_uses_verified_ams_record(tmp_path) -> None:
    scholar_id = "CGSMDGwAAAAJ"
    author_pub_id = f"{scholar_id}:8AbLer7MMksC"
    doi = "10.1090/qam/1598"
    person_id = "person-noeljwalkington-c55e6e3b"
    source = json.loads(Path("data/curated/works.json").read_text())
    verified = [work for work in source["works"] if work["doi"] == doi]
    assert len(verified) == 1
    curated_path = tmp_path / "works.json"
    curated_path.write_text(
        json.dumps({"schema_version": 1, "external_coauthors": [], "works": verified})
    )
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": person_id,
                "display_name": "Noel J. Walkington",
                "scholar_id": scholar_id,
            }
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://www.cmu.edu/math/people/faculty/index.html",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematical Sciences",
                "directory_url": "https://www.cmu.edu/math/people/faculty/index.html",
            }
        ],
        schools=[
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://www.cmu.edu/mcs",
            }
        ],
    )
    registry = load_registry(*paths)
    curated = _load_curated_rows(curated_path, registry)[0]
    curated["embedding"] = [0.0, 1.0]
    scholar = {
        **curated,
        "scholar_id": scholar_id,
        "author_pub_id": author_pub_id,
        "source_type": "scholar",
        "source_doi": "",
        "authors": "M Duan",
        "venue": "Carnegie Mellon University, 2021",
        "citation": "",
        "citation_count": 1,
        "source_url": (
            "https://scholar.google.com/citations?view_op=view_citation"
            f"&user={scholar_id}&citation_for_view={author_pub_id}"
        ),
        "source_record_json": json.dumps(
            {
                "title": verified[0]["title"],
                "author": "M Duan",
                "pub_year": "2021",
                "venue": "Carnegie Mellon University, 2021",
                "author_pub_id": author_pub_id,
            }
        ),
        "source_verified_at": "",
        "source_verification_urls": [],
        "fetched_at_utc": "2026-10-24T12:00:00+00:00",
    }
    tables = build_dataset_tables(
        pandas.DataFrame([scholar, curated]),
        registry,
        load_exclusions(DEFAULT_EXCLUSIONS_PATH),
    )
    raw = (
        tables["profile_publications"]
        .loc[tables["profile_publications"]["author_pub_id"].eq(author_pub_id)]
        .iloc[0]
    )
    assert raw["authors"] == "M Duan"
    assert raw["venue"] == "Carnegie Mellon University, 2021"
    assert raw["review_exclusion_code"] == "corrupted_duplicate"
    assert not raw["map_eligible"]
    work = tables["works"].iloc[0]
    assert work["work_id"] == raw["work_id"]
    assert work["doi"] == doi
    assert work["authors"] == "Mochong Duan and Noel J. Walkington"
    assert work["venue"] == "Quarterly of Applied Mathematics"
    assert work["citation"] == "Quarterly of Applied Mathematics 79(4), 695–715 (2021)"
    assert work["source_types"] == ["curated_verified"]
    assert len(tables["profile_publications"]) == 2
    assert len(tables["authorships"]) == 1
    assert tables["authorships"].iloc[0]["source_type"] == "curated_verified"


def test_yasser_review_excludes_only_two_sourced_duplicates() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    scholar_id = "_VnEj3gAAAAJ"
    assert {
        author_pub_id for sid, author_pub_id in exclusions if sid == scholar_id
    } == {
        f"{scholar_id}:R3hNpaxXUhUC",
        f"{scholar_id}:qUcmZB5y_30C",
    }
    for suffix in ("2osOgNQ5qMEC", "zYLM7Y9cAGgC", "RHpTSmoSYBkC"):
        assert (scholar_id, f"{scholar_id}:{suffix}") not in exclusions
    assert all(
        exclusions[(scholar_id, f"{scholar_id}:{suffix}")].reason_code
        == "corrupted_duplicate"
        for suffix in ("R3hNpaxXUhUC", "qUcmZB5y_30C")
    )


@pytest.mark.parametrize(
    ("entries", "version", "message"),
    [
        ([], 2, "Unsupported Scholar-exclusion schema"),
        (None, 1, "Scholar exclusions must be a list"),
        ([None], 1, "must be an object"),
        ([_entry(scholar_id="")], 1, "invalid observation identity"),
        ([_entry(author_pub_id="other:paper")], 1, "invalid observation identity"),
        ([_entry(reason_code="unknown")], 1, "known code"),
        ([_entry(reason="")], 1, "known code"),
        ([_entry(evidence_url="http://example.test")], 1, "HTTPS evidence URL"),
        ([_entry(verified_at="yesterday")], 1, "verified_at"),
        ([_entry(), _entry()], 1, "Duplicate Scholar exclusion"),
    ],
)
def test_review_manifest_rejects_invalid_decisions(
    tmp_path, entries, version, message
) -> None:
    path = _write_manifest(tmp_path / "exclusions.json", entries, version=version)
    with pytest.raises(ValueError, match=message):
        load_exclusions(path)


def test_reviewed_false_attribution_cannot_attach_faculty_to_shared_work(
    tmp_path,
) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One",
                "scholar_id": "oneAAAAJ",
            },
            {
                "person_id": "person-two",
                "display_name": "Two",
                "scholar_id": "twoAAAAJ",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/directory",
                "verified_at": "2026-10-24",
            }
            for person_id, department_id in (
                ("person-one", "ece"),
                ("person-two", "math"),
            )
        ]
        + [
            {
                "person_id": "person-two",
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "source_url": "https://example.test/directory",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "ECE",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematics",
                "directory_url": "https://example.test/math",
            },
        ],
        schools=[
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://example.test/engineering",
            },
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://example.test/mcs",
            },
        ],
    )
    registry = load_registry(*paths)
    base = {
        "faculty": "",
        "authors": "Two and Collaborator",
        "title": "A Substantive Shared Research Article",
        "year": 2024,
        "venue": "Journal",
        "citation": "Journal 2024",
        "source_record_json": "{}",
        "fetched_at_utc": "2026-10-24T00:00:00+00:00",
        "embedding": [0.1, 0.2],
        "source_type": "scholar",
    }
    bad = {
        **base,
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "person_id": "person-one",
        "display_name": "One",
        "memberships": [{"department_id": "ece", "included": True}],
        "department_ids": ["ece"],
        "authors": "One and Incorrect",
        "venue": "Misattributed Preprint",
        "citation": "Bad venue 2024",
        "citation_count": 900,
        "source_url": "https://example.test/bad-source",
        "external_coauthors": [
            {
                "person_id": "external-bad",
                "display_name": "Bad Link",
                "source_url": "https://example.test/bad-author",
            }
        ],
    }
    good = {
        **base,
        "scholar_id": "twoAAAAJ",
        "author_pub_id": "twoAAAAJ:paper",
        "person_id": "person-two",
        "display_name": "Two",
        "memberships": [
            {"department_id": "math", "included": True},
            {"department_id": "ece", "included": False},
        ],
        "department_ids": ["math"],
        "citation_count": 4,
        "source_url": "https://example.test/good-source",
        "external_coauthors": [],
    }
    nonpublication = {
        **bad,
        "author_pub_id": "oneAAAAJ:award",
        "title": "NSF Award 1234567",
        "source_url": "https://example.test/award",
    }
    decisions = {
        ("oneAAAAJ", "oneAAAAJ:paper"): ReviewedExclusion(
            "false_profile_attribution",
            "The paper's author list excludes One.",
            "https://example.test/paper",
            "2026-10-24",
        ),
        ("oneAAAAJ", "oneAAAAJ:award"): ReviewedExclusion(
            "nonpublication",
            "Grant award, not a publication.",
            "https://example.test/award",
            "2026-10-24",
        ),
    }

    tables = build_dataset_tables(
        pandas.DataFrame([bad, good, nonpublication]), registry, decisions
    )
    raw = tables["profile_publications"].set_index("author_pub_id")
    assert len(raw) == 3
    assert not bool(raw.loc["oneAAAAJ:paper", "map_eligible"])
    assert raw.loc["oneAAAAJ:paper", "review_exclusion_code"] == (
        "false_profile_attribution"
    )
    assert raw.loc["oneAAAAJ:paper", "review_evidence_url"] == (
        "https://example.test/paper"
    )
    assert raw.loc["oneAAAAJ:paper", "review_verified_at"] == "2026-10-24"
    assert raw.loc["oneAAAAJ:award", "review_exclusion_reason"] == (
        "Grant award, not a publication."
    )
    assert raw.loc["twoAAAAJ:paper", "review_exclusion_code"] == ""
    assert tables["authorships"]["person_id"].tolist() == ["person-two"]
    shared = tables["works"].loc[tables["works"]["map_eligible"]].iloc[0]
    assert shared["faculty"] == ["Two"]
    assert shared["person_ids"] == ["person-two"]
    assert shared["department_ids"] == ["math"]
    assert shared["school_ids"] == ["mcs"]
    assert shared["authors"] == "Two and Collaborator"
    assert shared["venue"] == "Journal"
    assert shared["citation"] == "Journal 2024"
    assert shared["citation_count"] == 4
    assert shared["source_urls"] == ["https://example.test/good-source"]
    assert shared["author_pub_ids"] == ["twoAAAAJ:paper"]
    assert shared["observation_count"] == 1
    assert shared["external_coauthor_ids"] == []
    excluded = tables["works"].loc[~tables["works"]["map_eligible"]].iloc[0]
    assert excluded["has_included_source_profile"]
    assert excluded["memberships"] == []
    assert excluded["map_exclusion_reasons"] == ["reviewed_nonpublication"]
    people = tables["people"].set_index("person_id")
    assert people.loc["person-one", "publication_observation_count"] == 2
    assert people.loc["person-one", "unique_work_count"] == 0
    assert people.loc["person-one", "mapped_work_count"] == 0
    assert people.loc["person-two", "mapped_work_count"] == 1

    works = tables["works"].copy()
    works["x"] = 0.1
    works["y"] = 0.2
    works["tsne_x"] = 0.3
    works["tsne_y"] = 0.4
    works["keyword_id"] = "topic-one"
    works["keyword"] = "network theory"
    works["keyword_extracted"] = "network theory"
    works["keyword_label_reviewed"] = False
    works["detail_keyword_id"] = "topic-one-detail"
    works["detail_keyword"] = "random network estimates"
    works["detail_keyword_extracted"] = "random network estimates"
    works["detail_keyword_label_reviewed"] = False
    artifact = build_map_artifact(
        works, registry, generated_at_utc="2026-10-24T12:00:00+00:00"
    )
    assert artifact["point_count"] == 1
    assert artifact["excluded_work_count"] == 1
    assert artifact["points"][0]["faculty_ids"] == ["person-two"]
    assert artifact["points"][0]["department_ids"] == ["math"]
    assert artifact["points"][0]["source_url"] == ("https://example.test/good-source")
