"""Exact-ID gate for Scholar profiles known to contain mixed author identities."""

import csv
import json
from collections import Counter
from pathlib import Path

import pandas
import pytest

from map_of_research import publisher
from map_of_research.attribution import (
    DEFAULT_ATTRIBUTION_REVIEWS_PATH,
    AttributionReviews,
    ConfirmedAttribution,
    load_attribution_reviews,
)
from map_of_research.dataset import build_dataset_tables
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry

SCHOLAR_ID = "8iLLOQgAAAAJ"
LEDGER = Path("registry/reviews/2026-10-25-bio-chien-ho-582-disposition.csv")
FINAL_LEDGER = Path(
    "registry/reviews/2026-10-25-bio-chien-ho-582-followup-reconciled.csv"
)


def _manifest():
    return json.loads(DEFAULT_ATTRIBUTION_REVIEWS_PATH.read_text())


def _write(tmp_path, value):
    path = tmp_path / "attribution.json"
    path.write_text(json.dumps(value))
    return path


def test_upload_reads_attribution_from_producer_checkout(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/upload.yml").read_text()
    source_path = "source/data/curated/scholar-attribution-reviews.json"
    assert f"--attribution-reviews {source_path}" in workflow
    (tmp_path / "source").symlink_to(repo_root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    args = publisher._parse_args(
        [
            "--snapshot",
            "snapshot/snapshots/cmu-engineering-publications.parquet",
            "--manifest",
            "snapshot/snapshots/cmu-engineering-publications.parquet.manifest.json",
            "--attribution-reviews",
            source_path,
        ]
    )
    assert len(load_attribution_reviews(args.attribution_reviews).confirmed) == 437


def test_chien_gate_accounts_for_every_frozen_citation_id():
    attribution = load_attribution_reviews(DEFAULT_ATTRIBUTION_REVIEWS_PATH)
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    with LEDGER.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with FINAL_LEDGER.open(newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    assert len(rows) == len({row["author_pub_id"] for row in rows}) == 582
    assert [int(row["index"]) for row in rows] == list(range(1, 583))
    assert [row["author_pub_id"] for row in final_rows] == [
        row["author_pub_id"] for row in rows
    ]
    assert Counter(row["decision"] for row in rows) == {
        "confirmed_retained": 269,
        "confirmed_excluded": 35,
        "held_attribution_review": 274,
        "held_duplicate_review": 4,
    }
    assert attribution.held_profiles == {SCHOLAR_ID}
    assert len(attribution.confirmed) == 437
    assert sum(sid == SCHOLAR_ID for sid, _ in exclusions) == 92
    assert {
        row["author_pub_id"] for row in rows if row["decision"] == "confirmed_retained"
    } <= {pub_id for sid, pub_id in attribution.confirmed if sid == SCHOLAR_ID}
    assert {
        row["author_pub_id"] for row in rows if row["decision"] == "confirmed_excluded"
    } <= {pub_id for sid, pub_id in exclusions if sid == SCHOLAR_ID}
    positive = {"confirmed_retained", "confirmed_retained_candidate"}
    negative = {"confirmed_excluded", "confirmed_excluded_candidate"}
    assert {
        row["author_pub_id"] for row in final_rows if row["review_decision"] in positive
    } == {pub_id for sid, pub_id in attribution.confirmed if sid == SCHOLAR_ID}
    assert {
        row["author_pub_id"] for row in final_rows if row["review_decision"] in negative
    } == {pub_id for sid, pub_id in exclusions if sid == SCHOLAR_ID}
    assert Counter(row["review_decision"] for row in final_rows) == {
        "confirmed_retained": 304,
        "confirmed_retained_candidate": 133,
        "confirmed_excluded": 37,
        "confirmed_excluded_candidate": 55,
        "still_held": 41,
        "confirmed_author_context_hold": 2,
        "confirmed_author_metadata_hold": 9,
        "confirmed_author_duplicate_swap_hold": 1,
    }
    duplicate = final_rows[494]
    assert duplicate["index"] == "495"
    assert duplicate["related_row"] == "170"
    assert duplicate["review_decision"] == "confirmed_excluded_candidate"
    assert (
        exclusions[(SCHOLAR_ID, duplicate["author_pub_id"])].reason_code
        == "corrupted_duplicate"
    )
    for index in (429, 485):
        row = final_rows[index - 1]
        assert row["review_decision"] == "confirmed_excluded_candidate"
        assert row["decision_origin"] == "chrome_primary_identity_addendum"
        assert (
            exclusions[(SCHOLAR_ID, row["author_pub_id"])].reason_code
            == "false_profile_attribution"
        )
    assert all(
        final_rows[index - 1]["review_decision"] == "still_held"
        for index in (402, 513, 514)
    )
    for index in (476, 482, 496):
        row = final_rows[index - 1]
        assert row["review_decision"] == "confirmed_retained_candidate"
        assert (SCHOLAR_ID, row["author_pub_id"]) in attribution.confirmed
    for index in (116, 292, 328, 359, 394, 406, 424, 426, 446, 451):
        row = final_rows[index - 1]
        assert row["review_decision"] == "confirmed_retained_candidate"
        assert row["decision_origin"] == "typed_context_policy_review"
        assert (SCHOLAR_ID, row["author_pub_id"]) in attribution.confirmed
    for index in (425, 525):
        row = final_rows[index - 1]
        assert row["review_decision"] == "confirmed_author_context_hold"
        assert (SCHOLAR_ID, row["author_pub_id"]) not in attribution.confirmed
    for index, retained in ((430, "13"), (540, "202")):
        row = final_rows[index - 1]
        assert row["review_decision"] == "confirmed_excluded_candidate"
        assert row["related_row"] == retained
        assert (
            exclusions[(SCHOLAR_ID, row["author_pub_id"])].reason_code
            == "corrupted_duplicate"
        )
    assert final_rows[502]["review_decision"] == "still_held"
    assert (SCHOLAR_ID, final_rows[502]["author_pub_id"]) not in attribution.confirmed
    assert (SCHOLAR_ID, final_rows[502]["author_pub_id"]) not in exclusions
    assert all(
        ";" not in decision.evidence_url for decision in attribution.confirmed.values()
    )
    assert all(
        ";" not in decision.evidence_url
        for (sid, _), decision in exclusions.items()
        if sid == SCHOLAR_ID
    )
    assert not set(attribution.confirmed).intersection(exclusions)
    assert all(
        row["bibliography_entry"] and row["evidence_url"]
        for row in rows
        if row["decision"] == "confirmed_retained"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda doc: doc.update(schema_version=2), "Unsupported Scholar-attribution"),
        (lambda doc: doc.update(profiles=None), "profile and observation lists"),
        (
            lambda doc: doc.update(confirmed_observations=None),
            "profile and observation lists",
        ),
        (lambda doc: doc.update(profiles=[None]), "profile must be an object"),
        (
            lambda doc: doc["profiles"][0].update(scholar_id=""),
            "invalid identity or capture",
        ),
        (
            lambda doc: doc["profiles"].append(doc["profiles"][0]),
            "invalid identity or capture",
        ),
        (
            lambda doc: doc["profiles"][0].update(capture_sha256="bad"),
            "invalid identity or capture",
        ),
        (
            lambda doc: doc["profiles"][0].update(captured_observations=0),
            "invalid identity or capture",
        ),
        (
            lambda doc: doc["profiles"][0].update(evidence_url="http://bad"),
            "dated HTTPS evidence",
        ),
        (
            lambda doc: doc["profiles"][0].update(verified_at="bad"),
            "dated HTTPS evidence",
        ),
        (
            lambda doc: doc.update(confirmed_observations=[None]),
            "observation must be an object",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(scholar_id="other"),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(
                author_pub_id="other:paper"
            ),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(
                author_pub_id=f"{SCHOLAR_ID}:"
            ),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"].append(
                doc["confirmed_observations"][0]
            ),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(bibliography_entry=0),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"][-1].update(evidence_basis=""),
            "invalid identity",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(
                evidence_url="http://bad"
            ),
            "dated HTTPS evidence",
        ),
        (
            lambda doc: doc["confirmed_observations"][0].update(verified_at="bad"),
            "dated HTTPS evidence",
        ),
    ],
)
def test_invalid_attribution_manifest_fails_closed(tmp_path, change, message):
    document = _manifest()
    change(document)
    with pytest.raises(ValueError, match=message):
        load_attribution_reviews(_write(tmp_path, document))


def test_attribution_gate_preserves_raw_rows_and_blocks_unconfirmed_edges(tmp_path):
    person = {
        "person_id": "person-chien-ho",
        "display_name": "Chien Ho",
        "scholar_id": SCHOLAR_ID,
        "scholar_id_source_url": "https://www.cmu.edu/bio/people/faculty/ho.html",
        "scholar_id_verified_at": "2026-10-25",
        "orcid": "",
        "homepage_url": "",
        "notes": "",
    }
    paths = write_registry(
        tmp_path / "registry",
        people=[person],
        memberships=[
            {
                "person_id": "person-chien-ho",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "",
                "source_url": "https://www.cmu.edu/bio/people/faculty/ho.html",
                "verified_at": "2026-10-25",
            }
        ],
    )
    registry = load_registry(*paths)
    ids = [
        f"{SCHOLAR_ID}:u5HHmVD_uO8C",  # CMU bibliography entry 272.
        f"{SCHOLAR_ID}:zLWjf1WUPmwC",  # Exact abstract attribution remains held.
        f"{SCHOLAR_ID}:kWvqk_afx_IC",  # Different Chien-Hsien Ho, exact exclusion.
    ]
    rows = []
    for index, author_pub_id in enumerate(ids):
        rows.append(
            {
                "scholar_id": SCHOLAR_ID,
                "person_id": "person-chien-ho",
                "display_name": "Chien Ho",
                "faculty": "Chien Ho",
                "department_ids": ["ece"],
                "memberships": [
                    {"department_id": "ece", "role": "faculty", "included": True}
                ],
                "author_pub_id": author_pub_id,
                "title": f"Evidence-based spectroscopy paper {index}",
                "authors": "C Ho",
                "year": 2005,
                "venue": "Journal 1, 1-10, 2005",
                "citation": "Journal 1, 1-10, 2005",
                "citation_count": 1,
                "source_url": "",
                "source_record_json": "{}",
                "fetched_at_utc": "2026-10-25T00:00:00+00:00",
                "embedding": [0.0, 1.0],
            }
        )
    tables = build_dataset_tables(
        pandas.DataFrame(rows),
        registry,
        load_exclusions(DEFAULT_EXCLUSIONS_PATH),
        attribution_reviews=load_attribution_reviews(DEFAULT_ATTRIBUTION_REVIEWS_PATH),
    )
    raw = tables["profile_publications"].set_index("author_pub_id")
    assert len(raw) == 3
    assert bool(raw.loc[ids[0], "map_eligible"])
    assert raw.loc[ids[0], "review_attribution_status"] == "confirmed"
    assert raw.loc[ids[0], "review_attribution_evidence_url"].startswith("https://")
    assert not bool(raw.loc[ids[1], "map_eligible"])
    assert raw.loc[ids[1], "review_attribution_status"] == "held"
    assert "attribution_unconfirmed" in raw.loc[ids[1], "map_exclusion_reasons"]
    assert not bool(raw.loc[ids[2], "map_eligible"])
    assert raw.loc[ids[2], "review_exclusion_code"] == "false_profile_attribution"
    assert raw.loc[ids[2], "review_attribution_status"] == ""
    assert "attribution_unconfirmed" not in raw.loc[ids[2], "map_exclusion_reasons"]
    assert len(tables["authorships"]) == 1


def test_publish_rejects_conflicting_reviews(monkeypatch, tmp_path):
    key = (SCHOLAR_ID, f"{SCHOLAR_ID}:paper")
    monkeypatch.setattr(publisher, "load_registry", lambda *args: object())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (pandas.DataFrame({"scholar_id": []}), {}),
    )
    monkeypatch.setattr(publisher, "load_exclusions", lambda path: {key: object()})
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)
    monkeypatch.setattr(
        publisher, "apply_corrections", lambda frame, corrections: (frame, 0)
    )
    monkeypatch.setattr(
        publisher,
        "load_attribution_reviews",
        lambda path: AttributionReviews(
            frozenset({SCHOLAR_ID}),
            {key: ConfirmedAttribution("https://example.test", 1, "2026-10-25")},
        ),
    )
    with pytest.raises(ValueError, match="reviews conflict"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
        )
