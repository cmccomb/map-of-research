"""Bio publication cannot lose an old researcher link or pending topic label."""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from types import SimpleNamespace

import pandas
import pytest

import map_of_research.bio_release_gate as gate
from map_of_research.provisional_holds import ProvisionalHold

REVIEWED_DUPLICATES = dict(gate.REVIEWED_DUPLICATE_LOSSES)


def _candidate(monkeypatch):
    monkeypatch.setattr(gate, "REVIEWED_DUPLICATE_LOSSES", {})
    monkeypatch.setattr(gate, "BIO_SOURCE_ROWS", 5)
    monkeypatch.setattr(gate, "BIO_ROSTER_COUNT", 4)
    monkeypatch.setattr(gate, "BIO_SCHOLAR_PROFILES", 4)
    monkeypatch.setattr(gate, "BIO_SCHOLAR_ROWS", 5)
    monkeypatch.setattr(gate, "BIO29_COUNT", 4)
    monkeypatch.setattr(gate, "BIO29_SCHOLAR_ROWS", 5)
    monkeypatch.setattr(
        gate,
        "POSTER_SOURCE_SHA256",
        hashlib.sha256(b"poster-raw").hexdigest(),
    )
    people = ["person-0", "person-1", gate.POSTER_ID[0], "person-3"]
    raw = []
    works = []
    for index, (author_pub_id, (work_id, doi, native)) in enumerate(
        gate.REVIEWED_GAINS.items()
    ):
        raw.append(
            {
                "person_id": people[index],
                "scholar_id": author_pub_id.split(":")[0],
                "author_pub_id": author_pub_id,
                "source_type": "scholar",
                "source_record_json": f"raw-{index}",
                "work_id": work_id,
                "map_eligible": True,
            }
        )
        works.append(
            {
                "work_id": work_id,
                "map_eligible": True,
                "doi": doi or "",
                "source_native_ids": [native] if native else [],
            }
        )
    poster = dict(
        zip(
            gate.IDENTITY,
            gate.POSTER_ID,
            strict=True,
        )
    )
    poster.update(
        source_record_json="poster-raw",
        work_id=gate.POSTER_WORK_ID,
        map_eligible=False,
        review_attribution_status="held",
        review_attribution_evidence_url="https://scholar.google.com/poster",
        map_exclusion_reasons=["attribution_provisional_hold"],
    )
    raw.append(poster)
    works.append({"work_id": gate.POSTER_WORK_ID, "map_eligible": False})
    restored = [f"pending-{index:03d}" for index in range(15)]
    for work_id in restored:
        works.append(
            {
                "work_id": work_id,
                "map_eligible": True,
                **gate.PENDING_FIELDS,
            }
        )
    candidate_points = [
        {
            "work_id": item["work_id"],
            "faculty_ids": [],
            "keyword_ids": [gate.PENDING_BROAD_ID, gate.PENDING_DETAIL_ID],
        }
        for item in works
        if item["work_id"] in restored
    ]
    candidate_points.append(
        {
            "work_id": "collaboration",
            "faculty_ids": people[:2],
            "keyword_ids": ["ordinary", "ordinary-detail"],
        }
    )
    artifact = {
        "catalogs": {
            "faculty": [
                {
                    "person_id": person_id,
                    "memberships": [{"department_id": "bio"}],
                }
                for person_id in people
            ]
        },
        "points": candidate_points,
        "keywords": [
            {"keyword_id": gate.PENDING_BROAD_ID, "label": gate.PENDING_BROAD_LABEL},
            {
                "keyword_id": gate.PENDING_DETAIL_ID,
                "label": gate.PENDING_DETAIL_LABEL,
                "parent_keyword_id": gate.PENDING_BROAD_ID,
            },
        ],
    }
    registry = SimpleNamespace(
        memberships=[
            SimpleNamespace(person_id=person_id, included=True, department_id="bio")
            for person_id in people
        ],
        people=[
            SimpleNamespace(person_id=p, scholar_id=f"scholar-{i}")
            for i, p in enumerate(people)
        ],
    )
    return {
        "source_frame": pandas.DataFrame(raw),
        "source_manifest": {"snapshot_sha256": gate.BIO_SOURCE_SHA256},
        "registry": registry,
        "tables": {
            "profile_publications": pandas.DataFrame(raw),
            "works": pandas.DataFrame(works),
        },
        "candidate_artifact": artifact,
        "continuity_receipt": {
            "prior_revision": "a" * 40,
            "unexplained_losses": [],
            "unexplained_gains": [],
        },
        "prior_revision": "a" * 40,
        "baselines": {
            "bio29_ids": set(people),
            "bio29_profile": deepcopy(raw),
            "repair_profile": deepcopy(raw),
            "repair_works": deepcopy(works),
            "repair_artifact": {"points": [deepcopy(candidate_points[-1])]},
        },
        "repair_manifest": {"restore_eligible_work_ids": restored},
        "provisional_holds": {
            (gate.POSTER_ID[1], gate.POSTER_ID[2]): ProvisionalHold(
                scholar_id=gate.POSTER_ID[1],
                author_pub_id=gate.POSTER_ID[2],
                capture_row_index=1,
                reason="Poster attribution requires primary-source review.",
                evidence_url="https://scholar.google.com/poster",
                reviewed_at="2026-10-25",
            )
        },
    }


def _candidate_with_murphy_duplicates(monkeypatch, identities=None):
    candidate = _candidate(monkeypatch)
    reviewed = {
        identity: details
        for identity, details in REVIEWED_DUPLICATES.items()
        if identities is None or identity in identities
    }
    monkeypatch.setattr(gate, "REVIEWED_DUPLICATE_LOSSES", reviewed)
    row_count = 5 + 2 * len(reviewed)
    monkeypatch.setattr(gate, "BIO_SOURCE_ROWS", row_count)
    monkeypatch.setattr(gate, "BIO_ROSTER_COUNT", 5)
    monkeypatch.setattr(gate, "BIO_SCHOLAR_PROFILES", 5)
    monkeypatch.setattr(gate, "BIO_SCHOLAR_ROWS", row_count)
    monkeypatch.setattr(gate, "BIO29_COUNT", 5)
    monkeypatch.setattr(gate, "BIO29_SCHOLAR_ROWS", row_count)
    duplicate_rows = []
    canonical_rows = []
    excluded_rows = []
    work_rows = []
    for identity, (canonical_id, canonical_work_id, evidence_url) in reviewed.items():
        duplicate = dict(zip(gate.IDENTITY, identity, strict=True))
        duplicate.update(
            source_record_json=f"murphy-duplicate-raw-{identity[2]}",
            work_id=f"work-murphy-duplicate-{identity[2]}",
            map_eligible=True,
            collaboration_eligible=True,
        )
        canonical = {
            **duplicate,
            "author_pub_id": canonical_id,
            "source_record_json": f"murphy-canonical-raw-{canonical_id}",
            "work_id": canonical_work_id,
        }
        excluded = {
            **duplicate,
            "map_eligible": False,
            "collaboration_eligible": False,
            "map_exclusion_reasons": ["reviewed_corrupted_duplicate"],
            "review_exclusion_code": "corrupted_duplicate",
            "review_evidence_url": evidence_url,
            "review_verified_at": "2026-10-25",
        }
        duplicate_rows.append(duplicate)
        canonical_rows.append(canonical)
        excluded_rows.append(excluded)
        work_rows.extend(
            [
                {"work_id": duplicate["work_id"], "map_eligible": False},
                {
                    "work_id": canonical_work_id,
                    "map_eligible": True,
                    "collaboration_eligible": True,
                },
            ]
        )
    candidate["source_frame"] = pandas.concat(
        [candidate["source_frame"], pandas.DataFrame(duplicate_rows + canonical_rows)],
        ignore_index=True,
    )
    candidate["baselines"]["bio29_profile"].extend(
        deepcopy(duplicate_rows + canonical_rows)
    )
    candidate["baselines"]["repair_profile"].extend(
        deepcopy(duplicate_rows + canonical_rows)
    )
    candidate["tables"]["profile_publications"] = pandas.concat(
        [
            candidate["tables"]["profile_publications"],
            pandas.DataFrame(excluded_rows + canonical_rows),
        ],
        ignore_index=True,
    )
    candidate["baselines"]["repair_works"].extend(
        [{**row, "map_eligible": True} for row in work_rows]
    )
    candidate["tables"]["works"] = pandas.concat(
        [candidate["tables"]["works"], pandas.DataFrame(work_rows)],
        ignore_index=True,
    )
    candidate["registry"].memberships.append(
        SimpleNamespace(person_id=gate.MURPHY_ID, included=True, department_id="bio")
    )
    candidate["registry"].people.append(
        SimpleNamespace(person_id=gate.MURPHY_ID, scholar_id=gate.MURPHY_SCHOLAR_ID)
    )
    candidate["baselines"]["bio29_ids"].add(gate.MURPHY_ID)
    candidate["candidate_artifact"]["catalogs"]["faculty"].append(
        {"person_id": gate.MURPHY_ID, "memberships": [{"department_id": "bio"}]}
    )
    candidate["candidate_artifact"]["points"].extend(
        {"work_id": work_id, "faculty_ids": [gate.MURPHY_ID]}
        for _, work_id, _ in reviewed.values()
    )
    candidate["continuity_receipt"]["reviewed_losses"] = [
        "/".join(identity) for identity in reviewed
    ]
    return candidate


@pytest.mark.parametrize("identity", list(REVIEWED_DUPLICATES))
def test_exact_reviewed_murphy_duplicate_loss_keeps_canonical_work(
    monkeypatch, identity
):
    candidate = _candidate_with_murphy_duplicates(monkeypatch, [identity])
    receipt = gate.validate_bio_release_prewrite(**candidate)
    assert receipt["reviewed_duplicate_mapped_losses"] == 1

    duplicate_index = len(candidate["tables"]["profile_publications"]) - 2
    candidate["tables"]["profile_publications"].at[
        duplicate_index, "review_evidence_url"
    ] = "https://example.org/unreviewed"
    with pytest.raises(ValueError, match="reviewed duplicate differs"):
        gate.validate_bio_release_prewrite(**candidate)


def test_murphy_duplicate_requires_continuity_review_and_retained_work(monkeypatch):
    identity = next(iter(REVIEWED_DUPLICATES))
    candidate = _candidate_with_murphy_duplicates(monkeypatch, [identity])
    candidate["continuity_receipt"]["reviewed_losses"] = []
    with pytest.raises(ValueError, match="clean exact-observation region diff"):
        gate.validate_bio_release_prewrite(**candidate)
    candidate["continuity_receipt"]["reviewed_losses"] = ["/".join(identity)]

    canonical_work_id = REVIEWED_DUPLICATES[identity][1]
    index = candidate["tables"]["works"].index[
        candidate["tables"]["works"].work_id == canonical_work_id
    ][0]
    candidate["tables"]["works"].at[index, "map_eligible"] = False
    with pytest.raises(ValueError, match="mapped repaired-baseline observation"):
        gate.validate_bio_release_prewrite(**candidate)


def test_all_three_reviewed_duplicate_losses_are_required(monkeypatch):
    candidate = _candidate_with_murphy_duplicates(monkeypatch)
    receipt = gate.validate_bio_release_prewrite(**candidate)
    assert receipt["reviewed_duplicate_mapped_losses"] == 3

    identity = next(iter(REVIEWED_DUPLICATES))
    index = candidate["tables"]["profile_publications"].index[
        candidate["tables"]["profile_publications"].author_pub_id == identity[2]
    ][0]
    candidate["tables"]["profile_publications"].at[index, "map_eligible"] = True
    candidate["tables"]["profile_publications"].at[index, "collaboration_eligible"] = (
        True
    )
    duplicate_work_id = candidate["tables"]["profile_publications"].at[index, "work_id"]
    work_index = candidate["tables"]["works"].index[
        candidate["tables"]["works"].work_id == duplicate_work_id
    ][0]
    candidate["tables"]["works"].at[work_index, "map_eligible"] = True
    candidate["continuity_receipt"]["reviewed_losses"].remove("/".join(identity))
    with pytest.raises(
        ValueError, match="lacks exact reviewed Murphy duplicate losses"
    ):
        gate.validate_bio_release_prewrite(**candidate)


def test_bio_prewrite_gate_preserves_complete_source_graph_and_topics(
    monkeypatch,
) -> None:
    candidate = _candidate(monkeypatch)
    receipt = gate.validate_bio_release_prewrite(**candidate)
    assert receipt["status"] == "pass"
    assert receipt["bio_edges_preserved"] == 1
    assert receipt["pending_topic_works_preserved"] == 15

    candidate["candidate_artifact"]["points"][-1]["faculty_ids"] = ["person-0"]
    with pytest.raises(ValueError, match="graph loses"):
        gate.validate_bio_release_prewrite(**candidate)


def test_pending_topic_assignment_survives_refit_and_gate_rejects_drift(
    monkeypatch,
) -> None:
    candidate = _candidate(monkeypatch)
    candidate_works = candidate["tables"]["works"].copy()
    pending_index = candidate_works.index[
        candidate_works.work_id
        == candidate["repair_manifest"]["restore_eligible_work_ids"][0]
    ][0]
    candidate_works.at[pending_index, "keyword_id"] = "keyword-indoor-location"
    candidate["tables"]["works"] = candidate_works
    with pytest.raises(ValueError, match="Pending-review topic changed"):
        gate.validate_bio_release_prewrite(**candidate)

    restored = gate.preserve_pending_review_topics(
        candidate_works,
        candidate["baselines"]["repair_works"],
        candidate["repair_manifest"],
    )
    candidate["tables"]["works"] = restored
    assert gate.validate_bio_release_prewrite(**candidate)["status"] == "pass"


def test_bb88_visible_poster_requires_exact_source_bound_hold(monkeypatch) -> None:
    candidate = _candidate(monkeypatch)
    poster_work = next(
        row
        for row in candidate["baselines"]["repair_works"]
        if row["work_id"] == gate.POSTER_WORK_ID
    )
    poster_work["map_eligible"] = True
    candidate["baselines"]["repair_profile"][-1]["map_eligible"] = True
    with pytest.raises(ValueError, match="one-time poster-loss approval"):
        gate.validate_bio_release_prewrite(**candidate)
    candidate["continuity_receipt"]["approved_one_time_losses"] = [
        "/".join(gate.POSTER_ID)
    ]
    candidate["continuity_receipt"]["poster_loss_approval_sha256"] = "f" * 64
    receipt = gate.validate_bio_release_prewrite(**candidate)
    assert receipt["held_poster_mapped_loss"] is True

    candidate["continuity_receipt"]["approved_one_time_losses"] = []
    with pytest.raises(ValueError, match="one-time poster-loss approval"):
        gate.validate_bio_release_prewrite(**candidate)
    candidate["continuity_receipt"]["approved_one_time_losses"] = [
        "/".join(gate.POSTER_ID)
    ]

    candidate["tables"]["profile_publications"].at[4, "map_exclusion_reasons"] = (
        "other_reason"
    )
    with pytest.raises(ValueError, match="exact provisional hold"):
        gate.validate_bio_release_prewrite(**candidate)

    candidate["tables"]["profile_publications"].at[4, "map_exclusion_reasons"] = [
        "attribution_provisional_hold"
    ]
    candidate["tables"]["profile_publications"].at[
        4, "review_attribution_evidence_url"
    ] = "https://scholar.google.com/other"
    with pytest.raises(ValueError, match="exact provisional hold"):
        gate.validate_bio_release_prewrite(**candidate)

    candidate["tables"]["profile_publications"].at[
        4, "review_attribution_evidence_url"
    ] = "https://scholar.google.com/poster"
    candidate["provisional_holds"] = {}
    with pytest.raises(ValueError, match="exact provisional hold"):
        gate.validate_bio_release_prewrite(**candidate)


def test_bb88_without_pending_topics_needs_no_repair_manifest(monkeypatch) -> None:
    candidate = _candidate(monkeypatch)
    for row in candidate["baselines"]["repair_works"]:
        row.pop("keyword_id", None)
        row.pop("detail_keyword_id", None)
    candidate["repair_manifest"] = {"restore_eligible_work_ids": []}
    assert (
        gate.validate_bio_release_prewrite(**candidate)["pending_topic_works_preserved"]
        == 0
    )
    assert (
        gate.preserve_pending_review_topics(
            candidate["tables"]["works"],
            candidate["baselines"]["repair_works"],
            candidate["repair_manifest"],
        )
        is candidate["tables"]["works"]
    )
    candidate["repair_manifest"]["restore_eligible_work_ids"] = ["unexpected"]
    with pytest.raises(ValueError, match="absent from parent"):
        gate.preserve_pending_review_topics(
            candidate["tables"]["works"],
            candidate["baselines"]["repair_works"],
            candidate["repair_manifest"],
        )
    with pytest.raises(ValueError, match="absent from parent"):
        gate.validate_bio_release_prewrite(**candidate)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("source_hash", "frozen"),
        ("source_rows", "frozen"),
        ("roster", "roster"),
        ("scholar_ids", "roster"),
        ("scholar_rows", "captured rows"),
        ("candidate_source", "released source rows"),
        ("bio29_people", "Bio29 baseline"),
        ("bio29_rows", "Bio29 baseline"),
        ("bio29_missing", "Bio29 Scholar observation"),
        ("mapped_loss", "mapped repaired-baseline"),
        ("mapped_person", "mapped Bio29 researcher"),
        ("reviewed_work", "Reviewed Bio work"),
        ("poster", "McManus poster"),
        ("continuity", "region diff"),
        ("catalog", "catalog omits"),
        ("repair_manifest", "immutable parent"),
        ("pending_point", "Pending-review topic changed"),
        ("pending_catalog", "pending topic catalog"),
    ],
)
def test_bio_gate_fails_closed_on_cohort_or_graph_drift(
    monkeypatch, change, message
) -> None:
    candidate = _candidate(monkeypatch)
    if change == "source_hash":
        candidate["source_manifest"]["snapshot_sha256"] = "wrong"
    elif change == "source_rows":
        candidate["source_frame"] = candidate["source_frame"].iloc[:-1]
    elif change == "roster":
        candidate["registry"].memberships.pop()
    elif change == "scholar_ids":
        candidate["registry"].people[0].scholar_id = ""
    elif change == "scholar_rows":
        candidate["source_frame"].loc[0, "source_type"] = "curated_verified"
    elif change == "candidate_source":
        candidate["tables"]["profile_publications"] = candidate["tables"][
            "profile_publications"
        ].iloc[:-1]
    elif change == "bio29_people":
        candidate["baselines"]["bio29_ids"].remove("person-0")
    elif change == "bio29_rows":
        candidate["baselines"]["bio29_profile"].pop()
    elif change == "bio29_missing":
        candidate["baselines"]["bio29_profile"].append(
            {
                **candidate["baselines"]["bio29_profile"][0],
                "author_pub_id": "other:absent",
            }
        )
        monkeypatch.setattr(gate, "BIO29_SCHOLAR_ROWS", 6)
    elif change == "mapped_loss":
        candidate["tables"]["profile_publications"].loc[0, "map_eligible"] = False
    elif change == "mapped_person":
        candidate["baselines"]["repair_profile"][0]["map_eligible"] = False
        candidate["tables"]["profile_publications"].loc[0, "map_eligible"] = False
    elif change == "reviewed_work":
        candidate["tables"]["works"].loc[0, "doi"] = "wrong"
    elif change == "poster":
        candidate["tables"]["profile_publications"].loc[4, "source_record_json"] = (
            "changed"
        )
    elif change == "continuity":
        candidate["continuity_receipt"] = None
    elif change == "catalog":
        candidate["candidate_artifact"]["catalogs"]["faculty"].pop()
    elif change == "repair_manifest":
        candidate["repair_manifest"]["restore_eligible_work_ids"].pop()
    elif change == "pending_point":
        candidate["candidate_artifact"]["points"][0]["keyword_ids"] = ["wrong"]
    elif change == "pending_catalog":
        candidate["candidate_artifact"]["keywords"][0]["label"] = "wrong"
    with pytest.raises(ValueError, match=message):
        gate.validate_bio_release_prewrite(**candidate)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("short_manifest", "exact mapped pending-review"),
        ("missing_candidate", "loses a restored"),
        ("missing_baseline", "exact mapped pending-review"),
        ("baseline_topic", "baseline topic is not pending"),
        ("candidate_excluded", "candidate excludes"),
    ],
)
def test_pending_topic_preservation_rejects_unreviewed_input(
    monkeypatch, change, message
) -> None:
    candidate = _candidate(monkeypatch)
    works = candidate["tables"]["works"]
    baseline = candidate["baselines"]["repair_works"]
    manifest = candidate["repair_manifest"]
    pending_id = manifest["restore_eligible_work_ids"][0]
    if change == "short_manifest":
        manifest["restore_eligible_work_ids"].pop()
    elif change == "missing_candidate":
        works = works[works.work_id != pending_id]
    elif change == "missing_baseline":
        baseline = [row for row in baseline if row["work_id"] != pending_id]
    elif change == "baseline_topic":
        next(row for row in baseline if row["work_id"] == pending_id)["keyword"] = (
            "wrong"
        )
    elif change == "candidate_excluded":
        works.loc[works.work_id == pending_id, "map_eligible"] = False
    with pytest.raises(ValueError, match=message):
        gate.preserve_pending_review_topics(works, baseline, manifest)


def test_bio_baseline_loader_pins_two_revisions(monkeypatch, tmp_path) -> None:
    candidate = _candidate(monkeypatch)
    paths = {}
    by_revision = {
        gate.BIO29_REVISION: {
            "people/train-00000-of-00001.parquet": [
                {
                    "person_id": person_id,
                    "included_department_ids": ["bio"],
                    "mapped_work_count": 1,
                }
                for person_id in candidate["baselines"]["bio29_ids"]
            ],
            "profile_publications/train-00000-of-00001.parquet": candidate["baselines"][
                "bio29_profile"
            ],
        },
        "a" * 40: {
            "profile_publications/train-00000-of-00001.parquet": candidate["baselines"][
                "repair_profile"
            ],
            "works/train-00000-of-00001.parquet": candidate["baselines"][
                "repair_works"
            ],
            "maps/publications.json": candidate["baselines"]["repair_artifact"],
        },
    }
    for revision, files in by_revision.items():
        for filename, rows in files.items():
            path = tmp_path / revision / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if filename.endswith(".parquet"):
                pandas.DataFrame(rows).to_parquet(path, index=False)
            else:
                import json

                path.write_text(json.dumps(rows))
            paths[revision, filename] = path
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(paths[kwargs["revision"], filename])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    loaded = gate.load_bio_baselines(
        repo_id="test/dataset", prior_revision="a" * 40, hf_token="token"
    )
    assert loaded["bio29_ids"] == candidate["baselines"]["bio29_ids"]
    assert len(loaded["repair_works"]) == len(candidate["baselines"]["repair_works"])
    assert {item[2]["revision"] for item in calls} == {gate.BIO29_REVISION, "a" * 40}
    assert all(item[2]["token"] == "token" for item in calls)

    calls.clear()
    gate.load_bio_baselines(
        repo_id="test/dataset", prior_revision="a" * 40, hf_token=""
    )
    assert all(item[2]["token"] is None for item in calls)

    monkeypatch.setattr(gate, "BIO29_COUNT", 5)
    with pytest.raises(ValueError, match="baseline now has"):
        gate.load_bio_baselines(
            repo_id="test/dataset", prior_revision="a" * 40, hf_token="token"
        )


def test_bio_helpers_reject_duplicate_identity_and_ignore_noncollaboration() -> None:
    with pytest.raises(ValueError, match="Duplicate Bio release identity"):
        gate._indexed([{"work_id": "same"}, {"work_id": "same"}], ("work_id",))
    assert gate._as_list(None) == []
    assert gate._as_list(("a", "b")) == ["a", "b"]
    assert gate._as_list(pandas.Series(["a", "b"])) == ["a", "b"]
    assert gate._as_list("a") == ["a"]
    assert (
        gate._bio_edges(
            {"points": [{"faculty_ids": ["a", "b"], "collaboration_eligible": False}]},
            {"a", "b"},
        )
        == set()
    )


def test_bio_source_diff_ignores_other_colleges_and_detects_exact_raw_change() -> None:
    registry = SimpleNamespace(
        memberships=[
            SimpleNamespace(person_id="bio-person", included=True, department_id="bio"),
            SimpleNamespace(
                person_id="chem-sullivan", included=True, department_id="chem"
            ),
        ]
    )
    prior = [
        {
            "person_id": "bio-person",
            "scholar_id": "bio-scholar",
            "author_pub_id": "bio-scholar:one",
            "source_type": "scholar",
            "source_record_json": '{"title":"one"}',
        }
    ]
    prior_people = [{"person_id": "bio-person", "included_department_ids": ["bio"]}]
    assert not gate.bio_scholar_source_changed(prior, prior, registry, prior_people)
    unrelated = [
        *prior,
        {
            "person_id": "chem-sullivan",
            "scholar_id": "sullivan-scholar",
            "author_pub_id": "sullivan-scholar:one",
            "source_type": "scholar",
            "source_record_json": '{"title":"Sullivan chemistry"}',
        },
    ]
    assert not gate.bio_scholar_source_changed(unrelated, prior, registry, prior_people)
    changed = [dict(prior[0], source_record_json='{"title":"changed"}')]
    assert gate.bio_scholar_source_changed(changed, prior, registry, prior_people)
    assert gate.bio_scholar_source_changed(
        prior + [dict(prior[0], author_pub_id="bio-scholar:two")],
        prior,
        registry,
        prior_people,
    )
    with pytest.raises(ValueError, match="lacks raw source records"):
        gate.bio_scholar_source_changed(
            prior, [dict(prior[0], source_record_json="")], registry, prior_people
        )

    removed_bio = SimpleNamespace(
        memberships=[
            SimpleNamespace(person_id="bio-person", included=False, department_id="bio")
        ]
    )
    assert gate.bio_scholar_source_changed(changed, prior, removed_bio, prior_people)
    assert gate.bio_scholar_source_changed(prior, prior, removed_bio, prior_people)
    with pytest.raises(ValueError, match="lacks Bio membership baseline"):
        gate.bio_scholar_source_changed(prior, prior, registry, [])
