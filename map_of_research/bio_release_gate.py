"""Exact source, historical Bio, graph, and topic gates for the Bio release."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .region_continuity import require_revision

# Frozen combined source content at staged W3 commit 01c05970. Its Git commit
# may change when it is restacked onto the Sullivan upload receipt.
BIO_SOURCE_SHA256 = "d08fb3a422df8dee3b0d0f75a29d68fef9b7121229a9cf6d418917e1d4a19550"
BIO29_REVISION = "96fdc335fa0061be30a14698d2a89b9d21454b35"
BIO_SOURCE_ROWS = 51_883
BIO_ROSTER_COUNT = 47
BIO_SCHOLAR_PROFILES = 30
BIO_SCHOLAR_ROWS = 2_688
BIO29_COUNT = 29
BIO29_SCHOLAR_ROWS = 2_106
PENDING_BROAD_ID = "keyword-unclassified"
PENDING_DETAIL_ID = "keyword-unclassified-pending-review"
PENDING_BROAD_LABEL = "Unclassified publications"
PENDING_DETAIL_LABEL = "Pending topic review"
PENDING_FIELDS = {
    "keyword_id": PENDING_BROAD_ID,
    "keyword": PENDING_BROAD_LABEL,
    "keyword_extracted": PENDING_BROAD_LABEL,
    "keyword_label_reviewed": False,
    "detail_keyword_id": PENDING_DETAIL_ID,
    "detail_keyword": PENDING_DETAIL_LABEL,
    "detail_keyword_extracted": PENDING_DETAIL_LABEL,
    "detail_keyword_label_reviewed": False,
}
IDENTITY = ("person_id", "scholar_id", "author_pub_id", "source_type")
REVIEWED_GAINS = {
    "FXqaMl4AAAAJ:UebtZRa9Y70C": (
        "work-4cbb2b69a5fa9112cf7b3a3cbbca683d69d578ae290630ebd7597b7bf3618cb4",
        "10.1016/j.neuron.2017.06.006",
        None,
    ),
    "UZf8gAUAAAAJ:u5HHmVD_uO8C": (
        "work-8e4f8cbdef621e150de90229ae4180faa256885827107e7ae2eadb67880eff41",
        None,
        "pmcid:pmc3970999",
    ),
    "28yZqbcAAAAJ:UeHWp8X0CEIC": (
        "work-4a4c2ad38e9b2eebc2682edf65bb94b55739f97964b069fe95e817bb85c5eaa4",
        "10.1016/j.molcel.2008.06.010",
        None,
    ),
    "6ArMETUAAAAJ:nroGzMJTTpEC": (
        "work-5daf79858885bd71b90e858b7ee9bce367bae980a3ba69d64dd5b7e861d6a967",
        "10.1371/journal.pcbi.1014780",
        None,
    ),
}
POSTER_ID = (
    "person-cjoelmcmanus-f28918ce",
    "28yZqbcAAAAJ",
    "28yZqbcAAAAJ:HKviVsUxM5wC",
    "scholar",
)
POSTER_WORK_ID = "work-d701cf6129386f93f78754f5b2f78e2653904fecbc6428dfb27ca849547cf4e8"
POSTER_SOURCE_SHA256 = (
    "a2d3b3ebb010eea4f246edf2d67e7c4463440499ea15de380aff25486dddd164"
)
MURPHY_ID = "person-robertfmurphy-5930bd3c"
MURPHY_SCHOLAR_ID = "qQLlBH4AAAAJ"
# Each excluded Scholar row duplicates a retained, independently verified work.
# The tuple is (retained citation, retained work ID, exclusion evidence URL).
REVIEWED_DUPLICATE_LOSSES = {
    (MURPHY_ID, MURPHY_SCHOLAR_ID, "qQLlBH4AAAAJ:P7Ujq4OLJYoC", "scholar"): (
        "qQLlBH4AAAAJ:g5m5HwL7SMYC",
        "work-3f9429e5d668edf6eb960934479268ba9b5f5fbae31ec542a1b8bc9bfe4874e2",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4256960/",
    ),
    (MURPHY_ID, MURPHY_SCHOLAR_ID, "qQLlBH4AAAAJ:idthP5jqfYAC", "scholar"): (
        "qQLlBH4AAAAJ:AXPGKjj_ei8C",
        "work-e34bf1b2730e8840322680db92eb671f9c1b97ee1212febf6456e9eab3b294db",
        "https://murphylab.cbd.cmu.edu/publications/156-ahmed2009.pdf",
    ),
    (MURPHY_ID, MURPHY_SCHOLAR_ID, "qQLlBH4AAAAJ:unp9ATQDT5gC", "scholar"): (
        "qQLlBH4AAAAJ:pqnbT2bcN3wC",
        "work-4cfbe6eef8651fa63f73d1c31c8790b9e98fdd56876e14c1d591d29cd2e77f8a",
        "https://psb.stanford.edu/psb-online/proceedings/psb07/kou.pdf",
    ),
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _rows(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dict"):
        return frame.to_dict("records")
    return list(frame)


def _indexed(rows: Any, fields: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    result = {}
    for row in _rows(rows):
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"Duplicate Bio release identity: {key}")
        result[key] = row
    return result


def _by_work_id(rows: Any) -> dict[str, dict]:
    return {key[0]: row for key, row in _indexed(rows, ("work_id",)).items()}


def _mapped_observations(profiles: Any, works: Any, people: set[str]) -> set[tuple]:
    work_by_id = _by_work_id(works)
    return {
        key
        for key, row in _indexed(profiles, IDENTITY).items()
        if key[0] in people
        and key[3] == "scholar"
        and bool(row["map_eligible"])
        and bool(work_by_id.get(str(row["work_id"]), {}).get("map_eligible"))
    }


def _bio_edges(artifact: Mapping[str, Any], bio_ids: set[str]) -> set[tuple[str, str]]:
    edges = set()
    for point in artifact["points"]:
        if not point.get("collaboration_eligible", True):
            continue
        people = sorted(set(point["faculty_ids"]) & bio_ids)
        edges.update(itertools.combinations(people, 2))
    return edges


def bio_scholar_source_changed(
    source_rows: Any,
    prior_profiles: Any,
    registry: Any,
    prior_people: Any,
) -> bool:
    """Compare the union of current and immutable-parent Bio Scholar cohorts."""

    current_bio_ids = {
        item.person_id
        for item in registry.memberships
        if item.included and item.department_id == "bio"
    }
    prior_bio_ids = {
        str(row["person_id"])
        for row in _rows(prior_people)
        if "bio" in _as_list(row["included_department_ids"])
    }
    if not prior_bio_ids:
        raise ValueError("Immutable parent lacks Bio membership baseline")
    bio_ids = current_bio_ids | prior_bio_ids

    def records(rows: Any) -> dict[tuple[str, ...], str]:
        selected = [
            row
            for row in _rows(rows)
            if row.get("person_id") in bio_ids and row.get("source_type") == "scholar"
        ]
        indexed = _indexed(selected, IDENTITY)
        if any(not row.get("source_record_json") for row in indexed.values()):
            raise ValueError("Bio Scholar baseline lacks raw source records")
        return {key: str(row["source_record_json"]) for key, row in indexed.items()}

    return current_bio_ids != prior_bio_ids or records(source_rows) != records(
        prior_profiles
    )


def load_bio_baselines(*, repo_id: str, prior_revision: str, hf_token: str) -> dict:
    """Read only immutable Bio29 and repaired-release source/map files."""

    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    require_revision(prior_revision)

    def hosted(revision: str, name: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id,
                name,
                repo_type="dataset",
                revision=revision,
                token=hf_token or None,
            )
        )

    bio_people = parquet.read_table(
        hosted(BIO29_REVISION, "people/train-00000-of-00001.parquet"),
        columns=["person_id", "included_department_ids", "mapped_work_count"],
    ).to_pylist()
    bio29 = {
        row["person_id"]
        for row in bio_people
        if "bio" in row["included_department_ids"] and row["mapped_work_count"] > 0
    }
    if len(bio29) != BIO29_COUNT:
        raise ValueError(f"Original Bio29 baseline now has {len(bio29)} people")
    return {
        "bio29_ids": bio29,
        "bio29_profile": parquet.read_table(
            hosted(BIO29_REVISION, "profile_publications/train-00000-of-00001.parquet"),
            columns=list(IDENTITY),
        ).to_pylist(),
        "repair_profile": parquet.read_table(
            hosted(prior_revision, "profile_publications/train-00000-of-00001.parquet"),
            columns=[*IDENTITY, "work_id", "map_eligible", "source_record_json"],
        ).to_pylist(),
        "repair_works": parquet.read_table(
            hosted(prior_revision, "works/train-00000-of-00001.parquet"),
            columns=["work_id", "map_eligible", *PENDING_FIELDS],
        ).to_pylist(),
        "repair_artifact": json.loads(
            hosted(prior_revision, "maps/publications.json").read_text(encoding="utf-8")
        ),
    }


def preserve_pending_review_topics(
    candidate_works: Any,
    repair_works: Any,
    repair_manifest: Mapping[str, Any],
) -> Any:
    """Carry any explicit pending labels through a new topic-model refit."""

    prior = _by_work_id(repair_works)
    pending_in_parent = {
        work_id
        for work_id, row in prior.items()
        if row.get("keyword_id") == PENDING_BROAD_ID
        and row.get("detail_keyword_id") == PENDING_DETAIL_ID
        and bool(row["map_eligible"])
    }
    restored = repair_manifest.get("restore_eligible_work_ids")
    if not pending_in_parent:
        if restored not in ([], None):
            raise ValueError("Repair manifest names pending works absent from parent")
        return candidate_works
    if (
        not isinstance(restored, list)
        or len(restored) != len(pending_in_parent)
        or set(restored) != pending_in_parent
    ):
        raise ValueError("Expected the exact mapped pending-review works in parent")
    output = candidate_works.copy()
    current_ids = output["work_id"].astype(str)
    if current_ids.duplicated().any() or set(restored) - set(current_ids):
        raise ValueError("Bio candidate loses a restored pending-review work")
    for work_id in restored:
        baseline = prior[work_id]
        if any(baseline[field] != value for field, value in PENDING_FIELDS.items()):
            raise ValueError(f"Repair baseline topic is not pending: {work_id}")
        index = output.index[current_ids == work_id]
        if len(index) != 1 or not bool(output.at[index[0], "map_eligible"]):
            raise ValueError(f"Bio candidate excludes pending work {work_id}")
        for field, value in PENDING_FIELDS.items():
            output.at[index[0], field] = value
    return output


def validate_bio_release_prewrite(
    *,
    source_frame: Any,
    source_manifest: Mapping[str, Any],
    registry: Any,
    tables: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    continuity_receipt: Mapping[str, Any] | None,
    prior_revision: str,
    baselines: Mapping[str, Any],
    repair_manifest: Mapping[str, Any],
    provisional_holds: Mapping[tuple[str, str], Any],
    approved_person_hold_losses: frozenset[tuple[str, str, str, str]] = frozenset(),
) -> dict[str, Any]:
    """Fail before a Hub write if the whole Bio cohort or old links drift."""

    require_revision(prior_revision)
    if (
        source_manifest.get("snapshot_sha256") != BIO_SOURCE_SHA256
        or len(source_frame) != BIO_SOURCE_ROWS
    ):
        raise ValueError(
            "Bio release source differs from the frozen 51,883-row snapshot"
        )
    bio_ids = {
        item.person_id
        for item in registry.memberships
        if item.included and item.department_id == "bio"
    }
    bio_people = [person for person in registry.people if person.person_id in bio_ids]
    if (
        len(bio_ids) != BIO_ROSTER_COUNT
        or sum(bool(p.scholar_id) for p in bio_people) != BIO_SCHOLAR_PROFILES
    ):
        raise ValueError("Bio release roster or verified Scholar IDs changed")
    source_rows = _rows(source_frame)
    bio_scholar_rows = [
        row
        for row in source_rows
        if row["person_id"] in bio_ids and row["source_type"] == "scholar"
    ]
    if (
        len(bio_scholar_rows) != BIO_SCHOLAR_ROWS
        or len({row["scholar_id"] for row in bio_scholar_rows}) != BIO_SCHOLAR_PROFILES
    ):
        raise ValueError("Bio Scholar source does not contain all 2,688 captured rows")
    source_identities = _indexed(source_rows, IDENTITY)
    candidate_profiles = _indexed(tables["profile_publications"], IDENTITY)
    missing = set(source_identities) - set(candidate_profiles)
    if missing != approved_person_hold_losses:
        raise ValueError(f"Candidate loses {len(missing)} released source rows")
    original_bio29 = set(baselines["bio29_ids"])
    old_bio29_rows = {
        key
        for key in _indexed(baselines["bio29_profile"], IDENTITY)
        if key[0] in original_bio29 and key[3] == "scholar"
    }
    if len(original_bio29) != BIO29_COUNT or len(old_bio29_rows) != BIO29_SCHOLAR_ROWS:
        raise ValueError("Original Bio29 baseline differs from 2,106 Scholar rows")
    if old_bio29_rows - set(candidate_profiles):
        raise ValueError("Bio release loses an original Bio29 Scholar observation")
    prior_mapped = _mapped_observations(
        baselines["repair_profile"], baselines["repair_works"], original_bio29
    )
    new_mapped = _mapped_observations(
        tables["profile_publications"], tables["works"], original_bio29
    )
    mapped_losses = prior_mapped - new_mapped
    unexpected_losses = sorted(
        mapped_losses - {POSTER_ID} - REVIEWED_DUPLICATE_LOSSES.keys()
    )
    if unexpected_losses:
        raise ValueError(
            "Bio release loses "
            f"{len(unexpected_losses)} mapped repaired-baseline observation(s): "
            f"{unexpected_losses[:3]}"
        )
    current_works = _by_work_id(tables["works"])
    old_profiles = _indexed(baselines["repair_profile"], IDENTITY)
    reviewed_duplicate_losses = mapped_losses & REVIEWED_DUPLICATE_LOSSES.keys()
    if reviewed_duplicate_losses != REVIEWED_DUPLICATE_LOSSES.keys():
        missing = sorted(REVIEWED_DUPLICATE_LOSSES.keys() - reviewed_duplicate_losses)
        raise ValueError(
            "Bio release lacks exact reviewed Murphy duplicate losses: "
            f"{len(missing)} missing; {missing[:3]}"
        )
    for identity in reviewed_duplicate_losses:
        canonical_id, canonical_work_id, evidence_url = REVIEWED_DUPLICATE_LOSSES[
            identity
        ]
        duplicate = candidate_profiles.get(identity)
        canonical = candidate_profiles.get(
            (MURPHY_ID, MURPHY_SCHOLAR_ID, canonical_id, "scholar")
        )
        canonical_work = current_works.get(canonical_work_id)
        prior = old_profiles.get(identity)
        if (
            duplicate is None
            or prior is None
            or duplicate.get("source_record_json") != prior.get("source_record_json")
            or bool(duplicate.get("map_eligible"))
            or bool(duplicate.get("collaboration_eligible"))
            or duplicate.get("review_exclusion_code") != "corrupted_duplicate"
            or duplicate.get("review_evidence_url") != evidence_url
            or duplicate.get("review_verified_at") != "2026-10-25"
            or "reviewed_corrupted_duplicate"
            not in _as_list(duplicate.get("map_exclusion_reasons"))
            or canonical is None
            or canonical.get("work_id") != canonical_work_id
            or not bool(canonical.get("map_eligible"))
            or not bool(canonical.get("collaboration_eligible"))
            or canonical_work is None
            or not bool(canonical_work.get("map_eligible"))
            or not bool(canonical_work.get("collaboration_eligible"))
        ):
            raise ValueError(f"Bio reviewed duplicate differs: {identity}")
    if len({key[0] for key in new_mapped}) < BIO29_COUNT:
        raise ValueError("Bio release loses a mapped Bio29 researcher")
    for author_pub_id, (work_id, doi, native_id) in REVIEWED_GAINS.items():
        found = [
            row
            for key, row in candidate_profiles.items()
            if key[2] == author_pub_id and key[3] == "scholar"
        ]
        work = current_works.get(work_id)
        if (
            len(found) != 1
            or found[0]["work_id"] != work_id
            or not bool(found[0]["map_eligible"])
            or work is None
            or not bool(work["map_eligible"])
            or (doi is not None and str(work["doi"]).lower() != doi)
            or (
                native_id is not None
                and native_id not in _as_list(work["source_native_ids"])
            )
        ):
            raise ValueError(
                f"Reviewed Bio work is missing or unmapped: {author_pub_id}"
            )
    old_poster = _indexed(baselines["repair_profile"], IDENTITY).get(POSTER_ID)
    poster = candidate_profiles.get(POSTER_ID)
    prior_poster_work = (
        _by_work_id(baselines["repair_works"]).get(str(old_poster["work_id"]))
        if old_poster is not None
        else None
    )
    poster_hold = provisional_holds.get((POSTER_ID[1], POSTER_ID[2]))
    poster_source_hash = hashlib.sha256(
        str(poster.get("source_record_json", "") if poster else "").encode()
    ).hexdigest()
    if (
        old_poster is None
        or poster is None
        or prior_poster_work is None
        or poster["work_id"] != POSTER_WORK_ID
        or old_poster["work_id"] != POSTER_WORK_ID
        or poster_source_hash != POSTER_SOURCE_SHA256
        or hashlib.sha256(str(old_poster["source_record_json"]).encode()).hexdigest()
        != POSTER_SOURCE_SHA256
    ):
        raise ValueError("McManus poster must match the exact source-bound record")
    if (
        poster_hold is None
        or poster_hold.scholar_id != POSTER_ID[1]
        or poster_hold.author_pub_id != POSTER_ID[2]
        or not poster_hold.evidence_url
        or bool(poster["map_eligible"])
        or poster.get("review_attribution_status") != "held"
        or poster.get("review_attribution_evidence_url") != poster_hold.evidence_url
        or "attribution_provisional_hold"
        not in _as_list(poster.get("map_exclusion_reasons"))
        or POSTER_WORK_ID
        in {point["work_id"] for point in candidate_artifact["points"]}
    ):
        raise ValueError("McManus poster requires an exact provisional hold")
    if (
        continuity_receipt is None
        or continuity_receipt.get("prior_revision") != prior_revision
        or continuity_receipt.get("unexplained_losses")
        or continuity_receipt.get("unexplained_gains")
        or not {"/".join(identity) for identity in reviewed_duplicate_losses}.issubset(
            set(continuity_receipt.get("reviewed_losses") or [])
        )
        or set(continuity_receipt.get("approved_person_hold_losses") or [])
        != {"/".join(key) for key in approved_person_hold_losses}
    ):
        raise ValueError("Bio release lacks a clean exact-observation region diff")
    approved_losses = continuity_receipt.get("approved_one_time_losses") or []
    expected_poster_loss = ["/".join(POSTER_ID)] if POSTER_ID in mapped_losses else []
    if approved_losses != expected_poster_loss or (
        expected_poster_loss
        and not continuity_receipt.get("poster_loss_approval_sha256")
    ):
        raise ValueError("Bio release needs the exact one-time poster-loss approval")
    candidate_bio = {
        person["person_id"]
        for person in candidate_artifact["catalogs"]["faculty"]
        if any(
            membership["department_id"] == "bio" for membership in person["memberships"]
        )
    }
    if candidate_bio != bio_ids:
        raise ValueError("Bio map catalog omits included faculty")
    old_edges = _bio_edges(baselines["repair_artifact"], bio_ids)
    new_edges = _bio_edges(candidate_artifact, bio_ids)
    if old_edges - new_edges:
        raise ValueError(f"Bio graph loses {len(old_edges - new_edges)} faculty edges")

    restored = repair_manifest.get("restore_eligible_work_ids", [])
    pending_in_parent = {
        row["work_id"]
        for row in baselines["repair_works"]
        if row.get("keyword_id") == PENDING_BROAD_ID
        and row.get("detail_keyword_id") == PENDING_DETAIL_ID
        and bool(row["map_eligible"])
    }
    if pending_in_parent and (
        not isinstance(restored, list)
        or len(restored) != len(pending_in_parent)
        or set(restored) != pending_in_parent
    ):
        raise ValueError("Pending-topic list differs from the immutable parent")
    if not pending_in_parent and restored:
        raise ValueError("Repair manifest names pending works absent from parent")
    point_by_id = {point["work_id"]: point for point in candidate_artifact["points"]}
    for work_id in restored:
        work = current_works.get(work_id)
        point = point_by_id.get(work_id)
        if (
            work is None
            or not bool(work["map_eligible"])
            or any(work[field] != value for field, value in PENDING_FIELDS.items())
            or point is None
            or point["keyword_ids"] != [PENDING_BROAD_ID, PENDING_DETAIL_ID]
        ):
            raise ValueError(f"Pending-review topic changed in Bio release: {work_id}")
    if restored:
        keywords = {item["keyword_id"]: item for item in candidate_artifact["keywords"]}
        if (
            keywords.get(PENDING_BROAD_ID, {}).get("label") != PENDING_BROAD_LABEL
            or keywords.get(PENDING_DETAIL_ID, {}).get("label") != PENDING_DETAIL_LABEL
            or keywords[PENDING_DETAIL_ID].get("parent_keyword_id") != PENDING_BROAD_ID
        ):
            raise ValueError("Bio map loses its pending topic catalog entries")
    return {
        "status": "pass",
        "gate": "bio-full-source-graph-and-pending-topics-v1",
        "prior_revision": prior_revision,
        "source_snapshot_sha256": BIO_SOURCE_SHA256,
        "raw_rows": len(source_frame),
        "bio_people": len(bio_ids),
        "bio_verified_profiles": BIO_SCHOLAR_PROFILES,
        "bio_scholar_rows": len(bio_scholar_rows),
        "bio29_observations_preserved": len(old_bio29_rows),
        "bio29_mapped_observations_preserved": len(prior_mapped - mapped_losses),
        "held_poster_mapped_loss": POSTER_ID in mapped_losses,
        "reviewed_duplicate_mapped_losses": len(reviewed_duplicate_losses),
        "bio_edges_preserved": len(old_edges),
        "pending_topic_works_preserved": len(restored),
    }
