"""The next full rebuild must retain the published 19-row Chemistry review."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from map_of_research.chem_carryforward import (
    PRIOR_REVISION,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    validate_chem_carryforward,
)

FIXTURE = Path("tests/fixtures/chem_dcb_carryforward_rows.json")
FROZEN_BIO_PROFILES = Path("tests/fixtures/bio_approved_profiles.json")
FROZEN_BIO_EXCLUSIONS = Path("tests/fixtures/bio_approved_exclusions.json")
FROZEN_BIO_HOLDS = Path("tests/fixtures/bio_approved_provisional_holds.json")
FROZEN_CHEM_CONTEXTS = Path("tests/fixtures/chem_approved_contexts.json")


def _candidate():
    prior = json.loads(FIXTURE.read_text())
    current = copy.deepcopy(prior)
    kwargs = {
        "prior_revision": PRIOR_REVISION,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_rows": prior["source"],
        "prior_profiles": prior["profiles"],
        "prior_works": prior["works"],
        "prior_authorships": prior["authorships"],
        "current_profiles": current["profiles"],
        "current_works": current["works"],
        "current_authorships": current["authorships"],
        "releases_path": FROZEN_BIO_PROFILES,
        "exclusions_path": FROZEN_BIO_EXCLUSIONS,
        "holds_path": FROZEN_BIO_HOLDS,
        "contexts_path": FROZEN_CHEM_CONTEXTS,
    }
    return current, kwargs


def test_chem_review_survives_full_rebuild_with_exact_authorship_partition():
    _, kwargs = _candidate()
    receipt = validate_chem_carryforward(**kwargs)
    assert receipt["review_sha256"] == (
        "e2b888a3293b9515491770cf13ccfc0dbdd75e31bc4fda671698f5acbb41ec9b"
    )
    assert (receipt["included"], receipt["held"], receipt["excluded"]) == (
        10,
        5,
        4,
    )
    assert receipt["authorships"] == 10


@pytest.mark.parametrize("change", ["title", "authorship", "attribution", "hidden"])
def test_chem_review_rejects_changed_metadata_or_graph_identity(change):
    current, kwargs = _candidate()
    donahue = "ILambBMAAAAJ:eflP2zaiRacC"
    if change == "title":
        profile = next(
            row for row in current["profiles"] if row["author_pub_id"] == donahue
        )
        work = next(
            row for row in current["works"] if row["work_id"] == profile["work_id"]
        )
        profile["title"] = work["title"] = "Incorrect presentation title"
    elif change == "authorship":
        authored = next(
            row for row in current["authorships"] if row["author_pub_id"] == donahue
        )
        authored["person_id"] = "person-wrong"
    elif change == "attribution":
        profile = next(
            row for row in current["profiles"] if row["author_pub_id"] == donahue
        )
        work = next(
            row for row in current["works"] if row["work_id"] == profile["work_id"]
        )
        work["person_ids"] = ["person-wrong"]
        work["faculty"] = ["Wrong Person"]
        work["memberships"] = []
        work["department_ids"] = []
        work["school_ids"] = []
    else:
        hidden = next(row for row in current["profiles"] if not row["map_eligible"])
        current["authorships"].append(
            {
                **current["authorships"][0],
                "observation_id": hidden["observation_id"],
                "work_id": hidden["work_id"],
                "person_id": hidden["person_id"],
                "scholar_id": hidden["scholar_id"],
                "author_pub_id": hidden["author_pub_id"],
            }
        )
    with pytest.raises(ValueError, match="Chemistry carry-forward"):
        validate_chem_carryforward(**kwargs)


def test_chem_review_rejects_other_source_or_parent():
    _, kwargs = _candidate()
    for key, value in (
        ("prior_revision", "0" * 40),
        ("source_commit", "0" * 40),
        ("source_sha256", "0" * 64),
    ):
        changed = kwargs | {key: value}
        with pytest.raises(ValueError, match="parent, source, or review"):
            validate_chem_carryforward(**changed)
