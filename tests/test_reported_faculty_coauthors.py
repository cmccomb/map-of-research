"""Fail-closed checks for one-sided, reviewed faculty coauthor reports."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pandas
import pytest

from map_of_research.io import sha256_file
from map_of_research.reported_faculty_coauthors import (
    apply_reported_faculty_coauthors,
    require_reported_source_sync,
)

REPORTER = "person-reporter"
TARGET = "person-target"
SCHOLAR = "Reporter123"
AUTHOR_PUB_ID = f"{SCHOLAR}:publication-1"
OBSERVATION_ID = (
    "observation-" + hashlib.sha256(f"{SCHOLAR}\0{AUTHOR_PUB_ID}".encode()).hexdigest()
)
WORK_ID = "work-" + "a" * 64


def evidence() -> dict[str, str]:
    return {
        "person_id": TARGET,
        "source_person_id": REPORTER,
        "source_scholar_id": SCHOLAR,
        "source_author_pub_id": AUTHOR_PUB_ID,
        "source_observation_id": OBSERVATION_ID,
        "source_work_id": WORK_ID,
        "source_title": "A verified joint paper",
        "source_authors": "A Reporter, B Target",
        "source_url": "https://example.test/publisher-paper",
        "verified_at": "2026-10-30",
    }


def sources():
    raw = {"title": "A verified joint paper", "author": "A Reporter, B Target"}
    works = pandas.DataFrame(
        [{"work_id": WORK_ID, "map_eligible": True, "collaboration_eligible": True}]
    )
    profiles = pandas.DataFrame(
        [
            {
                "scholar_id": SCHOLAR,
                "author_pub_id": AUTHOR_PUB_ID,
                "observation_id": OBSERVATION_ID,
                "person_id": REPORTER,
                "work_id": WORK_ID,
                "source_type": "scholar",
                "map_eligible": True,
                "collaboration_eligible": True,
                "source_record_json": json.dumps(raw),
            }
        ]
    )
    authorships = pandas.DataFrame([{"work_id": WORK_ID, "person_id": REPORTER}])
    registry = SimpleNamespace(
        people_by_id={
            REPORTER: SimpleNamespace(scholar_id=SCHOLAR, display_name="A Reporter"),
            TARGET: SimpleNamespace(scholar_id="", display_name="B Target"),
        },
        memberships=[
            SimpleNamespace(person_id=REPORTER, included=True),
            SimpleNamespace(person_id=TARGET, included=True),
        ],
    )
    return works, profiles, authorships, registry


def run(tmp_path, assertions, sources_override=None, held_person_ids=frozenset()):
    path = tmp_path / "reported-faculty-coauthors.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assertions": assertions,
            }
        )
    )
    works, profiles, authorships, registry = sources_override or sources()
    result = apply_reported_faculty_coauthors(
        works,
        profiles,
        authorships,
        registry,
        source_path=path,
        source_sha256=sha256_file(path),
        held_person_ids=held_person_ids,
    )
    return result, (works, profiles, authorships)


def run_document(tmp_path, document, sources_override=None):
    path = tmp_path / "reported-faculty-coauthors.json"
    path.write_text(json.dumps(document))
    works, profiles, authorships, registry = sources_override or sources()
    return apply_reported_faculty_coauthors(
        works,
        profiles,
        authorships,
        registry,
        source_path=path,
        source_sha256=sha256_file(path),
    )


def test_reported_coauthor_adds_edge_provenance_without_authorship(tmp_path) -> None:
    (by_work, receipt), (works, profiles, authorships) = run(tmp_path, [evidence()])
    assert len(receipt) == 1
    assert by_work[WORK_ID] == [
        {
            "person_id": TARGET,
            "source_person_id": REPORTER,
            "source_observation_id": OBSERVATION_ID,
            "source_url": "https://example.test/publisher-paper",
            "verified_at": "2026-10-30",
        }
    ]
    assert "reported_faculty_coauthors" not in works
    assert profiles.iloc[0].person_id == REPORTER
    assert authorships.to_dict("records") == [
        {"work_id": WORK_ID, "person_id": REPORTER}
    ]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_author_pub_id", f"{SCHOLAR}:wrong", "identity"),
        ("source_observation_id", "observation-" + "0" * 64, "identity"),
        ("source_work_id", "work-" + "b" * 64, "unused"),
        ("source_authors", "A Reporter", "source binding"),
        ("person_id", "person-unknown", "identity"),
        ("source_url", "http://example.test/paper", "identity"),
    ],
)
def test_reported_coauthor_rejects_changed_or_unusable_assertion(
    tmp_path, field, value, error
) -> None:
    item = evidence()
    item[field] = value
    with pytest.raises(ValueError, match=error):
        run(tmp_path, [item])


def test_reported_coauthor_rejects_excluded_reporter_or_work(tmp_path) -> None:
    for frame, column in ((0, "map_eligible"), (1, "collaboration_eligible")):
        data = list(sources())
        data[frame] = data[frame].copy()
        data[frame].loc[0, column] = False
        with pytest.raises(ValueError, match="source binding"):
            run(tmp_path, [evidence()], tuple(data))


def test_reported_coauthor_rejects_duplicate_and_existing_authorship(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicated"):
        run(tmp_path, [evidence(), deepcopy(evidence())])
    data = list(sources())
    data[2] = pandas.concat(
        [data[2], pandas.DataFrame([{"work_id": WORK_ID, "person_id": TARGET}])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="superseded"):
        run(tmp_path, [evidence()], tuple(data))


def test_reported_coauthor_cannot_bypass_target_observation_decision(tmp_path) -> None:
    data = list(sources())
    target_row = data[1].iloc[0].to_dict()
    target_row.update(
        scholar_id="Target123",
        author_pub_id="Target123:publication-1",
        person_id=TARGET,
        map_eligible=False,
        collaboration_eligible=False,
    )
    data[1] = pandas.concat(
        [data[1], pandas.DataFrame([target_row])], ignore_index=True
    )
    with pytest.raises(ValueError, match="target already has an observation"):
        run(tmp_path, [evidence()], tuple(data))


def test_reported_coauthor_requires_included_target(tmp_path) -> None:
    data = list(sources())
    data[3].memberships[1].included = False
    with pytest.raises(ValueError, match="identity"):
        run(tmp_path, [evidence()], tuple(data))


def test_reported_coauthor_cannot_bypass_person_publication_hold(tmp_path) -> None:
    with pytest.raises(ValueError, match="publication hold"):
        run(tmp_path, [evidence()], held_person_ids=frozenset({TARGET}))


def test_reported_coauthor_requires_frozen_curated_checksum(tmp_path) -> None:
    path = tmp_path / "reported-faculty-coauthors.json"
    path.write_text(json.dumps({"schema_version": 1, "assertions": []}))
    with pytest.raises(ValueError, match="snapshot manifest"):
        apply_reported_faculty_coauthors(
            *sources(), source_path=path, source_sha256="0" * 64
        )


def test_reported_coauthor_requires_matching_frozen_source(tmp_path) -> None:
    producer = tmp_path / "producer.json"
    frozen = tmp_path / "frozen.json"
    assert require_reported_source_sync(producer, frozen) is False
    producer.write_text('{"schema_version":1,"assertions":[]}')
    with pytest.raises(ValueError, match="frozen source differ"):
        require_reported_source_sync(producer, frozen)
    frozen.write_bytes(producer.read_bytes())
    assert require_reported_source_sync(producer, frozen) is True
    frozen.write_text('{"schema_version":1,"assertions":[{}]}')
    with pytest.raises(ValueError, match="frozen source differ"):
        require_reported_source_sync(producer, frozen)


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ([], "source is invalid"),
        ({"schema_version": 2, "assertions": []}, "source is invalid"),
        ({"schema_version": 1, "assertions": {}}, "must be a list"),
    ],
)
def test_reported_coauthor_rejects_malformed_manifest(tmp_path, document, error):
    with pytest.raises(ValueError, match=error):
        run_document(tmp_path, document)


def test_reported_coauthor_allows_explicit_empty_manifest(tmp_path) -> None:
    assert run_document(tmp_path, {"schema_version": 1, "assertions": []}) == ({}, [])


def test_reported_coauthor_rejects_ambiguous_source_rows(tmp_path) -> None:
    for table_index, error in ((0, "work identity"), (1, "source observation")):
        data = list(sources())
        data[table_index] = pandas.concat(
            [data[table_index], data[table_index]], ignore_index=True
        )
        with pytest.raises(ValueError, match=error):
            run(tmp_path, [evidence()], tuple(data))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda item: item.pop("source_title"), "invalid fields"),
        (lambda item: item.update(source_title=""), "exact text"),
        (lambda item: item.update(verified_at="20261030"), "review date"),
        (lambda item: item.update(verified_at="2026-13-30"), "review date"),
        (
            lambda item: item.update(source_authors="A Reporter, B Alias"),
            "source binding",
        ),
    ],
)
def test_reported_coauthor_rejects_unreviewed_evidence_shape(
    tmp_path, change, error
) -> None:
    item = evidence()
    change(item)
    with pytest.raises(ValueError, match=error):
        run(tmp_path, [item])


def test_reported_coauthor_rejects_invalid_raw_observation(tmp_path) -> None:
    data = list(sources())
    data[1].loc[0, "source_record_json"] = "{not json"
    with pytest.raises(ValueError, match="raw source is invalid"):
        run(tmp_path, [evidence()], tuple(data))


def test_reported_coauthor_requires_target_surname_in_raw_byline(tmp_path) -> None:
    item = evidence()
    item["source_authors"] = "A Reporter, B Elsewhere"
    data = list(sources())
    data[1].loc[0, "source_record_json"] = json.dumps(
        {"title": item["source_title"], "author": item["source_authors"]}
    )
    with pytest.raises(ValueError, match="absent from source byline"):
        run(tmp_path, [item], tuple(data))
