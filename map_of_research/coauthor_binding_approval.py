"""Authorize three reviewed McComb work rebindings for two named coauthors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file

PRIOR_REVISION = "5073c24450e2f239ca49f674cbc81287b915045b"
SNAPSHOT_SHA256 = "51ebf5c5b5185ba1ac4d19beb29fccebca9d89c8e4628d8f8c9df2db839028a3"
CURATED_SHA256 = "9509555aac7661f0b155b180789fa1d5b97eb7239d370ca71bc04f8fe8b658ef"
CACHE_SHA256 = "8caef8d4f83afbf582281b2572202a23a820e8ab093e6d423dd9fd853287e56f"
MARTELARO_REVIEW_SHA256 = (
    "375d482cd81218bf1de5b3701f203cab76781fc6597a8f2fc14af42ea38ec8f6"
)
AKINCI_REVIEW_SHA256 = (
    "650d540027c3f8c7113432e7935502e9b97df1de8784de525546509b3a859478"
)
APPROVAL_SHA256 = "865f36b105198a8e425edfa9ff426f44e4eef4b3484149e7e9d8ed88a8adf269"
DEFAULT_APPROVAL_PATH = Path("data/curated/mccomb-coauthor-bindings-5073.json")
DEFAULT_MARTELARO_REVIEW_PATH = Path(
    "registry/reviews/2026-10-30-mccomb-martelaro-curated-coauthorship.md"
)
DEFAULT_AKINCI_REVIEW_PATH = Path(
    "registry/reviews/2026-10-30-mccomb-akinci-curated-coauthorship.md"
)

ObservationKey = tuple[str, str, str, str]
BindingChange = tuple[ObservationKey, str, str, bool, bool]


@dataclass(frozen=True)
class ExactCoauthorBindingApproval:
    manifest_sha256: str
    binding_changes: frozenset[BindingChange]


def _indexed(rows: list[Mapping[str, Any]]) -> dict[ObservationKey, Mapping[str, Any]]:
    result = {
        (
            str(row.get("person_id") or ""),
            str(row.get("scholar_id") or ""),
            str(row.get("author_pub_id") or ""),
            str(row.get("source_type") or ""),
        ): row
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("Coauthor binding approval found duplicate observations")
    return result


def validate_exact_coauthor_bindings(
    approval_path: Path,
    *,
    prior_revision: str,
    source_snapshot_sha256: str,
    curated_works_path: Path,
    mccomb_cache_path: Path,
    martelaro_review_path: Path,
    akinci_review_path: Path,
    prior_profiles: list[Mapping[str, Any]],
    current_profiles: list[Mapping[str, Any]],
) -> ExactCoauthorBindingApproval:
    """Return only exact visible transitions backed by the frozen source content."""

    if (
        prior_revision != PRIOR_REVISION
        or source_snapshot_sha256 != SNAPSHOT_SHA256
        or sha256_file(curated_works_path) != CURATED_SHA256
        or sha256_file(mccomb_cache_path) != CACHE_SHA256
        or sha256_file(martelaro_review_path) != MARTELARO_REVIEW_SHA256
        or sha256_file(akinci_review_path) != AKINCI_REVIEW_SHA256
        or sha256_file(approval_path) != APPROVAL_SHA256
    ):
        raise ValueError("Coauthor binding approval differs from reviewed source")
    document = load_json(approval_path)
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "prior_revision",
            "source_snapshot_sha256",
            "curated_works_sha256",
            "mccomb_cache_sha256",
            "martelaro_review_sha256",
            "akinci_review_sha256",
            "binding_changes",
        }
        or document.get("schema_version") != 1
        or document.get("prior_revision") != PRIOR_REVISION
        or document.get("source_snapshot_sha256") != SNAPSHOT_SHA256
        or document.get("curated_works_sha256") != CURATED_SHA256
        or document.get("mccomb_cache_sha256") != CACHE_SHA256
        or document.get("martelaro_review_sha256") != MARTELARO_REVIEW_SHA256
        or document.get("akinci_review_sha256") != AKINCI_REVIEW_SHA256
        or not isinstance(document.get("binding_changes"), list)
        or len(document["binding_changes"]) != 3
    ):
        raise ValueError("Coauthor binding approval manifest is invalid")
    curated = load_json(curated_works_path)
    if not isinstance(curated, dict) or not isinstance(curated.get("works"), list):
        raise ValueError("Coauthor curated source is invalid")
    curated_targets = {
        (str(row.get("person_id") or ""), str(row.get("doi") or "").lower())
        for row in curated["works"]
    }
    old = _indexed(prior_profiles)
    new = _indexed(current_profiles)
    approved: set[BindingChange] = set()
    reviewed_coauthors: list[str] = []
    for item in document["binding_changes"]:
        if not isinstance(item, dict) or set(item) != {
            "observation",
            "prior_work_id",
            "current_work_id",
            "prior_collaboration_eligible",
            "current_collaboration_eligible",
            "doi",
            "coauthor_person_id",
        }:
            raise ValueError("Coauthor binding entry is invalid")
        identity = tuple(str(item["observation"]).split("/"))
        if (
            len(identity) != 4
            or identity[:2] != ("person-mccomb-03260969", "0P9w_S0AAAAJ")
            or identity[3] != "scholar"
        ):
            raise ValueError("Coauthor binding observation is not McComb Scholar")
        before, after = old.get(identity), new.get(identity)
        if (
            before is None
            or after is None
            or str(before.get("work_id") or "") != item["prior_work_id"]
            or str(after.get("work_id") or "") != item["current_work_id"]
            or item["prior_work_id"] == item["current_work_id"]
            or item["prior_collaboration_eligible"] is not True
            or item["current_collaboration_eligible"] is not True
            or not bool(before.get("map_eligible"))
            or not bool(after.get("map_eligible"))
            or not bool(before.get("collaboration_eligible"))
            or not bool(after.get("collaboration_eligible"))
            or (item["coauthor_person_id"], str(item["doi"]).lower())
            not in curated_targets
        ):
            raise ValueError("Coauthor binding differs from exact reviewed transition")
        approved.add(
            (
                identity,
                item["prior_work_id"],
                item["current_work_id"],
                True,
                True,
            )
        )
        reviewed_coauthors.append(item["coauthor_person_id"])
    if len(approved) != 3 or sorted(reviewed_coauthors) != [
        "person-akinci-3e3b7cc1",
        "person-nikolasmartelaro-f2c38c73",
        "person-nikolasmartelaro-f2c38c73",
    ]:
        raise ValueError("Coauthor binding approval does not cover the exact trio")
    return ExactCoauthorBindingApproval(
        manifest_sha256=APPROVAL_SHA256,
        binding_changes=frozenset(approved),
    )
