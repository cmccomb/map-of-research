"""Exact, source-bound review of Bio work rekeys on the frozen 38f parent.

The 11 still-visible Scholar observations retain their collaboration eligibility.
Seven acquire reviewed DOI identities; four join independently verified native
source records. This approval covers only those exact work bindings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .io import load_json, sha256_file

DEFAULT_APPROVAL_PATH = Path("data/curated/bio-exact-binding-approval-38f.json")
PRIOR_REVISION = "38f119d71692cdb9112210217b532a7dd7c1633a"
SOURCE_COMMIT = "1fabb9b4213171d214c8c1f42cff2f8f2919a83f"
SOURCE_SHA256 = "d08fb3a422df8dee3b0d0f75a29d68fef9b7121229a9cf6d418917e1d4a19550"
APPROVAL_SHA256 = "b9f35512683570167c2cbe379f6f9eaf81363242c218be94e2ea395f6537a0ea"
BINDINGS_SHA256 = "7e4058f40cc5da9f4ac29f4c1cfbb1bc544fa20542632b56310a32e40d5b2471"
DOCUMENT_FIELDS = {
    "schema_version",
    "reviewed_at",
    "prior_revision",
    "source_snapshot_commit",
    "source_snapshot_sha256",
    "doi_aliases_sha256",
    "curated_works_sha256",
    "profile_releases_sha256",
    "provisional_holds_sha256",
    "poster_loss_approval_sha256",
    "approved_binding_changes",
}
ENTRY_FIELDS = {
    "person_id",
    "scholar_id",
    "author_pub_id",
    "source_type",
    "prior_work_id",
    "current_work_id",
    "prior_collaboration_eligible",
    "current_collaboration_eligible",
    "evidence_kind",
    "evidence_id",
    "evidence_url",
}
ObservationKey = tuple[str, str, str, str]
BindingChange = tuple[ObservationKey, str, str, bool, bool]


@dataclass(frozen=True)
class ExactBioBindingApproval:
    binding_changes: frozenset[BindingChange]
    manifest_sha256: str


def binding_digest(bindings: frozenset[BindingChange]) -> str:
    lines = sorted(
        "/".join(identity)
        + f"|{prior_work_id}|{current_work_id}|"
        + str(prior_collaboration_eligible).lower()
        + "|"
        + str(current_collaboration_eligible).lower()
        for (
            identity,
            prior_work_id,
            current_work_id,
            prior_collaboration_eligible,
            current_collaboration_eligible,
        ) in bindings
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load_bio_binding_approval(
    *,
    approval_path: Path,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
    doi_aliases_path: Path,
    curated_works_path: Path,
    profile_releases_path: Path,
    provisional_holds_path: Path,
    poster_loss_approval_path: Path | None,
) -> ExactBioBindingApproval:
    """Require the reviewed manifest and its exact source and evidence files."""

    if (
        prior_revision != PRIOR_REVISION
        or source_snapshot_commit != SOURCE_COMMIT
        or source_snapshot_sha256 != SOURCE_SHA256
    ):
        raise ValueError("Bio binding approval is bound to a different release")
    if poster_loss_approval_path is None:
        raise ValueError("Bio binding approval requires the exact poster approval")
    if sha256_file(approval_path) != APPROVAL_SHA256:
        raise ValueError("Bio binding approval manifest changed")
    document = load_json(approval_path)
    if not isinstance(document, dict) or set(document) != DOCUMENT_FIELDS:
        raise ValueError("Bio binding approval schema changed")
    if (
        document["schema_version"] != 1
        or document["reviewed_at"] != "2026-10-25"
        or document["prior_revision"] != PRIOR_REVISION
        or document["source_snapshot_commit"] != SOURCE_COMMIT
        or document["source_snapshot_sha256"] != SOURCE_SHA256
        or any(
            document[field] != sha256_file(path)
            for field, path in (
                ("doi_aliases_sha256", doi_aliases_path),
                ("curated_works_sha256", curated_works_path),
                ("profile_releases_sha256", profile_releases_path),
                ("provisional_holds_sha256", provisional_holds_path),
                ("poster_loss_approval_sha256", poster_loss_approval_path),
            )
        )
    ):
        raise ValueError("Bio binding approval evidence or release binding changed")

    entries = document["approved_binding_changes"]
    if (
        not isinstance(entries, list)
        or len(entries) != 11
        or any(
            not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS
            for entry in entries
        )
    ):
        raise ValueError("Bio binding approval decision set changed")
    aliases = load_json(doi_aliases_path)["aliases"]
    native_works = load_json(curated_works_path)["works"]
    bindings: set[BindingChange] = set()
    identities: list[ObservationKey] = []
    evidence_counts = {"doi_alias": 0, "native_source": 0}
    for entry in entries:
        identity = tuple(
            entry[key]
            for key in ("person_id", "scholar_id", "author_pub_id", "source_type")
        )
        kind = entry["evidence_kind"]
        evidence_id = entry["evidence_id"]
        evidence_url = entry["evidence_url"]
        if (
            any(not isinstance(value, str) or not value for value in identity)
            or identity[3] != "scholar"
            or not identity[2].startswith(f"{identity[1]}:")
            or kind not in evidence_counts
            or not isinstance(evidence_id, str)
            or not isinstance(evidence_url, str)
            or not evidence_url.startswith("https://")
            or not isinstance(entry["prior_work_id"], str)
            or not isinstance(entry["current_work_id"], str)
            or entry["prior_work_id"] == entry["current_work_id"]
            or entry["prior_collaboration_eligible"] is not True
            or entry["current_collaboration_eligible"] is not True
        ):
            raise ValueError(f"Bio binding approval row differs: {identity}")
        if kind == "doi_alias":
            evidence = [
                alias
                for alias in aliases
                if alias.get("scholar_id") == identity[1]
                and alias.get("author_pub_id") == identity[2]
                and alias.get("published_doi") == evidence_id
                and evidence_url in alias.get("evidence_urls", [])
            ]
        else:
            evidence = [
                work
                for work in native_works
                if work.get("person_id") == identity[0]
                and work.get("source_native_id") == evidence_id
                and evidence_url
                in ([work.get("source_url"), *work.get("source_verification_urls", [])])
                and work.get("source_verified_at") == "2026-10-25"
            ]
        if len(evidence) != 1:
            raise ValueError(f"Bio binding lacks exact reviewed evidence: {identity}")
        evidence_counts[kind] += 1
        identities.append(identity)
        bindings.add(
            (
                identity,
                entry["prior_work_id"],
                entry["current_work_id"],
                True,
                True,
            )
        )
    if (
        identities != sorted(set(identities))
        or len(bindings) != 11
        or evidence_counts != {"doi_alias": 7, "native_source": 4}
    ):
        raise ValueError("Bio binding approval identities changed")
    result = ExactBioBindingApproval(frozenset(bindings), APPROVAL_SHA256)
    if binding_digest(result.binding_changes) != BINDINGS_SHA256:
        raise ValueError("Bio binding approval exact decision set changed")
    return result
