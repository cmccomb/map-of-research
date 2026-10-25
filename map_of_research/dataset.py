"""Loss-aware dataset normalization for people, works, and relationships."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .attribution import AttributionReviews
from .contexts import ReviewedContext
from .doi_aliases import ReviewedDoiAlias
from .exclusions import ReviewedExclusion
from .provisional_holds import ProvisionalHold
from .quality import QUALITY_ASSESSMENT_VERSION, assess_publication
from .registry import Registry

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def normalize_title(value: str) -> str:
    """Normalize exact title variants conservatively for work identity."""

    return NON_ALPHANUMERIC.sub(" ", value.casefold()).strip()


def _clean_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.casefold() in {"nan", "<na>"} else text


def _year(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doi(*values: Any) -> str:
    for value in values:
        match = DOI_PATTERN.search(_clean_text(value))
        if match:
            return match.group(0).rstrip(".,;)").casefold()
    return ""


def _hash_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _observation_identity(row: Any) -> str:
    return _hash_id("observation", f"{row.scholar_id}\0{row.author_pub_id}")


def _work_identity(
    row: Any,
    *,
    curated_title_year_dois: Mapping[tuple[str, int], set[str]] | None = None,
    curated_title_year_native_ids: Mapping[tuple[str, int], set[str]] | None = None,
    context: ReviewedContext | None = None,
) -> tuple[str, str, str]:
    if context is not None and context.publication_type == "patent":
        return (
            _hash_id("work", f"patent-document:{context.document_id}"),
            "patent_document",
            "",
        )
    native_id = _clean_text(getattr(row, "source_native_id", ""))
    if native_id and not _doi(getattr(row, "source_doi", "")):
        return _hash_id("work", f"native:{native_id}"), "source_native_id", ""
    doi = _doi(
        getattr(row, "source_doi", ""),
        row.source_url,
        row.citation,
        row.source_record_json,
    )
    if doi:
        return _hash_id("work", f"doi:{doi}"), "doi", doi
    normalized_title = normalize_title(str(row.title))
    year = _year(row.year)
    if normalized_title and year is not None:
        candidate_dois = (curated_title_year_dois or {}).get(
            (normalized_title, year), set()
        )
        candidate_native_ids = (curated_title_year_native_ids or {}).get(
            (normalized_title, year), set()
        )
        if len(candidate_dois) == 1 and not candidate_native_ids:
            curated_doi = next(iter(candidate_dois))
            return (
                _hash_id("work", f"doi:{curated_doi}"),
                "curated_title_year",
                curated_doi,
            )
        if not candidate_dois and len(candidate_native_ids) == 1:
            curated_native_id = next(iter(candidate_native_ids))
            return (
                _hash_id("work", f"native:{curated_native_id}"),
                "curated_native_title_year",
                "",
            )
        value = f"title-year:{normalized_title}\0{year}"
        return _hash_id("work", value), "normalized_title_year", ""
    value = f"profile-publication:{row.scholar_id}\0{row.author_pub_id}"
    return _hash_id("work", value), "profile_publication", ""


def _best_text(values: list[Any]) -> str:
    cleaned = [_clean_text(value) for value in values]
    nonempty = [value for value in cleaned if value]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    return min(
        counts,
        key=lambda value: (-counts[value], -len(value), value.casefold()),
    )


def _source_values(records: list[Any], field: str) -> list[Any]:
    """Use Scholar for a populated field, then verified records for gaps."""

    scholar_values = [
        getattr(record, field)
        for record in records
        if record.source_type == "scholar" and _clean_text(getattr(record, field))
    ]
    return scholar_values or [getattr(record, field) for record in records]


def _variants(values: list[Any]) -> list[str]:
    return sorted(
        {_clean_text(value) for value in values if _clean_text(value)},
        key=lambda value: (value.casefold(), value),
    )


def _mode_year(values: list[Any]) -> int | None:
    years = [_year(value) for value in values]
    present = [value for value in years if value is not None]
    if not present:
        return None
    counts = Counter(present)
    return min(counts, key=lambda value: (-counts[value], value))


def _normalize_memberships(value: Any) -> list[dict[str, Any]]:
    memberships = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            raise ValueError("Observation membership must be an object")
        included = item.get("included", False)
        if isinstance(included, str):
            included = included.casefold() == "true"
        memberships.append(
            {
                "department_id": _clean_text(item.get("department_id")),
                "role": _clean_text(item.get("role")),
                "included": bool(included),
                "legacy_label": _clean_text(item.get("legacy_label")),
                "source_url": _clean_text(item.get("source_url")),
                "verified_at": _clean_text(item.get("verified_at")),
            }
        )
    return memberships


def build_dataset_tables(
    observations: Any,
    registry: Registry,
    exclusions: Mapping[tuple[str, str], ReviewedExclusion] | None = None,
    doi_aliases: Mapping[tuple[str, str], ReviewedDoiAlias] | None = None,
    attribution_reviews: AttributionReviews | None = None,
    contexts: Mapping[tuple[str, str], ReviewedContext] | None = None,
    provisional_holds: Mapping[tuple[str, str], ProvisionalHold] | None = None,
) -> dict[str, Any]:
    """Build retained observations plus canonical people/work relationship tables."""

    import pandas

    reviewed_aliases = doi_aliases or {}
    profile_publications = observations.copy()
    if "source_type" not in profile_publications:
        profile_publications["source_type"] = "scholar"
    for column, default in (
        ("source_doi", ""),
        ("source_native_id", ""),
        ("source_verified_at", ""),
        ("source_verification_urls", []),
        ("external_coauthors", []),
    ):
        if column not in profile_publications:
            profile_publications[column] = [
                default for _ in range(len(profile_publications))
            ]
    curated_title_year_dois: dict[tuple[str, int], set[str]] = {}
    curated_title_year_native_ids: dict[tuple[str, int], set[str]] = {}
    for row in profile_publications.itertuples(index=False):
        if row.source_type != "curated_verified":
            continue
        doi = _doi(row.source_doi)
        title = normalize_title(str(row.title))
        year = _year(row.year)
        if doi and title and year is not None:
            curated_title_year_dois.setdefault((title, year), set()).add(doi)
        native_id = _clean_text(row.source_native_id)
        if native_id and title and year is not None:
            curated_title_year_native_ids.setdefault((title, year), set()).add(
                native_id
            )
    observation_ids: list[str] = []
    work_ids: list[str] = []
    match_methods: list[str] = []
    dois: list[str] = []
    map_eligibility: list[bool] = []
    exclusion_reasons: list[list[str]] = []
    review_codes: list[str] = []
    review_reasons: list[str] = []
    review_evidence_urls: list[str] = []
    review_dates: list[str] = []
    attribution_statuses: list[str] = []
    attribution_evidence_urls: list[str] = []
    publication_types: list[str] = []
    document_ids: list[str] = []
    context_reasons: list[str] = []
    context_evidence_urls: list[str] = []
    context_dates: list[str] = []
    context_display_contributors: list[str] = []
    collaboration_eligibility: list[bool] = []
    unmapped_work_reasons: dict[str, set[str]] = {}
    known_person_titles = {
        normalize_title(person.display_name) for person in registry.people
    }
    for position, row in enumerate(profile_publications.itertuples(index=False)):
        key = (str(row.scholar_id), str(row.author_pub_id))
        context = (contexts or {}).get(key)
        observation_ids.append(_observation_identity(row))
        alias = reviewed_aliases.get(key)
        if alias is not None and row.source_type == "scholar":
            source_doi = _doi(row.source_doi)
            if source_doi and source_doi != alias.doi:
                raise ValueError("Reviewed DOI alias conflicts with a source DOI")
            doi = alias.doi
            work_id = _hash_id("work", f"doi:{doi}")
            method = "reviewed_version_alias"
            index = profile_publications.index[position]
            profile_publications.at[index, "source_verified_at"] = max(
                _clean_text(row.source_verified_at), alias.verified_at
            )
            profile_publications.at[index, "source_verification_urls"] = sorted(
                {*_as_list(row.source_verification_urls), *alias.evidence_urls}
            )
        else:
            work_id, method, doi = _work_identity(
                row,
                curated_title_year_dois=curated_title_year_dois,
                curated_title_year_native_ids=curated_title_year_native_ids,
                context=context,
            )
        work_ids.append(work_id)
        match_methods.append(method)
        dois.append(doi)
        quality = assess_publication(
            title=str(row.title),
            year=_year(row.year),
            doi=doi,
            source_url=_clean_text(row.source_url),
        )
        reasons = list(quality.exclusion_reasons)
        if normalize_title(str(row.title)) in known_person_titles:
            reasons.append("person_or_citation_index")
        review = (exclusions or {}).get(key)
        if review is not None:
            reasons.append(f"reviewed_{review.reason_code}")
        attribution = (
            attribution_reviews.confirmed.get(key) if attribution_reviews else None
        )
        provisional_hold = (provisional_holds or {}).get(key)
        held_profile = (
            attribution_reviews is not None
            and row.source_type == "scholar"
            and str(row.scholar_id) in attribution_reviews.held_profiles
        )
        if held_profile and attribution is None and review is None:
            reasons.append("attribution_unconfirmed")
        if provisional_hold is not None:
            reasons.append("attribution_provisional_hold")
        attribution_statuses.append(
            "confirmed"
            if attribution
            else "held"
            if (held_profile and not review) or provisional_hold is not None
            else ""
        )
        attribution_evidence_urls.append(
            attribution.evidence_url
            if attribution
            else provisional_hold.evidence_url
            if provisional_hold
            else ""
        )
        if context is not None:
            context_exclusion = ""
            if context.publication_type == "withdrawn_article":
                context_exclusion = "reviewed_withdrawn_article"
            elif not context.map_eligible:
                context_exclusion = "reviewed_context_not_mapped"
            if context_exclusion:
                reasons.append(context_exclusion)
                unmapped_work_reasons.setdefault(work_id, set()).add(context_exclusion)
        review_codes.append(review.reason_code if review else "")
        review_reasons.append(review.reason if review else "")
        review_evidence_urls.append(review.evidence_url if review else "")
        review_dates.append(review.verified_at if review else "")
        publication_types.append(context.publication_type if context else "publication")
        document_ids.append(context.document_id if context else "")
        context_reasons.append(context.reason if context else "")
        context_evidence_urls.append(context.evidence_url if context else "")
        context_dates.append(context.verified_at if context else "")
        context_display_contributors.append(
            context.display_contributors if context else ""
        )
        stable_reasons = list(dict.fromkeys(reasons))
        map_eligibility.append(not stable_reasons)
        exclusion_reasons.append(stable_reasons)
        collaboration_eligibility.append(not stable_reasons and context is None)
    for index, work_id in enumerate(work_ids):
        if work_id in unmapped_work_reasons:
            map_eligibility[index] = False
            collaboration_eligibility[index] = False
            for reason in sorted(unmapped_work_reasons[work_id]):
                if reason not in exclusion_reasons[index]:
                    exclusion_reasons[index].append(reason)
    profile_publications.insert(0, "observation_id", observation_ids)
    profile_publications.insert(1, "work_id", work_ids)
    profile_publications.insert(2, "work_match_method", match_methods)
    profile_publications.insert(3, "doi", dois)
    profile_publications.insert(4, "map_eligible", map_eligibility)
    profile_publications.insert(5, "map_exclusion_reasons", exclusion_reasons)
    profile_publications.insert(6, "review_exclusion_code", review_codes)
    profile_publications.insert(7, "review_exclusion_reason", review_reasons)
    profile_publications.insert(8, "review_evidence_url", review_evidence_urls)
    profile_publications.insert(9, "review_verified_at", review_dates)
    profile_publications.insert(10, "publication_type", publication_types)
    profile_publications.insert(11, "source_document_id", document_ids)
    profile_publications.insert(12, "context_review_reason", context_reasons)
    profile_publications.insert(13, "context_evidence_url", context_evidence_urls)
    profile_publications.insert(14, "context_verified_at", context_dates)
    profile_publications.insert(
        15, "context_display_contributors", context_display_contributors
    )
    profile_publications.insert(16, "collaboration_eligible", collaboration_eligibility)
    profile_publications.insert(
        17, "quality_assessment_version", QUALITY_ASSESSMENT_VERSION
    )
    profile_publications.insert(11, "review_attribution_status", attribution_statuses)
    profile_publications.insert(
        12, "review_attribution_evidence_url", attribution_evidence_urls
    )

    school_titles = registry.school_catalog()
    school_by_department = {
        department.department_id: department.school_id
        for department in registry.departments
    }
    authorship_rows: list[dict[str, Any]] = []
    for row in profile_publications.itertuples(index=False):
        if not row.map_eligible:
            continue
        memberships = _normalize_memberships(row.memberships)
        authorship_rows.append(
            {
                "work_id": row.work_id,
                "person_id": row.person_id,
                "observation_id": row.observation_id,
                "scholar_id": row.scholar_id,
                "author_pub_id": row.author_pub_id,
                "display_name": row.display_name,
                "department_ids": sorted(
                    {
                        membership["department_id"]
                        for membership in memberships
                        if membership["included"]
                    }
                ),
                "school_ids": sorted(
                    {
                        school_by_department[membership["department_id"]]
                        for membership in memberships
                        if membership["included"]
                    }
                ),
                "memberships": memberships,
                "fetched_at_utc": row.fetched_at_utc,
                "source_type": row.source_type,
                "contributor_role": {
                    "patent": "inventor",
                    "edited_book": "editor",
                    "panel_transcript": "participant",
                }.get(row.publication_type, "author"),
            }
        )
    authorships = pandas.DataFrame(
        authorship_rows,
        columns=(
            "work_id",
            "person_id",
            "observation_id",
            "scholar_id",
            "author_pub_id",
            "display_name",
            "department_ids",
            "school_ids",
            "memberships",
            "fetched_at_utc",
            "source_type",
            "contributor_role",
        ),
    )

    department_titles = registry.department_catalog()
    work_rows: list[dict[str, Any]] = []
    for work_id, group in profile_publications.groupby("work_id", sort=True):
        records = list(group.itertuples(index=False))
        eligible_records = [record for record in records if record.map_eligible]
        canonical_records = eligible_records or records
        contextual_records = (
            [
                record
                for record in records
                if "attribution_provisional_hold" not in record.map_exclusion_reasons
            ]
            if eligible_records
            else records
        )
        map_eligible = bool(eligible_records)
        reviewed_types = {
            record.publication_type
            for record in contextual_records
            if record.publication_type != "publication"
        }
        if len(reviewed_types) > 1:
            raise ValueError(f"Conflicting reviewed publication types for {work_id}")
        publication_type = next(iter(reviewed_types), "publication")
        collaboration_eligible = (
            map_eligible
            and publication_type == "publication"
            and all(record.collaboration_eligible for record in eligible_records)
        )
        people: dict[str, str] = {}
        included_memberships: dict[tuple[str, str], dict[str, str]] = {}
        for record in eligible_records:
            people[str(record.person_id)] = str(record.display_name)
            for membership in _normalize_memberships(record.memberships):
                if not membership["included"]:
                    continue
                key = (str(record.person_id), membership["department_id"])
                included_memberships[key] = {
                    "person_id": str(record.person_id),
                    "display_name": str(record.display_name),
                    "department_id": membership["department_id"],
                    "department_title": department_titles[membership["department_id"]],
                    "school_id": school_by_department[membership["department_id"]],
                    "school_title": school_titles[
                        school_by_department[membership["department_id"]]
                    ],
                    "role": membership["role"],
                }
        memberships = sorted(
            included_memberships.values(),
            key=lambda item: (
                item["department_id"],
                item["display_name"].casefold(),
            ),
        )
        department_ids = sorted(
            {membership["department_id"] for membership in memberships}
        )
        school_ids = sorted({membership["school_id"] for membership in memberships})
        years = [record.year for record in canonical_records]
        published_records = []
        for record in canonical_records:
            alias = reviewed_aliases.get(
                (str(record.scholar_id), str(record.author_pub_id))
            )
            if alias is not None and alias.relationship == "version_of_record":
                published_records.append(record)
        fetched = sorted(
            str(record.fetched_at_utc)
            for record in canonical_records
            if record.source_type == "scholar" and str(record.fetched_at_utc)
        )
        external_coauthors: dict[str, dict[str, str]] = {}
        for record in eligible_records:
            if not collaboration_eligible or not record.collaboration_eligible:
                continue
            for coauthor in _as_list(record.external_coauthors):
                if not isinstance(coauthor, Mapping):
                    raise ValueError("External coauthor provenance must be an object")
                person_id = _clean_text(coauthor.get("person_id"))
                details = {
                    "person_id": person_id,
                    "display_name": _clean_text(coauthor.get("display_name")),
                    "source_url": _clean_text(coauthor.get("source_url")),
                }
                if (
                    person_id in external_coauthors
                    and external_coauthors[person_id] != details
                ):
                    raise ValueError("Conflicting external coauthor provenance")
                external_coauthors[person_id] = details
        work_rows.append(
            {
                "work_id": work_id,
                "map_eligible": map_eligible,
                "publication_type": publication_type,
                "source_document_id": _best_text(
                    [record.source_document_id for record in contextual_records]
                ),
                "collaboration_eligible": collaboration_eligible,
                "context_evidence_urls": _variants(
                    [record.context_evidence_url for record in contextual_records]
                ),
                "display_contributors": _best_text(
                    [
                        record.context_display_contributors
                        for record in contextual_records
                    ]
                ),
                "has_included_source_profile": any(
                    membership["included"]
                    for record in canonical_records
                    for membership in _normalize_memberships(record.memberships)
                ),
                "map_exclusion_reasons": []
                if map_eligible
                else sorted(
                    {
                        reason
                        for record in records
                        for reason in record.map_exclusion_reasons
                    }
                ),
                "quality_assessment_version": QUALITY_ASSESSMENT_VERSION,
                "work_match_method": _best_text(
                    [record.work_match_method for record in canonical_records]
                ),
                "doi": _best_text([record.doi for record in canonical_records]),
                "title": _best_text(
                    _source_values(published_records or canonical_records, "title")
                ),
                "title_variants": _variants(
                    [record.title for record in canonical_records]
                ),
                "authors": _best_text(_source_values(canonical_records, "authors")),
                "author_variants": _variants(
                    [record.authors for record in canonical_records]
                ),
                "year": _mode_year(
                    [record.year for record in published_records]
                    if published_records
                    else _source_values(canonical_records, "year")
                ),
                "year_variants": sorted(
                    {
                        year
                        for year in (_year(value) for value in years)
                        if year is not None
                    }
                ),
                "venue": _best_text(
                    [record.venue for record in published_records]
                    if published_records
                    else _source_values(canonical_records, "venue")
                ),
                "venue_variants": _variants(
                    [record.venue for record in canonical_records]
                ),
                "citation": _best_text(_source_values(canonical_records, "citation")),
                "citation_variants": _variants(
                    [record.citation for record in canonical_records]
                ),
                "citation_count": max(
                    (
                        int(value)
                        for value in _source_values(canonical_records, "citation_count")
                    ),
                    default=0,
                ),
                "source_urls": _variants(
                    [record.source_url for record in canonical_records]
                ),
                "source_types": sorted(
                    {str(record.source_type) for record in canonical_records}
                ),
                "source_native_ids": sorted(
                    {
                        str(record.source_native_id)
                        for record in canonical_records
                        if record.source_native_id
                    }
                ),
                "source_verified_at": max(
                    (str(record.source_verified_at) for record in canonical_records),
                    default="",
                ),
                "source_verification_urls": sorted(
                    {
                        str(url)
                        for record in canonical_records
                        for url in _as_list(record.source_verification_urls)
                        if url
                    }
                ),
                "external_coauthors": [
                    external_coauthors[person_id]
                    for person_id in sorted(external_coauthors)
                ],
                "external_coauthor_ids": sorted(external_coauthors),
                "person_ids": sorted(people),
                "faculty": [people[person_id] for person_id in sorted(people)],
                "department_ids": department_ids,
                "department_titles": [
                    department_titles[department_id] for department_id in department_ids
                ],
                "school_ids": school_ids,
                "school_titles": [school_titles[school_id] for school_id in school_ids],
                "memberships": memberships,
                "scholar_ids": sorted(
                    {
                        str(record.scholar_id)
                        for record in canonical_records
                        if record.scholar_id
                    }
                ),
                "author_pub_ids": sorted(
                    {str(record.author_pub_id) for record in canonical_records}
                ),
                "observation_ids": sorted(
                    {str(record.observation_id) for record in canonical_records}
                ),
                "observation_count": len(canonical_records),
                "first_fetched_at_utc": fetched[0] if fetched else "",
                "last_fetched_at_utc": fetched[-1] if fetched else "",
                "embedding": list(canonical_records[0].embedding),
            }
        )
    works = pandas.DataFrame(work_rows)

    observation_count = Counter(profile_publications["person_id"])
    scholar_observation_count = Counter(
        profile_publications.loc[
            profile_publications["source_type"] == "scholar", "person_id"
        ]
    )
    curated_observation_count = Counter(
        profile_publications.loc[
            profile_publications["source_type"] == "curated_verified", "person_id"
        ]
    )
    unique_work_count = {
        person_id: int(group["work_id"].nunique())
        for person_id, group in authorships.groupby("person_id")
    }
    mapped_work_count: Counter[str] = Counter()
    for row in works.itertuples(index=False):
        if not row.map_eligible:
            continue
        for person_id in {membership["person_id"] for membership in row.memberships}:
            mapped_work_count[person_id] += 1
    memberships_by_person: dict[str, list[dict[str, Any]]] = {}
    for membership in registry.memberships:
        memberships_by_person.setdefault(membership.person_id, []).append(
            {
                "department_id": membership.department_id,
                "department_title": department_titles[membership.department_id],
                "school_id": school_by_department[membership.department_id],
                "school_title": school_titles[
                    school_by_department[membership.department_id]
                ],
                "role": membership.role,
                "included": membership.included,
                "legacy_label": membership.legacy_label,
                "source_url": membership.source_url,
                "verified_at": membership.verified_at,
            }
        )
    people_rows = []
    for person in registry.people:
        primary = registry.primary_affiliation(person.person_id)
        memberships = sorted(
            memberships_by_person.get(person.person_id, []),
            key=lambda item: (item["department_id"], item["role"]),
        )
        people_rows.append(
            {
                "person_id": person.person_id,
                "display_name": person.display_name,
                "scholar_id": person.scholar_id,
                "scholar_id_source_url": person.scholar_id_source_url,
                "scholar_id_verified_at": person.scholar_id_verified_at,
                "orcid": person.orcid,
                "homepage_url": person.homepage_url,
                "notes": person.notes,
                **primary.published_fields(),
                "memberships": memberships,
                "included_department_ids": sorted(
                    {
                        membership["department_id"]
                        for membership in memberships
                        if membership["included"]
                    }
                ),
                "included_school_ids": sorted(
                    {
                        membership["school_id"]
                        for membership in memberships
                        if membership["included"]
                    }
                ),
                "publication_observation_count": observation_count.get(
                    person.person_id, 0
                ),
                "scholar_observation_count": scholar_observation_count.get(
                    person.person_id, 0
                ),
                "curated_observation_count": curated_observation_count.get(
                    person.person_id, 0
                ),
                "unique_work_count": unique_work_count.get(person.person_id, 0),
                "mapped_work_count": mapped_work_count.get(person.person_id, 0),
            }
        )
    people = pandas.DataFrame(people_rows)
    return {
        "people": people,
        "works": works,
        "authorships": authorships,
        "profile_publications": profile_publications,
    }


def table_summary(tables: Mapping[str, Any]) -> str:
    """Return a stable compact summary for receipts and tests."""

    return json.dumps(
        {name: len(frame) for name, frame in sorted(tables.items())},
        sort_keys=True,
    )
