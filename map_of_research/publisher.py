"""Publish retained source observations and the unified map artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .atomic_release import AtomicReleaseReconciliationRequired, commit_atomic_release
from .attribution import (
    DEFAULT_ATTRIBUTION_REVIEWS_PATH,
    load_attribution_reviews,
)
from .bio_binding_approval import load_bio_binding_approval
from .bio_release_gate import (
    BIO_SOURCE_SHA256,
    bio_scholar_source_changed,
    load_bio_baselines,
    preserve_pending_review_topics,
    validate_bio_release_prewrite,
)
from .chem_carryforward import (
    AUTHORSHIP_FIELDS as CHEM_CARRYFORWARD_AUTHORSHIP_FIELDS,
)
from .chem_carryforward import (
    PRIOR_REVISION as CHEM_CARRYFORWARD_PARENT,
)
from .chem_carryforward import (
    PROFILE_FIELDS as CHEM_CARRYFORWARD_PROFILE_FIELDS,
)
from .chem_carryforward import (
    SOURCE_SHA256 as CHEM_CARRYFORWARD_SOURCE_SHA256,
)
from .chem_carryforward import (
    WORK_FIELDS as CHEM_CARRYFORWARD_WORK_FIELDS,
)
from .chem_carryforward import (
    validate_chem_carryforward,
)
from .chem_continuity import (
    CHEMISTRY_HUB_REVISION,
    load_chemistry_parent,
    merge_chemistry_review_inputs,
    needs_chemistry_continuity,
    reconcile_chemistry_person_counts,
    validate_chemistry_candidate,
)
from .chem_region_review import (
    CHEM_REVIEW_SHA256,
    DEFAULT_CHEM_REGION_REVIEW_PATH,
    load_chem_region_review,
)
from .coauthor_binding_approval import (
    DEFAULT_AKINCI_REVIEW_PATH,
    DEFAULT_MARTELARO_REVIEW_PATH,
    validate_exact_coauthor_bindings,
)
from .coauthor_binding_approval import (
    DEFAULT_APPROVAL_PATH as DEFAULT_COAUTHOR_BINDING_APPROVAL_PATH,
)
from .coauthor_binding_approval import (
    PRIOR_REVISION as COAUTHOR_BINDING_PRIOR_REVISION,
)
from .coauthor_binding_approval import (
    SNAPSHOT_SHA256 as COAUTHOR_BINDING_SNAPSHOT_SHA256,
)
from .composite_region import (
    HISTORICAL_REVISION as COMPOSITE_HISTORICAL_REVISION,
)
from .composite_region import (
    MATH_REVIEW_MANIFEST_SHA256,
    SULLIVAN_PERSON_ID,
    TOPIC_FIELDS,
    freeze_composite_presentation,
    load_composite_region_plan,
    load_reviewed_sullivan_embeddings,
    require_baseline_map,
    validate_composite_map_points,
    validate_composite_staged_nonwork_tables,
    validate_composite_staged_works,
)
from .composite_region import (
    PRESENTATION_FIELDS as COMPOSITE_PRESENTATION_FIELDS,
)
from .composite_region import (
    VARIANT_FIELDS as COMPOSITE_VARIANT_FIELDS,
)
from .contexts import DEFAULT_CONTEXTS_PATH, load_contexts
from .corrections import (
    DEFAULT_CORRECTIONS_PATH,
    apply_corrections,
    load_corrections,
)
from .coverage import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_CURATED_WORKS_PATH,
    DEFAULT_PRIOR_ID_CROSSWALK_PATH,
    validate_coverage,
)
from .dataset import build_dataset_tables
from .doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    ReviewedDoiAlias,
    load_doi_aliases,
    validate_doi_aliases,
)
from .exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from .held_topic_freeze import (
    held_only_continuity_candidate,
    preserve_held_only_topics,
    retain_held_only_parent_works,
    validate_held_only_stage,
)
from .hub_card import build_updated_dataset_card
from .io import atomic_write_json, load_json, sha256_file
from .keywords import (
    KEYWORD_MODEL_VERSION,
    REGION_AUDIT_VERSION,
    add_publication_keywords,
    exclude_low_information_regions,
)
from .math_release_gate import validate_math_coverage
from .mccomb_continuity import (
    DEFAULT_APPROVAL_PATH as DEFAULT_MCCOMB_CONTINUITY_PATH,
)
from .mccomb_continuity import (
    DEFAULT_CROSSWALK_PATH as DEFAULT_MCCOMB_CROSSWALK_PATH,
)
from .mccomb_continuity import (
    PRIOR_REVISION as MCCOMB_CONTINUITY_PRIOR_REVISION,
)
from .mccomb_continuity import (
    SNAPSHOT_SHA256 as MCCOMB_CONTINUITY_SNAPSHOT_SHA256,
)
from .mccomb_continuity import validate_exact_mccomb_continuity
from .pending_publications import (
    PENDING_MAP_SCHEMA_VERSION,
    build_pending_publications,
)
from .person_publication_holds import (
    DEFAULT_PERSON_PUBLICATION_HOLDS_PATH,
    load_person_publication_holds,
    require_no_held_person_contributions,
    select_released_person_observations,
    validate_person_hold_losses,
)
from .poster_loss_approval import load_poster_loss_approval
from .presentation_drift import (
    PresentationDriftError,
    compare_presentation,
    preserve_additive_report_presentation,
    preserve_deletion_only_presentation,
    preserve_reviewed_additive_presentation,
    reported_coauthor_evidence_changed,
    require_held_additive_staged_works,
    require_presentation_review,
    require_preserved_additive_artifact,
)
from .private_subset_preflight import require_private_preflight_receipt
from .profile_releases import (
    DEFAULT_PROFILE_RELEASES_PATH,
    load_profile_releases,
    require_no_held_contributions,
    select_released_observations,
)
from .provisional_holds import (
    DEFAULT_PROVISIONAL_HOLDS_PATH,
    load_provisional_holds,
    require_no_provisional_hold_contributions,
    validate_provisional_holds,
)
from .public_subsets import ids_sha256, require_public_subset_boundary
from .quality import QUALITY_ASSESSMENT_VERSION
from .region_continuity import (
    PRIOR_OBSERVATION_FIELDS,
    PRIOR_WORK_FIELDS,
    RegionReleaseDiffError,
    reconcile_region_decisions,
    require_revision,
)
from .region_repair import (
    require_approved_comparison_revision,
    require_expected_repair_preview,
    require_frozen_producer_tip,
    require_frozen_source_tip,
    require_historical_region_decisions,
)
from .region_review import DEFAULT_REGION_REVIEW_PATH, load_region_review
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    Registry,
    implicit_primary_baseline_for_people,
    load_registry,
    validate_included_scholar_primary_homes,
)
from .reported_faculty_coauthors import (
    apply_reported_faculty_coauthors,
    require_reported_source_sync,
)
from .scs_four_topic_hold import (
    PRIOR_WORK_FIELDS as SCS_TOPIC_HOLD_PRIOR_WORK_FIELDS,
)
from .scs_four_topic_hold import apply_scs_four_topic_holds
from .snapshot import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_REPORTED_FACULTY_COAUTHORS_PATH,
    validate_snapshot,
)
from .sullivan_continuity_addendum import (
    DEFAULT_ADDENDUM_PATH as DEFAULT_SULLIVAN_CONTINUITY_ADDENDUM_PATH,
)
from .sullivan_continuity_addendum import (
    validate_sullivan_continuity_addendum,
)
from .sullivan_loss_approval import (
    DEFAULT_APPROVAL_PATH as DEFAULT_SULLIVAN_LOSS_APPROVAL_PATH,
)
from .sullivan_loss_approval import (
    DEFAULT_CROSSWALK_PATH as DEFAULT_SULLIVAN_CROSSWALK_PATH,
)
from .sullivan_loss_approval import (
    PRIOR_REVISION as SULLIVAN_PRIOR_REVISION,
)
from .sullivan_loss_approval import (
    SOURCE_SHA256 as SULLIVAN_SOURCE_SHA256,
)
from .sullivan_loss_approval import (
    validate_sullivan_exact_losses,
)
from .topic_labels import TOPIC_LABEL_REVIEW_VERSION

LOGGER = logging.getLogger(__name__)
REPO_ID = "ccm/cmu-engineering-publications"
DATASET_CONFIGS = (
    "people",
    "works",
    "authorships",
    "profile_publications",
)
REUSABLE_EMBEDDING_CONFIGS = (
    "profile_publications",
    "publications",
    "default",
)
HF_TOKEN_ENV = "HF_TOKEN"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
EMBEDDING_DIMENSION = 768
MAP_SCHEMA_VERSION = 7
LAYOUT_VERSION = "global-publications-multilayout-v3"
DEFAULT_LAYOUT_ID = "pca"
LAYOUTS = (
    {
        "layout_id": "pca",
        "label": "Global structure",
        "method": "PCA",
        "description": "Emphasizes broad variation across the full corpus.",
        "x_field": "x",
        "y_field": "y",
        "version": "global-publications-pca-v5",
    },
    {
        "layout_id": "tsne",
        "label": "Local neighborhoods",
        "method": "t-SNE",
        "description": (
            "Emphasizes nearby topical neighborhoods; distances between separate "
            "clusters are not comparable."
        ),
        "x_field": "tsne_x",
        "y_field": "tsne_y",
        "version": "global-publications-tsne-v3",
    },
)
KEYWORD_LEVELS = (
    {"level": 0, "label": "Topic regions"},
    {"level": 1, "label": "Detailed topics"},
)
MAP_ARTIFACT_NAME = "publications.json"
MAP_TITLE = "CMU Faculty Research"
MAX_REUSABLE_DOI_REKEYS = 25
STATUS_SCHEMA_VERSION = 2
DEFAULT_STATUS_PATH = Path("status/last-upload.json")
PHYSICS_REVIEW_STATUS_NAME = "physics-cremonesi-schumacher-source-2026-10-24.json"
PHYSICS_REVIEW_BATCH = "physics-cremonesi-schumacher-source"
PHYSICS_REVIEW_SCHOLAR_IDS = frozenset({"wJ_gQLoAAAAJ", "LibSWRMAAAAJ"})
BREIVIK_REVIEW_STATUS_NAME = "physics-breivik-source-2026-10-24.json"
BREIVIK_REVIEW_BATCH = "physics-breivik-source-slice"
BREIVIK_REVIEW_SCHOLAR_ID = "AcW2EXwAAAAJ"
STRING_LIST_COLUMNS = frozenset(
    {
        "author_pub_ids",
        "author_variants",
        "citation_variants",
        "context_evidence_urls",
        "included_department_ids",
        "included_school_ids",
        "map_exclusion_reasons",
        "department_ids",
        "department_titles",
        "school_ids",
        "school_titles",
        "observation_ids",
        "person_ids",
        "scholar_ids",
        "source_urls",
        "source_types",
        "source_native_ids",
        "source_verification_urls",
        "external_coauthor_ids",
        "title_variants",
        "venue_variants",
    }
)


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


def _published_rows(
    config_name: str,
    columns: tuple[str, ...],
    *,
    hf_token: str,
    revision: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Read published Parquet columns without applying stale Hub feature metadata."""

    import pyarrow.parquet as parquet
    from huggingface_hub import snapshot_download

    directory_name = "data" if config_name == "default" else config_name
    download_options: dict[str, Any] = {
        "repo_type": "dataset",
        "allow_patterns": f"{directory_name}/train-*.parquet",
        "token": hf_token or None,
    }
    if revision is not None:
        download_options["revision"] = require_revision(revision)
    snapshot_root = Path(snapshot_download(REPO_ID, **download_options))
    parquet_files = sorted((snapshot_root / directory_name).glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No published {config_name} Parquet shard")
    for parquet_path in parquet_files:
        shard = parquet.ParquetFile(parquet_path)
        missing = set(columns).difference(shard.schema_arrow.names)
        if missing:
            LOGGER.info("Published %s lacks reusable columns %s", config_name, missing)
            return
        for batch in shard.iter_batches(columns=list(columns), batch_size=512):
            yield from batch.to_pylist()


def _require_current_dataset_revision(*, hf_token: str, prior_revision: str) -> None:
    """Fail before writing if the pinned baseline is no longer the Hub head."""

    from huggingface_hub import HfApi

    expected = require_revision(prior_revision)
    actual = HfApi(token=hf_token).repo_info(REPO_ID, repo_type="dataset").sha
    if actual != expected:
        raise RuntimeError(
            f"Hosted dataset moved after preview: expected {expected}, found {actual}"
        )


def _existing_embeddings(
    *, hf_token: str, revision: str | None = None
) -> dict[tuple[str, str], list[float]]:
    """Load reusable observation vectors from any prior dataset generation."""

    for config_name in REUSABLE_EMBEDDING_CONFIGS:
        embeddings: dict[tuple[str, str], list[float]] = {}
        try:
            for row in _published_rows(
                config_name,
                ("author_pub_id", "title", "embedding"),
                hf_token=hf_token,
                **({"revision": revision} if revision is not None else {}),
            ):
                author_pub_id = str(row.get("author_pub_id") or "")
                title = str(row.get("title") or "")
                embedding = row.get("embedding")
                if not author_pub_id or not title or not isinstance(embedding, list):
                    continue
                if len(embedding) != EMBEDDING_DIMENSION:
                    continue
                embeddings[(author_pub_id, title)] = [
                    float(value) for value in embedding
                ]
        except FileNotFoundError:
            LOGGER.info("No reusable %s dataset config found", config_name)
            continue
        except Exception as error:
            raise RuntimeError(
                f"Could not read reusable {config_name} embeddings"
            ) from error
        if embeddings:
            LOGGER.info(
                "Reusing %d embeddings from config %s",
                len(embeddings),
                config_name,
            )
            return embeddings
    return {}


def add_embeddings(
    frame: Any,
    *,
    hf_token: str,
    encoder: Any | None = None,
    prior_revision: str | None = None,
    reviewed_new_vectors: Mapping[tuple[str, str], list[float]] | None = None,
) -> tuple[Any, int]:
    """Reuse published vectors and exact reviewed vectors before encoding titles."""

    existing = _existing_embeddings(
        hf_token=hf_token,
        **({"revision": prior_revision} if prior_revision is not None else {}),
    )
    keys = [
        (str(row.author_pub_id), str(row.title))
        for row in frame[["author_pub_id", "title"]].itertuples(index=False)
    ]
    # The encoder receives only the title. A new curated observation of an
    # already published title can therefore use the same vector, provided all
    # published vectors for that exact title agree.
    by_title: dict[str, list[float]] = {}
    ambiguous_titles: set[str] = set()
    for (_, title), vector in existing.items():
        previous = by_title.setdefault(title, vector)
        if previous != vector:
            ambiguous_titles.add(title)
    for key in keys:
        if (
            key not in existing
            and key[1] in by_title
            and key[1] not in ambiguous_titles
        ):
            existing[key] = by_title[key[1]]

    missing_keys = list(dict.fromkeys(key for key in keys if key not in existing))
    if reviewed_new_vectors is not None:
        if set(missing_keys) != set(reviewed_new_vectors):
            raise ValueError("New vectors differ from reviewed Sullivan release")
        existing.update(reviewed_new_vectors)
        output = frame.copy()
        output["embedding"] = [existing[key] for key in keys]
        return output, len(missing_keys)
    if missing_keys:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                MODEL_NAME,
                revision=MODEL_REVISION,
                trust_remote_code=False,
            )
        vectors = encoder.encode(
            [title for _, title in missing_keys],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        if len(vectors) != len(missing_keys):
            raise RuntimeError("Embedding model returned an unexpected row count")
        for key, vector in zip(missing_keys, vectors, strict=True):
            values = [float(value) for value in vector]
            if len(values) != EMBEDDING_DIMENSION:
                raise RuntimeError("Embedding model returned an unexpected dimension")
            existing[key] = values

    output = frame.copy()
    output["embedding"] = [existing[key] for key in keys]
    return output, len(missing_keys)


def _existing_layouts(
    *, hf_token: str, revision: str | None = None
) -> dict[str, dict[str, float]]:
    """Load a complete compatible layout that can be reused without refitting."""

    coordinate_fields = {
        field
        for layout in LAYOUTS
        for field in (str(layout["x_field"]), str(layout["y_field"]))
    }
    required_fields = ("work_id", "layout_version", *sorted(coordinate_fields))
    layouts: dict[str, dict[str, float]] = {}
    try:
        for row in _published_rows(
            "works",
            required_fields,
            hf_token=hf_token,
            **({"revision": revision} if revision is not None else {}),
        ):
            if row.get("layout_version") != LAYOUT_VERSION:
                continue
            work_id = str(row.get("work_id") or "")
            coordinates: dict[str, float] = {}
            for field in coordinate_fields:
                try:
                    value = float(row.get(field))
                except (TypeError, ValueError):
                    break
                if not math.isfinite(value):
                    break
                coordinates[field] = value
            if len(coordinates) != len(coordinate_fields) or not work_id:
                continue
            if work_id in layouts:
                return {}
            layouts[work_id] = coordinates
    except FileNotFoundError:
        LOGGER.info("No reusable works layout found")
        return {}
    except Exception as error:
        raise RuntimeError("Could not read reusable works layouts") from error
    if layouts:
        LOGGER.info("Found reusable coordinates for %d works", len(layouts))
    return layouts


def _existing_work_state(
    *, hf_token: str, revision: str | None = None
) -> list[dict[str, Any]]:
    """Read the published work order and fields needed to verify layout reuse."""

    fields = (
        "work_id",
        "title",
        "year",
        "doi",
        "work_match_method",
        "map_eligible",
        "map_region_outlier",
        "map_exclusion_reasons",
        "department_ids",
        "quality_assessment_version",
        "keyword_model_version",
        "topic_label_review_version",
        "map_region_audit_version",
        "layout_version",
    )
    try:
        return list(
            _published_rows(
                "works",
                fields,
                hf_token=hf_token,
                **({"revision": revision} if revision is not None else {}),
            )
        )
    except FileNotFoundError:
        LOGGER.info("No published works are available for order reuse")
        return []
    except Exception as error:
        raise RuntimeError("Could not read published works for order reuse") from error


def _year_key(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reusable_work_order(
    works: Any,
    previous: Sequence[Mapping[str, Any]],
    layouts: Mapping[str, Mapping[str, float]],
    doi_aliases: Mapping[tuple[str, str], ReviewedDoiAlias] | None = None,
) -> tuple[Any, dict[str, Mapping[str, float]]] | None:
    """Reuse prior coordinates and row order only for an identical work corpus.

    DOI-backed records may replace a few title/year work IDs. Reusing the old
    row order matters as much as reusing coordinates: k-means initialization
    depends on row order even when every coordinate is unchanged.
    """

    if len(previous) != len(works) or not previous or not layouts:
        return None
    prior_by_id = {str(row.get("work_id") or ""): row for row in previous}
    current_by_id = {str(row.work_id): row for row in works.itertuples(index=False)}
    if (
        "" in prior_by_id
        or len(prior_by_id) != len(previous)
        or len(current_by_id) != len(works)
    ):
        return None

    for row in previous:
        if (
            row.get("layout_version") != LAYOUT_VERSION
            or row.get("quality_assessment_version") != QUALITY_ASSESSMENT_VERSION
            or row.get("keyword_model_version") != KEYWORD_MODEL_VERSION
            or row.get("topic_label_review_version") != TOPIC_LABEL_REVIEW_VERSION
            or row.get("map_region_audit_version") != REGION_AUDIT_VERSION
        ):
            return None

    missing_prior = set(prior_by_id) - set(current_by_id)
    new_current = set(current_by_id) - set(prior_by_id)
    if len(missing_prior) > MAX_REUSABLE_DOI_REKEYS:
        return None

    # Require a title/year key to identify exactly one work across both full
    # corpora. This rules out accidental joins on common or duplicate titles.
    prior_keys: dict[tuple[str, int | None], list[str]] = {}
    current_keys: dict[tuple[str, int | None], list[str]] = {}
    for work_id, row in prior_by_id.items():
        key = (str(row.get("title") or ""), _year_key(row.get("year")))
        prior_keys.setdefault(key, []).append(work_id)
    for work_id, row in current_by_id.items():
        key = (str(row.title or ""), _year_key(row.year))
        current_keys.setdefault(key, []).append(work_id)

    rekeys: dict[str, str] = {}
    for old_id in missing_prior:
        row = prior_by_id[old_id]
        key = (str(row.get("title") or ""), _year_key(row.get("year")))
        if (
            not key[0].strip()
            or key[1] is None
            or row.get("work_match_method") != "normalized_title_year"
            or str(row.get("doi") or "").strip()
        ):
            return None
        old_matches = prior_keys.get(key, [])
        new_matches = current_keys.get(key, [])
        if len(old_matches) != 1 or len(new_matches) != 1:
            return None
        new_id = new_matches[0]
        if new_id not in new_current:
            return None
        current = current_by_id[new_id]
        doi = str(current.doi or "").casefold()
        expected_id = "work-" + hashlib.sha256(f"doi:{doi}".encode()).hexdigest()
        curated_rekey = current.work_match_method in {
            "doi",
            "curated_title_year",
        } and "curated_verified" in _as_list(current.source_types)
        reviewed_alias_rekey = (
            current.work_match_method == "reviewed_version_alias"
            and "scholar" in _as_list(current.source_types)
            and any(
                alias.doi == doi
                and alias.observed_title == str(current.title or "")
                and alias.observed_year == _year_key(current.year)
                and scholar_id in _as_list(current.scholar_ids)
                and author_pub_id in _as_list(current.author_pub_ids)
                and set(alias.evidence_urls).issubset(
                    _as_list(current.source_verification_urls)
                )
                for (scholar_id, author_pub_id), alias in (doi_aliases or {}).items()
            )
        )
        if (
            not doi.startswith("10.")
            or new_id != expected_id
            or not (curated_rekey or reviewed_alias_rekey)
            or not str(current.source_verified_at or "").strip()
            or not _as_list(current.source_verification_urls)
        ):
            return None
        rekeys[old_id] = new_id
    remapped_layouts: dict[str, Mapping[str, float]] = {}
    for old_id, prior in prior_by_id.items():
        new_id = rekeys.get(old_id, old_id)
        current = current_by_id[new_id]
        if str(prior.get("title") or "") != str(current.title or "") or _year_key(
            prior.get("year")
        ) != _year_key(current.year):
            return None
        prior_reasons = set(map(str, _as_list(prior.get("map_exclusion_reasons"))))
        was_region_outlier = bool(prior.get("map_region_outlier"))
        previous_eligible = bool(prior.get("map_eligible")) or was_region_outlier
        if was_region_outlier:
            prior_reasons.discard("low_information_region")
        if (
            previous_eligible != bool(current.map_eligible)
            or prior_reasons != set(map(str, _as_list(current.map_exclusion_reasons)))
            or (previous_eligible and bool(_as_list(prior.get("department_ids"))))
            != (bool(current.map_eligible) and bool(_as_list(current.department_ids)))
        ):
            return None
        if previous_eligible and bool(_as_list(prior.get("department_ids"))):
            if old_id not in layouts:
                return None
            remapped_layouts[new_id] = layouts[old_id]

    ordered_ids = [
        rekeys.get(str(row["work_id"]), str(row["work_id"])) for row in previous
    ]
    positions = {
        str(work_id): position for position, work_id in enumerate(works["work_id"])
    }
    ordered_works = works.iloc[
        [positions[work_id] for work_id in ordered_ids]
    ].reset_index(drop=True)
    LOGGER.info(
        "Reusing published layout and work order for %d works (%d DOI rekeys)",
        len(ordered_works),
        len(rekeys),
    )
    return ordered_works, remapped_layouts


def _normalize_layout(projected: Any, *, preserve_aspect: bool) -> Any:
    """Center coordinates and scale them to a stable browser-friendly range."""

    import numpy

    projected = numpy.asarray(projected, dtype=numpy.float32)
    projected -= projected.mean(axis=0, keepdims=True)
    if preserve_aspect:
        scale = float(numpy.max(numpy.abs(projected)))
        if scale == 0:
            scale = 1.0
    else:
        scale = numpy.max(numpy.abs(projected), axis=0)
        scale[scale == 0] = 1.0
    return projected / scale


def _fit_pca_layout(matrix: Any) -> Any:
    """Fit the current broad-structure PCA view."""

    from sklearn.decomposition import PCA

    projected = PCA(
        n_components=2,
        svd_solver="randomized",
        random_state=0,
    ).fit_transform(matrix)
    return _normalize_layout(projected, preserve_aspect=False)


def _fit_tsne_layout(matrix: Any) -> Any:
    """Fit a memory-bounded PCA, t-SNE, and final-orientation pipeline."""

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    sample_count = len(matrix)
    reduced_dimensions = min(50, matrix.shape[1], sample_count - 1)
    reduced = PCA(
        n_components=reduced_dimensions,
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(matrix)
    perplexity = min(30.0, max(1.0, (sample_count - 1) / 3))
    neighborhoods = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=42,
        method="barnes_hut",
        angle=0.5,
        n_jobs=1,
        verbose=1,
    ).fit_transform(reduced)
    oriented = PCA(
        n_components=2,
        svd_solver="full",
        random_state=42,
    ).fit_transform(neighborhoods)
    return _normalize_layout(oriented, preserve_aspect=True)


def add_global_layout(
    works: Any,
    *,
    existing_layouts: Mapping[str, Mapping[str, float]] | None = None,
) -> Any:
    """Attach both deterministic, full-corpus layouts to every included work."""

    import numpy

    output = works.copy()
    output["layout_version"] = LAYOUT_VERSION
    coordinate_fields = tuple(
        dict.fromkeys(
            field
            for layout in LAYOUTS
            for field in (str(layout["x_field"]), str(layout["y_field"]))
        )
    )
    for field in coordinate_fields:
        output[field] = math.nan
    included_mask = output["map_eligible"] & output["department_ids"].apply(
        lambda value: bool(_as_list(value))
    )
    included = output.loc[included_mask]
    if included.empty:
        raise ValueError("No included works are available for map generation")
    included_work_ids = [str(value) for value in included["work_id"]]
    if existing_layouts and set(existing_layouts) == set(included_work_ids):
        for field in coordinate_fields:
            output.loc[included_mask, field] = [
                round(float(existing_layouts[work_id][field]), 7)
                for work_id in included_work_ids
            ]
        LOGGER.info("Reused both layouts for %d unchanged works", len(included))
        return output

    matrix = numpy.asarray(included["embedding"].tolist(), dtype=numpy.float32)
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    if len(matrix) == 1:
        pca_projected = numpy.zeros((1, 2), dtype=numpy.float32)
        tsne_projected = numpy.zeros((1, 2), dtype=numpy.float32)
    else:
        pca_projected = _fit_pca_layout(matrix)
        tsne_projected = _fit_tsne_layout(matrix)
    output.loc[included_mask, "x"] = [
        round(float(value), 7) for value in pca_projected[:, 0]
    ]
    output.loc[included_mask, "y"] = [
        round(float(value), 7) for value in pca_projected[:, 1]
    ]
    output.loc[included_mask, "tsne_x"] = [
        round(float(value), 7) for value in tsne_projected[:, 0]
    ]
    output.loc[included_mask, "tsne_y"] = [
        round(float(value), 7) for value in tsne_projected[:, 1]
    ]
    return output


def _normalized_memberships(value: Any) -> list[dict[str, str]]:
    memberships = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            raise ValueError("Work membership must be an object")
        memberships.append(
            {
                "person_id": str(item.get("person_id") or ""),
                "display_name": str(item.get("display_name") or ""),
                "department_id": str(item.get("department_id") or ""),
                "department_title": str(item.get("department_title") or ""),
                "role": str(item.get("role") or ""),
            }
        )
    return memberships


def build_map_artifact(
    works: Any,
    registry: Registry,
    *,
    generated_at_utc: str,
    publication_hold_reasons: Mapping[str, str] | None = None,
    reported_faculty_coauthors_by_work: Mapping[str, list[dict[str, str]]]
    | None = None,
) -> dict[str, Any]:
    """Build the full work-centric artifact without changing global coordinates."""

    school_by_department = {
        department.department_id: department.school_id
        for department in registry.departments
    }

    related_rows: list[tuple[Any, list[dict[str, str]]]] = []
    selected_rows: list[tuple[Any, list[dict[str, str]]]] = []
    for row in works.itertuples(index=False):
        memberships = _normalized_memberships(row.memberships)
        source_included = bool(
            getattr(row, "has_included_source_profile", bool(memberships))
        )
        if source_included:
            related_rows.append((row, memberships))
            if row.map_eligible and memberships:
                selected_rows.append((row, memberships))
    points = []
    external_coauthors: dict[str, dict[str, str]] = {}
    keyword_metadata: dict[str, dict[str, Any]] = {}
    for row, memberships in selected_rows:
        collaboration_eligible = bool(getattr(row, "collaboration_eligible", True))
        reported_coauthors = list(
            (reported_faculty_coauthors_by_work or {}).get(str(row.work_id), [])
        )
        if reported_coauthors and not collaboration_eligible:
            raise ValueError(
                "Reported faculty coauthor work lost collaboration eligibility"
            )
        for coauthor in (
            _as_list(getattr(row, "external_coauthors", []))
            if collaboration_eligible
            else []
        ):
            if not isinstance(coauthor, Mapping):
                raise ValueError("External coauthor catalog entry must be an object")
            entry = {
                "person_id": str(coauthor.get("person_id") or ""),
                "display_name": str(coauthor.get("display_name") or ""),
                "source_url": str(coauthor.get("source_url") or ""),
            }
            person_id = entry["person_id"]
            if not person_id or not entry["display_name"]:
                raise ValueError("External coauthor catalog entry needs identity")
            if (
                person_id in external_coauthors
                and external_coauthors[person_id] != entry
            ):
                raise ValueError("Conflicting external coauthor catalog entry")
            external_coauthors[person_id] = entry
        source_urls = [str(value) for value in _as_list(row.source_urls) if value]
        keyword_id = str(row.keyword_id)
        keyword = str(row.keyword)
        keyword_extracted = str(row.keyword_extracted)
        keyword_label_reviewed = bool(row.keyword_label_reviewed)
        detail_keyword_id = str(row.detail_keyword_id)
        detail_keyword = str(row.detail_keyword)
        detail_keyword_extracted = str(row.detail_keyword_extracted)
        detail_keyword_label_reviewed = bool(row.detail_keyword_label_reviewed)
        if not all(
            (
                keyword_id,
                keyword,
                keyword_extracted,
                detail_keyword_id,
                detail_keyword,
                detail_keyword_extracted,
            )
        ):
            raise ValueError("Every mapped work must have both topic keyword levels")
        row_keywords = (
            (
                keyword_id,
                keyword,
                keyword_extracted,
                keyword_label_reviewed,
                0,
                None,
            ),
            (
                detail_keyword_id,
                detail_keyword,
                detail_keyword_extracted,
                detail_keyword_label_reviewed,
                1,
                keyword_id,
            ),
        )
        for (
            item_id,
            label,
            extracted_label,
            label_reviewed,
            level,
            parent_id,
        ) in row_keywords:
            metadata = {
                "label": label,
                "extracted_label": extracted_label,
                "label_reviewed": label_reviewed,
                "level": level,
                "parent_keyword_id": parent_id,
            }
            existing = keyword_metadata.setdefault(item_id, metadata)
            if existing != metadata:
                raise ValueError(f"Keyword {item_id} has inconsistent metadata")
        points.append(
            {
                "x": round(float(row.x), 7),
                "y": round(float(row.y), 7),
                "tsne_x": round(float(row.tsne_x), 7),
                "tsne_y": round(float(row.tsne_y), 7),
                "work_id": str(row.work_id),
                "title": str(row.title),
                "authors": str(row.authors),
                "faculty_ids": sorted(
                    {membership["person_id"] for membership in memberships}
                ),
                "department_ids": sorted(
                    {membership["department_id"] for membership in memberships}
                ),
                "school_ids": sorted(
                    {
                        school_by_department[membership["department_id"]]
                        for membership in memberships
                    }
                ),
                "year": None
                if row.year is None or math.isnan(float(row.year))
                else int(row.year),
                "venue": str(row.venue),
                "citation_count": int(row.citation_count),
                "doi": str(row.doi),
                "publication_type": str(
                    getattr(row, "publication_type", "publication")
                ),
                "source_document_id": str(getattr(row, "source_document_id", "")),
                "collaboration_eligible": collaboration_eligible,
                "context_evidence_urls": list(
                    _as_list(getattr(row, "context_evidence_urls", []))
                ),
                "display_contributors": str(getattr(row, "display_contributors", "")),
                "source_url": source_urls[0] if source_urls else "",
                "source_types": list(
                    _as_list(getattr(row, "source_types", ["scholar"]))
                ),
                "source_native_ids": list(
                    _as_list(getattr(row, "source_native_ids", []))
                ),
                "source_verified_at": str(getattr(row, "source_verified_at", "")),
                "source_verification_urls": list(
                    _as_list(getattr(row, "source_verification_urls", []))
                ),
                **(
                    {"reported_faculty_coauthors": reported_coauthors}
                    if reported_coauthors
                    else {}
                ),
                "external_coauthor_ids": list(
                    _as_list(getattr(row, "external_coauthor_ids", []))
                )
                if collaboration_eligible
                else [],
                "observation_count": int(row.observation_count),
                "keyword_ids": [keyword_id, detail_keyword_id],
            }
        )
    points.sort(key=lambda point: point["work_id"])
    unused_reports = set(reported_faculty_coauthors_by_work or {}).difference(
        {point["work_id"] for point in points}
    )
    if unused_reports:
        raise ValueError("Reported faculty coauthor assertion has no mapped point")

    keyword_members: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        for keyword_id in point["keyword_ids"]:
            keyword_members.setdefault(str(keyword_id), []).append(point)
    keywords = []
    for keyword_id in sorted(
        keyword_members,
        key=lambda value: (
            keyword_metadata[value]["level"],
            keyword_metadata[value]["label"].casefold(),
        ),
    ):
        members = keyword_members[keyword_id]
        coordinates = {}
        for layout in LAYOUTS:
            x_field = str(layout["x_field"])
            y_field = str(layout["y_field"])
            coordinates[str(layout["layout_id"])] = {
                "x": round(
                    sum(float(point[x_field]) for point in members) / len(members),
                    7,
                ),
                "y": round(
                    sum(float(point[y_field]) for point in members) / len(members),
                    7,
                ),
            }
        keywords.append(
            {
                "keyword_id": keyword_id,
                **keyword_metadata[keyword_id],
                "publication_count": len(members),
                "coordinates": coordinates,
            }
        )
    keyword_levels = []
    for level in KEYWORD_LEVELS:
        level_number = int(level["level"])
        keyword_levels.append(
            {
                **level,
                "keyword_count": sum(
                    keyword["level"] == level_number for keyword in keywords
                ),
            }
        )

    department_counts: dict[str, int] = {}
    school_counts: dict[str, int] = {}
    faculty_counts: dict[str, int] = {}
    for point in points:
        for department_id in point["department_ids"]:
            department_counts[department_id] = (
                department_counts.get(department_id, 0) + 1
            )
        for school_id in point["school_ids"]:
            school_counts[school_id] = school_counts.get(school_id, 0) + 1
        for person_id in point["faculty_ids"]:
            faculty_counts[person_id] = faculty_counts.get(person_id, 0) + 1

    departments = [
        {
            "department_id": department.department_id,
            "school_id": department.school_id,
            "unit_type": department.unit_type,
            "title": department.title,
            "directory_url": department.directory_url,
            "reviewed_at": department.reviewed_at,
            "publication_count": department_counts.get(
                department.department_id,
                0,
            ),
        }
        for department in sorted(
            registry.departments,
            key=lambda item: item.title.casefold(),
        )
    ]
    schools = [
        {
            "school_id": school.school_id,
            "title": school.title,
            "directory_url": school.directory_url,
            "publication_count": school_counts.get(school.school_id, 0),
        }
        for school in sorted(registry.schools, key=lambda item: item.title.casefold())
    ]
    memberships_by_person: dict[str, list[dict[str, str]]] = {}
    for membership in registry.memberships:
        if not membership.included:
            continue
        memberships_by_person.setdefault(membership.person_id, []).append(
            {
                "department_id": membership.department_id,
                "role": membership.role,
            }
        )
    faculty = []
    ordered_people = sorted(
        registry.people,
        key=lambda item: item.display_name.casefold(),
    )
    for person in ordered_people:
        memberships = memberships_by_person.get(person.person_id, [])
        if not memberships:
            continue
        primary = registry.primary_affiliation(person.person_id)
        faculty.append(
            {
                "person_id": person.person_id,
                "display_name": person.display_name,
                "scholar_id": person.scholar_id,
                "orcid": person.orcid,
                "homepage_url": person.homepage_url,
                "memberships": sorted(
                    memberships,
                    key=lambda item: (item["department_id"], item["role"]),
                ),
                **primary.published_fields(),
                "school_ids": sorted(
                    {
                        school_by_department[membership["department_id"]]
                        for membership in memberships
                    }
                ),
                "publication_count": faculty_counts.get(person.person_id, 0),
                "publication_hold_reason": (publication_hold_reasons or {}).get(
                    person.person_id, ""
                ),
            }
        )
    fetched = [
        str(row.last_fetched_at_utc)
        for row, _ in related_rows
        if "scholar" in _as_list(getattr(row, "source_types", ["scholar"]))
        and str(row.last_fetched_at_utc)
    ]
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "quality_assessment_version": QUALITY_ASSESSMENT_VERSION,
        "keyword_model_version": KEYWORD_MODEL_VERSION,
        "topic_label_review_version": TOPIC_LABEL_REVIEW_VERSION,
        "region_audit_version": REGION_AUDIT_VERSION,
        "layout_version": LAYOUT_VERSION,
        "default_layout_id": DEFAULT_LAYOUT_ID,
        "layouts": [dict(layout) for layout in LAYOUTS],
        "title": MAP_TITLE,
        "generated_at_utc": generated_at_utc,
        "source_data_oldest_at_utc": min(fetched) if fetched else None,
        "source_data_newest_at_utc": max(fetched) if fetched else None,
        "point_count": len(points),
        "excluded_work_count": len(related_rows) - len(selected_rows),
        "department_count": len(departments),
        "school_count": len(schools),
        "faculty_count": len(faculty),
        "keyword_count": len(keywords),
        "keyword_levels": keyword_levels,
        "catalogs": {
            "schools": schools,
            "departments": departments,
            "faculty": faculty,
            "external_coauthors": [
                external_coauthors[person_id]
                for person_id in sorted(external_coauthors)
            ],
        },
        "keywords": keywords,
        "points": points,
    }


def _write_map_artifacts(
    works: Any,
    registry: Registry,
    *,
    maps_dir: Path,
    generated_at_utc: str,
    dataset_commits: Mapping[str, str],
    continuity_receipt: Mapping[str, Any] | None = None,
    publication_hold_reasons: Mapping[str, str] | None = None,
    reported_faculty_coauthors_by_work: Mapping[str, list[dict[str, str]]]
    | None = None,
    pending_publications: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Render both map files before the Hub is touched."""

    maps_dir.mkdir(parents=True, exist_ok=True)
    artifact_schema_version = (
        PENDING_MAP_SCHEMA_VERSION
        if pending_publications is not None
        else MAP_SCHEMA_VERSION
    )
    map_manifest: dict[str, Any] = {
        "schema_version": artifact_schema_version,
        "quality_assessment_version": QUALITY_ASSESSMENT_VERSION,
        "keyword_model_version": KEYWORD_MODEL_VERSION,
        "topic_label_review_version": TOPIC_LABEL_REVIEW_VERSION,
        "region_audit_version": REGION_AUDIT_VERSION,
        "layout_version": LAYOUT_VERSION,
        "default_layout_id": DEFAULT_LAYOUT_ID,
        "layouts": [dict(layout) for layout in LAYOUTS],
        "generated_at_utc": generated_at_utc,
        "dataset_commits": dict(dataset_commits),
        "artifact": {},
    }
    if continuity_receipt is not None:
        map_manifest["region_continuity"] = dict(continuity_receipt)
    if not dataset_commits:
        map_manifest["dataset_revision_policy"] = "same_revision_as_map_manifest"
    artifact = build_map_artifact(
        works,
        registry,
        generated_at_utc=generated_at_utc,
        publication_hold_reasons=publication_hold_reasons,
        reported_faculty_coauthors_by_work=reported_faculty_coauthors_by_work,
    )
    if pending_publications is not None:
        artifact["schema_version"] = artifact_schema_version
        artifact["pending_publications"] = pending_publications
        artifact["pending_publication_count"] = len(pending_publications)
    artifact_path = maps_dir / MAP_ARTIFACT_NAME
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    map_manifest["artifact"] = {
        "title": artifact["title"],
        "point_count": artifact["point_count"],
        "excluded_work_count": artifact["excluded_work_count"],
        "department_count": artifact["department_count"],
        "school_count": artifact["school_count"],
        "faculty_count": artifact["faculty_count"],
        "keyword_count": artifact["keyword_count"],
        "keyword_levels": artifact["keyword_levels"],
        "file": artifact_path.name,
    }
    if pending_publications is not None:
        map_manifest["artifact"]["pending_publication_count"] = len(
            pending_publications
        )
    manifest_path = maps_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(map_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path, manifest_path, artifact


def _pinned_parent_map(
    *, prior_revision: str, hf_token: str
) -> tuple[dict[str, Any], str]:
    """Read the immutable parent map used by the presentation-drift gate."""

    from huggingface_hub import hf_hub_download

    parent_path = Path(
        hf_hub_download(
            REPO_ID,
            f"maps/{MAP_ARTIFACT_NAME}",
            repo_type="dataset",
            revision=prior_revision,
            token=hf_token or None,
        )
    )
    parent = load_json(parent_path)
    if not isinstance(parent, dict):
        raise ValueError("Pinned parent map is not an object")
    return parent, sha256_file(parent_path)


def _pinned_parent_presentation_source_sha256(
    *, prior_revision: str, hf_token: str
) -> str:
    """Read the raw source recorded by the immutable parent map manifest."""

    from huggingface_hub import hf_hub_download

    manifest_path = Path(
        hf_hub_download(
            REPO_ID,
            "maps/manifest.json",
            repo_type="dataset",
            revision=prior_revision,
            token=hf_token or None,
        )
    )
    manifest = load_json(manifest_path)
    presentation = (
        manifest.get("presentation_drift") if isinstance(manifest, dict) else None
    )
    source_sha256 = (
        presentation.get("source_snapshot_sha256")
        if isinstance(presentation, dict)
        else None
    )
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("Pinned parent map manifest lacks its raw source SHA-256")
    return source_sha256


def _to_hub_dataset(frame: Any) -> Any:
    import datasets

    dataset = datasets.Dataset.from_pandas(frame, preserve_index=False)
    features = dataset.features.copy()
    changed = False
    for column in STRING_LIST_COLUMNS.intersection(dataset.column_names):
        features[column] = datasets.List(datasets.Value("string"))
        changed = True
    if "embedding" in dataset.column_names:
        features["embedding"] = datasets.List(datasets.Value("float32"))
        changed = True
    if changed:
        dataset = dataset.cast(features)
    return dataset


def _require_physics_review_ready(
    frame: Any,
    snapshot_path: Path,
    exclusions: Mapping[tuple[str, str], Any],
) -> None:
    """Keep held Physics captures raw until their source review is released."""

    present_ids = PHYSICS_REVIEW_SCHOLAR_IDS.intersection(frame["scholar_id"])
    if not present_ids:
        return
    status_path = snapshot_path.parent.parent / "status" / PHYSICS_REVIEW_STATUS_NAME
    status = load_json(status_path)
    if not isinstance(status, dict) or status.get("batch") != PHYSICS_REVIEW_BATCH:
        raise ValueError(
            f"Physics map-quality gate: invalid source status: {status_path}"
        )
    batch_ready = (
        status.get("map_quality_hold") is False
        and status.get("producer_review_ready") is True
        and status.get("unresolved_exact_ids") == []
    )
    profile_decisions = status.get("paired_profile_release_ready")
    for scholar_id in present_ids:
        if batch_ready:
            continue
        decision = (
            profile_decisions.get(scholar_id)
            if isinstance(profile_decisions, dict)
            else None
        )
        if (
            not isinstance(decision, dict)
            or decision.get("map_quality_hold") is not False
            or decision.get("producer_review_ready") is not True
            or decision.get("unresolved_exact_ids") != []
        ):
            raise ValueError(
                "Physics map-quality gate: source review is held; "
                f"snapshot contains {scholar_id}"
            )
        ledger_path = decision.get("review_ledger_path")
        ledger_sha256 = decision.get("review_ledger_sha256")
        if (
            not isinstance(ledger_path, str)
            or Path(ledger_path).parts != ("status", Path(ledger_path).name)
            or not isinstance(ledger_sha256, str)
            or len(ledger_sha256) != 64
        ):
            raise ValueError("Physics map-quality gate: invalid profile review ledger")
        ledger_file = snapshot_path.parent.parent / ledger_path
        ledger_bytes = ledger_file.read_bytes()
        if hashlib.sha256(ledger_bytes).hexdigest() != ledger_sha256:
            raise ValueError("Physics map-quality gate: profile review ledger changed")
        ledger = json.loads(ledger_bytes)
        if (
            ledger.get("scholar_id") != scholar_id
            or ledger.get("reviewed_exclusion_count")
            != decision.get("reviewed_exclusion_count")
            or ledger.get("provisional_hold_count")
            != decision.get("provisional_hold_count")
        ):
            raise ValueError("Physics map-quality gate: profile review ledger differs")
    expected_counts = status.get("paired_producer_review_count_by_profile")
    if not isinstance(expected_counts, dict) or any(
        expected_counts.get(scholar_id)
        != sum(key[0] == scholar_id for key in exclusions)
        for scholar_id in present_ids
    ):
        raise ValueError(
            "Physics map-quality gate: reviewed-exclusion counts differ "
            "from the source status"
        )


def _require_breivik_review_ready(
    frame: Any,
    snapshot_path: Path,
    exclusions: Mapping[tuple[str, str], Any],
) -> None:
    """Keep Breivik's captured rows out of releases until review is complete."""

    if BREIVIK_REVIEW_SCHOLAR_ID not in frame["scholar_id"].values:
        return
    status_path = snapshot_path.parent.parent / "status" / BREIVIK_REVIEW_STATUS_NAME
    status = load_json(status_path)
    if not isinstance(status, dict) or status.get("batch") != BREIVIK_REVIEW_BATCH:
        raise ValueError(
            f"Breivik map-quality gate: invalid source status: {status_path}"
        )
    if (
        status.get("ingestion_ready") is not True
        or status.get("snapshot_release_hold") is not False
        or type(status.get("preliminary_candidate_count")) is not int
        or status["preliminary_candidate_count"] != 0
        or status.get("quality_review_complete") is not True
    ):
        raise ValueError("Breivik map-quality gate: source review is held")
    reviewed_count = status.get("reviewed_exclusion_count_present_in_producer_main")
    actual_count = sum(key[0] == BREIVIK_REVIEW_SCHOLAR_ID for key in exclusions)
    if type(reviewed_count) is not int or reviewed_count != actual_count:
        raise ValueError(
            "Breivik map-quality gate: reviewed-exclusion count differs "
            "from the source status"
        )


def _publish_atomic_release(
    tables: dict[str, Any],
    registry: Registry,
    *,
    source_frame: Any,
    source_manifest: Mapping[str, Any],
    release_receipt: list[dict[str, Any]] | None = None,
    hf_token: str,
    prior_revision: str,
    generated_at_utc: str,
    continuity_receipt: Mapping[str, Any] | None,
    repair_manifest_path: Path,
    math_region_review_path: Path = DEFAULT_REGION_REVIEW_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
    provisional_holds_path: Path = DEFAULT_PROVISIONAL_HOLDS_PATH,
    provisional_holds: Mapping[tuple[str, str], Any],
    prior_profiles: Any,
    prior_people: Any,
    atomic_intent_sink: Callable[[Mapping[str, Any]], None] | None = None,
    require_bio_release: bool = False,
    composite_plan: Any | None = None,
    prior_works: Sequence[Mapping[str, Any]] | None = None,
    prior_artifact: Mapping[str, Any] | None = None,
    approved_person_hold_losses: frozenset[tuple[str, str, str, str]] = frozenset(),
    held_topic_parent_artifact: Mapping[str, Any] | None = None,
    held_topic_parent_map_sha256: str | None = None,
    reported_faculty_coauthors_by_work: Mapping[str, list[dict[str, str]]]
    | None = None,
    enforce_presentation_drift: bool = False,
    presentation_drift_approval_path: Path | None = None,
    preserve_deletion_presentation: bool = False,
    additive_placement_review_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Stage every output, run Bio gates, then commit seven files once."""

    from huggingface_hub import hf_hub_download

    if atomic_intent_sink is None:
        raise ValueError("Live publish requires a durable prewrite intent sink")
    if additive_placement_review_path is not None and not enforce_presentation_drift:
        raise ValueError("Additive placement requires pinned presentation review")
    prior_revision = require_revision(prior_revision)
    bio_release = source_manifest["snapshot_sha256"] == BIO_SOURCE_SHA256
    if additive_placement_review_path is not None and (
        bio_release
        or composite_plan is not None
        or held_topic_parent_artifact is not None
    ):
        raise ValueError("Additive placement is restricted to ordinary releases")
    bio_source_changed = bio_scholar_source_changed(
        source_frame, prior_profiles, registry, prior_people
    )
    if (require_bio_release or bio_source_changed) and not bio_release:
        raise ValueError("Bio release source differs from the frozen snapshot")
    baselines = None
    repair_manifest = None
    if bio_release:
        baselines = load_bio_baselines(
            repo_id=REPO_ID,
            prior_revision=prior_revision,
            hf_token=hf_token,
        )
        # Carry every pending label in the immutable Hub parent. The historical
        # Bernhard manifest names 112 different decisions, while the current
        # Sullivan/Math parent contains 15 mapped pending labels.
        repair_manifest = {
            "restore_eligible_work_ids": sorted(
                str(row["work_id"])
                for row in baselines["repair_works"]
                if row.get("keyword_id") == "keyword-unclassified"
                and row.get("detail_keyword_id")
                == "keyword-unclassified-pending-review"
                and bool(row["map_eligible"])
            )
        }
        tables["works"] = preserve_pending_review_topics(
            tables["works"], baselines["repair_works"], repair_manifest
        )

    parent_map = None
    parent_map_sha256 = None
    preservation_receipt = None
    if enforce_presentation_drift:
        parent_map, parent_map_sha256 = _pinned_parent_map(
            prior_revision=prior_revision,
            hf_token=hf_token,
        )
        if additive_placement_review_path is not None:
            if presentation_drift_approval_path is not None:
                raise ValueError(
                    "Additive placement and refit approvals cannot be combined"
                )
            if prior_works is None:
                raise ValueError("Additive placement needs pinned parent works")
            public_profiles = tables["profile_publications"]
            if "author_pub_id" not in public_profiles.columns:
                raise ValueError("Additive placement lacks selected source IDs")
            selected_ids = public_profiles["author_pub_id"].astype(str).tolist()
            current_id_counts = Counter(selected_ids)
            prior_id_counts = Counter(
                str(row["author_pub_id"]) for row in prior_profiles
            )
            if (
                not selected_ids
                or public_profiles["author_pub_id"].isna().any()
                or any(not value.strip() for value in selected_ids)
                or any(
                    count > max(1, prior_id_counts[author_pub_id])
                    for author_pub_id, count in current_id_counts.items()
                )
            ):
                raise ValueError(
                    "Additive placement has missing or duplicate source IDs "
                    "beyond parent"
                )
            selected_ids_digest = ids_sha256(set(selected_ids))
            candidate_before_preservation = build_map_artifact(
                tables["works"],
                registry,
                generated_at_utc=generated_at_utc,
            )
            tables["works"], preservation_receipt = (
                preserve_reviewed_additive_presentation(
                    tables["works"],
                    prior_works,
                    parent_map,
                    candidate_before_preservation,
                    prior_revision=prior_revision,
                    source_snapshot_sha256=str(source_manifest["snapshot_sha256"]),
                    selected_ids_sha256=selected_ids_digest,
                    parent_map_sha256=parent_map_sha256,
                    review_path=additive_placement_review_path,
                )
            )
        if (
            preserve_deletion_presentation
            and additive_placement_review_path is None
            and presentation_drift_approval_path is None
        ):
            if prior_works is None:
                raise ValueError("Deletion-only preservation needs pinned parent works")
            candidate_before_preservation = build_map_artifact(
                tables["works"],
                registry,
                generated_at_utc=generated_at_utc,
                reported_faculty_coauthors_by_work=reported_faculty_coauthors_by_work,
            )
            if reported_coauthor_evidence_changed(
                parent_map, candidate_before_preservation
            ):
                parent_source_sha256 = _pinned_parent_presentation_source_sha256(
                    prior_revision=prior_revision, hf_token=hf_token
                )
                tables["works"], preservation_receipt = (
                    preserve_additive_report_presentation(
                        tables["works"],
                        prior_works,
                        parent_map,
                        candidate_before_preservation,
                        source_snapshot_sha256=str(source_manifest["snapshot_sha256"]),
                        parent_source_snapshot_sha256=parent_source_sha256,
                    )
                )
            else:
                tables["works"], preservation_receipt = (
                    preserve_deletion_only_presentation(
                        tables["works"],
                        prior_works,
                        parent_map,
                        candidate_before_preservation,
                    )
                )

    pending_publications = None
    if additive_placement_review_path is not None:
        if not isinstance(preservation_receipt, Mapping):
            raise ValueError("Additive pending publication receipt is missing")
        held_ids = preservation_receipt.get("held_new_work_ids")
        if not isinstance(held_ids, list):
            raise ValueError("Additive pending work IDs are missing")
        pending_publications = build_pending_publications(
            tables["works"],
            tables["profile_publications"],
            release_receipt or [],
            registry,
            held_ids,
        )

    with tempfile.TemporaryDirectory(prefix="map-of-research-atomic-release-") as temp:
        root = Path(temp)
        staged: dict[str, tuple[Any, Path]] = {}
        files: dict[str, Path] = {}
        for config_name in DATASET_CONFIGS:
            dataset = _to_hub_dataset(tables[config_name])
            path = root / config_name / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(str(path))
            staged[config_name] = (dataset, path)
            files[f"{config_name}/{path.name}"] = path
        if additive_placement_review_path is not None:
            require_held_additive_staged_works(
                staged["works"][1], preservation_receipt
            )
        held_table_counts = None
        if held_topic_parent_artifact is not None:
            import datasets

            if not held_topic_parent_map_sha256:
                raise ValueError("Held-only release lacks pinned parent map digest")
            tables["works"], held_table_counts = retain_held_only_parent_works(
                {name: staged[name][1] for name in DATASET_CONFIGS},
                prior_revision=prior_revision,
                hf_token=hf_token,
            )
            staged["works"] = (
                datasets.Dataset.from_parquet(str(staged["works"][1])),
                staged["works"][1],
            )
        artifact_path, map_manifest_path, artifact = _write_map_artifacts(
            tables["works"],
            registry,
            maps_dir=root / "maps",
            generated_at_utc=generated_at_utc,
            dataset_commits={},
            continuity_receipt=continuity_receipt,
            publication_hold_reasons=(
                dict(
                    zip(
                        tables["people"]["person_id"],
                        tables["people"]["publication_hold_reason"],
                        strict=True,
                    )
                )
                if {"person_id", "publication_hold_reason"}
                <= set(tables["people"].columns)
                else {}
            ),
            reported_faculty_coauthors_by_work=reported_faculty_coauthors_by_work,
            pending_publications=pending_publications,
        )
        presentation_receipt = None
        if enforce_presentation_drift:
            if additive_placement_review_path is not None:
                require_preserved_additive_artifact(
                    parent_map, artifact, preservation_receipt
                )
            presentation_report = compare_presentation(
                parent_map,
                artifact,
                prior_revision=prior_revision,
                source_snapshot_sha256=str(source_manifest["snapshot_sha256"]),
                parent_map_sha256=parent_map_sha256,
                candidate_map_sha256=sha256_file(artifact_path),
            )
            presentation_receipt = require_presentation_review(
                presentation_report,
                presentation_drift_approval_path,
            )
            map_manifest = load_json(map_manifest_path)
            if not isinstance(map_manifest, dict):
                raise ValueError("Staged map manifest is not an object")
            map_manifest["presentation_drift"] = presentation_receipt
            if preservation_receipt is not None:
                map_manifest["presentation_preservation"] = preservation_receipt
            atomic_write_json(map_manifest_path, map_manifest)
        require_public_subset_boundary(tables, release_receipt or [], artifact=artifact)
        files["maps/publications.json"] = artifact_path
        files["maps/manifest.json"] = map_manifest_path
        held_topic_receipt = None
        if held_topic_parent_artifact is not None:
            held_topic_receipt = validate_held_only_stage(
                artifact, held_topic_parent_artifact
            )
        composite_map_receipt = None
        composite_works_receipt = None
        composite_nonwork_receipt = None
        if composite_plan is not None:  # pragma: no cover - exact f8/bb88 preview
            if prior_works is None or prior_artifact is None:
                raise ValueError(
                    "Composite atomic stage lacks frozen bb88 presentation"
                )
            composite_map_receipt = validate_composite_map_points(
                load_json(artifact_path),
                prior_artifact,
                prior_works,
                prior_profiles,
                tables["profile_publications"].to_dict("records"),
                composite_plan,
                curated_dir=math_region_review_path.parent,
            )
            current_sullivan_work_ids = {
                str(row.work_id)
                for row in tables["profile_publications"].itertuples(index=False)
                if row.person_id == SULLIVAN_PERSON_ID
            }
            prior_sullivan_work_ids = {
                str(row["work_id"])
                for row in prior_profiles
                if row.get("person_id") == SULLIVAN_PERSON_ID
            }
            composite_works_receipt = validate_composite_staged_works(
                Path(
                    hf_hub_download(
                        REPO_ID,
                        "works/train-00000-of-00001.parquet",
                        repo_type="dataset",
                        revision=prior_revision,
                        token=hf_token or None,
                    )
                ),
                files["works/train-00000-of-00001.parquet"],
                plan=composite_plan,
                sullivan_work_ids=prior_sullivan_work_ids | current_sullivan_work_ids,
                current_sullivan_work_ids=current_sullivan_work_ids,
                curated_dir=math_region_review_path.parent,
            )
            composite_nonwork_receipt = validate_composite_staged_nonwork_tables(
                {
                    config_name: (
                        Path(
                            hf_hub_download(
                                REPO_ID,
                                f"{config_name}/train-00000-of-00001.parquet",
                                repo_type="dataset",
                                revision=prior_revision,
                                token=hf_token or None,
                            )
                        ),
                        files[f"{config_name}/train-00000-of-00001.parquet"],
                    )
                    for config_name in ("people", "authorships", "profile_publications")
                },
                plan=composite_plan,
                curated_dir=math_region_review_path.parent,
                exclusions_path=exclusions_path,
                holds_path=provisional_holds_path,
            )
        prior_readme = Path(
            hf_hub_download(
                REPO_ID,
                "README.md",
                repo_type="dataset",
                revision=prior_revision,
                token=hf_token,
            )
        ).read_text(encoding="utf-8")
        readme_path = root / "README.md"
        readme_path.write_text(
            build_updated_dataset_card(prior_readme, staged), encoding="utf-8"
        )
        files["README.md"] = readme_path

        if bio_release:
            gate = validate_bio_release_prewrite(
                source_frame=source_frame,
                source_manifest=source_manifest,
                registry=registry,
                tables=tables,
                candidate_artifact=artifact,
                continuity_receipt=continuity_receipt,
                prior_revision=prior_revision,
                baselines=baselines,
                repair_manifest=repair_manifest,
                provisional_holds=provisional_holds,
                approved_person_hold_losses=approved_person_hold_losses,
            )
        else:
            if (
                continuity_receipt is None
                or continuity_receipt.get("prior_revision") != prior_revision
                or continuity_receipt.get("unexplained_losses")
                or continuity_receipt.get("unexplained_gains")
            ):
                raise ValueError("Atomic release lacks a clean region-continuity gate")
            gate = {
                "status": "pass",
                "gate": "full-source-and-region-continuity-v1",
                "prior_revision": prior_revision,
                "source_snapshot_sha256": source_manifest["snapshot_sha256"],
                "raw_rows": len(source_frame),
                "map_points": artifact["point_count"],
            }
        if composite_plan is not None:  # pragma: no cover - exact f8/bb88 preview
            if (
                composite_map_receipt is None
                or composite_works_receipt is None
                or composite_nonwork_receipt is None
                or source_manifest["snapshot_sha256"] != SULLIVAN_SOURCE_SHA256
                or prior_revision != SULLIVAN_PRIOR_REVISION
            ):
                raise ValueError("Composite release lacks exact source and map parity")
            gate["gate"] = "source-bound-sullivan-reviewed-math-composite-v2"
            gate["composite_map_parity"] = composite_map_receipt
            gate["composite_works_parity"] = composite_works_receipt
            gate["composite_nonwork_parity"] = composite_nonwork_receipt
            gate["bernhard_repair_manifest_sha256"] = composite_plan.manifest_sha256
        if held_topic_receipt is not None:
            gate["held_only_topic_parity"] = {
                **held_topic_receipt,
                "parent_map_sha256": held_topic_parent_map_sha256,
                "exact_table_rows": held_table_counts,
            }
        if presentation_receipt is not None:
            gate["presentation_drift"] = presentation_receipt
            if preservation_receipt is not None:
                gate["presentation_preservation"] = preservation_receipt
        return commit_atomic_release(
            files,
            prewrite_gate=gate,
            repo_id=REPO_ID,
            hf_token=hf_token,
            prior_revision=prior_revision,
            commit_message=(
                "Publish complete Biological Sciences Scholar corpus atomically"
                if bio_release
                else "Publish central research dataset atomically"
            ),
            intent_sink=atomic_intent_sink,
            intent_metadata={
                "producer_commit": os.environ.get("MAP_PRODUCER_COMMIT"),
                "snapshot_commit": os.environ.get("MAP_SOURCE_COMMIT"),
                "workflow_url": _workflow_url(),
            },
        )


def publish_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    hf_token: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    encoder: Any | None = None,
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
    doi_aliases_path: Path = DEFAULT_DOI_ALIASES_PATH,
    attribution_reviews_path: Path = DEFAULT_ATTRIBUTION_REVIEWS_PATH,
    contexts_path: Path = DEFAULT_CONTEXTS_PATH,
    profile_releases_path: Path = DEFAULT_PROFILE_RELEASES_PATH,
    private_subset_preflight_receipt_path: Path | None = None,
    public_subset_selection_manifest_path: Path | None = None,
    person_publication_holds_path: Path = DEFAULT_PERSON_PUBLICATION_HOLDS_PATH,
    provisional_holds_path: Path = DEFAULT_PROVISIONAL_HOLDS_PATH,
    historical_loss_approval_path: Path | None = None,
    bio_binding_approval_path: Path | None = None,
    source_cache_dir: Path | None = None,
    primary_home_baseline_path: Path | None = None,
    prior_revision: str | None = None,
    region_comparison_revision: str | None = None,
    atomic_release: bool | None = None,
    atomic_intent_sink: Callable[[Mapping[str, Any]], None] | None = None,
    require_bio_release: bool = False,
    repair_manifest_path: Path = Path("data/curated/region-remediation-bernhard.json"),
    source_commit: str | None = None,
    sullivan_loss_approval_path: Path = DEFAULT_SULLIVAN_LOSS_APPROVAL_PATH,
    sullivan_continuity_addendum_path: Path = DEFAULT_SULLIVAN_CONTINUITY_ADDENDUM_PATH,
    sullivan_crosswalk_path: Path = DEFAULT_SULLIVAN_CROSSWALK_PATH,
    mccomb_continuity_path: Path = DEFAULT_MCCOMB_CONTINUITY_PATH,
    mccomb_crosswalk_path: Path = DEFAULT_MCCOMB_CROSSWALK_PATH,
    coauthor_binding_approval_path: Path = DEFAULT_COAUTHOR_BINDING_APPROVAL_PATH,
    presentation_drift_approval_path: Path | None = None,
    additive_placement_review_path: Path | None = None,
    curated_works_path: Path = DEFAULT_CURATED_WORKS_PATH,
    math_region_review_path: Path = DEFAULT_REGION_REVIEW_PATH,
    chem_region_review_path: Path = DEFAULT_CHEM_REGION_REVIEW_PATH,
) -> dict[str, Any]:
    """Publish all normalized dataset configs and work-centric map artifacts."""

    if prior_revision is None:
        raise ValueError("Live publish requires an immutable prior Hub revision")
    require_revision(prior_revision)
    if (
        additive_placement_review_path is not None
        and region_comparison_revision not in (None, prior_revision)
    ):
        raise ValueError("Additive placement requires the exact parent comparison")
    if atomic_release is False:
        raise ValueError("Live publish requires one atomic Hub commit")
    if atomic_intent_sink is None:
        raise ValueError("Live publish requires a durable prewrite intent sink")
    _require_current_dataset_revision(hf_token=hf_token, prior_revision=prior_revision)
    registry = load_registry(people_path, memberships_path, departments_path)
    validate_included_scholar_primary_homes(
        registry,
        legacy_implicit_primary=implicit_primary_baseline_for_people(
            people_path, primary_home_baseline_path
        ),
    )
    frame, manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        max_age_days=max_age_days,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        curated_path=curated_works_path,
    )
    sullivan_transition = (
        prior_revision == SULLIVAN_PRIOR_REVISION
        and manifest["snapshot_sha256"] == SULLIVAN_SOURCE_SHA256
    )
    if sullivan_transition and not source_commit:
        raise ValueError("Sullivan release needs its exact source commit")
    frozen_bio_source = manifest["snapshot_sha256"] == BIO_SOURCE_SHA256
    if require_bio_release and not frozen_bio_source:
        raise ValueError("Bio release source differs from the frozen snapshot")
    if region_comparison_revision is not None:
        require_approved_comparison_revision(
            current_revision=prior_revision,
            comparison_revision=region_comparison_revision,
            snapshot_sha256=str(manifest["snapshot_sha256"]),
        )
    exclusions = load_exclusions(exclusions_path)
    doi_aliases = load_doi_aliases(doi_aliases_path)
    validate_doi_aliases(frame, doi_aliases, exclusions)
    corrections = load_corrections(corrections_path)
    corrected_frame, validated_correction_count = apply_corrections(frame, corrections)
    contexts = load_contexts(contexts_path)
    attribution_reviews = load_attribution_reviews(attribution_reviews_path)
    if set(attribution_reviews.confirmed).intersection(exclusions):
        raise ValueError("Scholar attribution and exclusion reviews conflict")
    release_decisions = load_profile_releases(profile_releases_path)
    release_manifest_sha256 = sha256_file(profile_releases_path)
    person_holds = load_person_publication_holds(person_publication_holds_path)
    person_hold_manifest_sha256 = sha256_file(person_publication_holds_path)
    provisional_profiles = load_provisional_holds(provisional_holds_path)
    provisional_manifest_sha256 = sha256_file(provisional_holds_path)
    cache_dir = source_cache_dir or snapshot_path.parent.parent / "data" / "authors"
    private_subset_preflight = require_private_preflight_receipt(
        receipt_path=private_subset_preflight_receipt_path,
        selection_manifest_path=public_subset_selection_manifest_path,
        snapshot_path=snapshot_path,
        cache_dir=cache_dir,
        release_decisions=release_decisions,
    )
    released_frame, release_receipt, held_observation_ids = (
        select_released_observations(
            corrected_frame,
            registry,
            release_decisions,
            cache_dir=cache_dir,
        )
    )
    # Exact cache-bound holds remove these rows before their historical
    # map-quality gates; any unheld row still has to pass its source review.
    _require_physics_review_ready(released_frame, snapshot_path, exclusions)
    _require_breivik_review_ready(released_frame, snapshot_path, exclusions)
    released_frame, person_hold_receipt, held_person_observation_ids = (
        select_released_person_observations(
            released_frame,
            registry,
            person_holds,
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            prior_revision=prior_revision,
        )
    )
    provisional_holds, provisional_receipt = validate_provisional_holds(
        provisional_profiles,
        release_decisions,
        release_receipt,
        cache_dir=cache_dir,
        excluded_ids=set(exclusions),
        confirmed_ids=set(attribution_reviews.confirmed),
    )
    if (  # pragma: no cover - exact f8/bb88 manifest bytes
        sullivan_transition
        and sha256_file(math_region_review_path) != MATH_REVIEW_MANIFEST_SHA256
    ):
        raise ValueError("Math region review manifest bytes changed")
    reviewed_math_decisions = (
        load_region_review(
            math_region_review_path,
            snapshot_frame=frame,
            snapshot_sha256=str(manifest["snapshot_sha256"]),
            cache_dir=cache_dir,
            registry=registry,
            exclusions=set(exclusions),
            holds=set(provisional_holds),
        )
        if sullivan_transition
        else ()
    )
    composite_plan = (
        load_composite_region_plan(
            repair_manifest_path,
            reviewed_math_decisions,
            source_commit=source_commit or "",
            source_sha256=str(manifest["snapshot_sha256"]),
            prior_revision=prior_revision,
        )
        if sullivan_transition
        else None
    )
    poster_loss_approval = None
    if historical_loss_approval_path is not None:
        if not frozen_bio_source:
            raise ValueError("McManus poster approval requires the frozen Bio source")
        poster_loss_approval = load_poster_loss_approval(
            historical_loss_approval_path,
            prior_revision=prior_revision,
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            profile_releases_sha256=release_manifest_sha256,
            provisional_holds_sha256=provisional_manifest_sha256,
            release_decisions=release_decisions,
            validated_holds=provisional_holds,
        )
    chemistry_parent_state = None
    chemistry_decisions = ()
    if needs_chemistry_continuity(
        set(frame["scholar_id"].dropna().astype(str)),
        prior_revision,
        composite=composite_plan is not None,
        snapshot_sha256=str(manifest["snapshot_sha256"]),
    ):
        chemistry_decisions = load_chem_region_review(
            chem_region_review_path,
            snapshot_frame=frame,
            snapshot_sha256=str(manifest["snapshot_sha256"]),
            cache_dir=cache_dir,
            registry=registry,
            require_frozen_snapshot=False,
        )
        chemistry_parent_state = load_chemistry_parent(
            chemistry_decisions, prior_revision, hf_token
        )
        exclusions, provisional_holds, contexts = merge_chemistry_review_inputs(
            chemistry_decisions, exclusions, provisional_holds, contexts
        )
    published_correction_count = int(
        released_frame["metadata_correction_fields"].ne("").sum()
    )
    reviewed_new_vectors = None
    if composite_plan is not None:  # pragma: no cover - exact f8/bb88 release
        reviewed_new_vectors = load_reviewed_sullivan_embeddings(
            math_region_review_path.parent,
            source_commit=source_commit or "",
            source_sha256=str(manifest["snapshot_sha256"]),
            hub_parent=prior_revision,
        )
    enriched, new_embedding_count = add_embeddings(
        released_frame,
        hf_token=hf_token,
        encoder=encoder,
        prior_revision=prior_revision,
        reviewed_new_vectors=reviewed_new_vectors,
    )
    tables = build_dataset_tables(
        enriched,
        registry,
        exclusions,
        doi_aliases,
        attribution_reviews,
        contexts,
        provisional_holds,
    )
    person_hold_reasons = {
        item["person_id"]: item["display_reason"] for item in person_hold_receipt
    }
    tables["people"]["publication_hold_reason"] = (
        tables["people"]["person_id"].map(person_hold_reasons).fillna("")
        if person_hold_reasons
        else ""
    )
    if chemistry_parent_state is not None and prior_revision == CHEMISTRY_HUB_REVISION:
        tables["people"] = reconcile_chemistry_person_counts(
            tables["people"], chemistry_decisions
        )
    existing_layouts = _existing_layouts(
        hf_token=hf_token,
        revision=prior_revision,
    )
    layout_reused = False
    if existing_layouts:
        reuse = _reusable_work_order(
            tables["works"],
            _existing_work_state(
                hf_token=hf_token,
                revision=prior_revision,
            ),
            existing_layouts,
            doi_aliases,
        )
        if reuse is not None:
            tables["works"], existing_layouts = reuse
            layout_reused = True
    tables["works"] = add_global_layout(
        tables["works"],
        existing_layouts=existing_layouts,
    )
    tables["works"] = add_publication_keywords(tables["works"])
    tables["works"] = exclude_low_information_regions(
        tables["works"], doi_aliases=doi_aliases
    )
    producer_reports_path = (
        people_path.parent.parent / DEFAULT_REPORTED_FACULTY_COAUTHORS_PATH
    )
    frozen_reports_path = (
        snapshot_path.parent.parent / DEFAULT_REPORTED_FACULTY_COAUTHORS_PATH
    )
    has_frozen_reports = require_reported_source_sync(
        producer_reports_path, frozen_reports_path
    )
    reported_coauthors_by_work, reported_coauthor_receipt = (
        apply_reported_faculty_coauthors(
            tables["works"],
            tables["profile_publications"],
            tables["authorships"],
            registry,
            source_path=frozen_reports_path,
            source_sha256=str(manifest.get("reported_faculty_coauthors_sha256", "")),
            held_person_ids=frozenset(item.person_id for item in person_holds),
        )
        if has_frozen_reports
        else ({}, [])
    )
    comparison_revision = region_comparison_revision or prior_revision
    chem_carryforward = (
        comparison_revision == CHEM_CARRYFORWARD_PARENT
        and manifest["snapshot_sha256"] == CHEM_CARRYFORWARD_SOURCE_SHA256
    )
    prior_work_fields = tuple(
        dict.fromkeys(
            (
                *PRIOR_WORK_FIELDS,
                *SCS_TOPIC_HOLD_PRIOR_WORK_FIELDS,
                *(CHEM_CARRYFORWARD_WORK_FIELDS if chem_carryforward else ()),
                *(
                    ("map_exclusion_reasons",)
                    if (
                        comparison_revision != prior_revision
                        or composite_plan is not None
                    )
                    else ()
                ),
                *COMPOSITE_PRESENTATION_FIELDS,
                *(TOPIC_FIELDS if held_observation_ids else ()),
                *(COMPOSITE_VARIANT_FIELDS if composite_plan is not None else ()),
                *(
                    ("map_region_documented_override",)
                    if composite_plan is not None
                    else ()
                ),
            )
        )
    )
    prior_works = list(
        _published_rows(
            "works",
            prior_work_fields,
            hf_token=hf_token,
            revision=comparison_revision,
        )
    )
    prior_profiles = list(
        _published_rows(
            "profile_publications",
            tuple(
                dict.fromkeys(
                    (
                        *PRIOR_OBSERVATION_FIELDS,
                        *(
                            CHEM_CARRYFORWARD_PROFILE_FIELDS
                            if chem_carryforward
                            else ()
                        ),
                    )
                )
            ),
            hf_token=hf_token,
            revision=comparison_revision,
        )
    )
    prior_people = list(
        _published_rows(
            "people",
            ("person_id", "included_department_ids"),
            hf_token=hf_token,
            revision=prior_revision,
        )
    )
    if not prior_works or not prior_profiles:
        raise RuntimeError("Pinned prior dataset lacks region continuity tables")
    approved_person_hold_losses = validate_person_hold_losses(
        person_holds,
        prior_revision=prior_revision,
        prior_works=prior_works,
        prior_profiles=prior_profiles,
        current_profiles=tables["profile_publications"].to_dict("records"),
    )
    prior_artifact = None
    if composite_plan is not None:  # pragma: no cover - exact f8/bb88 preview
        from huggingface_hub import hf_hub_download

        prior_artifact = require_baseline_map(
            Path(
                hf_hub_download(
                    REPO_ID,
                    f"maps/{MAP_ARTIFACT_NAME}",
                    repo_type="dataset",
                    revision=prior_revision,
                    token=hf_token or None,
                )
            )
        )
    historical_works = (
        {
            str(row["work_id"]): row
            for row in _published_rows(
                "works",
                ("work_id", "map_eligible", "map_region_outlier"),
                hf_token=hf_token,
                revision=COMPOSITE_HISTORICAL_REVISION,
            )
            if str(row["work_id"]) in composite_plan.restore | composite_plan.reexclude
        }
        if composite_plan is not None
        else None
    )
    sullivan_loss_approval = (
        validate_sullivan_exact_losses(
            approval_path=sullivan_loss_approval_path,
            prior_revision=prior_revision,
            source_snapshot_commit=source_commit or "",
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            cache_path=cache_dir / "wXSgFr4AAAAJ.json",
            profile_releases_path=profile_releases_path,
            provisional_holds_path=provisional_holds_path,
            crosswalk_path=sullivan_crosswalk_path,
            curated_works_path=curated_works_path,
            exclusions_path=exclusions_path,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            source_rows=frame.to_dict("records"),
            current_profiles=tables["profile_publications"].to_dict("records"),
            validated_holds=provisional_holds,
        )
        if sullivan_transition
        else None
    )
    sullivan_continuity_addendum = (
        validate_sullivan_continuity_addendum(
            addendum_path=sullivan_continuity_addendum_path,
            prior_revision=prior_revision,
            source_snapshot_commit=source_commit or "",
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            source_rows=frame.to_dict("records"),
            current_works=tables["works"].to_dict("records"),
            current_profiles=tables["profile_publications"].to_dict("records"),
        )
        if sullivan_transition
        else None
    )
    bio_binding_approval = (
        load_bio_binding_approval(
            approval_path=bio_binding_approval_path,
            prior_revision=prior_revision,
            source_snapshot_commit=source_commit or "",
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            doi_aliases_path=doi_aliases_path,
            exclusions_path=exclusions_path,
            person_publication_holds_path=person_publication_holds_path,
            curated_works_path=curated_works_path,
            profile_releases_path=profile_releases_path,
            provisional_holds_path=provisional_holds_path,
            poster_loss_approval_path=historical_loss_approval_path,
        )
        if bio_binding_approval_path is not None
        else None
    )
    mccomb_continuity = (
        validate_exact_mccomb_continuity(
            mccomb_continuity_path,
            prior_revision=prior_revision,
            source_snapshot_commit=source_commit or "",
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            cache_path=cache_dir / "0P9w_S0AAAAJ.json",
            crosswalk_path=mccomb_crosswalk_path,
            prior_profiles=prior_profiles,
            current_profiles=tables["profile_publications"].to_dict("records"),
        )
        if prior_revision == MCCOMB_CONTINUITY_PRIOR_REVISION
        and manifest["snapshot_sha256"] == MCCOMB_CONTINUITY_SNAPSHOT_SHA256
        else None
    )
    coauthor_binding_approval = (
        validate_exact_coauthor_bindings(
            coauthor_binding_approval_path,
            prior_revision=prior_revision,
            source_snapshot_sha256=str(manifest["snapshot_sha256"]),
            curated_works_path=curated_works_path,
            mccomb_cache_path=cache_dir / "0P9w_S0AAAAJ.json",
            martelaro_review_path=(
                people_path.parent / "reviews" / DEFAULT_MARTELARO_REVIEW_PATH.name
            ),
            akinci_review_path=(
                people_path.parent / "reviews" / DEFAULT_AKINCI_REVIEW_PATH.name
            ),
            prior_profiles=prior_profiles,
            current_profiles=tables["profile_publications"].to_dict("records"),
        )
        if prior_revision == COAUTHOR_BINDING_PRIOR_REVISION
        and manifest["snapshot_sha256"] == COAUTHOR_BINDING_SNAPSHOT_SHA256
        else None
    )
    tables["works"], scs_four_topic_holds = apply_scs_four_topic_holds(
        tables["works"],
        tables["profile_publications"],
        prior_works,
        prior_profiles,
        prior_revision=prior_revision,
        snapshot_sha256=str(manifest["snapshot_sha256"]),
        comparison_revision=comparison_revision,
    )
    tables["works"], continuity_receipt = reconcile_region_decisions(
        tables["works"],
        tables["profile_publications"],
        prior_works,
        prior_profiles,
        prior_revision=comparison_revision,
        sullivan_loss_approval=sullivan_loss_approval,
        sullivan_continuity_addendum=sullivan_continuity_addendum,
        bio_binding_approval=bio_binding_approval,
        mccomb_continuity=mccomb_continuity,
        coauthor_binding_approval=coauthor_binding_approval,
        approved_person_hold_losses=approved_person_hold_losses,
        composite_plan=composite_plan,
        historical_works=historical_works,
        approved_poster_loss=poster_loss_approval,
    )
    if scs_four_topic_holds is not None:  # pragma: no cover - exact held works
        continuity_receipt["scs_four_topic_holds"] = scs_four_topic_holds
    if chem_carryforward:  # pragma: no cover - exact dcb/W3 release
        chem_receipt = validate_chem_carryforward(
            prior_revision=comparison_revision,
            source_commit=source_commit or "",
            source_sha256=str(manifest["snapshot_sha256"]),
            source_rows=frame.to_dict("records"),
            prior_profiles=prior_profiles,
            prior_works=prior_works,
            prior_authorships=_published_rows(
                "authorships",
                CHEM_CARRYFORWARD_AUTHORSHIP_FIELDS,
                hf_token=hf_token,
                revision=comparison_revision,
            ),
            current_profiles=tables["profile_publications"].to_dict("records"),
            current_works=tables["works"].to_dict("records"),
            current_authorships=tables["authorships"].to_dict("records"),
            review_path=math_region_review_path.with_name(
                "scholar-chem-region-review.json"
            ),
            exclusions_path=exclusions_path,
            holds_path=provisional_holds_path,
            contexts_path=contexts_path,
            releases_path=profile_releases_path,
        )
        continuity_receipt["chemistry_review_carryforward"] = chem_receipt
    bio_source_changed = bio_scholar_source_changed(
        frame, prior_profiles, registry, prior_people
    )
    if bio_source_changed and not frozen_bio_source:
        raise ValueError("Changed Bio Scholar source needs the reviewed Bio snapshot")
    if comparison_revision != prior_revision:
        continuity_receipt["current_hub_revision"] = prior_revision
    if composite_plan is not None:  # pragma: no cover - exact f8/bb88 preview
        continuity_receipt["composite_region_manifest_sha256"] = (
            composite_plan.manifest_sha256
        )
        continuity_receipt["math_region_review_sha256"] = sha256_file(
            math_region_review_path
        )
    if chemistry_parent_state is not None:
        continuity_receipt["chem_region_review_sha256"] = CHEM_REVIEW_SHA256
        continuity_receipt["chemistry_review_root_revision"] = CHEMISTRY_HUB_REVISION
    tables["works"] = add_publication_keywords(tables["works"])
    if chemistry_parent_state is not None:
        validate_chemistry_candidate(
            tables,
            chemistry_parent_state,
            allow_region_refit=not layout_reused,
            require_count_parity=bool(
                held_observation_ids
                and held_only_continuity_candidate(continuity_receipt, prior_revision)
            ),
        )
    if composite_plan is not None:  # pragma: no cover - exact f8/bb88 preview
        if prior_artifact is None:
            raise ValueError("Composite release lacks frozen bb88 map")
        tables["works"] = freeze_composite_presentation(
            tables["works"],
            prior_works,
            prior_artifact,
            composite_plan,
            {
                str(row.work_id)
                for row in tables["profile_publications"].itertuples(index=False)
                if row.person_id == SULLIVAN_PERSON_ID
            },
        )
    require_no_held_contributions(tables, release_decisions, held_observation_ids)
    require_no_held_person_contributions(
        tables,
        {item["person_id"] for item in person_hold_receipt},
        held_person_observation_ids,
    )
    require_no_provisional_hold_contributions(tables, provisional_holds)
    require_public_subset_boundary(tables, release_receipt)
    held_topic_parent_artifact = None
    held_topic_parent_map_sha256 = None
    if (
        held_observation_ids
        and composite_plan is None
        and not bio_source_changed
        and manifest["snapshot_sha256"] != BIO_SOURCE_SHA256
        and comparison_revision == prior_revision
        and held_only_continuity_candidate(continuity_receipt, prior_revision)
    ):
        from huggingface_hub import hf_hub_download

        parent_map_path = Path(
            hf_hub_download(
                REPO_ID,
                f"maps/{MAP_ARTIFACT_NAME}",
                repo_type="dataset",
                revision=prior_revision,
                token=hf_token or None,
            )
        )
        held_topic_parent_artifact = load_json(parent_map_path)
        held_topic_parent_map_sha256 = sha256_file(parent_map_path)
        tables["works"] = preserve_held_only_topics(
            tables["works"],
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=tables["profile_publications"],
            prior_artifact=held_topic_parent_artifact,
            candidate_artifact=build_map_artifact(
                tables["works"],
                registry,
                generated_at_utc=str(manifest["created_at_utc"]),
                reported_faculty_coauthors_by_work=reported_coauthors_by_work,
            ),
            held_observation_ids=held_observation_ids,
        )
    if comparison_revision != prior_revision:
        require_historical_region_decisions(prior_works, tables["works"])
        repair_related_mask = (
            tables["works"]["has_included_source_profile"]
            if "has_included_source_profile" in tables["works"]
            else tables["works"]["department_ids"].apply(
                lambda value: bool(_as_list(value))
            )
        )
        repair_mapped_mask = (
            repair_related_mask
            & tables["works"]["map_eligible"]
            & tables["works"]["memberships"].apply(lambda value: bool(_as_list(value)))
        )
        require_expected_repair_preview(
            work_count=len(tables["works"]),
            map_point_count=int(repair_mapped_mask.sum()),
            new_embedding_count=new_embedding_count,
            layout_reused=layout_reused,
            continuity_receipt=continuity_receipt,
        )

    _require_current_dataset_revision(hf_token=hf_token, prior_revision=prior_revision)
    if comparison_revision != prior_revision:
        require_frozen_source_tip(snapshot_path.parent.parent)
        require_frozen_producer_tip(
            snapshot_path.parent.parent,
            os.environ.get("MAP_PRODUCER_COMMIT", ""),
        )
    generated_at_utc = str(manifest["created_at_utc"])
    if comparison_revision != prior_revision:
        raise ValueError(
            "Ordinary atomic upload needs the current immutable Hub revision; "
            "historical region repair requires its dedicated atomic path"
        )
    artifact_commit, atomic_receipt = _publish_atomic_release(
        tables,
        registry,
        source_frame=frame,
        source_manifest=manifest,
        release_receipt=release_receipt,
        hf_token=hf_token,
        prior_revision=prior_revision,
        generated_at_utc=generated_at_utc,
        continuity_receipt=continuity_receipt,
        repair_manifest_path=repair_manifest_path,
        math_region_review_path=math_region_review_path,
        exclusions_path=exclusions_path,
        provisional_holds_path=provisional_holds_path,
        provisional_holds=provisional_holds,
        prior_profiles=prior_profiles,
        prior_people=prior_people,
        atomic_intent_sink=atomic_intent_sink,
        require_bio_release=require_bio_release,
        composite_plan=composite_plan,
        prior_works=prior_works,
        prior_artifact=prior_artifact,
        approved_person_hold_losses=approved_person_hold_losses,
        held_topic_parent_artifact=held_topic_parent_artifact,
        held_topic_parent_map_sha256=held_topic_parent_map_sha256,
        reported_faculty_coauthors_by_work=reported_coauthors_by_work,
        enforce_presentation_drift=True,
        presentation_drift_approval_path=presentation_drift_approval_path,
        preserve_deletion_presentation=True,
        additive_placement_review_path=additive_placement_review_path,
    )
    dataset_commits = dict.fromkeys(DATASET_CONFIGS, artifact_commit)
    related_mask = (
        tables["works"]["has_included_source_profile"]
        if "has_included_source_profile" in tables["works"]
        else tables["works"]["department_ids"].apply(
            lambda value: bool(_as_list(value))
        )
    )
    mapped_mask = related_mask & tables["works"]["map_eligible"]
    return {
        "dataset_commits": dataset_commits,
        "artifact_commit": artifact_commit,
        "atomic_release": atomic_receipt,
        "raw_snapshot_observations": len(frame),
        "raw_snapshot_profiles": int(
            frame.loc[frame["source_type"] == "scholar", "scholar_id"].nunique()
        ),
        "raw_snapshot_sha256": manifest["snapshot_sha256"],
        "profile_release_manifest_sha256": release_manifest_sha256,
        "profile_release_decisions": release_receipt,
        "private_subset_preflight": private_subset_preflight,
        "person_publication_hold_manifest_sha256": person_hold_manifest_sha256,
        "person_publication_holds": person_hold_receipt,
        "reported_faculty_coauthors": reported_coauthor_receipt,
        "reported_faculty_coauthor_count": len(reported_coauthor_receipt),
        "held_person_observation_rows": sum(
            item["held_source_rows"] for item in person_hold_receipt
        ),
        "provisional_hold_manifest_sha256": provisional_manifest_sha256,
        "provisional_hold_profiles": provisional_receipt,
        "provisional_hold_observation_rows": len(provisional_holds),
        "held_profile_ids": [
            item["scholar_id"] for item in release_receipt if item["state"] == "held"
        ],
        "held_observation_rows": sum(
            item["person_observation_rows"]
            for item in release_receipt
            if item["state"] == "held"
        ),
        "held_source_observation_rows": sum(
            item["source_observation_rows"]
            for item in release_receipt
            if item["state"] == "held"
        ),
        "people": len(tables["people"]),
        "works": len(tables["works"]),
        "mapped_works": int(mapped_mask.sum()),
        "excluded_works": int((related_mask & ~mapped_mask).sum()),
        "keywords": int(tables["works"].loc[mapped_mask, "keyword_id"].nunique()),
        "detail_keywords": int(
            tables["works"].loc[mapped_mask, "detail_keyword_id"].nunique()
        ),
        "authorships": len(tables["authorships"]),
        "profile_publications": len(tables["profile_publications"]),
        "reviewed_exclusions": int(
            tables["profile_publications"]["review_exclusion_code"].ne("").sum()
            if "review_exclusion_code" in tables["profile_publications"]
            else 0
        ),
        "reviewed_exclusions_sha256": sha256_file(exclusions_path),
        "reviewed_metadata_corrections": published_correction_count,
        "validated_metadata_corrections": validated_correction_count,
        "reviewed_metadata_corrections_sha256": sha256_file(corrections_path),
        "reviewed_doi_aliases": len(doi_aliases),
        "reviewed_doi_aliases_sha256": sha256_file(doi_aliases_path),
        "reviewed_contexts": int(
            tables["profile_publications"]["context_verified_at"].ne("").sum()
        ),
        "reviewed_contexts_sha256": sha256_file(contexts_path),
        "attribution_reviews_sha256": sha256_file(attribution_reviews_path),
        "attribution_confirmed": int(
            tables["profile_publications"]["review_attribution_status"]
            .eq("confirmed")
            .sum()
            if "review_attribution_status" in tables["profile_publications"]
            else 0
        ),
        "attribution_held": int(
            tables["profile_publications"]["review_attribution_status"].eq("held").sum()
            if "review_attribution_status" in tables["profile_publications"]
            else 0
        ),
        "profile_count": int(
            enriched.loc[
                enriched.get("source_type", "scholar") == "scholar", "scholar_id"
            ].nunique()
            if "source_type" in enriched
            else enriched["scholar_id"].nunique()
        ),
        "new_embedding_count": new_embedding_count,
        "map_artifact_count": 1,
        "layout_version": LAYOUT_VERSION,
        "region_audit_version": REGION_AUDIT_VERSION,
        "topic_label_review_version": TOPIC_LABEL_REVIEW_VERSION,
        "region_continuity": continuity_receipt,
        "region_comparison_revision": comparison_revision,
    }


def _workflow_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def _persist_atomic_intent(
    intent: Mapping[str, Any], *, status: dict[str, Any], status_path: Path
) -> None:
    """Checkpoint staged write intent before the Hub call can start."""

    status.update(
        {
            "status": "failure",
            "error_type": "AtomicWriteOutcomeUnknown",
            "error_message": "Inspect the atomic release intent before retrying",
            "atomic_release_intent": dict(intent),
        }
    )
    atomic_write_json(status_path, status)
    repo_name = os.environ.get("MAP_DURABLE_INTENT_REPO")
    if not repo_name:
        return
    context = intent.get("release_context") or {}
    if not isinstance(context, Mapping):
        raise ValueError("Atomic intent lacks release context")
    for field in ("producer_commit", "snapshot_commit"):
        require_revision(str(context.get(field) or ""))
    repo = Path(repo_name).resolve(strict=True)
    branch = os.environ.get("MAP_DURABLE_INTENT_BRANCH")
    if branch != "automation/map-snapshot":
        raise ValueError("Atomic intent checkpoint requires the snapshot branch")
    try:
        relative = status_path.resolve().relative_to(repo)
    except ValueError as error:
        raise ValueError(
            "Atomic intent status file is outside snapshot checkout"
        ) from error
    if relative.as_posix() != "status/last-upload.json":
        raise ValueError("Atomic intent must checkpoint status/last-upload.json")
    current_tip = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_tip = subprocess.run(
        ["git", "-C", str(repo), "ls-remote", "--heads", "origin", branch],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not remote_tip or remote_tip[0] != current_tip:
        raise RuntimeError("Snapshot branch moved before atomic intent checkpoint")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit",
            "--only",
            "-m",
            "Record atomic release intent [skip ci]",
            "--",
            relative.as_posix(),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "push", "origin", f"HEAD:{branch}"],
        check=True,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prior-revision",
        required=True,
        type=require_revision,
        help="Immutable SHA of the dataset release this upload replaces",
    )
    parser.add_argument(
        "--region-comparison-revision",
        type=require_revision,
        help="Exact region baseline; defaults to the current Hub revision",
    )
    parser.add_argument(
        "--repair-reviewed-work-manifest",
        type=Path,
        default=Path("data/curated/region-remediation-bernhard.json"),
        help="Reviewed 112-work topic assignments preserved during the Bio release",
    )
    parser.add_argument(
        "--math-region-review-manifest",
        type=Path,
        default=DEFAULT_REGION_REVIEW_PATH,
        help="Exact 31-row Math review applied within the f8 composite release",
    )
    parser.add_argument(
        "--chem-region-review-manifest",
        type=Path,
        default=DEFAULT_CHEM_REGION_REVIEW_PATH,
        help="Exact 19-row Chemistry review retained in ordinary snapshots",
    )
    parser.add_argument(
        "--require-bio-release",
        action="store_true",
        help="Fail closed unless the frozen complete Bio snapshot passes its gate",
    )
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument("--primary-home-baseline", type=Path)
    parser.add_argument(
        "--memberships",
        type=Path,
        default=DEFAULT_MEMBERSHIPS_PATH,
    )
    parser.add_argument(
        "--departments",
        type=Path,
        default=DEFAULT_DEPARTMENTS_PATH,
    )
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument(
        "--reviewed-exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH
    )
    parser.add_argument(
        "--reviewed-metadata-corrections",
        type=Path,
        default=DEFAULT_CORRECTIONS_PATH,
    )
    parser.add_argument(
        "--reviewed-doi-aliases", type=Path, default=DEFAULT_DOI_ALIASES_PATH
    )
    parser.add_argument(
        "--attribution-reviews", type=Path, default=DEFAULT_ATTRIBUTION_REVIEWS_PATH
    )
    parser.add_argument("--reviewed-contexts", type=Path, default=DEFAULT_CONTEXTS_PATH)
    parser.add_argument(
        "--profile-releases", type=Path, default=DEFAULT_PROFILE_RELEASES_PATH
    )
    parser.add_argument(
        "--private-subset-preflight-receipt",
        type=Path,
        help="Controlled local audit receipt for reviewed public subsets",
    )
    parser.add_argument(
        "--public-subset-selection-manifest",
        type=Path,
        help="Public selected-ID manifest pinned by the local audit receipt",
    )
    parser.add_argument(
        "--person-publication-holds",
        type=Path,
        default=DEFAULT_PERSON_PUBLICATION_HOLDS_PATH,
    )
    parser.add_argument(
        "--provisional-holds", type=Path, default=DEFAULT_PROVISIONAL_HOLDS_PATH
    )
    parser.add_argument(
        "--approved-poster-loss",
        type=Path,
        help="Reviewed parent-pinned approval for one McManus poster map-point loss",
    )
    parser.add_argument(
        "--bio-binding-approval",
        type=Path,
        help="Reviewed parent-pinned Bio work-binding changes",
    )
    parser.add_argument(
        "--mccomb-continuity-approval",
        type=Path,
        default=DEFAULT_MCCOMB_CONTINUITY_PATH,
        help="Exact source-pinned McComb rekeys and corrected-year binding",
    )
    parser.add_argument(
        "--coauthor-binding-approval",
        type=Path,
        default=DEFAULT_COAUTHOR_BINDING_APPROVAL_PATH,
        help="Exact Hub5073 McComb work bindings backed by curated coauthors",
    )
    parser.add_argument(
        "--presentation-drift-approval",
        type=Path,
        help="Exact-parent reviewed approval for material map refit",
    )
    parser.add_argument(
        "--additive-placement-review",
        type=Path,
        help="Review placement or map hold for each new work point",
    )
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--math-cache-dir", type=Path)
    parser.add_argument("--math-capture-dir", type=Path)
    parser.add_argument("--scholar-cache-dir", type=Path)
    parser.add_argument("--scholar-captures-dir", type=Path)
    parser.add_argument("--scholar-baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument(
        "--scholar-curated-works", type=Path, default=DEFAULT_CURATED_WORKS_PATH
    )
    parser.add_argument(
        "--scholar-prior-id-crosswalk",
        type=Path,
        default=DEFAULT_PRIOR_ID_CROSSWALK_PATH,
    )
    parser.add_argument(
        "--scholar-additional-prior-id-crosswalk",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sullivan-loss-approval",
        type=Path,
        default=DEFAULT_SULLIVAN_LOSS_APPROVAL_PATH,
    )
    parser.add_argument(
        "--sullivan-continuity-addendum",
        type=Path,
        default=DEFAULT_SULLIVAN_CONTINUITY_ADDENDUM_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = _parse_args(argv)
    started_at = dt.datetime.now(dt.UTC)
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "dataset_id": REPO_ID,
        "dataset_configs": list(DATASET_CONFIGS),
        "workflow_url": _workflow_url(),
        "producer_commit": os.environ.get("MAP_PRODUCER_COMMIT"),
        "snapshot_commit": os.environ.get("MAP_SOURCE_COMMIT"),
        "prior_revision": args.prior_revision,
        "region_comparison_revision": (
            args.region_comparison_revision or args.prior_revision
        ),
    }
    try:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if not hf_token:
            raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")
        registry = load_registry(args.people, args.memberships, args.departments)
        frame, _ = validate_snapshot(
            args.snapshot,
            args.manifest,
            max_age_days=args.max_age_days,
            people_path=args.people,
            memberships_path=args.memberships,
            departments_path=args.departments,
        )
        snapshot_root = args.snapshot.parent.parent
        math_coverage = validate_math_coverage(
            frame,
            registry,
            cache_dir=args.math_cache_dir or snapshot_root / "data/authors",
            capture_dir=args.math_capture_dir or snapshot_root / "data/captures",
        )
        status["math_coverage"] = math_coverage
        scholar_coverage = validate_coverage(
            people_path=args.people,
            memberships_path=args.memberships,
            departments_path=args.departments,
            cache_dir=args.scholar_cache_dir or snapshot_root / "data/authors",
            captures_dir=args.scholar_captures_dir or snapshot_root / "data/captures",
            baseline_path=args.scholar_baseline,
            prior_id_crosswalk_path=args.scholar_prior_id_crosswalk,
            additional_prior_id_crosswalk_paths=tuple(
                args.scholar_additional_prior_id_crosswalk
            ),
            curated_works_path=args.scholar_curated_works,
            exclusions_path=args.reviewed_exclusions,
            snapshot_path=args.snapshot,
        )
        status["scholar_coverage"] = scholar_coverage

        def checkpoint(intent: Mapping[str, Any]) -> None:
            _persist_atomic_intent(intent, status=status, status_path=args.status_file)

        result = publish_snapshot(
            args.snapshot,
            args.manifest,
            hf_token=hf_token,
            max_age_days=args.max_age_days,
            people_path=args.people,
            memberships_path=args.memberships,
            departments_path=args.departments,
            exclusions_path=args.reviewed_exclusions,
            corrections_path=args.reviewed_metadata_corrections,
            doi_aliases_path=args.reviewed_doi_aliases,
            attribution_reviews_path=args.attribution_reviews,
            contexts_path=args.reviewed_contexts,
            profile_releases_path=args.profile_releases,
            private_subset_preflight_receipt_path=args.private_subset_preflight_receipt,
            public_subset_selection_manifest_path=args.public_subset_selection_manifest,
            person_publication_holds_path=args.person_publication_holds,
            provisional_holds_path=args.provisional_holds,
            historical_loss_approval_path=args.approved_poster_loss,
            bio_binding_approval_path=args.bio_binding_approval,
            mccomb_continuity_path=args.mccomb_continuity_approval,
            coauthor_binding_approval_path=args.coauthor_binding_approval,
            presentation_drift_approval_path=args.presentation_drift_approval,
            additive_placement_review_path=args.additive_placement_review,
            source_cache_dir=args.scholar_cache_dir,
            primary_home_baseline_path=args.primary_home_baseline,
            prior_revision=args.prior_revision,
            region_comparison_revision=args.region_comparison_revision,
            repair_manifest_path=args.repair_reviewed_work_manifest,
            atomic_intent_sink=checkpoint,
            require_bio_release=args.require_bio_release,
            source_commit=os.environ.get("MAP_SOURCE_COMMIT"),
            sullivan_loss_approval_path=args.sullivan_loss_approval,
            sullivan_continuity_addendum_path=args.sullivan_continuity_addendum,
            sullivan_crosswalk_path=next(
                (
                    path
                    for path in args.scholar_additional_prior_id_crosswalk
                    if path.name == DEFAULT_SULLIVAN_CROSSWALK_PATH.name
                ),
                DEFAULT_SULLIVAN_CROSSWALK_PATH,
            ),
            mccomb_crosswalk_path=next(
                (
                    path
                    for path in args.scholar_additional_prior_id_crosswalk
                    if path.name == DEFAULT_MCCOMB_CROSSWALK_PATH.name
                ),
                DEFAULT_MCCOMB_CROSSWALK_PATH,
            ),
            curated_works_path=args.scholar_curated_works,
            math_region_review_path=args.math_region_review_manifest,
            chem_region_review_path=args.chem_region_review_manifest,
        )
        status.pop("atomic_release_intent", None)
        status.update(
            {
                "status": "success",
                **result,
                "error_type": None,
                "error_message": None,
            }
        )
        exit_code = 0
    except Exception as error:
        LOGGER.exception("Snapshot publication failed")
        if isinstance(error, RegionReleaseDiffError):
            status["region_continuity"] = error.receipt
        if isinstance(error, PresentationDriftError):
            status["presentation_drift"] = error.receipt
        if isinstance(error, AtomicReleaseReconciliationRequired):
            status["atomic_release_reconciliation"] = error.receipt
        status.update(
            {
                "status": "failure",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        )
        exit_code = 1
    status["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    atomic_write_json(args.status_file, status)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
