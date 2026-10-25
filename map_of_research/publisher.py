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
from .bio_release_gate import (
    BIO_SOURCE_SHA256,
    bio_scholar_source_changed,
    load_bio_baselines,
    preserve_pending_review_topics,
    validate_bio_release_prewrite,
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
from .hub_card import build_updated_dataset_card
from .io import atomic_write_json, load_json, sha256_file
from .keywords import (
    KEYWORD_MODEL_VERSION,
    REGION_AUDIT_VERSION,
    add_publication_keywords,
    exclude_low_information_regions,
)
from .math_release_gate import validate_math_coverage
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
from .region_review import (
    CURRENT_REVISION as MATH_REGION_BASELINE,
)
from .region_review import (
    DEFAULT_REGION_REVIEW_PATH,
    ReviewedRegionDecision,
    load_region_review,
)
from .region_review import (
    SNAPSHOT_SHA256 as MATH_REGION_SNAPSHOT_SHA256,
)
from .region_review import (
    SOURCE_COMMIT as MATH_REGION_SOURCE_COMMIT,
)
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    Registry,
    implicit_primary_baseline_for_people,
    load_registry,
    validate_included_scholar_primary_homes,
)
from .snapshot import DEFAULT_MAX_AGE_DAYS, validate_snapshot
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
MATH_BASELINE_MAP_POINTS = 42925
MATH_BASELINE_MAP_SHA256 = (
    "358e23af6774ea4fd53bab5ed8b55fdd1891a4e7730c0924d74e428e94cd4a88"
)
MATH_BASELINE_WORK_SCHEMA_SHA256 = (
    "5d941f5bde40baf970605373690b5798da8a85f034686c731a000a3099e2dfc4"
)
MATH_COORDINATE_FIELDS = ("x", "y", "tsne_x", "tsne_y")
MATH_VARIANT_FIELDS = (
    "title_variants",
    "venue_variants",
    "citation_variants",
    "author_variants",
)
MATH_TRANSIENT_REGION_FIELDS = (
    "map_region_prior_eligible_override",
    "map_region_prior_excluded_override",
)
MATH_TOPIC_FIELDS = (
    "keyword_id",
    "keyword",
    "keyword_extracted",
    "keyword_label_reviewed",
    "detail_keyword_id",
    "detail_keyword",
    "detail_keyword_extracted",
    "detail_keyword_label_reviewed",
)
MATH_PRESENTATION_FIELDS = (*MATH_COORDINATE_FIELDS, *MATH_TOPIC_FIELDS)
MATH_UNCLASSIFIED_TOPICS = {
    "keyword_id": "keyword-unclassified",
    "keyword": "Unclassified publications",
    "keyword_extracted": "Unclassified publications",
    "keyword_label_reviewed": False,
    "detail_keyword_id": "keyword-unclassified-pending-review",
    "detail_keyword": "Pending topic review",
    "detail_keyword_extracted": "Pending topic review",
    "detail_keyword_label_reviewed": False,
}
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
) -> tuple[Any, int]:
    """Reuse unchanged source-observation vectors and encode only new titles."""

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


def _published_map_artifact(*, hf_token: str, revision: str) -> dict[str, Any]:
    """Read the immutable map artifact used for exact point-payload parity."""

    from huggingface_hub import hf_hub_download

    if revision != MATH_REGION_BASELINE:
        raise ValueError("Math presentation requires the frozen bb88 artifact")
    path = hf_hub_download(
        REPO_ID,
        filename=f"maps/{MAP_ARTIFACT_NAME}",
        repo_type="dataset",
        revision=require_revision(revision),
        token=hf_token or None,
    )
    if sha256_file(Path(path)) != MATH_BASELINE_MAP_SHA256:
        raise ValueError("Math presentation bb88 map artifact bytes changed")
    return load_json(Path(path))


def _math_artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Hash all staged map content independently of JSON formatting."""

    return hashlib.sha256(
        json.dumps(
            artifact, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def preserve_reviewed_math_presentation(
    works: Any,
    prior_works: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
    decisions: tuple[ReviewedRegionDecision, ...],
    registry: Registry,
    *,
    generated_at_utc: str,
) -> tuple[Any, dict[str, int | str]]:
    """Keep bb88 point payloads exact while adding four reviewed Math works."""

    gains = {
        item.work_id
        for item in decisions
        if item.transition == "restore" and item.visible
    }
    losses = {
        item.work_id
        for item in decisions
        if item.transition == "reexclude" and not item.visible
    }
    if len(gains) != 4 or len(losses) != 2 or gains & losses:
        raise ValueError("Math presentation requires exact four gains and two losses")
    points = prior_artifact.get("points")
    keywords = prior_artifact.get("keywords")
    if (
        prior_artifact.get("schema_version") != MAP_SCHEMA_VERSION
        or prior_artifact.get("quality_assessment_version")
        != QUALITY_ASSESSMENT_VERSION
        or prior_artifact.get("layout_version") != LAYOUT_VERSION
        or prior_artifact.get("keyword_model_version") != KEYWORD_MODEL_VERSION
        or prior_artifact.get("topic_label_review_version")
        != TOPIC_LABEL_REVIEW_VERSION
        or prior_artifact.get("region_audit_version") != REGION_AUDIT_VERSION
        or prior_artifact.get("default_layout_id") != DEFAULT_LAYOUT_ID
        or prior_artifact.get("layouts") != [dict(layout) for layout in LAYOUTS]
        or not isinstance(points, list)
        or len(points) != MATH_BASELINE_MAP_POINTS
        or prior_artifact.get("point_count") != len(points)
        or not isinstance(keywords, list)
    ):
        raise ValueError("Math presentation baseline artifact differs from bb88")
    old_points = {str(item["work_id"]): item for item in points}
    old_keywords = {str(item["keyword_id"]): item for item in keywords}
    old_works = {str(item["work_id"]): item for item in prior_works}
    current_indices = {str(row.work_id): index for index, row in works.iterrows()}
    if (
        len(old_points) != len(points)
        or len(old_keywords) != len(keywords)
        or len(old_works) != len(prior_works)
        or len(current_indices) != len(works)
        or not set(old_points).issubset(old_works)
        or set(old_keywords).intersection(
            {
                MATH_UNCLASSIFIED_TOPICS["keyword_id"],
                MATH_UNCLASSIFIED_TOPICS["detail_keyword_id"],
            }
        )
    ):
        raise ValueError("Math presentation baseline identities are invalid")
    visible = {
        str(row.work_id)
        for row in works.itertuples(index=False)
        if bool(row.map_eligible)
        and bool(_as_list(row.memberships))
        and bool(getattr(row, "has_included_source_profile", True))
    }
    common = visible & old_points.keys()
    if visible - old_points.keys() != gains or old_points.keys() - visible != losses:
        raise ValueError("Math presentation map IDs exceed exact +4/-2 review")
    if not set(MATH_TRANSIENT_REGION_FIELDS).issubset(works.columns):
        raise ValueError("Math presentation lacks reconciled region diagnostics")

    output = works.drop(columns=list(MATH_TRANSIENT_REGION_FIELDS)).copy()
    for work_id in sorted(common | gains):
        old = old_works[work_id]
        old_coords = {}
        for field in MATH_COORDINATE_FIELDS:
            try:
                value = float(old[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Math presentation has no frozen coordinates: {work_id}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"Math presentation has no frozen coordinates: {work_id}"
                )
            old_coords[field] = value
        if work_id in gains:
            if bool(old["map_eligible"]) or not bool(old["map_region_outlier"]):
                raise ValueError("Math restored point is not a prior region outlier")
            if any(
                old.get(field) != (False if field.endswith("_label_reviewed") else "")
                for field in MATH_TOPIC_FIELDS
            ):
                raise ValueError("Math restored point already has a frozen topic")
            topics = MATH_UNCLASSIFIED_TOPICS
        else:
            point = old_points[work_id]
            if (
                not bool(old["map_eligible"])
                or any(
                    round(old_coords[field], 7) != point[field]
                    for field in MATH_COORDINATE_FIELDS
                )
                or point["keyword_ids"] != [old["keyword_id"], old["detail_keyword_id"]]
            ):
                raise ValueError(
                    f"Math shared point differs from frozen works: {work_id}"
                )
            topics = {field: old[field] for field in MATH_TOPIC_FIELDS}
        index = current_indices[work_id]
        for field, value in (*old_coords.items(), *topics.items()):
            output.at[index, field] = value
        if work_id in common:
            for field in MATH_VARIANT_FIELDS:
                previous = _as_list(old[field])
                current = _as_list(output.at[index, field])
                if Counter(previous) != Counter(current):
                    raise ValueError(
                        f"Math shared work changed source variants: {work_id}: {field}"
                    )
                output.at[index, field] = previous

    candidate = build_map_artifact(output, registry, generated_at_utc=generated_at_utc)
    for field in (
        "schema_version",
        "quality_assessment_version",
        "keyword_model_version",
        "topic_label_review_version",
        "region_audit_version",
        "layout_version",
        "default_layout_id",
        "layouts",
        "title",
    ):
        if candidate.get(field) != prior_artifact.get(field):
            raise ValueError(f"Math presentation changed artifact metadata: {field}")
    candidate_points = {str(item["work_id"]): item for item in candidate["points"]}
    candidate_keywords = {
        str(item["keyword_id"]): item for item in candidate["keywords"]
    }
    if len(candidate_points) != len(visible) or set(candidate_points) != visible:
        raise ValueError("Math presentation artifact point IDs differ")
    for field, expected in (
        ("point_count", len(points) + len(gains) - len(losses)),
        ("keyword_count", len(keywords) + 2),
        ("department_count", prior_artifact["department_count"]),
        ("school_count", prior_artifact["school_count"]),
        ("faculty_count", prior_artifact["faculty_count"]),
    ):
        if candidate.get(field) != expected:
            raise ValueError(f"Math presentation changed artifact count: {field}")
    for work_id in sorted(common):
        if candidate_points[work_id] != old_points[work_id]:
            changed = sorted(
                field
                for field in old_points[work_id].keys()
                | candidate_points[work_id].keys()
                if old_points[work_id].get(field)
                != candidate_points[work_id].get(field)
            )
            raise ValueError(
                f"Math shared point payload changed: {work_id}: {', '.join(changed)}"
            )
    for work_id in gains:
        if candidate_points[work_id]["keyword_ids"] != [
            MATH_UNCLASSIFIED_TOPICS["keyword_id"],
            MATH_UNCLASSIFIED_TOPICS["detail_keyword_id"],
        ] or any(
            candidate_points[work_id][field]
            != round(float(old_works[work_id][field]), 7)
            for field in MATH_COORDINATE_FIELDS
        ):
            raise ValueError("Math restored point placement or topics changed")
    removed_keyword_counts = Counter(
        keyword_id
        for work_id in losses
        for keyword_id in old_points[work_id]["keyword_ids"]
    )
    new_keyword_ids = {
        MATH_UNCLASSIFIED_TOPICS["keyword_id"],
        MATH_UNCLASSIFIED_TOPICS["detail_keyword_id"],
    }
    if candidate_keywords.keys() != old_keywords.keys() | new_keyword_ids:
        raise ValueError("Math presentation changed topic IDs beyond exact review")
    gain_members = [candidate_points[work_id] for work_id in sorted(gains)]
    new_coordinates = {
        str(layout["layout_id"]): {
            axis: round(
                sum(
                    float(point[str(layout[f"{axis}_field"])])
                    for point in gain_members
                )
                / len(gain_members),
                7,
            )
            for axis in ("x", "y")
        }
        for layout in LAYOUTS
    }
    for keyword_id, label, extracted, reviewed, level, parent in (
        (
            MATH_UNCLASSIFIED_TOPICS["keyword_id"],
            MATH_UNCLASSIFIED_TOPICS["keyword"],
            MATH_UNCLASSIFIED_TOPICS["keyword_extracted"],
            MATH_UNCLASSIFIED_TOPICS["keyword_label_reviewed"],
            0,
            None,
        ),
        (
            MATH_UNCLASSIFIED_TOPICS["detail_keyword_id"],
            MATH_UNCLASSIFIED_TOPICS["detail_keyword"],
            MATH_UNCLASSIFIED_TOPICS["detail_keyword_extracted"],
            MATH_UNCLASSIFIED_TOPICS["detail_keyword_label_reviewed"],
            1,
            MATH_UNCLASSIFIED_TOPICS["keyword_id"],
        ),
    ):
        if candidate_keywords[keyword_id] != {
            "keyword_id": keyword_id,
            "label": label,
            "extracted_label": extracted,
            "label_reviewed": reviewed,
            "level": level,
            "parent_keyword_id": parent,
            "publication_count": len(gains),
            "coordinates": new_coordinates,
        }:
            raise ValueError(f"Math new topic metadata changed: {keyword_id}")
    for keyword_id, previous in old_keywords.items():
        current = candidate_keywords.get(keyword_id)
        if keyword_id not in removed_keyword_counts:
            expected = previous
        else:
            members = [
                point
                for point in points
                if point["work_id"] not in losses
                and keyword_id in point["keyword_ids"]
            ]
            if not members or len(members) != (
                previous["publication_count"] - removed_keyword_counts[keyword_id]
            ):
                raise ValueError(f"Math shared topic membership changed: {keyword_id}")
            expected = {
                **previous,
                "publication_count": len(members),
                "coordinates": {
                    str(layout["layout_id"]): {
                        axis: round(
                            sum(
                                float(point[str(layout[f"{axis}_field"])])
                                for point in members
                            )
                            / len(members),
                            7,
                        )
                        for axis in ("x", "y")
                    }
                    for layout in LAYOUTS
                },
            }
        if current != expected:
            raise ValueError(f"Math shared topic metadata changed: {keyword_id}")
    for level in range(len(KEYWORD_LEVELS)):
        current = candidate["keyword_levels"][level]
        previous = prior_artifact["keyword_levels"][level]
        if current != {**previous, "keyword_count": previous["keyword_count"] + 1}:
            raise ValueError("Math presentation changed keyword-level metadata")
    for catalog_name, key, point_field in (
        ("schools", "school_id", "school_ids"),
        ("departments", "department_id", "department_ids"),
        ("faculty", "person_id", "faculty_ids"),
    ):
        old_catalog = {
            str(item[key]): item for item in prior_artifact["catalogs"][catalog_name]
        }
        new_catalog = {
            str(item[key]): item for item in candidate["catalogs"][catalog_name]
        }
        added_counts = Counter(
            item_id
            for work_id in gains
            for item_id in candidate_points[work_id][point_field]
        )
        removed_counts = Counter(
            item_id
            for work_id in losses
            for item_id in old_points[work_id][point_field]
        )
        if old_catalog.keys() != new_catalog.keys() or any(
            new_catalog[item_id]
            != {
                **old_item,
                "publication_count": old_item["publication_count"]
                + added_counts[item_id]
                - removed_counts[item_id],
            }
            for item_id, old_item in old_catalog.items()
        ):
            raise ValueError(f"Math presentation changed {catalog_name} catalog")
    old_external = {
        str(item["person_id"]): item
        for item in prior_artifact["catalogs"]["external_coauthors"]
    }
    new_external = {
        str(item["person_id"]): item
        for item in candidate["catalogs"]["external_coauthors"]
    }
    removed_external = old_external.keys() - new_external.keys()
    if removed_external - {
        person_id
        for work_id in losses
        for person_id in old_points[work_id]["external_coauthor_ids"]
    } or removed_external & {
        person_id
        for work_id in common
        for person_id in old_points[work_id]["external_coauthor_ids"]
    } or any(
        new_external[person_id] != old_external[person_id]
        for person_id in old_external.keys() & new_external.keys()
    ) or not (new_external.keys() - old_external.keys()).issubset(
        {
            person_id
            for work_id in gains
            for person_id in candidate_points[work_id]["external_coauthor_ids"]
        }
    ):
        raise ValueError("Math presentation changed external coauthor catalog")
    return output, {
        "unchanged_shared_points": len(common),
        "restored_points": len(gains),
        "removed_points": len(losses),
        "frozen_coordinate_rows": len(common | gains),
        "map_artifact_sha256": _math_artifact_digest(candidate),
    }


def _published_field_value(value: Any) -> Any:
    """Compare Arrow and pandas values without weakening source-field parity."""

    if hasattr(value, "tolist"):
        return _published_field_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _published_field_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_published_field_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def require_reviewed_math_work_table_parity(
    works: Any,
    prior_rows: Iterator[dict[str, Any]],
    common_ids: set[str],
) -> int:
    """Reject changes to any published column of the 42,923 retained map works."""

    import numpy as np

    fields = tuple(str(field) for field in works.columns)
    schema_sha256 = hashlib.sha256(
        json.dumps(sorted(fields), separators=(",", ":")).encode()
    ).hexdigest()
    if (
        schema_sha256 != MATH_BASELINE_WORK_SCHEMA_SHA256
        or len(common_ids) != MATH_BASELINE_MAP_POINTS - 2
        or "embedding" not in fields
    ):
        raise ValueError("Math work-table schema or reviewed point count changed")
    field_positions = {field: index for index, field in enumerate(fields)}
    candidate = {}
    for row in works.itertuples(index=False, name=None):
        work_id = str(row[field_positions["work_id"]])
        if work_id in common_ids:
            if work_id in candidate:
                raise ValueError("Math common works are missing or duplicated")
            candidate[work_id] = row
    if len(candidate) != len(common_ids):
        raise ValueError("Math common works are missing or duplicated")
    seen: set[str] = set()
    for old in prior_rows:
        work_id = str(old["work_id"])
        if work_id not in common_ids:
            continue
        if work_id in seen:
            raise ValueError("Math baseline has duplicate common work IDs")
        seen.add(work_id)
        current = candidate[work_id]
        for field, position in field_positions.items():
            previous_value = old[field]
            current_value = current[position]
            equal = (
                np.array_equal(
                    np.asarray(previous_value, dtype=np.float32),
                    np.asarray(current_value, dtype=np.float32),
                    equal_nan=True,
                )
                if field == "embedding"
                else _published_field_value(previous_value)
                == _published_field_value(current_value)
            )
            if not equal:
                raise ValueError(
                    f"Math shared work-table field changed: {work_id}: {field}"
                )
    if seen != common_ids:
        raise ValueError("Math baseline is missing common work rows")
    return len(seen)


def _staged_math_work_rows(
    path: Path,
    *,
    expected_fields: Sequence[str],
    expected_rows: int,
) -> Iterator[dict[str, Any]]:
    """Read staged Arrow rows only after validating their complete schema."""

    import pyarrow.parquet as pq

    staged = pq.ParquetFile(path)
    if (
        set(staged.schema_arrow.names) != set(expected_fields)
        or len(staged.schema_arrow.names) != len(expected_fields)
        or staged.metadata.num_rows != expected_rows
    ):
        raise ValueError("Math staged works Parquet schema or row count changed")
    for batch in staged.iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def _write_map_artifacts(
    works: Any,
    registry: Registry,
    *,
    maps_dir: Path,
    generated_at_utc: str,
    dataset_commits: Mapping[str, str],
    continuity_receipt: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Render both map files before the Hub is touched."""

    maps_dir.mkdir(parents=True, exist_ok=True)
    map_manifest: dict[str, Any] = {
        "schema_version": MAP_SCHEMA_VERSION,
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
    artifact = build_map_artifact(works, registry, generated_at_utc=generated_at_utc)
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
    manifest_path = maps_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(map_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path, manifest_path, artifact


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
    if (
        status.get("map_quality_hold") is not False
        or status.get("producer_review_ready") is not True
        or status.get("unresolved_exact_ids") != []
    ):
        raise ValueError(
            "Physics map-quality gate: source review is held; "
            f"snapshot contains {', '.join(sorted(present_ids))}"
        )
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
    hf_token: str,
    prior_revision: str,
    generated_at_utc: str,
    continuity_receipt: Mapping[str, Any] | None,
    repair_manifest_path: Path,
    provisional_holds: Mapping[tuple[str, str], Any],
    prior_profiles: Any,
    prior_people: Any,
    atomic_intent_sink: Callable[[Mapping[str, Any]], None] | None = None,
    require_bio_release: bool = False,
    math_region_review: bool = False,
    math_common_work_ids: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Stage every output, run Bio gates, then commit seven files once."""

    from huggingface_hub import hf_hub_download

    if atomic_intent_sink is None:
        raise ValueError("Live publish requires a durable prewrite intent sink")
    prior_revision = require_revision(prior_revision)
    bio_release = source_manifest["snapshot_sha256"] == BIO_SOURCE_SHA256
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
        pending_in_parent = sum(
            row.get("keyword_id") == "keyword-unclassified"
            and row.get("detail_keyword_id") == "keyword-unclassified-pending-review"
            and bool(row["map_eligible"])
            for row in baselines["repair_works"]
        )
        if pending_in_parent:
            if not repair_manifest_path.is_file():
                raise FileNotFoundError(
                    "Bio release needs the reviewed repair manifest: "
                    f"{repair_manifest_path}"
                )
            repair_manifest = load_json(repair_manifest_path)
        else:
            repair_manifest = {"restore_eligible_work_ids": []}
        tables["works"] = preserve_pending_review_topics(
            tables["works"], baselines["repair_works"], repair_manifest
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
        artifact_path, map_manifest_path, artifact = _write_map_artifacts(
            tables["works"],
            registry,
            maps_dir=root / "maps",
            generated_at_utc=generated_at_utc,
            dataset_commits={},
            continuity_receipt=continuity_receipt,
        )
        files["maps/publications.json"] = artifact_path
        files["maps/manifest.json"] = map_manifest_path
        if math_region_review:
            if math_common_work_ids is None:
                raise ValueError("Atomic Math release lacks retained work identities")
            staged_artifact = load_json(artifact_path)
            if not isinstance(staged_artifact, dict):
                raise ValueError("Atomic Math staged map artifact is invalid")
            artifact = staged_artifact
            require_reviewed_math_work_table_parity(
                tables["works"],
                _staged_math_work_rows(
                    files["works/train-00000-of-00001.parquet"],
                    expected_fields=tuple(tables["works"].columns),
                    expected_rows=len(tables["works"]),
                ),
                math_common_work_ids,
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
        if math_region_review:
            expected_parity = {
                "unchanged_shared_points": MATH_BASELINE_MAP_POINTS - 2,
                "restored_points": 4,
                "removed_points": 2,
                "frozen_coordinate_rows": MATH_BASELINE_MAP_POINTS + 2,
                "shared_work_rows": MATH_BASELINE_MAP_POINTS - 2,
                "map_artifact_sha256": _math_artifact_digest(artifact),
            }
            if (
                bio_release
                or prior_revision != MATH_REGION_BASELINE
                or source_manifest["snapshot_sha256"] != MATH_REGION_SNAPSHOT_SHA256
                or artifact["point_count"] != MATH_BASELINE_MAP_POINTS + 2
                or continuity_receipt is None
                or continuity_receipt.get("math_presentation_parity")
                != expected_parity
            ):
                raise ValueError("Atomic Math release lacks exact presentation parity")
            gate["gate"] = "frozen-math-presentation-parity-v1"
            gate["math_presentation_parity"] = expected_parity
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


def require_math_region_mode(
    *,
    requested: bool,
    prior_revision: str | None,
    region_comparison_revision: str | None,
    source_commit: str | None,
) -> None:
    """Keep the frozen Math review atomic and separate from all-280 repair."""

    if prior_revision == MATH_REGION_BASELINE and not requested:
        raise ValueError(
            "Frozen bb88 Math review requires its exact reviewed publish flag"
        )
    if requested and (
        prior_revision != MATH_REGION_BASELINE
        or region_comparison_revision not in (None, prior_revision)
        or source_commit != MATH_REGION_SOURCE_COMMIT
    ):
        raise ValueError("Math region review requires frozen bb88/f7 ordinary publish")


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
    provisional_holds_path: Path = DEFAULT_PROVISIONAL_HOLDS_PATH,
    source_cache_dir: Path | None = None,
    primary_home_baseline_path: Path | None = None,
    prior_revision: str | None = None,
    region_comparison_revision: str | None = None,
    atomic_release: bool | None = None,
    atomic_intent_sink: Callable[[Mapping[str, Any]], None] | None = None,
    require_bio_release: bool = False,
    repair_manifest_path: Path = Path("data/curated/region-remediation-bernhard.json"),
    math_region_review: bool = False,
    math_region_review_path: Path = DEFAULT_REGION_REVIEW_PATH,
) -> dict[str, Any]:
    """Publish all normalized dataset configs and work-centric map artifacts."""

    require_math_region_mode(
        requested=math_region_review,
        prior_revision=prior_revision,
        region_comparison_revision=region_comparison_revision,
        source_commit=os.environ.get("MAP_SOURCE_COMMIT"),
    )
    if prior_revision is None:
        raise ValueError("Live publish requires an immutable prior Hub revision")
    require_revision(prior_revision)
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
    )
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
    _require_physics_review_ready(frame, snapshot_path, exclusions)
    _require_breivik_review_ready(frame, snapshot_path, exclusions)
    corrections = load_corrections(corrections_path)
    corrected_frame, validated_correction_count = apply_corrections(frame, corrections)
    contexts = load_contexts(contexts_path)
    attribution_reviews = load_attribution_reviews(attribution_reviews_path)
    if set(attribution_reviews.confirmed).intersection(exclusions):
        raise ValueError("Scholar attribution and exclusion reviews conflict")
    release_decisions = load_profile_releases(profile_releases_path)
    release_manifest_sha256 = sha256_file(profile_releases_path)
    provisional_profiles = load_provisional_holds(provisional_holds_path)
    provisional_manifest_sha256 = sha256_file(provisional_holds_path)
    cache_dir = source_cache_dir or snapshot_path.parent.parent / "data" / "authors"
    released_frame, release_receipt, held_observation_ids = (
        select_released_observations(
            corrected_frame,
            registry,
            release_decisions,
            cache_dir=cache_dir,
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
        if math_region_review
        else ()
    )
    published_correction_count = int(
        released_frame["metadata_correction_fields"].ne("").sum()
    )
    enriched, new_embedding_count = add_embeddings(
        released_frame,
        hf_token=hf_token,
        encoder=encoder,
        prior_revision=prior_revision,
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
    tables["works"] = exclude_low_information_regions(tables["works"])
    comparison_revision = region_comparison_revision or prior_revision
    prior_work_fields = (
        (*PRIOR_WORK_FIELDS, "map_exclusion_reasons")
        if comparison_revision != prior_revision
        else (*PRIOR_WORK_FIELDS, *MATH_PRESENTATION_FIELDS, *MATH_VARIANT_FIELDS)
        if math_region_review
        else PRIOR_WORK_FIELDS
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
            PRIOR_OBSERVATION_FIELDS,
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
    tables["works"], continuity_receipt = reconcile_region_decisions(
        tables["works"],
        tables["profile_publications"],
        prior_works,
        prior_profiles,
        prior_revision=comparison_revision,
        reviewed_math_decisions=reviewed_math_decisions,
    )
    bio_source_changed = bio_scholar_source_changed(
        frame, prior_profiles, registry, prior_people
    )
    if bio_source_changed and not frozen_bio_source:
        raise ValueError("Changed Bio Scholar source needs the reviewed Bio snapshot")
    if comparison_revision != prior_revision:
        continuity_receipt["current_hub_revision"] = prior_revision
    tables["works"] = add_publication_keywords(tables["works"])
    math_common_work_ids = None
    if math_region_review:
        prior_artifact = _published_map_artifact(
            hf_token=hf_token, revision=MATH_REGION_BASELINE
        )
        tables["works"], presentation_receipt = preserve_reviewed_math_presentation(
            tables["works"],
            prior_works,
            prior_artifact,
            reviewed_math_decisions,
            registry,
            generated_at_utc=str(manifest["created_at_utc"]),
        )
        removed_ids = {
            decision.work_id
            for decision in reviewed_math_decisions
            if decision.transition == "reexclude" and not decision.visible
        }
        math_common_work_ids = {
            str(point["work_id"]) for point in prior_artifact["points"]
        } - removed_ids
        presentation_receipt["shared_work_rows"] = (
            require_reviewed_math_work_table_parity(
                tables["works"],
                _published_rows(
                    "works",
                    tuple(tables["works"].columns),
                    hf_token=hf_token,
                    revision=MATH_REGION_BASELINE,
                ),
                math_common_work_ids,
            )
        )
        continuity_receipt["math_presentation_parity"] = presentation_receipt
    require_no_held_contributions(tables, release_decisions, held_observation_ids)
    require_no_provisional_hold_contributions(tables, provisional_holds)
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
        hf_token=hf_token,
        prior_revision=prior_revision,
        generated_at_utc=generated_at_utc,
        continuity_receipt=continuity_receipt,
        repair_manifest_path=repair_manifest_path,
        provisional_holds=provisional_holds,
        prior_profiles=prior_profiles,
        prior_people=prior_people,
        atomic_intent_sink=atomic_intent_sink,
        require_bio_release=require_bio_release,
        math_region_review=math_region_review,
        math_common_work_ids=math_common_work_ids,
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
        "provisional_hold_manifest_sha256": provisional_manifest_sha256,
        "provisional_hold_profiles": provisional_receipt,
        "provisional_hold_observation_rows": len(provisional_holds),
        "math_region_review_decisions": len(reviewed_math_decisions),
        "math_region_review_manifest_sha256": (
            sha256_file(math_region_review_path) if reviewed_math_decisions else None
        ),
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
        "--require-bio-release",
        action="store_true",
        help="Fail closed unless the frozen complete Bio snapshot passes its gate",
    )
    parser.add_argument(
        "--math-region-review",
        choices=("true", "false"),
        default="false",
        help="Apply the exact frozen 31-row Math review against bb88",
    )
    parser.add_argument(
        "--math-region-review-manifest",
        type=Path,
        default=DEFAULT_REGION_REVIEW_PATH,
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
        "--provisional-holds", type=Path, default=DEFAULT_PROVISIONAL_HOLDS_PATH
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
            provisional_holds_path=args.provisional_holds,
            source_cache_dir=args.scholar_cache_dir,
            primary_home_baseline_path=args.primary_home_baseline,
            prior_revision=args.prior_revision,
            region_comparison_revision=args.region_comparison_revision,
            repair_manifest_path=args.repair_reviewed_work_manifest,
            atomic_intent_sink=checkpoint,
            require_bio_release=args.require_bio_release,
            math_region_review=args.math_region_review == "true",
            math_region_review_path=args.math_region_review_manifest,
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
