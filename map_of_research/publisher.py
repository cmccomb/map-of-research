"""Publish retained source observations and the unified map artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset import build_dataset_tables
from .io import atomic_write_json
from .registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    Registry,
    load_registry,
)
from .snapshot import DEFAULT_MAX_AGE_DAYS, validate_snapshot

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
MAP_SCHEMA_VERSION = 3
LAYOUT_VERSION = "global-publications-pca-v3"
MAP_ARTIFACT_NAME = "publications.json"
MAP_TITLE = "CMU Engineering Research"
STATUS_SCHEMA_VERSION = 2
DEFAULT_STATUS_PATH = Path("status/last-upload.json")
STRING_LIST_COLUMNS = frozenset(
    {
        "author_pub_ids",
        "author_variants",
        "citation_variants",
        "included_department_ids",
        "department_ids",
        "department_titles",
        "observation_ids",
        "person_ids",
        "scholar_ids",
        "source_urls",
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


def _existing_embeddings(*, hf_token: str) -> dict[tuple[str, str], list[float]]:
    """Load reusable observation vectors from any prior dataset generation."""

    import datasets

    for config_name in REUSABLE_EMBEDDING_CONFIGS:
        try:
            existing = datasets.load_dataset(
                REPO_ID,
                name=config_name,
                split="train",
                token=hf_token or None,
            )
        except Exception:
            LOGGER.info("No reusable %s dataset config found", config_name)
            continue
        if "embedding" not in existing.column_names:
            continue
        embeddings: dict[tuple[str, str], list[float]] = {}
        for row in existing:
            author_pub_id = str(row.get("author_pub_id") or "")
            title = str(row.get("title") or "")
            embedding = row.get("embedding")
            if not author_pub_id or not title or not isinstance(embedding, list):
                continue
            if len(embedding) != EMBEDDING_DIMENSION:
                continue
            embeddings[(author_pub_id, title)] = [float(value) for value in embedding]
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
) -> tuple[Any, int]:
    """Reuse unchanged source-observation vectors and encode only new titles."""

    existing = _existing_embeddings(hf_token=hf_token)
    keys = [
        (str(row.author_pub_id), str(row.title))
        for row in frame[["author_pub_id", "title"]].itertuples(index=False)
    ]
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


def add_global_layout(works: Any) -> Any:
    """Fit one deterministic layout shared by every current map."""

    import numpy
    from sklearn.decomposition import PCA

    output = works.copy()
    output["layout_version"] = LAYOUT_VERSION
    output["x"] = math.nan
    output["y"] = math.nan
    included_mask = output["department_ids"].apply(lambda value: bool(_as_list(value)))
    included = output.loc[included_mask]
    if included.empty:
        raise ValueError("No included works are available for map generation")
    matrix = numpy.asarray(included["embedding"].tolist(), dtype=numpy.float32)
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    if len(matrix) == 1:
        projected = numpy.zeros((1, 2), dtype=numpy.float32)
    else:
        projected = PCA(
            n_components=2,
            svd_solver="randomized",
            random_state=0,
        ).fit_transform(matrix)
        projected -= projected.mean(axis=0, keepdims=True)
        scale = numpy.max(numpy.abs(projected), axis=0)
        scale[scale == 0] = 1.0
        projected /= scale
    output.loc[included_mask, "x"] = [
        round(float(value), 7) for value in projected[:, 0]
    ]
    output.loc[included_mask, "y"] = [
        round(float(value), 7) for value in projected[:, 1]
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

    selected_rows: list[tuple[Any, list[dict[str, str]]]] = []
    for row in works.itertuples(index=False):
        memberships = _normalized_memberships(row.memberships)
        if memberships:
            selected_rows.append((row, memberships))
    points = []
    for row, memberships in selected_rows:
        source_urls = [str(value) for value in _as_list(row.source_urls) if value]
        points.append(
            {
                "x": round(float(row.x), 7),
                "y": round(float(row.y), 7),
                "work_id": str(row.work_id),
                "title": str(row.title),
                "authors": str(row.authors),
                "faculty_ids": sorted(
                    {membership["person_id"] for membership in memberships}
                ),
                "department_ids": sorted(
                    {membership["department_id"] for membership in memberships}
                ),
                "year": None
                if row.year is None or math.isnan(float(row.year))
                else int(row.year),
                "venue": str(row.venue),
                "citation_count": int(row.citation_count),
                "doi": str(row.doi),
                "source_url": source_urls[0] if source_urls else "",
                "observation_count": int(row.observation_count),
            }
        )
    points.sort(key=lambda point: point["work_id"])

    department_counts: dict[str, int] = {}
    faculty_counts: dict[str, int] = {}
    for point in points:
        for department_id in point["department_ids"]:
            department_counts[department_id] = (
                department_counts.get(department_id, 0) + 1
            )
        for person_id in point["faculty_ids"]:
            faculty_counts[person_id] = faculty_counts.get(person_id, 0) + 1

    departments = [
        {
            "department_id": department.department_id,
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
                "publication_count": faculty_counts.get(person.person_id, 0),
            }
        )
    fetched = [
        str(row.last_fetched_at_utc)
        for row, _ in selected_rows
        if str(row.last_fetched_at_utc)
    ]
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "title": MAP_TITLE,
        "generated_at_utc": generated_at_utc,
        "source_data_oldest_at_utc": min(fetched) if fetched else None,
        "source_data_newest_at_utc": max(fetched) if fetched else None,
        "point_count": len(points),
        "department_count": len(departments),
        "faculty_count": len(faculty),
        "catalogs": {
            "departments": departments,
            "faculty": faculty,
        },
        "points": points,
    }


def _upload_map_artifacts(
    works: Any,
    registry: Registry,
    *,
    hf_token: str,
    generated_at_utc: str,
    dataset_commits: Mapping[str, str],
) -> str:
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="map-of-research-") as temporary_dir:
        maps_dir = Path(temporary_dir)
        map_manifest: dict[str, Any] = {
            "schema_version": MAP_SCHEMA_VERSION,
            "layout_version": LAYOUT_VERSION,
            "generated_at_utc": generated_at_utc,
            "dataset_commits": dict(dataset_commits),
            "artifact": {},
        }
        artifact = build_map_artifact(
            works,
            registry,
            generated_at_utc=generated_at_utc,
        )
        artifact_path = maps_dir / MAP_ARTIFACT_NAME
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        map_manifest["artifact"] = {
            "title": artifact["title"],
            "point_count": artifact["point_count"],
            "department_count": artifact["department_count"],
            "faculty_count": artifact["faculty_count"],
            "file": artifact_path.name,
        }
        (maps_dir / "manifest.json").write_text(
            json.dumps(map_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        commit = HfApi(token=hf_token).upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=maps_dir,
            path_in_repo="maps",
            delete_patterns="*.json",
            commit_message="Publish unified research map artifact",
        )
    commit_id = getattr(commit, "oid", None)
    if not commit_id:
        raise RuntimeError("Map artifact upload did not return a commit")
    return str(commit_id)


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
) -> dict[str, Any]:
    """Publish all normalized dataset configs and work-centric map artifacts."""

    registry = load_registry(people_path, memberships_path, departments_path)
    frame, manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        max_age_days=max_age_days,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
    )
    enriched, new_embedding_count = add_embeddings(
        frame,
        hf_token=hf_token,
        encoder=encoder,
    )
    tables = build_dataset_tables(enriched, registry)
    tables["works"] = add_global_layout(tables["works"])

    dataset_commits: dict[str, str] = {}
    for config_name in DATASET_CONFIGS:
        commit = _to_hub_dataset(tables[config_name]).push_to_hub(
            REPO_ID,
            config_name=config_name,
            split="train",
            token=hf_token,
            commit_message=f"Publish {config_name} dataset config",
        )
        commit_id = getattr(commit, "oid", None)
        if not commit_id:
            raise RuntimeError(f"{config_name} upload did not return a commit")
        dataset_commits[config_name] = str(commit_id)

    generated_at_utc = str(manifest["created_at_utc"])
    artifact_commit = _upload_map_artifacts(
        tables["works"],
        registry,
        hf_token=hf_token,
        generated_at_utc=generated_at_utc,
        dataset_commits=dataset_commits,
    )
    return {
        "dataset_commits": dataset_commits,
        "artifact_commit": artifact_commit,
        "people": len(tables["people"]),
        "works": len(tables["works"]),
        "authorships": len(tables["authorships"]),
        "profile_publications": len(tables["profile_publications"]),
        "profile_count": int(enriched["scholar_id"].nunique()),
        "new_embedding_count": new_embedding_count,
        "map_artifact_count": 1,
        "layout_version": LAYOUT_VERSION,
    }


def _workflow_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
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
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
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
    }
    try:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if not hf_token:
            raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")
        result = publish_snapshot(
            args.snapshot,
            args.manifest,
            hf_token=hf_token,
            max_age_days=args.max_age_days,
            people_path=args.people,
            memberships_path=args.memberships,
            departments_path=args.departments,
        )
        status.update({"status": "success", **result, "error_type": None})
        exit_code = 0
    except Exception as error:
        LOGGER.exception("Snapshot publication failed")
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
