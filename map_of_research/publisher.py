"""Publish a validated snapshot and precomputed thin-site map artifacts."""

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

from .io import atomic_write_json
from .snapshot import DEFAULT_MAX_AGE_DAYS, validate_snapshot

LOGGER = logging.getLogger(__name__)
REPO_ID = "ccm/cmu-engineering-publications"
DATASET_CONFIG = "publications"
LEGACY_DATASET_CONFIG = "default"
HF_TOKEN_ENV = "HF_TOKEN"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
EMBEDDING_DIMENSION = 768
MAP_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
DEFAULT_STATUS_PATH = Path("status/last-upload.json")


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


def _normalize_memberships(value: Any) -> list[dict[str, str]]:
    memberships: list[dict[str, str]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            raise ValueError("Dataset membership must be an object")
        membership = {
            "map_slug": str(item.get("map_slug") or ""),
            "department": str(item.get("department") or ""),
            "faculty": str(item.get("faculty") or ""),
        }
        if not all(membership.values()):
            raise ValueError("Dataset membership fields cannot be blank")
        memberships.append(membership)
    if not memberships:
        raise ValueError("Each publication row must have at least one membership")
    return memberships


def _existing_embeddings(*, hf_token: str) -> dict[tuple[str, str], list[float]]:
    """Load reusable vectors from either the normalized or legacy dataset."""

    import datasets

    for config_name in (DATASET_CONFIG, LEGACY_DATASET_CONFIG):
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
    """Reuse unchanged vectors and encode only new profile-publication rows."""

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
            normalize_embeddings=False,
        )
        if len(vectors) != len(missing_keys):
            raise RuntimeError("Embedding model returned an unexpected row count")
        for key, vector in zip(missing_keys, vectors, strict=True):
            values = [float(value) for value in vector]
            if len(values) != EMBEDDING_DIMENSION:
                raise RuntimeError("Embedding model returned an unexpected dimension")
            existing[key] = values

    output = frame.copy()
    output["map_slugs"] = output["map_slugs"].apply(
        lambda value: [str(item) for item in _as_list(value)]
    )
    output["memberships"] = output["memberships"].apply(_normalize_memberships)
    output["embedding"] = [existing[key] for key in keys]
    return output, len(missing_keys)


def _coordinates(embeddings: list[list[float]]) -> list[tuple[float, float]]:
    """Create deterministic, bounded 2-D semantic coordinates."""

    import numpy
    from sklearn.decomposition import PCA

    matrix = numpy.asarray(embeddings, dtype=numpy.float32)
    if len(matrix) == 1:
        return [(0.0, 0.0)]
    projected = PCA(
        n_components=2,
        random_state=42,
        svd_solver="randomized",
    ).fit_transform(matrix)
    projected -= projected.mean(axis=0, keepdims=True)
    scale = numpy.max(numpy.abs(projected), axis=0)
    scale[scale == 0] = 1.0
    projected /= scale
    return [(round(float(x), 7), round(float(y), 7)) for x, y in projected]


def _membership_for_slug(memberships: Any, map_slug: str) -> dict[str, str] | None:
    for membership in _normalize_memberships(memberships):
        if membership["map_slug"] == map_slug:
            return membership
    return None


def build_map_artifact(
    frame: Any,
    *,
    map_slug: str,
    title: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Expand normalized memberships only for one browser-facing map."""

    if map_slug == "map-of-eng":
        selected = frame.copy()
        memberships = [
            _normalize_memberships(value)[0] for value in selected["memberships"]
        ]
        groups = [membership["department"] for membership in memberships]
        faculty_labels = selected["faculty"].astype(str).tolist()
    else:
        membership_rows = [
            _membership_for_slug(value, map_slug) for value in frame["memberships"]
        ]
        mask = [membership is not None for membership in membership_rows]
        selected = frame.loc[mask].copy()
        memberships = [
            membership for membership in membership_rows if membership is not None
        ]
        groups = [membership["faculty"] for membership in memberships]
        faculty_labels = groups

    if selected.empty:
        raise ValueError(f"No publication rows are available for {map_slug}")
    coordinates = _coordinates(selected["embedding"].tolist())
    points = []
    for row, (x, y), group, faculty in zip(
        selected.itertuples(index=False),
        coordinates,
        groups,
        faculty_labels,
        strict=True,
    ):
        raw_year = row.year
        year = (
            None
            if raw_year is None
            or (isinstance(raw_year, float) and math.isnan(raw_year))
            else int(raw_year)
        )
        source_url = str(row.source_url or "")
        if source_url == "nan":
            source_url = ""
        points.append(
            {
                "x": x,
                "y": y,
                "title": str(row.title),
                "faculty": str(faculty),
                "group": str(group),
                "year": year,
                "citation_count": int(row.citation_count),
                "source_url": source_url,
            }
        )
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "map_slug": map_slug,
        "title": title,
        "generated_at_utc": generated_at_utc,
        "point_count": len(points),
        "points": points,
    }


def _map_catalog(frame: Any) -> dict[str, str]:
    catalog = {"map-of-eng": "CMU Engineering Research"}
    for value in frame["memberships"]:
        for membership in _normalize_memberships(value):
            slug = membership["map_slug"]
            department = membership["department"]
            previous = catalog.setdefault(slug, department)
            if previous != department:
                raise ValueError(f"Conflicting titles for map {slug}")
    return dict(sorted(catalog.items()))


def _upload_map_artifacts(
    frame: Any,
    *,
    hf_token: str,
    generated_at_utc: str,
    dataset_commit: str,
) -> str:
    from huggingface_hub import HfApi

    catalog = _map_catalog(frame)
    with tempfile.TemporaryDirectory(prefix="map-of-research-") as temporary_dir:
        maps_dir = Path(temporary_dir)
        map_manifest: dict[str, Any] = {
            "schema_version": MAP_SCHEMA_VERSION,
            "generated_at_utc": generated_at_utc,
            "dataset_commit": dataset_commit,
            "maps": {},
        }
        for map_slug, title in catalog.items():
            artifact = build_map_artifact(
                frame,
                map_slug=map_slug,
                title=title,
                generated_at_utc=generated_at_utc,
            )
            artifact_path = maps_dir / f"{map_slug}.json"
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            map_manifest["maps"][map_slug] = {
                "title": title,
                "point_count": artifact["point_count"],
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
            commit_message="Publish normalized research map artifacts",
        )
    commit_id = getattr(commit, "oid", None)
    if not commit_id:
        raise RuntimeError("Map artifact upload did not return a commit")
    return str(commit_id)


def publish_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    hf_token: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Publish the normalized dataset and all browser map artifacts."""

    import datasets

    frame, manifest = validate_snapshot(
        snapshot_path,
        manifest_path,
        max_age_days=max_age_days,
    )
    enriched, new_embedding_count = add_embeddings(
        frame,
        hf_token=hf_token,
        encoder=encoder,
    )
    publication_dataset = datasets.Dataset.from_pandas(
        enriched,
        preserve_index=False,
    ).select_columns(list(enriched.columns))
    features = publication_dataset.features.copy()
    features["embedding"] = datasets.List(datasets.Value("float32"))
    publication_dataset = publication_dataset.cast(features)
    commit = publication_dataset.push_to_hub(
        REPO_ID,
        config_name=DATASET_CONFIG,
        split="train",
        token=hf_token,
        commit_message="Publish normalized engineering publication snapshot",
    )
    dataset_commit = getattr(commit, "oid", None)
    if not dataset_commit:
        raise RuntimeError("Dataset upload did not return a commit")
    generated_at_utc = str(manifest["created_at_utc"])
    artifact_commit = _upload_map_artifacts(
        enriched,
        hf_token=hf_token,
        generated_at_utc=generated_at_utc,
        dataset_commit=str(dataset_commit),
    )
    return {
        "dataset_commit": str(dataset_commit),
        "artifact_commit": artifact_commit,
        "publication_rows": len(enriched),
        "profile_count": int(enriched["scholar_id"].nunique()),
        "new_embedding_count": new_embedding_count,
        "map_count": len(_map_catalog(enriched)),
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
        "dataset_config": DATASET_CONFIG,
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
