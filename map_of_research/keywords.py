"""Deterministic topic keywords for the publication landscape."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

KEYWORD_MODEL_VERSION = "hierarchical-visible-tsne-kmeans-ctfidf-v3"
REGION_AUDIT_VERSION = "low-information-semantic-region-v1"
DEFAULT_KEYWORD_COUNT = 30
DEFAULT_DETAIL_KEYWORD_COUNT = 120
MINIMUM_DETAIL_CLUSTER_SIZE = 50
MINIMUM_AUDIT_REGION_SIZE = 100
MAXIMUM_AUDIT_REGION_COHERENCE = 0.38
MINIMUM_MISSING_YEAR_FRACTION = 0.15
MINIMUM_UNCITED_FRACTION = 0.50
MINIMUM_SHORT_TITLE_FRACTION = 0.30
SHORT_TITLE_LENGTH = 30
MINIMUM_CLUSTER_SUPPORT = 2
MINIMUM_CLUSTER_COVERAGE = 0.01
RANDOM_STATE = 42

_TOKEN_PATTERN = re.compile(r"(?u)\b[^\W\d_][^\W_]+\b")
_DOMAIN_STOP_WORDS = frozenset(
    {
        "analysis",
        "application",
        "applications",
        "approach",
        "award",
        "awards",
        "based",
        "carnegie",
        "cmu",
        "conference",
        "department",
        "engineering",
        "ieee",
        "ices",
        "issue",
        "journal",
        "mellon",
        "method",
        "methods",
        "member",
        "members",
        "model",
        "models",
        "new",
        "paper",
        "pc",
        "proceedings",
        "report",
        "reports",
        "research",
        "session",
        "special",
        "study",
        "summary",
        "symposium",
        "university",
        "using",
        "volume",
    }
)
_UNINFORMATIVE_PHRASES = frozenset(
    {
        "early career",
        "low cost",
        "science technology",
        "ultra high",
        "united states",
    }
)


def _stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS) | _DOMAIN_STOP_WORDS


def _fallback_keyword(titles: list[str], *, used: set[str]) -> str:
    """Build a readable fallback when a cluster has no repeated phrase."""

    stop_words = _stop_words()
    counts: Counter[str] = Counter()
    for title in titles:
        counts.update(
            token.casefold()
            for token in _TOKEN_PATTERN.findall(title)
            if token.casefold() not in stop_words
        )
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    for start in range(len(ordered)):
        label = " ".join(ordered[start : start + 2])
        if label and label not in used:
            return label
    return f"research topic {len(used) + 1}"


def _cluster_keywords(
    titles: list[str],
    labels: Any,
    *,
    reserved: set[str] | None = None,
    prefer_readable: bool = False,
) -> dict[int, str]:
    """Select a coverage-aware, cluster-specific phrase for every cluster."""

    import numpy
    from sklearn.feature_extraction.text import CountVectorizer

    cluster_ids = sorted({int(label) for label in labels})
    try:
        vectorizer = CountVectorizer(
            binary=True,
            lowercase=True,
            ngram_range=(2, 3),
            stop_words=sorted(_stop_words()),
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(titles)
        phrases = vectorizer.get_feature_names_out()
        global_support = numpy.asarray(matrix.sum(axis=0)).ravel()
    except ValueError:
        matrix = None
        phrases = numpy.asarray([], dtype=str)
        global_support = numpy.asarray([], dtype=float)

    results: dict[int, str] = {}
    used = set(reserved or ())
    labels_array = numpy.asarray(labels, dtype=int)
    total_documents = len(titles)
    for cluster_id in cluster_ids:
        mask = labels_array == cluster_id
        cluster_titles = [
            title for title, included in zip(titles, mask, strict=True) if included
        ]
        cluster_size = len(cluster_titles)
        candidates: list[int] = []
        scores: Any = numpy.asarray([], dtype=float)
        local_support: Any = numpy.asarray([], dtype=float)
        if matrix is not None and phrases.size:
            local_support = numpy.asarray(matrix[mask].sum(axis=0)).ravel()
            required_support = min(
                cluster_size,
                max(
                    MINIMUM_CLUSTER_SUPPORT,
                    math.ceil(cluster_size * MINIMUM_CLUSTER_COVERAGE),
                ),
            )
            specificity = numpy.log((total_documents + 1) / (global_support + 1)) + 1
            scores = (local_support / max(cluster_size, 1)) * specificity
            if prefer_readable:
                readability = numpy.asarray(
                    [
                        0.75 if any(character.isdigit() for character in phrase) else 1
                        for phrase in phrases
                    ],
                    dtype=float,
                )
                scores *= readability
            candidates = [
                index
                for index, support in enumerate(local_support)
                if support >= required_support
            ]
            candidates.sort(
                key=lambda index: (
                    -float(scores[index]),
                    -float(local_support[index]),
                    str(phrases[index]),
                )
            )
        label = next(
            (
                str(phrases[index])
                for index in candidates
                if str(phrases[index]) not in used
                and str(phrases[index]) not in _UNINFORMATIVE_PHRASES
            ),
            "",
        )
        if not label:
            label = _fallback_keyword(cluster_titles, used=used)
        used.add(label)
        results[cluster_id] = label
    return results


def _detail_cluster_allocations(
    labels: Any,
    coordinates: Any,
    *,
    target_count: int,
    minimum_cluster_size: int,
) -> dict[int, int]:
    """Allocate detailed regions proportionally without creating tiny clusters."""

    import numpy

    labels_array = numpy.asarray(labels, dtype=int)
    cluster_ids = sorted({int(label) for label in labels_array})
    sizes = {
        cluster_id: int(numpy.count_nonzero(labels_array == cluster_id))
        for cluster_id in cluster_ids
    }
    capacities = {}
    for cluster_id in cluster_ids:
        cluster_coordinates = coordinates[labels_array == cluster_id]
        distinct_coordinates = len(numpy.unique(cluster_coordinates, axis=0))
        size_capacity = max(1, sizes[cluster_id] // minimum_cluster_size)
        capacities[cluster_id] = min(distinct_coordinates, size_capacity)

    allocated = {cluster_id: 1 for cluster_id in cluster_ids}
    effective_target = min(target_count, sum(capacities.values()))
    while sum(allocated.values()) < effective_target:
        candidates = [
            cluster_id
            for cluster_id in cluster_ids
            if allocated[cluster_id] < capacities[cluster_id]
        ]
        chosen = max(
            candidates,
            key=lambda cluster_id: (
                sizes[cluster_id] / (allocated[cluster_id] + 1),
                -cluster_id,
            ),
        )
        allocated[chosen] += 1
    return allocated


def _merge_small_clusters(
    coordinates: Any,
    labels: Any,
    *,
    minimum_cluster_size: int,
) -> Any:
    """Merge sparse k-means fragments into their nearest sibling region."""

    import numpy

    merged = numpy.asarray(labels, dtype=int).copy()
    while True:
        cluster_ids = sorted({int(label) for label in merged})
        if len(cluster_ids) == 1:
            break
        sizes = {
            cluster_id: int(numpy.count_nonzero(merged == cluster_id))
            for cluster_id in cluster_ids
        }
        source = min(
            cluster_ids, key=lambda cluster_id: (sizes[cluster_id], cluster_id)
        )
        if sizes[source] >= minimum_cluster_size:
            break
        centroids = {
            cluster_id: coordinates[merged == cluster_id].mean(axis=0)
            for cluster_id in cluster_ids
        }
        target = min(
            (cluster_id for cluster_id in cluster_ids if cluster_id != source),
            key=lambda cluster_id: (
                float(numpy.sum((centroids[source] - centroids[cluster_id]) ** 2)),
                cluster_id,
            ),
        )
        merged[merged == source] = target
    remapping = {
        cluster_id: index
        for index, cluster_id in enumerate(sorted({int(label) for label in merged}))
    }
    return numpy.asarray([remapping[int(label)] for label in merged], dtype=int)


def exclude_low_information_regions(works: Any) -> Any:
    """Withhold only large regions that fail several independent quality checks.

    The preliminary detailed topics expose diffuse catch-all regions that title-level
    rules cannot identify safely. A region is excluded only when it is simultaneously
    incoherent in embedding space, citation-poor, metadata-poor, and dominated by
    short titles. Every row and its audit reason remain in the published dataset.
    """

    import numpy
    import pandas

    output = works.copy()
    output["map_region_audit_version"] = REGION_AUDIT_VERSION
    output["map_region_outlier"] = False
    included_mask = output["map_eligible"] & output["detail_keyword_id"].ne("")
    for _, region in output.loc[included_mask].groupby(
        "detail_keyword_id",
        sort=True,
    ):
        if len(region) < MINIMUM_AUDIT_REGION_SIZE:
            continue
        matrix = numpy.asarray(region["embedding"].tolist(), dtype=numpy.float32)
        norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        centroid = matrix.mean(axis=0)
        centroid_norm = float(numpy.linalg.norm(centroid))
        coherence = (
            0.0
            if centroid_norm == 0
            else float((matrix @ (centroid / centroid_norm)).mean())
        )
        missing_year_fraction = float(region["year"].apply(pandas.isna).mean())
        uncited_fraction = float(region["citation_count"].fillna(0).eq(0).mean())
        short_title_fraction = float(
            region["title"].astype(str).str.len().lt(SHORT_TITLE_LENGTH).mean()
        )
        should_exclude = bool(
            coherence < MAXIMUM_AUDIT_REGION_COHERENCE
            and missing_year_fraction >= MINIMUM_MISSING_YEAR_FRACTION
            and uncited_fraction >= MINIMUM_UNCITED_FRACTION
            and short_title_fraction >= MINIMUM_SHORT_TITLE_FRACTION
        )
        if not should_exclude:
            continue
        indices = region.index
        output.loc[indices, "map_eligible"] = False
        output.loc[indices, "map_region_outlier"] = True
        output.loc[indices, "map_exclusion_reasons"] = output.loc[
            indices,
            "map_exclusion_reasons",
        ].apply(
            lambda reasons: list(dict.fromkeys([*reasons, "low_information_region"]))
        )
    return output


def add_publication_keywords(
    works: Any,
    *,
    keyword_count: int = DEFAULT_KEYWORD_COUNT,
    detail_keyword_count: int = DEFAULT_DETAIL_KEYWORD_COUNT,
    minimum_detail_cluster_size: int = MINIMUM_DETAIL_CLUSTER_SIZE,
) -> Any:
    """Assign broad and detailed visible-region keywords without dropping works."""

    import numpy
    from sklearn.cluster import KMeans

    if keyword_count < 1:
        raise ValueError("keyword_count must be positive")
    if detail_keyword_count < keyword_count:
        raise ValueError("detail_keyword_count must be at least keyword_count")
    if minimum_detail_cluster_size < 1:
        raise ValueError("minimum_detail_cluster_size must be positive")

    output = works.copy()
    output["keyword_model_version"] = KEYWORD_MODEL_VERSION
    output["keyword_id"] = ""
    output["keyword"] = ""
    output["detail_keyword_id"] = ""
    output["detail_keyword"] = ""
    included_mask = output["map_eligible"] & output["department_ids"].apply(
        lambda value: len(value) > 0
    )
    included = output.loc[included_mask]
    if included.empty:
        raise ValueError("No included works are available for keyword generation")

    coordinates = numpy.asarray(
        included[["tsne_x", "tsne_y"]].to_numpy(),
        dtype=numpy.float32,
    )
    if not numpy.isfinite(coordinates).all():
        raise ValueError("Keyword generation requires finite t-SNE coordinates")
    distinct_coordinates = len(numpy.unique(coordinates, axis=0))
    cluster_count = min(keyword_count, len(included), distinct_coordinates)
    if cluster_count == 1:
        labels = numpy.zeros(len(included), dtype=int)
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=RANDOM_STATE,
        ).fit_predict(coordinates)

    titles = [str(value) for value in included["title"]]
    labels_by_cluster = _cluster_keywords(titles, labels, prefer_readable=True)
    centroids = {
        cluster_id: coordinates[labels == cluster_id].mean(axis=0)
        for cluster_id in labels_by_cluster
    }
    ordered_clusters = sorted(
        labels_by_cluster,
        key=lambda cluster_id: (
            labels_by_cluster[cluster_id].casefold(),
            float(centroids[cluster_id][0]),
            float(centroids[cluster_id][1]),
        ),
    )
    keyword_ids = {
        cluster_id: f"keyword-{index + 1:02d}"
        for index, cluster_id in enumerate(ordered_clusters)
    }

    allocations = _detail_cluster_allocations(
        labels,
        coordinates,
        target_count=detail_keyword_count,
        minimum_cluster_size=minimum_detail_cluster_size,
    )
    detail_labels = numpy.empty(len(included), dtype=int)
    detail_parents: dict[int, int] = {}
    next_detail_cluster = 0
    for cluster_id in sorted(labels_by_cluster):
        positions = numpy.flatnonzero(labels == cluster_id)
        cluster_coordinates = coordinates[positions]
        allocation = allocations[cluster_id]
        if allocation == 1:
            local_labels = numpy.zeros(len(positions), dtype=int)
        else:
            local_labels = KMeans(
                n_clusters=allocation,
                n_init=10,
                random_state=RANDOM_STATE + cluster_id + 1,
            ).fit_predict(cluster_coordinates)
            local_labels = _merge_small_clusters(
                cluster_coordinates,
                local_labels,
                minimum_cluster_size=minimum_detail_cluster_size,
            )
        for local_cluster in sorted({int(label) for label in local_labels}):
            detail_labels[positions[local_labels == local_cluster]] = (
                next_detail_cluster
            )
            detail_parents[next_detail_cluster] = cluster_id
            next_detail_cluster += 1

    detail_labels_by_cluster = _cluster_keywords(
        titles,
        detail_labels,
        reserved=set(labels_by_cluster.values()),
    )
    detail_centroids = {
        cluster_id: coordinates[detail_labels == cluster_id].mean(axis=0)
        for cluster_id in detail_labels_by_cluster
    }
    detail_keyword_ids: dict[int, str] = {}
    for parent_cluster in ordered_clusters:
        children = sorted(
            (
                cluster_id
                for cluster_id, parent in detail_parents.items()
                if parent == parent_cluster
            ),
            key=lambda cluster_id: (
                detail_labels_by_cluster[cluster_id].casefold(),
                float(detail_centroids[cluster_id][0]),
                float(detail_centroids[cluster_id][1]),
            ),
        )
        for index, cluster_id in enumerate(children):
            detail_keyword_ids[cluster_id] = (
                f"{keyword_ids[parent_cluster]}-{index + 1:02d}"
            )

    output.loc[included_mask, "keyword_id"] = [
        keyword_ids[int(cluster_id)] for cluster_id in labels
    ]
    output.loc[included_mask, "keyword"] = [
        labels_by_cluster[int(cluster_id)] for cluster_id in labels
    ]
    output.loc[included_mask, "detail_keyword_id"] = [
        detail_keyword_ids[int(cluster_id)] for cluster_id in detail_labels
    ]
    output.loc[included_mask, "detail_keyword"] = [
        detail_labels_by_cluster[int(cluster_id)] for cluster_id in detail_labels
    ]
    return output
