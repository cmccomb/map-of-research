"""Deterministic topic keywords for the publication landscape."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

KEYWORD_MODEL_VERSION = "visible-tsne-kmeans-ctfidf-v1"
DEFAULT_KEYWORD_COUNT = 30
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
        "based",
        "carnegie",
        "cmu",
        "conference",
        "engineering",
        "ieee",
        "journal",
        "mellon",
        "method",
        "methods",
        "model",
        "models",
        "new",
        "paper",
        "proceedings",
        "research",
        "study",
        "using",
        "volume",
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


def _cluster_keywords(titles: list[str], labels: Any) -> dict[int, str]:
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
    used: set[str] = set()
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
            ),
            "",
        )
        if not label:
            label = _fallback_keyword(cluster_titles, used=used)
        used.add(label)
        results[cluster_id] = label
    return results


def add_publication_keywords(
    works: Any,
    *,
    keyword_count: int = DEFAULT_KEYWORD_COUNT,
) -> Any:
    """Assign visible t-SNE regions concise keywords without dropping works."""

    import numpy
    from sklearn.cluster import KMeans

    if keyword_count < 1:
        raise ValueError("keyword_count must be positive")

    output = works.copy()
    output["keyword_model_version"] = KEYWORD_MODEL_VERSION
    output["keyword_id"] = ""
    output["keyword"] = ""
    included_mask = output["map_eligible"] & output["department_ids"].apply(bool)
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
    labels_by_cluster = _cluster_keywords(titles, labels)
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
    output.loc[included_mask, "keyword_id"] = [
        keyword_ids[int(cluster_id)] for cluster_id in labels
    ]
    output.loc[included_mask, "keyword"] = [
        labels_by_cluster[int(cluster_id)] for cluster_id in labels
    ]
    return output
